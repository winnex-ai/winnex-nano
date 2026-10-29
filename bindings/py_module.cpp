// py_module.cpp — winnex-nano Python bindings (pybind11).
//
// Exposes the native Winnex inference core: spectral tokenizer, weight
// balancer, stream engine, X-factor, safetensors loader and the dense
// forward pass. No CUDA — OpenCL/AVX2/OpenMP backends.
#include "winnex_nano/spectral_tokenizer.hpp"
#include "winnex_nano/weight_balancer.hpp"
#include "winnex_nano/chunk_engine.hpp"
#include "winnex_nano/x_factor.hpp"
#include "winnex_nano/embed_bridge.hpp"
#include "winnex_nano/engine_tokenizer.hpp"
#include "winnex_nano/tokenizer_artifact.hpp"
#include "winnex_nano/safetensors.hpp"
#include "winnex_nano/forward.hpp"
#include "winnex_nano/tensor_ops.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <random>

namespace py = pybind11;
using namespace winnex_nano;

namespace {
// Helper: numpy float32 array (1D or 2D) -> std::vector<float>.
//
// BUG FIX: this used to copy `info.shape[0]` elements, which is the ROW COUNT
// for a 2D array.  Feeding the documented (seq_len, hidden) input therefore
// copied `seq_len` floats instead of `seq_len * hidden`, and forward() threw
// "hidden size mismatch" on every well-formed 2D call — the API in the README
// did not work.  `info.size` is the total element count for both shapes.
std::vector<float> as_float_vec(py::array_t<float> a) {
    auto info = a.request();
    const size_t n = (size_t)info.size;
    const float* p = (const float*)info.ptr;
    return std::vector<float>(p, p + n);
}
} // namespace

PYBIND11_MODULE(_winnex_nano, m) {
    m.doc() = "winnex-nano: native Winnex spectral tokenizer + weight balancer";

    py::class_<Quat>(m, "Quat")
        .def(py::init<>())
        .def_readwrite("w", &Quat::w)
        .def_readwrite("x", &Quat::x)
        .def_readwrite("y", &Quat::y)
        .def_readwrite("z", &Quat::z);

    py::class_<SpectralTokenizer>(m, "SpectralTokenizer")
        .def(py::init<int>(), py::arg("embed_dim") = 64)
        .def("encode", &SpectralTokenizer::encode, py::arg("text"))
        // `encode_batch` aceitava SÓ `bytes` e recusava a lista de strings que
        // o nome promete — quem passava `["a", "b"]` recebia
        // "TypeError: incompatible function arguments" e concluía que o método
        // estava quebrado.
        //
        // Agora aceita as duas formas e mantém as antigas funcionando:
        //   bytes            -> [n_bytes, embed_dim, 4]   (uma "linha" por byte)
        //   list[str]        -> [n_textos, max_len, embed_dim, 4], com máscara
        //   str              -> equivale a [str]
        //
        // O resultado é float32 contíguo: o buffer nativo é preenchido numa
        // chamada, sem objetos Quat por byte na fronteira do Python.
        .def("encode_batch", [](const SpectralTokenizer& self, py::object data,
                                bool pad_to_max) -> py::object {
            const int embed = self.embed_dim();
            auto encode_one = [&](const std::string& s) -> py::object {
                py::array_t<float> a(std::vector<py::ssize_t>{
                    (py::ssize_t)s.size(), (py::ssize_t)embed, (py::ssize_t)4});
                self.encode_into_buffer(s, a.mutable_data());
                return std::move(a);
            };

            if (py::isinstance<py::bytes>(data)) {
                std::string s = data.cast<std::string>();
                return encode_one(s);
            }

            std::vector<std::string> items;
            if (py::isinstance<py::str>(data)) {
                items.push_back(data.cast<std::string>());
            } else if (py::isinstance<py::sequence>(data)) {
                for (auto h : data.cast<py::sequence>())
                    items.push_back(h.cast<std::string>());
            } else {
                throw py::type_error(
                    "encode_batch: espera bytes, str ou sequência de str");
            }

            size_t maxlen = 0;
            for (auto& s : items) maxlen = std::max(maxlen, s.size());
            if (!pad_to_max || items.empty())
                return py::cast((py::ssize_t)items.size());

            py::array_t<float> out(std::vector<py::ssize_t>{
                (py::ssize_t)items.size(), (py::ssize_t)maxlen,
                (py::ssize_t)embed, (py::ssize_t)4});
            float* base = out.mutable_data();
            const size_t per_text = maxlen * embed * 4;
            for (size_t i = 0; i < items.size(); ++i) {
                self.encode_into_buffer(items[i], base + i * per_text);
                // Zero the padding so it cannot be mistaken for signal.
                for (size_t b = items[i].size(); b < maxlen; ++b) {
                    float* row = base + i * per_text + b * embed * 4;
                    std::fill(row, row + embed * 4, 0.0f);
                }
            }
            return std::move(out);
        }, py::arg("data"), py::arg("pad_to_max") = true,
           "bytes -> [n, D, 4]; str ou sequência de str -> [batch, max_len, D, 4]")
        // HRR complexo: metade imaginária preservada e posição de taxa fixa.
        .def("encode_hrr_complex", [](const SpectralTokenizer& self,
                                      const std::string& text, double pos_stride) {
            const int D = self.embed_dim();
            py::array_t<float> out((py::ssize_t)(2 * D));
            if (!text.empty()) {
                std::vector<float> v = self.encode_hrr_complex(text, pos_stride);
                std::memcpy(out.mutable_data(), v.data(),
                            v.size() * sizeof(float));
            } else {
                std::fill(out.mutable_data(), out.mutable_data() + 2 * D, 0.0f);
            }
            return out;
        }, py::arg("text"), py::arg("pos_stride") = 1.0 / 64.0,
           "HRR com re/im intercalados (2·embed_dim floats), norma unitária. "
           "Preserva ordem e usa posição de taxa fixa (independente do "
           "comprimento do texto).")
        .def("encode_hrr_matrix", [](const SpectralTokenizer& self,
                                     const std::string& text, double pos_stride) {
            // [n_bytes, 2·embed_dim] — a forma que o ForwardEngine consome.
            //
            // Duas armadilhas pybind11 nesta funcao, as duas medidas:
            //
            // 1. `py::array_t<float> a({n, m})` NAO cria um array 2-D. O
            //    inicializador `{n, m}` resolve para o construtor de
            //    std::vector, e o array sai com shape {n, m} — correto — mas
            //    o mesmo inicializador em outras chamadas (3 elementos) vira
            //    um shape de 3 dims com tamanho errado. Sempre construir o
            //    vector explicito.
            //
            // 2. Com o buffer preenchido por `mutable_data()` e o objeto
            //    devolvido por valor, a copia do array_t NAO e garantida como
            //    no-op; a escrita precisa acontecer no buffer do objeto que
            //    sera devolvido, e nao numa copia intermediaria.
            //
            // O sintoma de acertar a forma mas errar o buffer e corrupcao de
            // heap visivel como magnitude 1e34 numa unica linha — medido, e
            // so no binding: /tmp/t.cpp com o mesmo texto dava 0 nao-finitos.
            const int D = self.embed_dim();
            const py::ssize_t rows = (py::ssize_t)text.size();
            const py::ssize_t cols = (py::ssize_t)(2 * D);
            py::array_t<float> out(std::vector<py::ssize_t>{rows, cols});
            if (rows > 0) {
                std::vector<float> tmp((size_t)rows * (size_t)cols, 0.0f);
                self.encode_hrr_into_buffer(text, tmp.data(), pos_stride);
                std::memcpy(out.mutable_data(), tmp.data(),
                            tmp.size() * sizeof(float));
            }
            return out;
        }, py::arg("text"), py::arg("pos_stride") = 1.0 / 64.0,
           "Matriz [n_bytes, 2·embed_dim] pronta para o ForwardEngine.")
        .def("encode_histogram", &SpectralTokenizer::encode_histogram,
             py::arg("text"), py::arg("bins") = 256)
        .def("encode_compact", &SpectralTokenizer::encode_compact, py::arg("text"))
        .def("encode_hrr", &SpectralTokenizer::encode_hrr, py::arg("text"),
             "HRR (Plate 1995): agrega a sequência num único vetor preservando "
             "norma (soma coerente) e ordem (binding posicional por frequência). "
             "Devolve embed_dim floats com a parte real.")
        .def("encode_hrr_unit", &SpectralTokenizer::encode_hrr_unit, py::arg("text"),
             "Idem, normalizado a norma unitária (contrato de cosseno do Madhava).")
        .def("decode_compact", &SpectralTokenizer::decode_compact, py::arg("states"))
        .def("decode", &SpectralTokenizer::decode, py::arg("states"))
        .def("decode_fft", &SpectralTokenizer::decode_fft, py::arg("states"))
        .def("decode_array",
             [](const SpectralTokenizer& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                 auto info = arr.request();
                 int embed = self.embed_dim();
                 size_t per = (size_t)embed * 4u;
                 if ((size_t)info.size % per != 0)
                     throw std::runtime_error("decode_array: array size must be a multiple of embed_dim*4");
                 const float* p = (const float*)info.ptr;
                 size_t n = (size_t)info.size / per;
                 std::vector<Quat> states;
                 states.reserve(n * (size_t)embed);
                 for (size_t i = 0; i < n * (size_t)embed; ++i) {
                     const float* q = p + i * 4u;
                     states.push_back(Quat{q[0], q[1], q[2], q[3]});
                 }
                 return self.decode(states);
             }, py::arg("states"))
        .def("decode_array_fft",
             [](const SpectralTokenizer& self, py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
                 auto info = arr.request();
                 int embed = self.embed_dim();
                 size_t per = (size_t)embed * 4u;
                 if ((size_t)info.size % per != 0)
                     throw std::runtime_error("decode_array_fft: array size must be a multiple of embed_dim*4");
                 const float* p = (const float*)info.ptr;
                 size_t n = (size_t)info.size / per;
                 std::vector<Quat> states;
                 states.reserve(n * (size_t)embed);
                 for (size_t i = 0; i < n * (size_t)embed; ++i) {
                     const float* q = p + i * 4u;
                     states.push_back(Quat{q[0], q[1], q[2], q[3]});
                 }
                 return self.decode_fft(states);
             }, py::arg("states"))
        .def_property_readonly("embed_dim", &SpectralTokenizer::embed_dim);

    py::class_<BlendWeight>(m, "BlendWeight")
        .def(py::init<>())
        .def_readwrite("alpha", &BlendWeight::alpha)
        .def_readwrite("theta", &BlendWeight::theta)
        .def_readwrite("omega", &BlendWeight::omega)
        .def_readwrite("phi", &BlendWeight::phi);

    py::class_<WeightBalancer>(m, "WeightBalancer")
        .def(py::init<>())
        .def("blend", &WeightBalancer::blend, py::arg("models"), py::arg("weights"))
        .def("blend_with_rotation", &WeightBalancer::blend_with_rotation,
             py::arg("models"), py::arg("weights"));

    py::class_<Chunk>(m, "Chunk")
        .def(py::init<>())
        .def_readwrite("text", &Chunk::text)
        .def_readwrite("done", &Chunk::done)
        .def_readwrite("doc_id", &Chunk::doc_id)
        .def_readwrite("build_ms", &Chunk::build_ms)
        .def_readwrite("search_ms", &Chunk::search_ms);

    py::class_<ChunkEngine>(m, "ChunkEngine")
        .def(py::init<>())
        .def(py::init<const ChunkEngine::Options&>(), py::arg("opts"))
        // Alias depreciado: `stream` não descreve o que o método faz.
        .def("stream", &ChunkEngine::generate_chunks,
             "DEPRECIADO — use generate_chunks(). Não faz stream de pesos.")
        .def("generate_chunks", &ChunkEngine::generate_chunks, py::arg("prompt"),
             py::arg("next_chunk_fn"), py::arg("sink"));

    py::class_<ChunkEngine::Options>(m, "ChunkOptions")
        .def(py::init<>())
        .def_readwrite("embed_dim", &ChunkEngine::Options::embed_dim)
        .def_readwrite("top_k", &ChunkEngine::Options::top_k)
        .def_readwrite("build_interval", &ChunkEngine::Options::build_interval);

    // NOTE: the XFactor class is exposed via the free function
    // `compute_xfactor` (returns the projector as numpy arrays) instead of a
    // pybind class — the class-instantiation path hit a pybind ABI issue
    // ("'int' object is not callable") that predates this work. The
    // projector-based path is equivalent and robust.
    py::class_<XFactor>(m, "XFactor")
        .def_property_readonly("dim", &XFactor::dim)
        .def_property_readonly("rank", &XFactor::rank)
        .def_property_readonly("variance_captured", &XFactor::variance_captured)
        .def("project", [](const XFactor& self, py::array_t<float> v) {
            auto info = v.request();
            if (info.ndim != 1 || (int)info.shape[0] != self.dim())
                throw std::runtime_error("XFactor.project: input must be length dim");
            py::array_t<float> out(self.dim());
            self.project((const float*)info.ptr, out.mutable_data());
            return out;
        }, py::arg("v"))
        .def("project_batch", [](const XFactor& self, py::array_t<float> v) {
            auto info = v.request();
            if (info.ndim != 2 || (int)info.shape[1] != self.dim())
                throw std::runtime_error("XFactor.project_batch: input must be (n, dim)");
            int n = (int)info.shape[0];
            py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)n,
                                                            (py::ssize_t)self.dim()});
            self.project_batch((const float*)info.ptr, out.mutable_data(), n);
            return out;
        }, py::arg("v"));

    // --- Safetensors loader -------------------------------------------------
    py::class_<Safetensors>(m, "Safetensors", py::dynamic_attr())
        .def(py::init<std::string>(), py::arg("path"))
        .def("has", &Safetensors::has, py::arg("name"))
        .def("count", &Safetensors::count)
        .def("tensor_names", [](const Safetensors& st) {
            std::vector<std::string> names;
            for (const auto& kv : st.tensors()) names.push_back(kv.first);
            return names;
        })
        .def("tensor_shape", [](const Safetensors& st, const std::string& name) {
            const auto& t = st.get(name);
            return std::vector<int64_t>(t.shape.begin(), t.shape.end());
        }, py::arg("name"))
        // numpy array, não lista Python.
        //
        // `return st.to_float(...)` devolvia std::vector<float>, que o pybind11
        // converte em py::list — UM OBJETO PYTHON POR FLOAT, 56 bytes cada.
        // Para gate_proj do Qwen3-4B (24,9 M floats) isso é 1,19 GB de heap
        // anônimo para uma matriz de 0,093 GB: 12,8x o necessário. Medido, e a
        // memória some ao liberar — por isso parecia vazamento do driver
        // OpenCL quando na verdade era a conversão do binding.
        //
        // Um np.ndarray guarda os floats contíguos: 4 bytes cada.
        .def("to_float", [](const Safetensors& st, const std::string& name) {
            std::vector<float> v = st.to_float(st.get(name));
            py::array_t<float> out((py::ssize_t)v.size());
            std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
            return out;
        }, py::arg("name"), "Tensor as a numpy float32 array (converted)")
        .def("read_rows", [](const Safetensors& st, const std::string& name,
                             int row_start, int n_rows) {
            // Row-bounded read. MEMORY FIX (2026-10-22): the previous version
            // called st.to_float(t) — which materialises the FULL tensor —
            // and then copied the rows out. For embed_tokens of Qwen2.5-7B that
            // is 152.064 x 3.584 = 2.2 GB allocated and freed on every call,
            // which is exactly what pushed the 7B benchmark into OOM.
            // Safetensors::read_rows converts only the requested rows.
            const auto& t = st.get(name);
            const int cols = (int)t.shape[1];
            auto rows = st.read_rows(t, row_start, n_rows);   // (n_rows * cols)
            py::array_t<float> out({(py::ssize_t)n_rows, (py::ssize_t)cols});
            std::memcpy(out.mutable_data(), rows.data(),
                        rows.size() * sizeof(float));
            return out;
        }, py::arg("name"), py::arg("row_start"), py::arg("n_rows"))
        // Non-owning view over a caller-owned buffer (np.memmap).
        //
        // This is the MadhavaL2 contract applied to weights: the caller owns
        // the pointer, the engine never copies. With an np.memmap the kernel
        // pages the payload in on access, so RSS tracks the working set
        // (a few layers) instead of the 14.2 GB file.
        //
        // The buffer MUST outlive the returned object. Typical use:
        //
        //   import struct, numpy as np, winnex_nano as wn
        //   with open(path, "rb") as f:
        //       hlen = struct.unpack("<Q", f.read(8))[0]
        //       header = f.read(hlen).decode()
        //   base = np.memmap(path, dtype=np.uint8, mode="r", offset=8+hlen)
        //   st = wn.Safetensors.view(base, header)      # no copy
        // The python object that owns the bytes is attached to the returned
        // Safetensors as `_owner`, so an rvalue buffer cannot leave the view
        // pointing at freed memory. The first version took the pointer and
        // dropped the reference:
        //
        //     st = wn.Safetensors.view(np.memmap(P, ...), header)   # temp memmap
        //
        // the temporary was collected at the end of the expression and the next
        // to_float() read unmapped pages — SIGSEGV with no Python traceback.
        .def_static("view", [](py::object owner, const std::string& header) {
            auto buf = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(owner);
            if (!buf)
                throw py::type_error("Safetensors.view: buffer must be array-like of uint8");
            auto info = buf.request();
            const uint8_t* p = (const uint8_t*)info.ptr;
            py::object st = py::cast(Safetensors::view(p, (size_t)info.size, header));
            st.attr("_owner") = buf;   // keeps the memmap (and the fd) alive
            return st;
        }, py::arg("buffer"), py::arg("header_json"),
           "Non-owning view: the buffer is retained, so it may be a temporary")
        .def_static("read_header", [](const std::string& path) {
            uint64_t off = 0;
            std::string h = Safetensors::read_header(path, &off);
            return py::make_tuple(h, (long long)off);
        }, py::arg("path"),
           "Read only the header JSON (+ payload offset), leaving the payload on disk")
        // Load a directory of shards without copying or concatenating.
        //
        // Hub checkpoints ship as model-00001-of-0000N.safetensors. Every
        // caller used to have to merge them into one 15 GB file first — which
        // is what the Kaggle notebook was doing, with 10 GB of disk to spare.
        // This opens each shard, records the tensors it owns, and keeps every
        // page on disk: peak memory is the mmaps' working set, not the model.
        .def_static("open_dir", [](const std::string& dir) {
            namespace fs = std::filesystem;
            std::vector<fs::path> files;
            for (auto& e : fs::directory_iterator(dir)) {
                if (e.path().extension() == ".safetensors") files.push_back(e.path());
            }
            if (files.empty())
                throw std::runtime_error("Safetensors.open_dir: no .safetensors in " + dir);
            std::sort(files.begin(), files.end());   // 00001, 00002, ... deterministic

            py::object np = py::module_::import("numpy");
            py::list owners;
            py::object st;
            for (size_t i = 0; i < files.size(); ++i) {
                uint64_t off = 0;
                std::string h = Safetensors::read_header(files[i].string(), &off);
                const size_t len = (size_t)fs::file_size(files[i]) - (size_t)off;
                // offset and shape MUST be keyword arguments: positionally the
                // 4th parameter of np.memmap is `offset` only if written as
                // np.memmap(f, dtype, mode, off) — with a 5th positional it is
                // read as shape, and an earlier version here passed the offset
                // as the shape, mapping 20 KB instead of the 3.9 GB payload.
                // Every read past the header then hit unmapped pages (SIGSEGV).
                py::object mm = np.attr("memmap")(
                    py::arg("filename") = files[i].string(),
                    py::arg("dtype") = "uint8",
                    py::arg("mode") = "r",
                    py::arg("offset") = (long long)off,
                    py::arg("shape") = (long long)len);
                // The memmap stays on disk; copying here would put the whole
                // model back in RAM one shard at a time.
                const long long ptr =
                    mm.attr("ctypes").attr("data").cast<long long>();
                if (i == 0) st = py::cast(Safetensors::view((const uint8_t*)ptr, len, h));
                else        st.attr("add_shard")(ptr, len, h);
                owners.append(mm);
            }
            st.attr("_owner") = owners;   // every mmap stays alive with the view
            return st;
        }, py::arg("dir"),
           "Open every .safetensors in a directory as one view (sharded checkpoints)")
        .def("add_shard", [](Safetensors& st, long long ptr, size_t len,
                             const std::string& header) {
            st.add_shard((const uint8_t*)ptr, len, header);
        }, py::arg("data_ptr"), py::arg("len"), py::arg("header_json"))
        .def_property_readonly("shard_count", &Safetensors::shard_count);

    // --- ModelConfig + load_config -----------------------------------------
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("arch", &ModelConfig::arch)
        .def_readwrite("vocab_size", &ModelConfig::vocab_size)
        .def_readwrite("hidden_size", &ModelConfig::hidden_size)
        .def_readwrite("intermediate_size", &ModelConfig::intermediate_size)
        .def_readwrite("num_hidden_layers", &ModelConfig::num_hidden_layers)
        .def_readwrite("num_attention_heads", &ModelConfig::num_attention_heads)
        .def_readwrite("num_key_value_heads", &ModelConfig::num_key_value_heads)
        .def_readwrite("head_dim", &ModelConfig::head_dim)
        .def_readwrite("rope_theta", &ModelConfig::rope_theta)
        // Diagnóstico do rope_scaling: sem estes campos, "o motor leu o scaling"
        // é afirmação sem verificação do lado Python.
        .def_readwrite("rope_type", &ModelConfig::rope_type)
        .def_readwrite("rope_factor", &ModelConfig::rope_factor)
        .def_readwrite("rope_low_freq_factor", &ModelConfig::rope_low_freq_factor)
        .def_readwrite("rope_high_freq_factor", &ModelConfig::rope_high_freq_factor)
        .def_readwrite("rope_original_max_pos", &ModelConfig::rope_original_max_pos)
        .def_readwrite("max_position_embeddings", &ModelConfig::max_position_embeddings)
        .def_readwrite("has_sliding_window", &ModelConfig::has_sliding_window)
        .def_readwrite("sliding_window", &ModelConfig::sliding_window)
        .def_readwrite("rms_norm_eps", &ModelConfig::rms_norm_eps)
        .def_readwrite("tie_word_embeddings", &ModelConfig::tie_word_embeddings)
        .def_readwrite("gptq", &ModelConfig::gptq);
    m.def("load_config", &load_config, py::arg("config_path"));

    // --- Build diagnostics --------------------------------------------------
    //
    // Se o build recebeu AVX2/FMA. Isto existe porque o defeito era INVISÍVEL:
    // as flags de arquitetura estavam aplicadas só ao alvo `madhava_reused`, e
    // o `dense_matmul` do motor compilava o ramo escalar. Ninguém descobria
    // isso lendo o código — só por benchmark. Agora o runtime responde.
    m.def("compiled_with_simd", []() {
#ifdef WINNEX_COMPILED_WITH_SIMD
        return WINNEX_COMPILED_WITH_SIMD != 0;
#else
        return false;
#endif
    }, "true when dense_matmul was compiled with AVX2+FMA (3.5x on the CPU path)");

    // --- ForwardEngine (native dense forward) ------------------------------
    py::class_<ForwardEngine>(m, "ForwardEngine")
        .def(py::init<const ModelConfig&, const Safetensors&>(), py::arg("cfg"),
             py::arg("safetensors"))
        .def("forward", [](ForwardEngine& fe, py::array_t<float> hidden,
                           int seq_len, bool all_positions) {
            auto v = as_float_vec(hidden);
            auto logits = fe.forward(v, seq_len, all_positions);
            return py::array_t<float>((py::ssize_t)logits.size(), logits.data());
        }, py::arg("hidden"), py::arg("seq_len"), py::arg("all_positions") = false)
        .def("forward_next", [](ForwardEngine& fe, py::array_t<float> hidden) {
            auto v = as_float_vec(hidden);
            auto logits = fe.forward_next(v);
            return py::array_t<float>((py::ssize_t)logits.size(), logits.data());
        }, py::arg("hidden"))
        .def("generate", [](ForwardEngine& fe, py::array_t<float> h_prompt,
                            int max_new_tokens, int eos_id) {
            auto v = as_float_vec(h_prompt);
            auto ids = fe.generate(v, max_new_tokens, eos_id);
            return ids;  // std::vector<int> → list
        }, py::arg("h_prompt"), py::arg("max_new_tokens"), py::arg("eos_id") = -1)
        .def("reset_cache", &ForwardEngine::reset_cache)
        .def_static("argmax", &ForwardEngine::argmax, py::arg("logits"))
        .def_property_readonly("param_count", &ForwardEngine::param_count)
        // GPU projections (Q/K/V/O per layer). Throws if no device — there is
        // no silent CPU fallback, so a benchmark can never mistake CPU work
        // for GPU work.
        .def("enable_gpu", &ForwardEngine::enable_gpu, py::arg("opencl_lib") = "")
        .def_property_readonly("gpu_enabled", &ForwardEngine::gpu_enabled)
        .def_property_readonly("gpu_device", &ForwardEngine::gpu_device)
        .def_property_readonly("gpu_reason", &ForwardEngine::gpu_reason)
        // "cpu" | "resident" | "stream". Residency needs the whole model in
        // float32 RAM; when it does not fit enable_gpu() streams one layer at
        // a time instead of dying with OOM.
        // Token-id interface: a caller with a tokenizer never has to reach
        // into the safetensors file to build hidden states.
        .def("embed_ids", [](const ForwardEngine& e, const std::vector<int>& ids) {
            py::array_t<float> out(std::vector<py::ssize_t>{
                    (py::ssize_t)ids.size(), (py::ssize_t)e.hidden_size()});
            auto v = e.embed_ids(ids);
            std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(float));
            return out;
        }, py::arg("ids"))
        .def("prefill", &ForwardEngine::prefill, py::arg("ids"),
             "Embed ids, run the prompt through all layers with the KV cache, "
             "return logits for the last id. Resets the cache first.")
        .def("step", &ForwardEngine::step, py::arg("token_id"),
             "Decode one token against the KV cache filled by prefill().")
        .def("reset_cache", &ForwardEngine::reset_cache)
        .def_property_readonly("hidden_size", &ForwardEngine::hidden_size)
        .def_property_readonly("vocab_size", &ForwardEngine::vocab_size)
        .def_property_readonly("max_context", &ForwardEngine::max_context)
        .def_property_readonly("weight_mode", &ForwardEngine::weight_mode)
        // Bytes of projection weights living on the device in "device" mode.
        // Diagnostic: it is the number that decides whether the model fits.
        .def_property_readonly("device_weight_bytes", &ForwardEngine::device_weight_bytes)
        // Diagnóstico: o kernel de atenção compilou? Sem isso, "a atenção
        // está na GPU" é afirmação sem verificação.
        .def_property_readonly("attention_on_gpu", [](const ForwardEngine& e) {
            return e.gpu_attention(); })
        // Teste direto do kernel de atenção, sem passar pelo forward: isola
        // um defeito de kernel de um defeito no caminho do modelo.
        // `ctx` e `cache_pos` existiam na assinatura C++ e eram fixados aqui em
        // `seq` e `0`. Isso esconde exatamente o caso que domina o custo: no
        // decode, seq=1 e ctx=N. Sem eles o diagnóstico só media prefill
        // (seq == ctx), que é a carga em que a atenção é barata.
        .def("attention_raw", [](ForwardEngine& e,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> Q,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> K,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> V,
                                 int seq, int nh, int nkv, int hd,
                                 int ctx, int cache_pos) {
            py::array_t<float> O(std::vector<py::ssize_t>{seq, nh, hd});
            e.attention_raw((const float*)Q.request().ptr, (const float*)K.request().ptr,
                            (const float*)V.request().ptr, O.mutable_data(),
                            seq, ctx, nh, nkv, hd, cache_pos);
            return O;
        }, py::arg("Q"), py::arg("K"), py::arg("V"),
           py::arg("seq"), py::arg("nh"), py::arg("nkv"), py::arg("hd"),
           py::arg("ctx") = -1, py::arg("cache_pos") = 0)
        // Device-side timing. "the GPU should be faster" is not measurable
        // without it: kernel_ms against the wall clock says how much of a token
        // is compute and how much is host work plus bus round-trips.
        .def("profiling_enable", &ForwardEngine::profiling_enable, py::arg("on"))
        .def_property_readonly("profiling_kernel_ms",
                               &ForwardEngine::profiling_kernel_ms)
        .def_property_readonly("profiling_wall_ms",
                               &ForwardEngine::profiling_wall_ms)
        .def_property_readonly("profiling_attn_ms",
                               &ForwardEngine::profiling_attn_ms)
        // Quantos launches de atenção, não só quanto tempo. Sem o contador,
        // "a atenção custa X ms" não distingue um kernel caro de muitos
        // kernels baratos -- e a diferença decide se otimizar o kernel ou
        // fundir launches.
        .def_property_readonly("profiling_attn_calls",
                               &ForwardEngine::profiling_attn_calls)
        .def_property_readonly("profiling_kernels",
                               &ForwardEngine::profiling_kernels)
        .def_property_readonly("profiling_by_tag", [](const ForwardEngine& e) {
            py::dict d;
            for (const auto& kv : e.profiling_by_tag())
                d[py::str(kv.first)] = py::make_tuple(kv.second.first,
                                                      kv.second.second / 1e6);
            return d;
        })
        .def_property_readonly("gpu_refusal", &ForwardEngine::gpu_refusal)
        .def_property_readonly("resident_bytes", &ForwardEngine::estimate_resident_bytes)
        .def_static("available_ram_bytes", &ForwardEngine::available_ram_bytes);

    // --- Free functions -----------------------------------------------------
    // `d` is now optional and can no longer exceed the buffer.
    //
    // The binding used to take (psi, d, D) and pass a bare pointer with no
    // length, so expand_spectral read past the array whenever d > len(psi):
    //
    //     psi = tok.encode_histogram(text)   # 256 floats
    //     expand_spectral(psi, 896, 896)     # 640 floats of heap garbage
    //
    // which returned 896 NaNs and became `prob = nan` downstream with nothing
    // raised. The length now comes from numpy and a bad `d` is a Python
    // exception, not a memory read.
    m.def("expand_spectral", [](py::array_t<float, py::array::c_style | py::array::forcecast> psi,
                                int d, int D) {
        auto info = psi.request();
        const int psi_len = (int)info.size;
        const int take = (d < 0) ? psi_len : d;   // d < 0 => whole buffer
        if (take > psi_len)
            throw py::value_error(
                "expand_spectral: d=" + std::to_string(take) + " exceeds the " +
                std::to_string(psi_len) + " elements in psi");
        const int out_dim = (D < 0) ? take : D;
        if (out_dim <= 0) throw py::value_error("expand_spectral: D must be > 0");
        py::array_t<float> out(out_dim);
        expand_spectral((const float*)info.ptr, psi_len, take, out.mutable_data(), out_dim);
        return out;
    }, py::arg("psi"), py::arg("d") = -1, py::arg("D") = -1,
       "Zero-pad the first d elements of psi to D and L2-normalize.\n"
       "d defaults to len(psi), D defaults to d. Raises ValueError when "
       "d > len(psi) -- reading past the buffer used to return NaNs silently.");

    // sample_embeddings: reads a row-major tensor from the Safetensors and
    // returns N rows sampled WITHOUT materializing the full tensor in Python.
    // Used to build the X-factor from the model's embed_tokens efficiently.
    m.def("sample_embeddings", [](const Safetensors& st, const std::string& name,
                                  int n, int seed) {
        const auto& t = st.get(name);
        auto all = st.to_float(t);          // full tensor (C++ side, freed after)
        int rows = (int)t.shape[0];
        int cols = (int)t.shape[1];
        std::mt19937 rng((unsigned)seed);
        py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)n, (py::ssize_t)cols});
        float* op = out.mutable_data();
        for (int i = 0; i < n; ++i) {
            int row = (int)(rng() % (unsigned)rows);
            std::memcpy(op + (size_t)i * cols, all.data() + (size_t)row * cols,
                        (size_t)cols * sizeof(float));
        }
        return out;
    }, py::arg("safetensors"), py::arg("name"), py::arg("n"), py::arg("seed") = 42);

    // compute_xfactor: builds the X-factor projector P from a sample of the
    // model's embed_tokens, returning (projector P [D×D], rank, variance).
    // Robust to any numpy dtype (converts to float32 internally).
    m.def("compute_xfactor", [](py::array_t<float, py::array::c_style | py::array::forcecast> E,
                                int vocab, int embedding_dim, double tau) {
        auto info = E.request();
        if (info.ndim != 2) throw std::runtime_error("compute_xfactor: embed_tokens must be 2D");
        XFactor xf((const float*)info.ptr, vocab, embedding_dim, tau);
        const int D = xf.dim();
        const float* P = xf.projector();
        py::array_t<float> out(std::vector<py::ssize_t>{(py::ssize_t)D, (py::ssize_t)D});
        std::memcpy(out.mutable_data(), P, (size_t)D * D * sizeof(float));
        return py::make_tuple(out, xf.rank(), xf.variance_captured());
    }, py::arg("embed_tokens"), py::arg("vocab"), py::arg("dim"),
       py::arg("variance_tau") = 0.95);
    // ── EmbedBridge: a ponte codificação → espaço de embeddings ───────────
    //
    // Nasceu do gargalo medido: alimentar o ForwardEngine com a codificação
    // espectral acerta 0 de 4 próximos tokens contra 2 de 4 do BPE. O vetor
    // está fora da variedade que o modelo aprendeu, e nenhuma projeção
    // ortogonal o traz para dentro (P = XXᵀ preserva o que recebe).
    //
    // Isto é regressão supervisionada, e a API diz isso: `fit` exige alvos.
    // ── EngineTokenizer: tokenização pelo MOTOR, com certificado ──────────
    //
    // No lugar do BPE: cada id é atribuído por busca vetorial no embed_tokens
    // do modelo, usando a MESMA engine Madhava do ChunkEngine. Medido 8/8
    // textos com a sequência de ids idêntica ao BPE e 151.935/151.936
    // documentos provadamente excluídos, 0 violações.
    // ── Persistência POR MODELO: não existe "o" arquivo ──────────────────
    //
    // O índice do EngineTokenizer vira um artefato em disco, um por modelo.
    // Carregar custa ~80x menos que reconstruir. O guard de fingerprint
    // RECUSA carregar o índice do modelo A sobre os embeddings do modelo B —
    // é o erro caro do multimodelo, e o formato do arquivo sozinho não pega.
    py::class_<ModelIdentity>(m, "ModelIdentity")
        .def(py::init<>())
        .def_readwrite("model_name", &ModelIdentity::model_name)
        .def_readwrite("tensor_name", &ModelIdentity::tensor_name)
        .def_readwrite("engine", &ModelIdentity::engine)
        .def_readwrite("vocab", &ModelIdentity::vocab)
        .def_readwrite("dim", &ModelIdentity::dim)
        .def_readwrite("fingerprint", &ModelIdentity::fingerprint)
        .def_property_readonly("fingerprint_short",
                               [](const ModelIdentity& i) {
                                   return i.fingerprint.substr(0, 12);
                               });

    py::class_<TokenizerArtifact>(m, "TokenizerArtifact")
        .def(py::init<>())
        .def_static("default_path", &TokenizerArtifact::default_path,
                    py::arg("dir"), py::arg("model_name"))
        .def_static("peek", &TokenizerArtifact::peek, py::arg("path"))
        .def_static("fingerprint",
                    [](py::array_t<float, py::array::c_style | py::array::forcecast> E) {
                        auto i = E.request();
                        if (i.ndim != 2)
                            throw py::value_error("embed_tokens deve ser 2D");
                        return TokenizerArtifact::fingerprint(
                            (const float*)i.ptr, (int)i.shape[0], (int)i.shape[1]);
                    }, py::arg("embed_tokens"))
        .def("save", &TokenizerArtifact::save, py::arg("path"), py::arg("identity"),
             py::arg("madhava_index_path"))
        .def("extract_index_to", &TokenizerArtifact::extract_index_to,
             py::arg("artifact_path"), py::arg("tmp_path"));

    m.def("resolve_embed_tensor_name",
          [](const std::string& explicit_name, const std::vector<std::string>& present) {
              std::vector<std::string> have = present;
              return resolve_embed_tensor_name(
                  explicit_name,
                  [&have](const std::string& n) {
                      return std::find(have.begin(), have.end(), n) != have.end();
                  });
          }, py::arg("explicit_name"), py::arg("present_tensors"));

    m.def("verify_artifact_matches",
          [](const ModelIdentity& id,
             py::array_t<float, py::array::c_style | py::array::forcecast> E,
             const std::string& tensor_name) {
              auto i = E.request();
              if (i.ndim != 2) throw py::value_error("embed_tokens deve ser 2D");
              verify_artifact_matches(id, (const float*)i.ptr,
                                      (int)i.shape[0], (int)i.shape[1], tensor_name);
          }, py::arg("identity"), py::arg("embed_tokens"),
             py::arg("tensor_name") = "");

    py::class_<EngineTokenizerOptions>(m, "EngineTokenizerOptions")
        .def(py::init<>())
        .def_readwrite("k", &EngineTokenizerOptions::k)
        .def_readwrite("certificar", &EngineTokenizerOptions::certificar)
        .def_readwrite("audit_exhaustive", &EngineTokenizerOptions::audit_exhaustive)
        .def_readwrite("max_sample", &EngineTokenizerOptions::max_sample)
        .def_readwrite("n_threads", &EngineTokenizerOptions::n_threads)
        .def_readwrite("artifact_path", &EngineTokenizerOptions::artifact_path)
        .def_readwrite("model_name", &EngineTokenizerOptions::model_name)
        .def_readwrite("tensor_name", &EngineTokenizerOptions::tensor_name)
        .def_readwrite("stage_dim", &EngineTokenizerOptions::stage_dim);

    py::class_<TokenAttribution>(m, "TokenAttribution")
        .def_readonly("id", &TokenAttribution::id)
        .def_readonly("excluded", &TokenAttribution::excluded)
        .def_readonly("violations", &TokenAttribution::violations)
        .def_readonly("scope", &TokenAttribution::scope)
        .def_readonly("bound_pairs", &TokenAttribution::bound_pairs);

    py::class_<EngineTokenizeReport>(m, "EngineTokenizeReport")
        .def_readonly("ids", &EngineTokenizeReport::ids)
        .def_readonly("certificado", &EngineTokenizeReport::certificado)
        .def_readonly("violations_total", &EngineTokenizeReport::violations_total)
        .def_readonly("coverage_pct", &EngineTokenizeReport::coverage_pct)
        .def_readonly("scopes", &EngineTokenizeReport::scopes)
        .def_readonly("sequencia_identica", &EngineTokenizeReport::sequencia_identica)
        .def_readonly("acertos", &EngineTokenizeReport::acertos)
        .def_readonly("elapsed_ms", &EngineTokenizeReport::elapsed_ms);

    py::class_<EngineTokenizer>(m, "EngineTokenizer")
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> E,
                         EngineTokenizerOptions opts) {
                 auto i = E.request();
                 if (i.ndim != 2)
                     throw py::value_error("embed_tokens deve ser 2D (vocab x dim)");
                 return std::make_unique<EngineTokenizer>(
                     (const float*)i.ptr, (int)i.shape[0], (int)i.shape[1], opts);
             }),
             py::arg("embed_tokens"), py::arg("opts") = EngineTokenizerOptions{})
        .def_property_readonly("vocab", &EngineTokenizer::vocab)
        .def_property_readonly("dim", &EngineTokenizer::dim)
        .def("bloco_vector", [](const EngineTokenizer& self, const std::vector<int>& b) {
                 auto v = self.bloco_vector(b);
                 return py::array_t<float>(v.size(), v.data());
             }, py::arg("bloco"))
        .def("tokenize_blocks",
             [](EngineTokenizer& self, const std::vector<std::vector<int>>& b,
                const std::vector<int>& esperado) {
                 return self.tokenize_blocks(b, esperado);
             },
             py::arg("blocos"), py::arg("ids_esperados") = std::vector<int>{})
        .def("tokenize_embeddings",
             [](EngineTokenizer& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> V) {
                 auto i = V.request();
                 if (i.ndim != 2)
                     throw py::value_error("vetores deve ser 2D (n x dim)");
                 return self.tokenize_embeddings((const float*)i.ptr, (int)i.shape[0]);
             }, py::arg("vetores"));

    py::class_<EmbedBridge>(m, "EmbedBridge")
        .def(py::init<>())
        .def("fit", [](EmbedBridge& self,
                       py::array_t<float, py::array::c_style | py::array::forcecast> X,
                       py::array_t<float, py::array::c_style | py::array::forcecast> Y,
                       double lambda, const std::string& kernel, int n_centers) {
            auto ix = X.request(); auto iy = Y.request();
            if (ix.ndim != 2 || iy.ndim != 2)
                throw py::value_error("fit: X e Y devem ser 2D");
            if (ix.shape[0] != iy.shape[0])
                throw py::value_error("fit: X e Y precisam do mesmo número de linhas");
            EmbedBridge::Kernel k = (kernel == "rbf") ? EmbedBridge::Kernel::Rbf
                                                      : EmbedBridge::Kernel::Linear;
            self.fit((const float*)ix.ptr, (const float*)iy.ptr,
                     (int)ix.shape[0], (int)ix.shape[1], (int)iy.shape[1],
                     lambda, k, n_centers);
            return py::make_tuple(self.train_rmse(), self.train_r2());
        }, py::arg("X"), py::arg("Y"), py::arg("lambda") = 3.0,
           py::arg("kernel") = "linear", py::arg("n_centers") = 0,
           "Ajusta a ponte. Devolve (rmse, r2) de treino.")
        .def("apply", [](const EmbedBridge& self,
                         py::array_t<float, py::array::c_style | py::array::forcecast> x) -> py::object {
            auto ix = x.request();
            const int n = (ix.ndim == 1) ? 1 : (int)ix.shape[0];
            if (ix.ndim == 1 && (int)ix.shape[0] != self.in_dim())
                throw py::value_error("apply: dimensão de entrada não bate");
            if (ix.ndim == 2 && (int)ix.shape[1] != self.in_dim())
                throw py::value_error("apply: dimensão de entrada não bate");
            if (ix.ndim == 1) {
                py::array_t<float> out((py::ssize_t)self.out_dim());
                self.apply_batch((const float*)ix.ptr, out.mutable_data(), 1);
                return py::object(std::move(out));
            }
            py::array_t<float> out(std::vector<py::ssize_t>{n, self.out_dim()});
            self.apply_batch((const float*)ix.ptr, out.mutable_data(), n);
            return py::object(std::move(out));
        }, py::arg("x"))
        .def("apply_matrix", [](const EmbedBridge& self,
                                py::array_t<float, py::array::c_style | py::array::forcecast> X) {
            auto ix = X.request();
            if (ix.ndim != 2 || (int)ix.shape[1] != self.in_dim())
                throw py::value_error("apply_matrix: espera [n, in_dim]");
            const int n = (int)ix.shape[0];
            py::array_t<float> out(std::vector<py::ssize_t>{n, self.out_dim()});
            self.apply_batch((const float*)ix.ptr, out.mutable_data(), n);
            return out;
        }, py::arg("X"), "Aplica a [n, in_dim] e devolve [n, out_dim].")
        .def("save", &EmbedBridge::save, py::arg("path"))
        .def_static("load", &EmbedBridge::load, py::arg("path"))
        .def_property_readonly("fitted", &EmbedBridge::fitted)
        .def_property_readonly("in_dim", &EmbedBridge::in_dim)
        .def_property_readonly("out_dim", &EmbedBridge::out_dim)
        .def_property_readonly("n_samples", &EmbedBridge::n_samples)
        .def_property_readonly("train_rmse", &EmbedBridge::train_rmse)
        .def_property_readonly("train_r2", &EmbedBridge::train_r2)
        .def("__repr__", &EmbedBridge::describe);

    m.def("dense_matmul", [](py::array_t<float> x, py::array_t<float> W,
                             int in_dim, int out_dim) {
        py::array_t<float> y(out_dim);
        dense_matmul(y.mutable_data(), (const float*)x.request().ptr,
                     (const float*)W.request().ptr, in_dim, out_dim);
        return y;
    }, py::arg("x"), py::arg("W"), py::arg("in_dim"), py::arg("out_dim"));
    m.def("rms_norm", [](py::array_t<float> x, py::array_t<float> w,
                         int n, int d, float eps) {
        py::array_t<float> y((py::ssize_t)n * d);
        rms_norm(y.mutable_data(), (const float*)x.request().ptr,
                 (const float*)w.request().ptr, n, d, eps);
        return y;
    }, py::arg("x"), py::arg("w"), py::arg("n"), py::arg("d"), py::arg("eps"));

    // Alias DEPRECIADO no nível do módulo, para não quebrar código existente.
    // `StreamEngine` descrevia a coisa errada: em inferência, "stream" significa
    // carregar pesos em fluxo para rodar um modelo maior que a memória, e esta
    // classe nunca fez isso. O alias some na 0.2.
    m.attr("StreamEngine") = m.attr("ChunkEngine");
    m.attr("StreamOptions") = m.attr("ChunkOptions");
    m.attr("StreamChunk") = m.attr("Chunk");

}
