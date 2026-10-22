// py_module.cpp — winnex-nano Python bindings (pybind11).
//
// Exposes the native Winnex inference core: spectral tokenizer, weight
// balancer, stream engine, X-factor, safetensors loader and the dense
// forward pass. No CUDA — OpenCL/AVX2/OpenMP backends.
#include "winnex_nano/spectral_tokenizer.hpp"
#include "winnex_nano/weight_balancer.hpp"
#include "winnex_nano/stream_engine.hpp"
#include "winnex_nano/x_factor.hpp"
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
        .def("encode_batch", [](const SpectralTokenizer& self, py::bytes data) {
            // Encode bytes → numpy array (n, embed_dim, 4) float32.
            // Avoids per-Quat Python objects: the native buffer is filled in
            // one call and exposed as a contiguous array.
            char* buf = nullptr;
            Py_ssize_t len = 0;
            PyBytes_AsStringAndSize(data.ptr(), &buf, &len);
            if (len < 0) throw std::runtime_error("encode_batch: invalid bytes");
            int embed = self.embed_dim();
            py::array_t<float> out({(py::ssize_t)len, (py::ssize_t)embed, (py::ssize_t)4});
            self.encode_into_buffer(std::string(buf, (size_t)len), out.mutable_data());
            return out;
        }, py::arg("data"))
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

    py::class_<StreamChunk>(m, "StreamChunk")
        .def(py::init<>())
        .def_readwrite("text", &StreamChunk::text)
        .def_readwrite("done", &StreamChunk::done)
        .def_readwrite("doc_id", &StreamChunk::doc_id)
        .def_readwrite("build_ms", &StreamChunk::build_ms)
        .def_readwrite("search_ms", &StreamChunk::search_ms);

    py::class_<StreamEngine>(m, "StreamEngine")
        .def(py::init<>())
        .def(py::init<const StreamEngine::Options&>(), py::arg("opts"))
        .def("stream", &StreamEngine::stream, py::arg("prompt"),
             py::arg("next_chunk_fn"), py::arg("sink"));

    py::class_<StreamEngine::Options>(m, "StreamOptions")
        .def(py::init<>())
        .def_readwrite("embed_dim", &StreamEngine::Options::embed_dim)
        .def_readwrite("top_k", &StreamEngine::Options::top_k)
        .def_readwrite("build_interval", &StreamEngine::Options::build_interval);

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
        .def("to_float", [](const Safetensors& st, const std::string& name) {
            return st.to_float(st.get(name));  // std::vector<float> -> list
        }, py::arg("name"))
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
        .def_readwrite("rms_norm_eps", &ModelConfig::rms_norm_eps)
        .def_readwrite("tie_word_embeddings", &ModelConfig::tie_word_embeddings)
        .def_readwrite("gptq", &ModelConfig::gptq);
    m.def("load_config", &load_config, py::arg("config_path"));

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
        .def_property_readonly("weight_mode", &ForwardEngine::weight_mode)
        // Bytes of projection weights living on the device in "device" mode.
        // Diagnostic: it is the number that decides whether the model fits.
        .def_property_readonly("device_weight_bytes", &ForwardEngine::device_weight_bytes)
        .def_property_readonly("resident_bytes", &ForwardEngine::estimate_resident_bytes)
        .def_static("available_ram_bytes", &ForwardEngine::available_ram_bytes);

    // --- Free functions -----------------------------------------------------
    m.def("expand_spectral", [](py::array_t<float> psi, int d, int D) {
        py::array_t<float> out(D);
        expand_spectral((const float*)psi.request().ptr, d, out.mutable_data(), D);
        return out;
    }, py::arg("psi"), py::arg("d"), py::arg("D"));

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
}
