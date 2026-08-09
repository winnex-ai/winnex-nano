// py_module.cpp — winnex-nano Python bindings (pybind11).
//
// Exposes the spectral tokenizer and weight balancer so the benchmark and the
// OpenAI-compatible server can call the native core.
#include "winnex_nano/spectral_tokenizer.hpp"
#include "winnex_nano/weight_balancer.hpp"
#include "winnex_nano/stream_engine.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace winnex_nano;

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
        .def("decode", &SpectralTokenizer::decode, py::arg("states"))
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
}
