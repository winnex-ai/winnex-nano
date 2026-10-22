// forward_gpu.cpp — OpenCL dense projections for the native forward pass.
//
// Moves the DENSE MATRIX work of the transformer layers to the GPU. RoPE, the
// causal mask, softmax, residuals and the KV cache stay on the CPU: the
// projections are the dominant FLOPs, and the attention path is the part whose
// numerical correctness was verified most carefully — moving only the dense
// work keeps perplexity parity intact while removing the bottleneck.
//
// OpenCL is REQUIRED. There is no CPU fallback and no silent degradation: if
// the device cannot be brought up, the constructor throws and the engine does
// not pretend to run. A fallback that hides a missing GPU is how a benchmark
// ends up comparing CPU against CPU while reporting "GPU".
//
// Loader resolution (in order, no hidden vendor default):
//   1. the path passed by the caller
//   2. $WINNEX_OPENCL_LIB
//   3. the vendor ICD driver names below
//   4. the platform ICD loader
// Every attempt is recorded in ForwardGpu::reason().

#include "winnex_nano/forward_gpu.hpp"

#include <CL/cl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// ── Portable dynamic loading ───────────────────────────────────────────────
// dlopen/dlsym/dlclose are POSIX; Windows uses LoadLibrary/GetProcAddress.
// Same shim winnex-madhava and winnex-xfactor-c already use.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
static void* fg_open(const char* n) { return (void*)LoadLibraryA(n); }
static void* fg_sym(void* h, const char* n) { return (void*)GetProcAddress((HMODULE)h, n); }
#else
#include <dlfcn.h>
static void* fg_open(const char* n) { return dlopen(n, RTLD_NOW | RTLD_LOCAL); }
static void* fg_sym(void* h, const char* n) { return dlsym(h, n); }
#endif

namespace {

// The function pointers we resolve. The types come from the real CL headers,
// so a signature drift is a compile error rather than silent corruption.
struct FgApi {
    cl_int (*clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int (*clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    cl_int (*clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
    cl_context (*clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                  void (CL_CALLBACK*)(const char*, const void*, size_t, void*),
                                  void*, cl_int*);
    cl_command_queue (*clCreateCommandQueue)(cl_context, cl_device_id,
                                             cl_command_queue_properties, cl_int*);
    cl_program (*clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
    cl_int (*clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*,
                             void (CL_CALLBACK*)(cl_program, void*), void*);
    cl_int (*clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info,
                                    size_t, void*, size_t*);
    cl_kernel (*clCreateKernel)(cl_program, const char*, cl_int*);
    cl_mem (*clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    cl_int (*clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
    cl_int (*clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*,
                                     const size_t*, const size_t*, cl_uint,
                                     const cl_event*, cl_event*);
    cl_int (*clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                  void*, cl_uint, const cl_event*, cl_event*);
    cl_int (*clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                                   const void*, cl_uint, const cl_event*, cl_event*);
    cl_int (*clFinish)(cl_command_queue);
    cl_int (*clReleaseMemObject)(cl_mem);
    cl_int (*clReleaseKernel)(cl_kernel);
    cl_int (*clReleaseProgram)(cl_program);
    cl_int (*clReleaseCommandQueue)(cl_command_queue);
    cl_int (*clReleaseContext)(cl_context);
};

FgApi* g_api = nullptr;
std::string g_reason = "not initialized";

void load_opencl(const std::string& explicit_lib) {
    if (g_api) return;

    std::vector<std::string> tried;
    std::vector<std::string> candidates;
    if (!explicit_lib.empty()) candidates.push_back(explicit_lib);
    if (const char* env = std::getenv("WINNEX_OPENCL_LIB")) {
        if (*env) candidates.push_back(env);
    }
    // Vendor ICD drivers first — loading them registers the platform with the
    // generic loader. Then the platform loader itself.
    candidates.push_back("libnvidia-opencl.so.1");
    candidates.push_back("libOpenCL.so.1");
    candidates.push_back("libOpenCL.so");
    candidates.push_back("/System/Library/Frameworks/OpenCL.framework/OpenCL");
    candidates.push_back("OpenCL.dll");

    void* h = nullptr;
    for (const auto& c : candidates) {
        tried.push_back(c);
        h = fg_open(c.c_str());
        if (h) { g_reason = "loader: " + c; break; }
    }
    if (!h) {
        std::string msg = "no OpenCL loader found. Tried:";
        for (const auto& t : tried) msg += " " + t;
        msg += ". Set WINNEX_OPENCL_LIB or install an ICD loader (ocl-icd).";
        g_reason = msg;
        throw std::runtime_error("ForwardGpu: " + msg);
    }

    auto* api = new FgApi;
    std::vector<std::string> missing;
#define FG_LOAD(name) \
    *(void**)(&api->name) = fg_sym(h, #name); \
    if (!api->name) missing.push_back(#name)
    FG_LOAD(clGetPlatformIDs);          FG_LOAD(clGetDeviceIDs);
    FG_LOAD(clGetDeviceInfo);           FG_LOAD(clCreateContext);
    FG_LOAD(clCreateCommandQueue);      FG_LOAD(clCreateProgramWithSource);
    FG_LOAD(clBuildProgram);            FG_LOAD(clGetProgramBuildInfo);
    FG_LOAD(clCreateKernel);            FG_LOAD(clCreateBuffer);
    FG_LOAD(clSetKernelArg);            FG_LOAD(clEnqueueNDRangeKernel);
    FG_LOAD(clEnqueueReadBuffer);       FG_LOAD(clEnqueueWriteBuffer);
    FG_LOAD(clFinish);
    FG_LOAD(clReleaseMemObject);        FG_LOAD(clReleaseKernel);
    FG_LOAD(clReleaseProgram);          FG_LOAD(clReleaseCommandQueue);
    FG_LOAD(clReleaseContext);
#undef FG_LOAD
    if (!missing.empty()) {
        std::string msg = "OpenCL loader is missing symbols:";
        for (const auto& m : missing) msg += " " + m;
        delete api;
        g_reason = msg;
        throw std::runtime_error("ForwardGpu: " + msg);
    }
    g_api = api;
}

// ── Fused GEMM + bias kernel ───────────────────────────────────────────────
//
//   C[t, o] = bias[o] + Σ_i A[t, i] · W[o, i]
//
// A [seq, in] row-major activations, W [out, in] row-major (safetensors
// layout), C [seq, out] row-major.
//
// The bias is fused into the accumulation: the CPU path does a matmul and then
// a separate add loop per projection, one extra pass over the output per call.
//
// Work-item = one output element. The inner loop is unrolled by 8 so the loads
// from A and W pipeline. A tiled shared-memory GEMM would cut global traffic
// further; this version is deliberately simple so the first measurement
// isolates "is the GPU faster at all" from "is the tiling optimal".
const char* kGemmSrc = R"(
__kernel void gemm_bias(
    __global const float* A,      // [seq, in]
    __global const float* W,      // [out, in]
    __global const float* bias,   // [out] (has_bias=0 -> ignored)
    __global float* C,            // [seq, out]
    int seq, int in_dim, int out_dim, int has_bias)
{
    int gid = (int)get_global_id(0);
    if (gid >= seq * out_dim) return;
    int t = gid / out_dim;
    int o = gid - t * out_dim;

    const __global float* wrow = W + (size_t)o * in_dim;
    const __global float* arow = A + (size_t)t * in_dim;

    float acc = 0.0f;
    int i = 0;
    for (; i + 8 <= in_dim; i += 8) {
        acc += arow[i+0] * wrow[i+0];
        acc += arow[i+1] * wrow[i+1];
        acc += arow[i+2] * wrow[i+2];
        acc += arow[i+3] * wrow[i+3];
        acc += arow[i+4] * wrow[i+4];
        acc += arow[i+5] * wrow[i+5];
        acc += arow[i+6] * wrow[i+6];
        acc += arow[i+7] * wrow[i+7];
    }
    for (; i < in_dim; ++i) acc += arow[i] * wrow[i];

    if (has_bias) acc += bias[o];
    C[(size_t)t * out_dim + o] = acc;
}
)";

} // namespace

namespace winnex_nano {

struct ForwardGpu::Impl {
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel k_gemm = nullptr;
    char dev_name[256] = {0};

    // ── Resident weight cache ──────────────────────────────────────────────
    // MEASURED MOTIVATION (2026-10-22). The first working version uploaded the
    // weight matrix and allocated fresh buffers on EVERY matmul call. With 7
    // projections per layer and 24 layers, one forward is 168 calls, each
    // creating 4 buffers and tearing them down:
    //
    //     Q/K/V/O only on GPU : 254 ms/forward vs 326 ms CPU  (1.28x faster)
    //     all 7 on GPU        :  3.51 s        vs 2.80 s CPU  (0.80x SLOWER)
    //
    // Moving MORE work to the GPU made it SLOWER — the signature of per-call
    // overhead dominating. The weights are constant for the life of the
    // engine, so they belong on the device once.
    //
    // The cache is keyed by the host pointer plus the element count. A caller
    // that loads weights per layer (as ForwardEngine does) gets a fresh
    // LayerWeights each time, so the pointers change and the cache would thrash.
    // ForwardEngine therefore keeps its layers resident — see enable_gpu().
    std::unordered_map<const void*, std::pair<cl_mem, size_t>> wcache;
    // Reusable scratch (activations / output / bias), grown on demand.
    cl_mem ba = nullptr, bc = nullptr, bb = nullptr;
    size_t a_cap = 0, c_cap = 0, bias_cap = 0;

    cl_mem weight_buffer(const float* W, size_t n_floats) {
        auto it = wcache.find((const void*)W);
        if (it != wcache.end() && it->second.second == n_floats) return it->second.first;
        cl_int err = CL_SUCCESS;
        cl_mem b = g_api->clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         n_floats * sizeof(float), (void*)W, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("ForwardGpu: weight buffer failed");
        if (it != wcache.end()) g_api->clReleaseMemObject(it->second.first);
        wcache[(const void*)W] = {b, n_floats};
        return b;
    }

    ~Impl() {
        for (auto& kv : wcache) g_api->clReleaseMemObject(kv.second.first);
        if (ba) g_api->clReleaseMemObject(ba);
        if (bc) g_api->clReleaseMemObject(bc);
        if (bb) g_api->clReleaseMemObject(bb);
        if (k_gemm) g_api->clReleaseKernel(k_gemm);
        if (prog) g_api->clReleaseProgram(prog);
        if (q) g_api->clReleaseCommandQueue(q);
        if (ctx) g_api->clReleaseContext(ctx);
    }
};

ForwardGpu::ForwardGpu(const std::string& opencl_lib) : impl_(new Impl) {
    // No try/catch: a missing device is a hard error, by design.
    load_opencl(opencl_lib);

    cl_uint nplat = 0;
    if (g_api->clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0)
        throw std::runtime_error("ForwardGpu: no OpenCL platform (" + g_reason + ")");
    std::vector<cl_platform_id> plats(nplat);
    g_api->clGetPlatformIDs(nplat, plats.data(), nullptr);

    cl_device_id dev = nullptr;
    for (cl_uint p = 0; p < nplat && !dev; ++p) {
        cl_uint ndev = 0;
        g_api->clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev);
        if (ndev == 0) continue;
        g_api->clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    }
    if (!dev) throw std::runtime_error("ForwardGpu: no OpenCL GPU device (" + g_reason + ")");

    size_t nm = 0;
    g_api->clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &nm);
    if (nm > 0 && nm < sizeof(impl_->dev_name))
        g_api->clGetDeviceInfo(dev, CL_DEVICE_NAME, nm, impl_->dev_name, nullptr);

    cl_int err = CL_SUCCESS;
    impl_->ctx = g_api->clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !impl_->ctx)
        throw std::runtime_error("ForwardGpu: clCreateContext failed (" + std::to_string(err) + ")");
    impl_->q = g_api->clCreateCommandQueue(impl_->ctx, dev, 0, &err);
    if (err != CL_SUCCESS || !impl_->q)
        throw std::runtime_error("ForwardGpu: clCreateCommandQueue failed (" + std::to_string(err) + ")");

    const char* src = kGemmSrc;
    impl_->prog = g_api->clCreateProgramWithSource(impl_->ctx, 1, &src, nullptr, &err);
    if (err != CL_SUCCESS || !impl_->prog)
        throw std::runtime_error("ForwardGpu: clCreateProgramWithSource failed");

    err = g_api->clBuildProgram(impl_->prog, 1, &dev, "-cl-fast-relaxed-math",
                                nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t ln = 0;
        g_api->clGetProgramBuildInfo(impl_->prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln);
        std::vector<char> log(ln + 1, 0);
        if (ln) g_api->clGetProgramBuildInfo(impl_->prog, dev, CL_PROGRAM_BUILD_LOG,
                                             ln, log.data(), nullptr);
        throw std::runtime_error(std::string("ForwardGpu: clBuildProgram failed:\n") + log.data());
    }
    impl_->k_gemm = g_api->clCreateKernel(impl_->prog, "gemm_bias", &err);
    if (err != CL_SUCCESS || !impl_->k_gemm)
        throw std::runtime_error("ForwardGpu: clCreateKernel failed");

    g_reason = std::string("ok: ") + impl_->dev_name;
}

ForwardGpu::~ForwardGpu() { delete impl_; }

bool ForwardGpu::ready() const { return impl_ != nullptr && impl_->k_gemm != nullptr; }
std::string ForwardGpu::device_name() const { return impl_ ? impl_->dev_name : ""; }
std::string ForwardGpu::reason() { return g_reason; }

void ForwardGpu::matmul(const float* A, int seq,
                        const float* W, int out_dim, int in_dim,
                        const float* bias, float* C) {
    const size_t a_bytes = (size_t)seq * in_dim * sizeof(float);
    const size_t c_bytes = (size_t)seq * out_dim * sizeof(float);
    cl_int err = CL_SUCCESS;

    // Weights are resident: uploaded once per (pointer, size) for the life of
    // the engine. This is the fix for the measured per-call overhead — see the
    // note in Impl::weight_buffer.
    cl_mem bw = impl_->weight_buffer(W, (size_t)out_dim * in_dim);

    // Activation and output buffers are reallocated only when the shape grows.
    // They are pure scratch, so a WRITE_ONLY buffer reused across calls needs
    // no copy-in.
    if (impl_->a_cap < a_bytes) {
        if (impl_->ba) g_api->clReleaseMemObject(impl_->ba);
        impl_->ba = g_api->clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY, a_bytes, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("ForwardGpu: buffer A failed");
        impl_->a_cap = a_bytes;
    }
    if (impl_->c_cap < c_bytes) {
        if (impl_->bc) g_api->clReleaseMemObject(impl_->bc);
        impl_->bc = g_api->clCreateBuffer(impl_->ctx, CL_MEM_WRITE_ONLY, c_bytes, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("ForwardGpu: buffer C failed");
        impl_->c_cap = c_bytes;
    }

    int has_bias = bias ? 1 : 0;
    //
    // BIAS BUFFER SIZE (bug fixed 2026-10-22). This allocated `sizeof(float)` —
    // FOUR bytes — while passing CL_MEM_COPY_HOST_PTR with a pointer to the
    // caller's bias[out_dim] array. Only bias[0] was copied; the kernel then
    // read bias[o] for o up to out_dim, past the end of the buffer.
    //
    // Measured: the first three projections (Q/K/V, all with bias) had
    // max_rel_err of 9.19, 1.26 and 8.25; the fourth (o_proj, no bias) had
    // 1.07e-05. That asymmetry is what a truncated bias buffer looks like — it
    // produced a perplexity 1,022,696x the CPU value.
    if (bias) {
        if (impl_->bias_cap < (size_t)out_dim * sizeof(float) || impl_->bb == nullptr) {
            if (impl_->bb) g_api->clReleaseMemObject(impl_->bb);
            impl_->bb = g_api->clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY,
                                              (size_t)out_dim * sizeof(float), nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("ForwardGpu: buffer bias failed");
            impl_->bias_cap = (size_t)out_dim * sizeof(float);
        }
        g_api->clEnqueueWriteBuffer(impl_->q, impl_->bb, CL_TRUE, 0,
                                    (size_t)out_dim * sizeof(float), bias, 0, nullptr, nullptr);
    } else if (impl_->bb == nullptr) {
        float zero = 0.0f;
        impl_->bb = g_api->clCreateBuffer(impl_->ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float), &zero, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("ForwardGpu: buffer bias failed");
        impl_->bias_cap = sizeof(float);
    }

    // Upload the activations (they change every call; the weights do not).
    g_api->clEnqueueWriteBuffer(impl_->q, impl_->ba, CL_TRUE, 0, a_bytes, A, 0, nullptr, nullptr);

    cl_mem ba = impl_->ba, bb = impl_->bb, bc = impl_->bc;
    g_api->clSetKernelArg(impl_->k_gemm, 0, sizeof(cl_mem), &ba);
    g_api->clSetKernelArg(impl_->k_gemm, 1, sizeof(cl_mem), &bw);
    g_api->clSetKernelArg(impl_->k_gemm, 2, sizeof(cl_mem), &bb);
    g_api->clSetKernelArg(impl_->k_gemm, 3, sizeof(cl_mem), &bc);
    g_api->clSetKernelArg(impl_->k_gemm, 4, sizeof(int), &seq);
    g_api->clSetKernelArg(impl_->k_gemm, 5, sizeof(int), &in_dim);
    g_api->clSetKernelArg(impl_->k_gemm, 6, sizeof(int), &out_dim);
    g_api->clSetKernelArg(impl_->k_gemm, 7, sizeof(int), &has_bias);

    size_t gsz = (size_t)seq * out_dim;
    err = g_api->clEnqueueNDRangeKernel(impl_->q, impl_->k_gemm, 1, nullptr, &gsz,
                                        nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("ForwardGpu: enqueue failed (" + std::to_string(err) + ")");
    g_api->clEnqueueReadBuffer(impl_->q, bc, CL_TRUE, 0, c_bytes, C, 0, nullptr, nullptr);
    g_api->clFinish(impl_->q);

    // SANITY (opt-in): compare the first few GPU outputs against a scalar
    // reference. A wrong GEMM produces garbage that looks like a wrong kernel
    // argument order until you print both sides. Set WINNEX_GPU_DEBUG=1.
    if (std::getenv("WINNEX_GPU_DEBUG")) {
        static int calls = 0;
        if (calls++ < 4) {
            double max_rel = 0.0;
            int bad = -1;
            for (int t = 0; t < seq && t < 2; ++t) {
                for (int o = 0; o < out_dim && o < 16; ++o) {
                    double ref = 0.0;
                    for (int i = 0; i < in_dim; ++i)
                        ref += (double)A[(size_t)t * in_dim + i] * (double)W[(size_t)o * in_dim + i];
                    if (bias) ref += bias[o];
                    const double got = C[(size_t)t * out_dim + o];
                    const double rel = std::fabs(got - ref) / (std::fabs(ref) + 1e-6);
                    if (rel > max_rel) { max_rel = rel; bad = o; }
                }
            }
            std::fprintf(stderr, "[gpu] seq=%d in=%d out=%d max_rel_err=%.3e (o=%d)\n",
                         seq, in_dim, out_dim, max_rel, bad);
        }
    }
}

} // namespace winnex_nano
