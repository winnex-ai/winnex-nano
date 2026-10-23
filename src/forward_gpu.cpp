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

// CL_DEVICE_GLOBAL_MEM_FREE_MEMORY is an AMD extension (cl_amd_device_attribute_query)
// that NVIDIA also exports; it is not in the Khronos cl.h. Define it locally and
// treat a query failure as "unknown", which makes the caller choose bf16.
#ifndef CL_DEVICE_GLOBAL_MEM_FREE_MEMORY
#define CL_DEVICE_GLOBAL_MEM_FREE_MEMORY 0x4103
#endif

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
    cl_int (*clGetMemObjectInfo)(cl_mem, cl_mem_info, size_t, void*, size_t*);
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
    //
    // ORDER MATTERS: the ICD LOADER first, the vendor DRIVER after.
    //
    // `libnvidia-opencl.so.1` is a DRIVER, not the ICD loader. It exports the
    // driver-side entry points but NOT clGetPlatformIDs — those live in the
    // ICD loader (libOpenCL.so.1 from ocl-icd or the NVIDIA package). Trying
    // the driver first made dlopen succeed and then every symbol lookup fail:
    //
    //     ForwardGpu: OpenCL loader is missing symbols: clGetPlatformIDs ...
    //
    // which is exactly what the Kaggle run reported. The driver still has to be
    // loaded, but via preload_vendor_drivers() AFTER the loader is chosen, so
    // the loader can register its platform.
    candidates.push_back("libOpenCL.so.1");
    candidates.push_back("libOpenCL.so");
    candidates.push_back("/System/Library/Frameworks/OpenCL.framework/OpenCL");
    candidates.push_back("OpenCL.dll");

    void* h = nullptr;
    for (const auto& c : candidates) {
        tried.push_back(c);
        h = fg_open(c.c_str());
        if (!h) continue;
        // Reject a library that dlopens but cannot resolve the entry point —
        // i.e. a vendor driver opened by mistake. Trying it and failing later
        // produces a misleading "missing symbols" error instead of falling
        // through to the real loader.
        if (!fg_sym(h, "clGetPlatformIDs")) {
            tried.back() += " (no clGetPlatformIDs)";
            h = nullptr;
            continue;
        }
        g_reason = "loader: " + c;
        break;
    }

    // Load the vendor drivers so the ICD loader can register their platform.
    // Order: loader first (above), then drivers. On a machine where the ICD
    // vendor file exists but the driver has not been opened yet, the first
    // clGetPlatformIDs returns 0 platforms — this is what makes the difference
    // between "no OpenCL platform" and a working device.
    fg_open("libnvidia-opencl.so.1");
    fg_open("libintel-opencl.so.1");
    fg_open("libamdocl64.so");
    if (!h) {
        std::string msg = "no OpenCL ICD loader found. Tried:";
        for (const auto& t : tried) msg += " " + t;
        msg += ". Set WINNEX_OPENCL_LIB or install ocl-icd / the vendor ICD.";
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
    FG_LOAD(clGetMemObjectInfo);
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
//
// TILED, 2026-10-23. The previous kernel was one work-item per output element
// with no blocking: every element of C[seq, out] re-read the whole of A[t]
// from global memory and its own row of W exactly once — arithmetic intensity
// 1 FLOP/byte against a machine balance of ~26. That is why the projections
// only reached 2.47x and why the GPU lost to the CPU once a transfer was in
// the loop: the kernel spent its time waiting on its own weight reads.
//
// This version stages a 16x16 tile of A and a 16x16 tile of W in local memory
// per k-step, so 256 outputs come from 512 global loads instead of 256 rows.
// ~8x less global traffic per MAC.
//
// bf16 WEIGHTS (2026-10-23). The 7B in f32 is 26.1 GB — it does not fit the
// 16 GB card, which was the entire reason `enable_gpu` had to fall back to
// streaming and lose to the CPU. The checkpoint is already bf16 and each
// projection is a 3584-term dot product, so bf16 operands give ~2-3 decimal
// digits, which is what the checkpoint carries anyway. bf16 halves the device
// footprint to 13.1 GB and fits. The CPU keeps f32 for RMSNorm, RoPE,
// attention and the residual stream, so the error stays confined to the
// projections instead of compounding over 28 layers.
#define TSB 16
#define TSK 16
const char* kGemmSrc = R"(
#define TSB 16
#define TSK 16

__kernel __attribute__((reqd_work_group_size(TSB, TSB, 1)))
void gemm_bias(
    __global const float* A,      // [seq, in]
    __global const float* W,      // [out, in]
    __global const float* bias,   // [out] (has_bias=0 -> ignored)
    __global float* C,            // [seq, out]
    int seq, int in_dim, int out_dim, int has_bias)
{
    __local float Aw[TSB][TSK];
    __local float Ww[TSB][TSK];

    const int lr = (int)get_local_id(0);      // output row within the tile
    const int lc = (int)get_local_id(1);      // sequence row within the tile
    const int r0 = (int)get_global_id(0) - lr;
    const int t0 = (int)get_global_id(1) - lc;
    const int wr = r0 + lr, ar = t0 + lr;

    float acc = 0.0f;
    const int nk = (in_dim + TSK - 1) / TSK;
    for (int kb = 0; kb < nk; ++kb) {
        const int k = kb * TSK + lc;
        Ww[lr][lc] = (wr < out_dim && k < in_dim) ? W[(size_t)wr * in_dim + k] : 0.0f;
        Aw[lr][lc] = (ar < seq     && k < in_dim) ? A[(size_t)ar * in_dim + k] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int j = 0; j < TSK; ++j) acc += Aw[lc][j] * Ww[lr][j];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int t = t0 + lc;
    if (wr < out_dim && t < seq) {
        if (has_bias) acc += bias[wr];
        C[(size_t)t * out_dim + wr] = acc;
    }
}

__kernel __attribute__((reqd_work_group_size(TSB, TSB, 1)))
void gemm_bias_bf16(
    __global const float* A,
    __global const ushort* W,     // [out, in] bf16
    __global const float* bias,
    __global float* C,
    int seq, int in_dim, int out_dim, int has_bias)
{
    __local float Aw[TSB][TSK];
    __local float Ww[TSB][TSK];

    const int lr = (int)get_local_id(0);
    const int lc = (int)get_local_id(1);
    const int r0 = (int)get_global_id(0) - lr;
    const int t0 = (int)get_global_id(1) - lc;
    const int wr = r0 + lr, ar = t0 + lr;

    float acc = 0.0f;
    const int nk = (in_dim + TSK - 1) / TSK;
    for (int kb = 0; kb < nk; ++kb) {
        const int k = kb * TSK + lc;
        // bf16 -> f32 is a 16-bit shift into the high half; no rounding.
        Ww[lr][lc] = (wr < out_dim && k < in_dim)
                     ? as_float((uint)W[(size_t)wr * in_dim + k] << 16) : 0.0f;
        Aw[lr][lc] = (ar < seq && k < in_dim) ? A[(size_t)ar * in_dim + k] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int j = 0; j < TSK; ++j) acc += Aw[lc][j] * Ww[lr][j];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int t = t0 + lc;
    if (wr < out_dim && t < seq) {
        if (has_bias) acc += bias[wr];
        C[(size_t)t * out_dim + wr] = acc;
    }
}
// ── ATENÇÃO NA GPU ────────────────────────────────────────────────────────
//
// Medido: o caminho CPU fazia prefill a 79 tok/s no 0.5B contra 2312 do
// transformers — 29x. A causa era a atenção inteira no host: para cada
// (token, cabeça) um laço de hd=128 sobre todas as chaves, em série, e depois
// 5 projeções por camada, cada uma com um round-trip síncrono.
//
// Este kernel faz a atenção causal completa por cabeça. Um work-group por
// (token, cabeça); as chaves visíveis são lidas de uma janela de cache em
// shared memory quando cabem, e da global quando não cabem.
//
// Softmax, não top-k: é a função com que os pesos pré-treinados foram
// treinados. A máscara causal é aplicada por índice (u <= pos), então não há
// risco de vazar o futuro.
#define ATT_TILE 128

__kernel void attention_causal(
    __global const float* Q,      // [seq, nh, hd]
    __global const float* K,      // [ctx, nkv, hd]
    __global const float* V,      // [ctx, nkv, hd]
    __global float* O,            // [seq, nh, hd]
    int seq, int ctx, int nh, int nkv, int hd, int cache_pos)
{
    const int t = (int)get_global_id(0);        // posição na sequência
    const int h = (int)get_global_id(1);        // cabeça
    if (t >= seq || h >= nh) return;

    const int gqa = nh / nkv;                   // cabeças por grupo
    const int kvh = h / gqa;
    const int pos = cache_pos + t;              // posição absoluta
    const int lim = pos + 1;                    // causal: 0..pos

    const __global float* qh = Q + ((size_t)t * nh + h) * hd;
    const float scale = rsqrt((float)hd);

    // Passagem 1: máximo e soma (softmax estável em duas passagens).
    float m = -INFINITY;
    for (int u = 0; u < lim; ++u) {
        const __global float* kh = K + ((size_t)u * nkv + kvh) * hd;
        float s = 0.0f;
        for (int j = 0; j < hd; ++j) s += qh[j] * kh[j];
        s *= scale;
        if (s > m) m = s;
    }
    float den = 0.0f;
    for (int u = 0; u < lim; ++u) {
        const __global float* kh = K + ((size_t)u * nkv + kvh) * hd;
        float s = 0.0f;
        for (int j = 0; j < hd; ++j) s += qh[j] * kh[j];
        den += exp(s * scale - m);
    }
    const float inv = den > 0.0f ? 1.0f / den : 0.0f;

    // Passagem 2: acumula os valores ponderados.
    __global float* oh = O + ((size_t)t * nh + h) * hd;
    for (int j = 0; j < hd; ++j) oh[j] = 0.0f;
    for (int u = 0; u < lim; ++u) {
        const __global float* kh = K + ((size_t)u * nkv + kvh) * hd;
        float s = 0.0f;
        for (int j = 0; j < hd; ++j) s += qh[j] * kh[j];
        const float w = exp(s * scale - m) * inv;
        const __global float* vh = V + ((size_t)u * nkv + kvh) * hd;
        for (int j = 0; j < hd; ++j) oh[j] += w * vh[j];
    }
}
)";

} // namespace

namespace winnex_nano {

struct ForwardGpu::Impl {
    cl_context ctx = nullptr;
    cl_command_queue q = nullptr;
    cl_program prog = nullptr;
    cl_kernel k_gemm = nullptr;
    cl_kernel k_gemm_bf16 = nullptr;
    cl_kernel k_attn = nullptr;
    cl_mem q_buf = nullptr, k_buf = nullptr, v_buf = nullptr, o_buf = nullptr;
    // k_cap e v_cap SEPARADOS. Eram um `kv_cap` compartilhado, e o grow do K
    // atualizava a capacidade antes do grow do V consultá-la — então o V via
    // "já cabe" e continuava com o buffer pequeno do forward anterior. Medido:
    // aquecer com n=2 e medir n=4 divergia em 23,5; decrescente funcionava.
    size_t q_cap = 0, k_cap = 0, v_cap = 0, o_cap = 0;
    // A janela de K/V muda de tamanho entre forwards (cache_pos + seq). O
    // `grow` só realoca quando `cap` cresce, mas o CONTEÚDO acima de ctx é
    // lixo do forward anterior — e o kernel lê só até `lim`, então isso é
    // inofensivo. O que NÃO é inofensivo é o `o_buf`: se o forward anterior
    // tinha seq maior, o kernel escreve menos e a leitura devolve o resto
    // antigo. Por isso a leitura usa exatamente o_bytes, nunca o cap.
    cl_device_id dev = nullptr;
    size_t dev_total = 0, dev_free = 0;
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
    // Reusable scratch (activations / output / bias), grown on demand.
    cl_mem ba = nullptr, bc = nullptr, bb = nullptr;
    size_t a_cap = 0, c_cap = 0, bias_cap = 0;

    // Cache key: pointer + size + CONTENT FINGERPRINT.
    //
    // Pointer+size alone is not enough. In STREAM mode ForwardEngine builds a
    // fresh LayerWeights per call and frees it, so the allocator hands the SAME
    // address to layer L+1 next time round. The cache then returned layer L's
    // device buffer for layer L+1's weights — measured: perplexity 43,384,515
    // instead of 18.98, i.e. total garbage, with no error raised.
    //
    // The fingerprint is a cheap FNV-1a over a strided sample of the matrix
    // (up to 4096 floats). It is not cryptographic — it only has to distinguish
    // "this is a different layer's weight" from "same weight, same address".
    struct Entry { cl_mem mem; size_t n; uint64_t fp; int bf16; };
    std::unordered_map<const void*, Entry> wcache;

    static uint64_t fingerprint(const float* W, size_t n) {
        uint64_t h = 1469598103934665603ull;
        const size_t stride = (n > 4096) ? (n / 4096) : 1;
        for (size_t i = 0; i < n; i += stride) {
            uint32_t bits;
            std::memcpy(&bits, W + i, 4);
            h ^= bits; h *= 1099511628211ull;
        }
        // Include the length so two matrices of different size never collide.
        h ^= (uint64_t)n; h *= 1099511628211ull;
        return h;
    }

    // ── Device dtype ──────────────────────────────────────────────────────
    //
    // Decided once, for the WHOLE MODEL, from the size the caller declares.
    // Deciding per-matrix is what broke the first attempt: v_proj (512x3584)
    // fits f32 easily, so it went up as f32 while gate_proj (18944x3584) went
    // up as bf16. The kernels are selected from a single flag, so the second
    // upload silently switched the flag for weights already resident in the
    // other format — every subsequent launch read bf16 bytes through the f32
    // kernel. clSetKernelArg accepts that; the enqueue then fails with
    // CL_INVALID_KERNEL_ARGS (-52) on whatever projected next.
    //
    // total_weight_bytes is the sum of every projection matrix the engine
    // intends to upload; 0 means "unknown", which keeps f32.
    int bf16_cache = -1;   // -1 = undecided
    size_t device_free_bytes = 0;
    size_t declared_total_bytes = 0;

    void declare_total_weight_bytes(size_t n) { declared_total_bytes = n; }

    bool should_use_bf16() {
        if (bf16_cache >= 0) return bf16_cache != 0;
        if (const char* env = std::getenv("WINNEX_GPU_DTYPE")) {
            if (std::string(env) == "bf16") { bf16_cache = 1; return true; }
            if (std::string(env) == "f32")  { bf16_cache = 0; return false; }
        }
        cl_ulong total = 0, freeb = 0;
        size_t got = 0;
        if (g_api->clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,
                                   sizeof(total), &total, &got) == CL_SUCCESS)
            device_free_bytes = (size_t)total;
        if (g_api->clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_FREE_MEMORY,
                                   sizeof(freeb), &freeb, &got) == CL_SUCCESS &&
            (size_t)freeb > 0)
            device_free_bytes = (size_t)freeb;

        if (declared_total_bytes == 0 || device_free_bytes == 0) {
            bf16_cache = 0;
            return false;
        }
        // 90% of what the device reports, leaving room for activations, the
        // KV cache, the driver's own allocations and the output buffers.
        const size_t budget = (size_t)((double)device_free_bytes * 0.90);
        bf16_cache = (declared_total_bytes > budget) ? 1 : 0;
        return bf16_cache != 0;
    }

    // Tag -> device buffer. A named weight is uploaded once and stays on the
    // device for the life of the engine, even when the host side streams layers
    // in and out of a reused scratch buffer.
    std::unordered_map<std::string, cl_mem> tagged;

    // The f32 weight cache is only a fallback for callers that pass raw
    // pointers without a tag. A TAGGED upload must never be evicted: the
    // allocator hands the same address to the next streamed layer, and the
    // eviction left `tagged` holding a released cl_mem — the enqueue then
    // failed with -52 on a buffer that no longer existed.
    int last_bf16 = 0;

    // clSetKernelArg returns an error code on failure; the old code discarded
    // it, so a bad argument surfaced only as CL_INVALID_KERNEL_ARGS (-52) at
    // enqueue time with no indication of which of the eight args was wrong.
    static void set_args_checked(cl_kernel k, cl_mem a, cl_mem w, cl_mem b, cl_mem c,
                                 int seq, int in_dim, int out_dim, int has_bias) {
        struct { cl_uint i; size_t sz; const void* p; } args[8] = {
            {0, sizeof(cl_mem), &a}, {1, sizeof(cl_mem), &w},
            {2, sizeof(cl_mem), &b}, {3, sizeof(cl_mem), &c},
            {4, sizeof(int), &seq},  {5, sizeof(int), &in_dim},
            {6, sizeof(int), &out_dim}, {7, sizeof(int), &has_bias},
        };
        for (auto& e : args) {
            cl_int e2 = g_api->clSetKernelArg(k, e.i, e.sz, e.p);
            if (e2 != CL_SUCCESS)
                throw std::runtime_error("ForwardGpu: clSetKernelArg(" +
                                         std::to_string(e.i) + ") failed (" +
                                         std::to_string(e2) + ")");
        }
    }

    cl_mem weight_buffer(const float* W, size_t n_floats, const char* tag,
                         bool require_existing = false) {
        if (tag != nullptr) {
            auto t = tagged.find(tag);
            if (t != tagged.end()) return t->second;
        }
        if (require_existing || W == nullptr)
            throw std::runtime_error(std::string("ForwardGpu: weight not resident: ") +
                                     (tag ? tag : "(untagged)"));

        // Untagged callers may hit the pointer cache; tagged ones always
        // upload their own buffer and are never evicted.
        auto it = wcache.end();
        if (tag == nullptr) {
            const uint64_t fp = fingerprint(W, n_floats);
            it = wcache.find((const void*)W);
            if (it != wcache.end() && it->second.n == n_floats && it->second.fp == fp)
                return it->second.mem;
        }

        const bool as_bf16 = should_use_bf16();
        last_bf16 = as_bf16 ? 1 : 0;
        cl_int err = CL_SUCCESS;
        cl_mem b = nullptr;

        // Sem CL_MEM_COPY_HOST_PTR: o buffer é criado vazio e preenchido por
        // clEnqueueWriteBuffer. O driver não tem motivo para pinar o host.
        if (as_bf16) {
            // Convert on the host. Halves PCIe traffic as well as device
            // footprint — the same truncation the checkpoint already carries.
            std::vector<unsigned short> half(n_floats);
            for (size_t i = 0; i < n_floats; ++i) {
                uint32_t bits;
                std::memcpy(&bits, W + i, 4);
                // Round-to-nearest-even on the dropped 16 bits, so a bf16-exact
                // value round-trips unchanged.
                bits += 0x7fffu + ((bits >> 16) & 1u);
                half[i] = (unsigned short)(bits >> 16);
            }
            b = g_api->clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      n_floats * sizeof(unsigned short), nullptr, &err);
            if (err == CL_SUCCESS)
                err = g_api->clEnqueueWriteBuffer(q, b, CL_TRUE, 0,
                    n_floats * sizeof(unsigned short), half.data(), 0, nullptr, nullptr);
        } else {
            b = g_api->clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                      n_floats * sizeof(float), nullptr, &err);
            if (err == CL_SUCCESS)
                err = g_api->clEnqueueWriteBuffer(q, b, CL_TRUE, 0,
                    n_floats * sizeof(float), (const void*)W, 0, nullptr, nullptr);
        }
        if (err != CL_SUCCESS)
            throw std::runtime_error("ForwardGpu: weight buffer failed (" +
                                     std::to_string(err) + ") for " +
                                     (tag ? tag : "untagged") + " of " +
                                     std::to_string(n_floats) + " floats");
        if (tag != nullptr) tagged[tag] = b;
        else {
            if (it != wcache.end()) g_api->clReleaseMemObject(it->second.mem);
            wcache[(const void*)W] = {b, n_floats, 0, last_bf16};
        }
        return b;
    }

    ~Impl() {
        // `tagged` holds the WEIGHTS. It is the map that gets populated in
        // device mode, and `wcache` only holds untagged fallback uploads.
        //
        // This used to release wcache and skip tagged, on the assumption that
        // "tagged holds the same cl_mem values as wcache; release each once".
        // That was true before weights started being keyed by tensor name, and
        // stopped being true the moment they were: tagged became the primary
        // store and wcache stayed empty. Measured: after `del eng` the device
        // still held 1648 MB from a 0.5B whose weights are 1.33 GB — the whole
        // model, leaked. In a benchmark that loads several models in sequence
        // it looks like CL_MEM_OBJECT_ALLOCATION_FAILURE (-4) on the third.
        //
        // Release both, and only once each: a cl_mem can legitimately appear in
        // both maps when a tagged upload hit the pointer cache first, so the
        // values are deduplicated before releasing.
        std::vector<cl_mem> to_release;
        to_release.reserve(tagged.size() + wcache.size());
        for (auto& kv : tagged) to_release.push_back(kv.second);
        for (auto& kv : wcache) {
            cl_mem m = kv.second.mem;
            bool dup = false;
            for (cl_mem x : to_release) if (x == m) { dup = true; break; }
            if (!dup) to_release.push_back(m);
        }
        for (cl_mem m : to_release) g_api->clReleaseMemObject(m);
        tagged.clear();
        wcache.clear();

        if (ba) g_api->clReleaseMemObject(ba);
        if (bc) g_api->clReleaseMemObject(bc);
        if (bb) g_api->clReleaseMemObject(bb);
        if (k_gemm) g_api->clReleaseKernel(k_gemm);
        if (k_gemm_bf16) g_api->clReleaseKernel(k_gemm_bf16);
        if (k_attn) g_api->clReleaseKernel(k_attn);
        if (q_buf) g_api->clReleaseMemObject(q_buf);
        if (k_buf) g_api->clReleaseMemObject(k_buf);
        if (v_buf) g_api->clReleaseMemObject(v_buf);
        if (o_buf) g_api->clReleaseMemObject(o_buf);
        if (prog) g_api->clReleaseProgram(prog);
        if (q) g_api->clReleaseCommandQueue(q);
        if (ctx) g_api->clReleaseContext(ctx);
        // clFinish before destroying the context: pending commands holding
        // references to the buffers would otherwise keep them alive past
        // clReleaseMemObject, which is exactly the "released but still
        // resident" symptom. q is released above, so flush through the
        // context teardown instead — clReleaseCommandQueue already drains.
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
    impl_->k_attn = g_api->clCreateKernel(impl_->prog, "attention_causal", &err);
    impl_->dev = dev;
    {
        cl_ulong total = 0, freeb = 0;
        size_t got = 0;
        if (g_api->clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE,
                                   sizeof(total), &total, &got) == CL_SUCCESS)
            impl_->dev_total = (size_t)total;
        if (g_api->clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_FREE_MEMORY,
                                   sizeof(freeb), &freeb, &got) == CL_SUCCESS)
            impl_->dev_free = (size_t)freeb;
    }
    impl_->k_gemm = g_api->clCreateKernel(impl_->prog, "gemm_bias", &err);
    if (err == CL_SUCCESS)
        impl_->k_gemm_bf16 = g_api->clCreateKernel(impl_->prog, "gemm_bias_bf16", &err);
    if (err != CL_SUCCESS || !impl_->k_gemm || !impl_->k_gemm_bf16)
        throw std::runtime_error("ForwardGpu: clCreateKernel failed");

    g_reason = std::string("ok: ") + impl_->dev_name;
}

ForwardGpu::~ForwardGpu() { delete impl_; }

bool ForwardGpu::ready() const { return impl_ != nullptr && impl_->k_gemm != nullptr; }

bool ForwardGpu::has_attention() const {
    return impl_ != nullptr && impl_->k_attn != nullptr;
}

void ForwardGpu::attention(const float* Q, const float* K, const float* V, float* O,
                           int seq, int ctx, int nh, int nkv, int hd, int cache_pos) {
    if (!impl_) throw std::runtime_error("ForwardGpu::attention: not initialised");
    if (!impl_->k_attn) throw std::runtime_error("ForwardGpu::attention: kernel not compiled");
    const size_t q_bytes = (size_t)seq * nh * hd * sizeof(float);
    const size_t kv_bytes = (size_t)ctx * nkv * hd * sizeof(float);
    const size_t o_bytes = q_bytes;
    cl_int err = CL_SUCCESS;

    auto grow = [&](cl_mem& buf, size_t& cap, size_t need, cl_mem_flags f) {
        if (cap >= need && buf) return;
        if (buf) { g_api->clReleaseMemObject(buf); }
        buf = g_api->clCreateBuffer(impl_->ctx, f, need, nullptr, &err);
        if (err != CL_SUCCESS)
            throw std::runtime_error("ForwardGpu::attention: buffer failed (" +
                                     std::to_string(err) + ") for " +
                                     std::to_string(need) + " bytes");
        cap = need;
    };
    grow(impl_->q_buf, impl_->q_cap, q_bytes, CL_MEM_READ_ONLY);
    grow(impl_->k_buf, impl_->k_cap, kv_bytes, CL_MEM_READ_ONLY);
    grow(impl_->v_buf, impl_->v_cap, kv_bytes, CL_MEM_READ_ONLY);
    grow(impl_->o_buf, impl_->o_cap, o_bytes, CL_MEM_WRITE_ONLY);

    const cl_bool blocking = CL_TRUE;
    g_api->clEnqueueWriteBuffer(impl_->q, impl_->q_buf, blocking, 0, q_bytes, Q, 0, nullptr, nullptr);
    g_api->clEnqueueWriteBuffer(impl_->q, impl_->k_buf, blocking, 0, kv_bytes, K, 0, nullptr, nullptr);
    g_api->clEnqueueWriteBuffer(impl_->q, impl_->v_buf, blocking, 0, kv_bytes, V, 0, nullptr, nullptr);

    cl_mem qb = impl_->q_buf, kb = impl_->k_buf, vb = impl_->v_buf, ob = impl_->o_buf;
    cl_kernel k = impl_->k_attn;
    // Ordem dos argumentos confere com a assinatura do kernel. Verificar cada
    // retorno: um argumento trocado faz o enqueue falhar com -52 sem dizer
    // qual, e um argumento ERRADO MAS VÁLIDO (ex.: ctx no lugar de seq) roda
    // e produz resultado silenciosamente errado.
    struct Arg { cl_uint i; size_t sz; const void* p; };
    const Arg args[10] = {
        {0, sizeof(cl_mem), &qb}, {1, sizeof(cl_mem), &kb},
        {2, sizeof(cl_mem), &vb}, {3, sizeof(cl_mem), &ob},
        {4, sizeof(int), &seq},   {5, sizeof(int), &ctx},
        {6, sizeof(int), &nh},    {7, sizeof(int), &nkv},
        {8, sizeof(int), &hd},    {9, sizeof(int), &cache_pos},
    };
    for (const Arg& a : args) {
        cl_int e2 = g_api->clSetKernelArg(k, a.i, a.sz, a.p);
        if (e2 != CL_SUCCESS)
            throw std::runtime_error("ForwardGpu::attention: clSetKernelArg(" +
                                     std::to_string(a.i) + ") failed (" +
                                     std::to_string(e2) + ")");
    }

    const size_t gsz[2] = {(size_t)seq, (size_t)nh};
    err = g_api->clEnqueueNDRangeKernel(impl_->q, k, 2, nullptr, gsz, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("ForwardGpu::attention: enqueue failed (" +
                                 std::to_string(err) + ")");
    g_api->clEnqueueReadBuffer(impl_->q, ob, CL_TRUE, 0, o_bytes, O, 0, nullptr, nullptr);
    g_api->clFinish(impl_->q);
}
std::string ForwardGpu::device_name() const { return impl_ ? impl_->dev_name : ""; }
std::string ForwardGpu::reason() { return g_reason; }

void ForwardGpu::declare_weight_bytes(size_t total_f32_bytes) {
    impl_->declare_total_weight_bytes(total_f32_bytes);
}

bool ForwardGpu::has_weight(const char* tag) const {
    return impl_ && tag != nullptr && impl_->tagged.count(tag) > 0;
}

void ForwardGpu::matmul_device(const char* tag, const float* A, int seq,
                               int out_dim, int in_dim, const float* bias, float* C) {
    // Same body as matmul(), but the weight buffer is looked up by name and is
    // REQUIRED to exist. No host pointer is dereferenced: in device mode the
    // host has already dropped every projection matrix.
    const size_t a_bytes = (size_t)seq * in_dim * sizeof(float);
    const size_t c_bytes = (size_t)seq * out_dim * sizeof(float);
    cl_int err = CL_SUCCESS;
    cl_mem bw = impl_->weight_buffer(nullptr, (size_t)out_dim * in_dim, tag, true);
    const bool is_bf16 = impl_->last_bf16 != 0;
    cl_kernel kern = is_bf16 ? impl_->k_gemm_bf16 : impl_->k_gemm;

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

    g_api->clEnqueueWriteBuffer(impl_->q, impl_->ba, CL_TRUE, 0, a_bytes, A, 0, nullptr, nullptr);

    cl_mem ba = impl_->ba, bb = impl_->bb, bc = impl_->bc;
    const int has_bias = bias ? 1 : 0;
    Impl::set_args_checked(kern, ba, bw, bb, bc, seq, in_dim, out_dim, has_bias);

    const size_t lws[2] = {(size_t)TSB, (size_t)TSB};
    const size_t gsz[2] = {((size_t)out_dim + TSB - 1) / TSB * TSB,
                           ((size_t)seq + TSB - 1) / TSB * TSB};
    err = g_api->clEnqueueNDRangeKernel(impl_->q, kern, 2, nullptr, gsz, lws,
                                        0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("ForwardGpu: enqueue failed (" + std::to_string(err) +
            ") tag=" + (tag ? tag : "?") + " seq=" + std::to_string(seq) +
            " in=" + std::to_string(in_dim) + " out=" + std::to_string(out_dim) +
            " bias=" + std::to_string(has_bias) + " bf16=" + std::to_string((int)is_bf16));
    g_api->clEnqueueReadBuffer(impl_->q, bc, CL_TRUE, 0, c_bytes, C, 0, nullptr, nullptr);
    g_api->clFinish(impl_->q);
}

bool ForwardGpu::weights_fit(size_t total_f32_bytes) const {
    if (!impl_) return false;
    // NVIDIA does not answer CL_DEVICE_GLOBAL_MEM_FREE_MEMORY (it returns 0),
    // so fall back to the total. The card is otherwise idle when this runs.
    const size_t avail = impl_->dev_free > 0 ? impl_->dev_free : impl_->dev_total;
    if (avail == 0) return true;   // driver will not say; let the upload decide

    // 95%, leaving room for activations, the KV cache and the output buffers.
    // 85%, não 95%. Medido no Qwen2.5-14B (27,5 GB de pesos, 15,5 GB de placa):
    // com 95% o motor aceitava a conta de bf16 (13,8 GB) e mesmo assim estourava
    // com CL_MEM_OBJECT_ALLOCATION_FAILURE (-4) no meio do upload. O que falta
    // não é só o peso: são as ativações por camada, o cache KV e a fragmentação
    // do alocador. Uma guarda que aceita e falha depois é pior que uma que
    // recusa — a falha acontece 20 minutos depois e parece bug de kernel.
    const double budget = (double)avail * 0.85;
    // The question is "can the engine run this model", not "can it do so in
    // f32". f32 is preferred, but bf16 is the fallback — so a model whose f32
    // form (24.3 GB for the 7B) exceeds a 16 GB card is still viable at bf16
    // (12.15 GB), and rejecting it there would have been wrong.
    if ((double)total_f32_bytes <= budget) return true;
    if (const char* env = std::getenv("WINNEX_GPU_DTYPE")) {
        if (std::string(env) == "f32") return false;
        if (std::string(env) == "bf16") return (double)total_f32_bytes * 0.5 <= budget;
    }
    return (double)total_f32_bytes * 0.5 <= budget;
}

void ForwardGpu::release_weights() {
    if (!impl_) return;
    // Os dois mapas, sem duplicar: um cl_mem pode estar em ambos quando um
    // upload tagueado acertou o cache por ponteiro primeiro.
    std::vector<cl_mem> rel;
    for (auto& kv : impl_->tagged) rel.push_back(kv.second);
    for (auto& kv : impl_->wcache) {
        cl_mem m = kv.second.mem;
        bool dup = false;
        for (cl_mem x : rel) if (x == m) { dup = true; break; }
        if (!dup) rel.push_back(m);
    }
    for (cl_mem m : rel) g_api->clReleaseMemObject(m);
    impl_->tagged.clear();
    impl_->wcache.clear();
    impl_->bf16_cache = -1;
    impl_->declared_total_bytes = 0;
}

size_t ForwardGpu::device_memory_total() const { return impl_ ? impl_->dev_total : 0; }
size_t ForwardGpu::device_memory_free() const  { return impl_ ? impl_->dev_free : 0; }

size_t ForwardGpu::device_weight_bytes() const {
    if (!impl_) return 0;
    // Tagged weights are held by cl_mem only (no host pointer, no element
    // count), so ask the driver for the allocation size rather than guessing.
    size_t n = 0;
    for (auto& kv : impl_->tagged) {
        size_t bytes = 0;
        if (g_api->clGetMemObjectInfo(kv.second, CL_MEM_SIZE, sizeof(bytes), &bytes, nullptr)
                == CL_SUCCESS)
            n += bytes;
    }
    return n;
}
bool ForwardGpu::weights_are_bf16() const { return impl_ && impl_->bf16_cache > 0; }

void ForwardGpu::matmul(const float* A, int seq,
                        const float* W, int out_dim, int in_dim,
                        const float* bias, float* C,
                        const char* tag) {
    const size_t a_bytes = (size_t)seq * in_dim * sizeof(float);
    const size_t c_bytes = (size_t)seq * out_dim * sizeof(float);
    cl_int err = CL_SUCCESS;

    // Weights are resident: uploaded once per (pointer, size) for the life of
    // the engine. This is the fix for the measured per-call overhead — see the
    // note in Impl::weight_buffer.
    cl_mem bw = impl_->weight_buffer(W, (size_t)out_dim * in_dim, tag);

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

    // ── Non-blocking transfers ─────────────────────────────────────────────
    //
    // These were CL_TRUE (blocking) for both the write of A and the read of C.
    // Each blocking transfer is a full host<->device round-trip plus a driver
    // sync, and a decode step issues 7 projections x 28 layers = 196 of them.
    // Measured on Qwen2.5-7B: 0.59 s per decoded token, 1.7 tok/s.
    //
    // Neither transfer needs to block. The write is consumed by the kernel
    // that is queued immediately after it on the SAME in-order queue, so
    // ordering already guarantees the data is there; the read is followed by
    // clFinish, which is the only synchronisation the caller actually needs.
    //
    // Correctness is unchanged because this is one in-order command queue:
    // commands complete in submission order. An out-of-order queue would need
    // explicit events, and there is none here.
    // BLOCKING, e isso não é conservadorismo: com CL_FALSE o drive lê `A` de
    // forma assíncrona DEPOIS que a função retorna, e `A` é um std::vector
    // temporário do chamador que pode ser liberado antes. Medido: sequências
    // crescentes (n=2 -> 4 -> 13) divergiam a partir de n=4 enquanto
    // decrescentes funcionavam — o padrão exato de leitura de memória já
    // liberada. A ordem de comando da fila não protege o buffer do HOST; só
    // protege os cl_mem.
    //
    // O ganho da escrita não-bloqueante era zero de qualquer forma: medido
    // 0,592 -> 0,612 s/token ao torná-la assíncrona, dentro do ruído.
    g_api->clEnqueueWriteBuffer(impl_->q, impl_->ba, CL_TRUE, 0, a_bytes, A,
                                0, nullptr, nullptr);

    cl_mem ba = impl_->ba, bb = impl_->bb, bc = impl_->bc;
    const bool is_bf16 = impl_->last_bf16 != 0;
    cl_kernel kern = is_bf16 ? impl_->k_gemm_bf16 : impl_->k_gemm;
    g_api->clSetKernelArg(kern, 0, sizeof(cl_mem), &ba);
    g_api->clSetKernelArg(kern, 1, sizeof(cl_mem), &bw);
    g_api->clSetKernelArg(kern, 2, sizeof(cl_mem), &bb);
    g_api->clSetKernelArg(kern, 3, sizeof(cl_mem), &bc);
    g_api->clSetKernelArg(kern, 4, sizeof(int), &seq);
    g_api->clSetKernelArg(kern, 5, sizeof(int), &in_dim);
    g_api->clSetKernelArg(kern, 6, sizeof(int), &out_dim);
    g_api->clSetKernelArg(kern, 7, sizeof(int), &has_bias);

    // 2-D tiled launch: dim0 = output rows, dim1 = sequence rows. The global
    // size is rounded UP to whole tiles; the kernel masks the edge.
    const size_t lws[2] = {(size_t)TSB, (size_t)TSB};
    const size_t gsz[2] = {((size_t)out_dim + TSB - 1) / TSB * TSB,
                           ((size_t)seq + TSB - 1) / TSB * TSB};
    err = g_api->clEnqueueNDRangeKernel(impl_->q, kern, 2, nullptr, gsz, lws,
                                        0, nullptr, nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("ForwardGpu: enqueue failed (" + std::to_string(err) + ")");
    g_api->clEnqueueReadBuffer(impl_->q, bc, CL_FALSE, 0, c_bytes, C, 0, nullptr, nullptr);
    g_api->clFinish(impl_->q);   // the one sync this call needs

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
