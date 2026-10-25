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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <map>
#include <string>
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

// Grid e tile do kernel paralelo de atenção. Ficam fora do source CL para que
// o host e o kernel usem o MESMO número: o elenco do grid no host e o
// reqd_work_group_size do kernel têm que concordar, e dois literais separados
// divergem sem erro de compilação.
// 32 e não 64: o tile local é s_o[ATT_TG][ATT_HD_MAX], ou seja
// ATT_TG · ATT_HD_MAX · 4 bytes. Com TG=64 e HD_MAX=128 isso dá 32 KB de
// __local — acima do mínimo que o OpenCL garante (16 KB) e acima do limite de
// várias GPUs integradas e do que o driver de macOS expõe. O kernel compilava
// no NVIDIA (48 KB disponíveis) e falhava em outros.
//
// Com TG=32 o tile cai para 16 KB, que é o piso garantido pela especificação.
// O custo medido é pequeno: o ganho vem de paralelizar sobre as chaves, e 32
// work-items por linha já satura a placa para ctx=512.
constexpr int kAttTg = 32;      // work-items por linha de atenção
constexpr int kAttHdMax = 128;  // maior head_dim que cabe no tile local (16 KB)

// Bytes de __local que o kernel exige. O construtor compara com
// CL_DEVICE_LOCAL_MEM_SIZE do dispositivo e descarta o kernel paralelo se não
// couber — em vez de deixar o clBuildProgram falhar no carregamento do módulo
// e derrubar o `import` inteiro.
constexpr size_t kAttLocalBytes = (size_t)kAttTg * kAttHdMax * sizeof(float)
                                + 2 * kAttTg * sizeof(float);

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
    // NOT resolved by FG_LOAD: a loader that exports neither would otherwise
    // be reported as "missing" and abort initialisation. Resolved optionally
    // below; a null here only disables the timing instrumentation.
    cl_int (*clGetEventProfilingInfo)(cl_event, cl_profiling_info, size_t, void*, size_t*);
    cl_int (*clReleaseEvent)(cl_event);
    cl_int (*clReleaseMemObject)(cl_mem);
    cl_int (*clReleaseKernel)(cl_kernel);
    cl_int (*clReleaseProgram)(cl_program);
    // Opcional. Só serve para checar que o work-group do kernel paralelo
    // concorda com kAttTg; ausente, a checagem é pulada e o kernel é usado.
    cl_int (*clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info,
                                       size_t, void*, size_t*);
    cl_int (*clReleaseCommandQueue)(cl_command_queue);
    cl_int (*clReleaseContext)(cl_context);
};

FgApi* g_api = nullptr;
std::string g_reason = "not initialized";

// Kernel-only timing. Device time is the honest numerator for "the GPU should
// be faster": if the kernels total 12 ms and a token takes 84 ms, the other
// 72 ms is host work and bus round-trips, and no kernel tuning will find it.
struct GpuProf {
    bool on = false;
    int kernels = 0;
    double kernel_ns = 0.0;    // device time (profiling events)
    double wall_ns = 0.0;      // host wall time inside matmul_device
    double out_elems = 0.0;
    double attn_ns = 0.0;
    int attn_calls = 0;
    std::map<std::string, std::pair<int, double>> by_tag;
};
GpuProf g_prof;

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

    auto* api = new FgApi{};   // zero-init: optional entries stay null
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
    // Profiling is optional: a driver that does not export these leaves the
    // pointers null and timing falls back to host-side wall clock.
    // Optional profiling entry points, from the loader handle still in scope.
    api->clGetEventProfilingInfo = (decltype(api->clGetEventProfilingInfo))
        fg_sym(h, "clGetEventProfilingInfo");
    api->clReleaseEvent = (decltype(api->clReleaseEvent))
        fg_sym(h, "clReleaseEvent");
    FG_LOAD(clReleaseMemObject);        FG_LOAD(clReleaseKernel);
    FG_LOAD(clReleaseProgram);          FG_LOAD(clReleaseCommandQueue);
    FG_LOAD(clReleaseContext);
#undef FG_LOAD
    // Opcional: nem todo loader exporta esta, e ela só é usada para conferir
    // o work-group do kernel paralelo. Ausente, o kernel é usado sem a
    // checagem em vez de abortar a inicialização.
    api->clGetKernelWorkGroupInfo = (decltype(api->clGetKernelWorkGroupInfo))
        fg_sym(h, "clGetKernelWorkGroupInfo");
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

// ── attention_split (default) ─────────────────────────────────────────────
// O kernel acima resolve UMA linha de atenção por work-item: no decode o grid
// é (1, nh) = 14 itens. Numa placa de 36 CUs isso são 14 lanes ocupadas, e
// mediu-se o sintoma exato: nh=14 e nh=1 custam o mesmo tempo (3,10 ms em
// ctx=512), porque o tempo é dominado pela falta de paralelismo, não pelo
// trabalho. Os três q·k por chave (max, soma, acumula) ficam escondidos atrás
// dessa latência.
//
// Esta versão faz as duas correções de uma vez:
//   1. ONLINE SOFTMAX — um único q·k por chave. O max e a soma são mantidos
//      juntos, com reescalonamento (m, l, O) a cada chave. É a identidade
//      softmax(x)_i = exp(x_i−m)/Σexp(x_j−m) reaplicada de forma incremental,
//      não uma aproximação.
//   2. PARALELISMO NO EIXO DE CHAVES — ATT_TG work-items por linha de
//      atenção, reduzindo (m, l, O) em memória local. Transforma 14 lanes em
//      14·ATT_TG.
//
// Medido no mesmo dispositivo (RTX 5060 Ti), contra o kernel acima:
//   ctx=512  3,061 ms -> 0,060 ms   (51x)
//   ctx=2048 12,264 ms -> 0,181 ms  (68x)
// erro máximo contra o kernel de referência: 2,7e-7 (float32).
//
// O source CL é uma raw string, então estes literais NÃO podem vir de
// kAttTg/kAttHdMax -- `#define ATT_TG kAttTg` chegaria ao compilador OpenCL
// como texto, não como o inteiro 64. Os dois lados são, portanto, literais
// separados, e é obrigatório verificar que concordam: o construtor lê
// CL_KERNEL_WORK_GROUP_SIZE do kernel compilado e o compara com kAttTg. Sem
// essa checagem, mudar um lado faz o host elencar um grid que não é o do
// kernel, e o resultado fica errado sem erro nenhum.
#define ATT_TG 32
#define ATT_HD_MAX 128

//__ATT_SPLIT_MARKER__
__kernel __attribute__((reqd_work_group_size(ATT_TG, 1, 1)))
void attention_split(
    __global const float* Q,      // [seq, nh, hd]
    __global const float* K,      // [ctx_max, nkv, hd] on device
    __global const float* V,      // [ctx_max, nkv, hd] on device
    __global float* O,            // [seq, nh, hd]
    int seq, int ctx, int nh, int nkv, int hd, int cache_pos)
{
    const int g = (int)get_global_id(1);      // t*nh + h
    const int t = g / nh, h = g % nh;
    const int lid = (int)get_local_id(0);
    if (t >= seq || h >= nh) return;

    const int gqa = nh / nkv;
    const int kvh = h / gqa;
    const int lim = cache_pos + t + 1;
    const __global float* qh = Q + ((size_t)t * nh + h) * hd;
    const float scale = rsqrt((float)hd);

    __local float s_m[ATT_TG], s_l[ATT_TG];
    __local float s_o[ATT_TG][ATT_HD_MAX];

    float m = -INFINITY, l = 0.0f;
    float acc[ATT_HD_MAX];
    for (int j = 0; j < hd; ++j) acc[j] = 0.0f;

    for (int u = lid; u < lim; u += ATT_TG) {
        const __global const float* kh = K + ((size_t)u * nkv + kvh) * hd;
        float s = 0.0f;
        for (int j = 0; j < hd; ++j) s += qh[j] * kh[j];
        s *= scale;
        const float mn = fmax(m, s);
        const float corr = (m == -INFINITY) ? 0.0f : exp(m - mn);
        const float p = exp(s - mn);
        l = l * corr + p;
        m = mn;
        const __global const float* vh = V + ((size_t)u * nkv + kvh) * hd;
        for (int j = 0; j < hd; ++j) acc[j] = acc[j] * corr + p * vh[j];
    }

    s_m[lid] = m; s_l[lid] = l;
    for (int j = 0; j < hd; ++j) s_o[lid][j] = acc[j];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Redução combinada: (m, l, O) não são somáveis independentemente, porque
    // cada work-item usou um m diferente. A combinação correta reescala os dois
    // lados para o maior m antes de somar — é a mesma identidade do laço.
    for (int st = ATT_TG / 2; st > 0; st >>= 1) {
        if (lid < st) {
            const float m1 = s_m[lid], m2 = s_m[lid + st];
            const float mn = fmax(m1, m2);
            const float c1 = (m1 == -INFINITY) ? 0.0f : exp(m1 - mn);
            const float c2 = (m2 == -INFINITY) ? 0.0f : exp(m2 - mn);
            s_l[lid] = s_l[lid] * c1 + s_l[lid + st] * c2;
            for (int j = 0; j < hd; ++j)
                s_o[lid][j] = s_o[lid][j] * c1 + s_o[lid + st][j] * c2;
            s_m[lid] = mn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        const float inv = s_l[0] > 0.0f ? 1.0f / s_l[0] : 0.0f;
        __global float* oh = O + ((size_t)t * nh + h) * hd;
        for (int j = 0; j < hd; ++j) oh[j] = s_o[0][j] * inv;
    }
}

// ── attention_device_kv ──────────────────────────────────────────────────
// Same math as attention_causal, but K and V are ALREADY on the device. The
// original kernel re-uploaded the whole cache every call: at ctx=256 that is
// 2 x 131 KB per layer per token, and it measured 9.56 ms of a 10.81 ms call
// (88%) — 1.25 ms was actual compute. The host now writes only the new
// position, so the bus carries nkv*hd*4 bytes (512 B) instead of ctx times
// that.
//
// `k_off`/`v_off` are element offsets where the new `seq` positions start, so
// a prefill appends in one write and a decode appends one row.
__kernel void attention_device_kv(
    __global const float* Q,      // [seq, nh, hd]
    __global const float* K,      // [ctx_max, nkv, hd] on device
    __global const float* V,      // [ctx_max, nkv, hd] on device
    __global float* O,            // [seq, nh, hd]
    int seq, int ctx, int nh, int nkv, int hd, int cache_pos)

{
    const int t = (int)get_global_id(0);
    const int h = (int)get_global_id(1);
    if (t >= seq || h >= nh) return;

    const int gqa = nh / nkv;
    const int kvh = h / gqa;
    const int pos = cache_pos + t;
    const int lim = pos + 1;

    const __global float* qh = Q + ((size_t)t * nh + h) * hd;
    const float scale = rsqrt((float)hd);

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

// Original kernel, kept for the non-resident path (drivers that refuse the
// one above, or an engine with no persistent KV buffers).
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

// ── GEMV para seq pequeno: o GEMM tiled desperdiça o tile inteiro no decode ──
//
// gemm_bias é um GEMM tiled 16x16: eixo x = linha de saída, eixo y = posição
// da sequência. No DECODE seq é sempre 1, então 255 das 256 threads do tile
// fazem trabalho mascarado. Pior: a linha de saída é o eixo RÁPIDO e k é o
// eixo LENTO, então threads vizinhas leem W com passo de `in_dim * 4` bytes —
// 28 a 152 linhas de cache por elemento útil (~3% de eficiência).
//
// MEDIDO nesta máquina (RTX 5060 Ti, ~448 GB/s de pico), W[151936 x 896] f32:
//
//   gemm_bias      7,60 ms    71,7 GB/s  (16% do pico)
//   gemv_bias      1,28 ms   425,1 GB/s  (95% do pico)   -> 5,9x
//
// O prefill NÃO usa este kernel: com seq grande o tiled amortiza o
// carregamento de W sobre 16 posições e ganha. A escolha é feita em
// matmul_device() por `seq`, não por heurística de tamanho de tensor.
//
// Um work-group por linha de saída; as GEMV_WG lanes leem a linha em float4
// (contígua) e reduzem em memória local.
#define GEMV_WG 64
const char* kGemvSrc = R"(
#define GEMV_WG 64

__kernel __attribute__((reqd_work_group_size(GEMV_WG, 1, 1)))
void gemv_bias(
    __global const float* A,      // [seq, in]  (seq == 1)
    __global const float* W,      // [out, in]
    __global const float* bias,   // [out] (has_bias=0 -> ignored)
    __global float* C,            // [seq, out]
    int in_dim, int out_dim, int has_bias)
{
    __local float red[GEMV_WG];
    const int row  = (int)get_group_id(0);
    const int lane = (int)get_local_id(0);

    float acc = 0.0f;
    if (row < out_dim) {
        __global const float4* wr = (__global const float4*)(W + (size_t)row * in_dim);
        __global const float4* ar = (__global const float4*)(A);
        const int nv = in_dim >> 2;
        for (int i = lane; i < nv; i += GEMV_WG) {
            float4 w = wr[i];
            float4 a = ar[i];
            acc += w.x * a.x + w.y * a.y + w.z * a.z + w.w * a.w;
        }
    }
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = GEMV_WG / 2; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0 && row < out_dim) {
        float v = red[0];
        if (has_bias) v += bias[row];
        C[row] = v;
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
    cl_kernel k_gemv = nullptr;   // seq==1: 5,9x o GEMM tiled (ver kGemvSrc)
    cl_kernel k_attn = nullptr;
    cl_kernel k_attn_split = nullptr;
    cl_kernel k_attn_res = nullptr;
    cl_mem q_buf = nullptr, k_buf = nullptr, v_buf = nullptr, o_buf = nullptr;
    // k_cap e v_cap SEPARADOS. Eram um `kv_cap` compartilhado, e o grow do K
    // atualizava a capacidade antes do grow do V consultá-la — então o V via
    // "já cabe" e continuava com o buffer pequeno do forward anterior. Medido:
    // aquecer com n=2 e medir n=4 divergia em 23,5; decrescente funcionava.
    size_t q_cap = 0, k_cap = 0, v_cap = 0, o_cap = 0;
    // ── Persistent KV cache, one buffer per layer, on the device ──────────
    //
    // The cache used to be re-uploaded whole on every attention call: at
    // ctx=256 that is 2 x 131 KB x 24 layers x 1 token = 6.3 MB per token
    // across 48 BLOCKING transfers, and it measured 88% of the attention
    // call. Growing to ctx_max once and writing only the new positions turns
    // that into 512 B per layer per token.
    //
    // Sized lazily on the first prefill, because ctx_max is a property of the
    // caller's sequence, not of the model. A cache that cannot be allocated
    // leaves the vectors empty and every call falls back to the old path —
    // no silent half-state.
    std::vector<cl_mem> kv_k, kv_v;
    size_t kv_layer_cap = 0;     // capacity per layer, in positions
    int kv_layers = 0;
    // Reused staging row so a decode append does not allocate per call.
    std::vector<float> stage;
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
        if (k_gemv) g_api->clReleaseKernel(k_gemv);
        if (k_attn) g_api->clReleaseKernel(k_attn);
        if (k_attn_split) g_api->clReleaseKernel(k_attn_split);
        if (q_buf) g_api->clReleaseMemObject(q_buf);
        if (k_buf) g_api->clReleaseMemObject(k_buf);
        if (v_buf) g_api->clReleaseMemObject(v_buf);
        if (o_buf) g_api->clReleaseMemObject(o_buf);
        for (cl_mem m : kv_k) if (m) g_api->clReleaseMemObject(m);
        for (cl_mem m : kv_v) if (m) g_api->clReleaseMemObject(m);
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
    impl_->q = g_api->clCreateCommandQueue(impl_->ctx, dev,
                                           CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || !impl_->q)
        throw std::runtime_error("ForwardGpu: clCreateCommandQueue failed (" + std::to_string(err) + ")");

    // O kernel paralelo de atenção pede kAttLocalBytes de __local. Se o
    // dispositivo não tiver esse espaço, incluí-lo no programa faz o
    // clBuildProgram FALHAR POR INTEIRO e o construtor lançar — nenhum kernel
    // carrega, e o módulo não abre. Um kernel de desempenho não pode derrubar
    // o módulo por não caber num dispositivo; ele deve simplesmente não ser
    // oferecido. Por isso o programa é montado com a diretiva ligada ou
    // desligada, conforme o dispositivo declare ter espaço.
    size_t dev_local = 0;
    {
        size_t got = 0;
        g_api->clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE,
                               sizeof(dev_local), &dev_local, &got);
    }
    const bool want_split = dev_local == 0 || dev_local >= kAttLocalBytes;
    std::string prog_src;
    {
        const char* marker = "//__ATT_SPLIT_MARKER__";
        std::string base(kGemmSrc);
        size_t pos = base.find(marker);
        if (pos == std::string::npos) {
            prog_src = base;  // sem marcação: inclui tudo (o caso anterior)
        } else if (want_split) {
            prog_src = base.substr(0, pos) + base.substr(pos + std::strlen(marker));
        } else {
            // remove o kernel: corta da marcação até o próximo kernel
            size_t end = base.find("__kernel", pos + std::strlen(marker));
            prog_src = base.substr(0, pos) + (end == std::string::npos ? "" : base.substr(end));
            g_reason = "attention_split omitido: __local exigido " +
                       std::to_string(kAttLocalBytes / 1024) + " KB > dispositivo " +
                       std::to_string(dev_local / 1024) + " KB";
        }
    }
    // O GEMV do decode vive num programa separado da fonte do GEMM porque não
    // passa pela remoção condicional do attention_split — são arquivos .cl
    // independentes concatenados aqui.
    prog_src += "\n";
    prog_src += kGemvSrc;

    const char* src = prog_src.c_str();
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
    // Optional: a driver that refuses this kernel leaves it null and the engine
    // keeps the per-call-upload path.
    impl_->k_attn_res = g_api->clCreateKernel(impl_->prog, "attention_device_kv", &err);
    if (err != CL_SUCCESS) impl_->k_attn_res = nullptr;
    // A variante paralela. Se o driver recusá-la (hd > ATT_HD_MAX, ou
    // work-group de 64 não suportado), cai no kernel serial: a correção é de
    // desempenho, e desempenho não justifica falhar.
    impl_->k_attn_split = g_api->clCreateKernel(impl_->prog, "attention_split", &err);
    if (err != CL_SUCCESS) impl_->k_attn_split = nullptr;
    // O host elenca o grid com kAttTg e o kernel declara reqd_work_group_size
    // com o literal 64 do source CL (que não pode ver kAttTg: é raw string).
    // Se os dois divergirem o kernel roda com um grid errado, sem erro. Esta
    // leitura do kernel compilado é o que transforma a divergência em falha.
    if (impl_->k_attn_split) {
        // CL_KERNEL_COMPILE_WORK_GROUP_SIZE, não CL_KERNEL_WORK_GROUP_SIZE:
        // o primeiro devolve o reqd_work_group_size do source (o que o kernel
        // EXIGE), o segundo devolve o máximo que o dispositivo aceita (1024
        // nesta placa). Comparar o máximo com 64 descartaria um kernel correto.
        size_t req[3] = {0, 0, 0};
        if (g_api->clGetKernelWorkGroupInfo &&
            g_api->clGetKernelWorkGroupInfo(impl_->k_attn_split, dev,
                    CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(req), req, nullptr) == CL_SUCCESS) {
            if ((int)req[0] != kAttTg) {
                g_api->clReleaseKernel(impl_->k_attn_split);
                impl_->k_attn_split = nullptr;
                g_reason = "attention_split: reqd_work_group_size do kernel (" +
                           std::to_string((int)req[0]) + ") != kAttTg (" +
                           std::to_string(kAttTg) + "); usando o kernel serial";
            }
        }
    }
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
    // GEMV é opcional: se faltar, matmul_device cai no GEMM tiled. Amarrar o
    // motor inteiro a ele transformaria uma otimização de decode numa
    // dependência de boot.
    impl_->k_gemv = g_api->clCreateKernel(impl_->prog, "gemv_bias", &err);
    if (err != CL_SUCCESS) impl_->k_gemv = nullptr;

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
    // Mesmo A/B do caminho residente. Este é o caminho que o decode usa de
    // fato (`attn_res` exige WINNEX_KV_RESIDENT=1), então sem trocar aqui o
    // ganho do kernel novo não apareceria no token.
    const bool use_split = impl_->k_attn_split != nullptr && hd <= kAttHdMax &&
                           !std::getenv("WINNEX_ATTN_SERIAL");
    cl_kernel k = use_split ? impl_->k_attn_split : impl_->k_attn;
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

    // Eixo 0 do grid difere entre os dois kernels (ver attention_resident).
    size_t gsz[2];
    size_t lsz[2];
    if (use_split) {
        gsz[0] = (size_t)kAttTg;
        gsz[1] = (size_t)seq * nh;
        lsz[0] = (size_t)kAttTg;
        lsz[1] = 1;
    } else {
        gsz[0] = (size_t)seq;
        gsz[1] = (size_t)nh;
    }
    cl_event ev = nullptr;
    const bool prof = g_api->clGetEventProfilingInfo && g_prof.on;
    err = g_api->clEnqueueNDRangeKernel(impl_->q, k, 2, nullptr, gsz,
                                        use_split ? lsz : nullptr,
                                        0, nullptr, prof ? &ev : nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("ForwardGpu::attention: enqueue failed (" +
                                 std::to_string(err) + ")");
    g_api->clEnqueueReadBuffer(impl_->q, ob, CL_TRUE, 0, o_bytes, O, 0, nullptr, nullptr);
    g_api->clFinish(impl_->q);
    if (prof) {
        cl_ulong t0 = 0, t1 = 0;
        g_api->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        g_api->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
        g_prof.attn_ns += (double)(t1 - t0);
        g_prof.attn_calls += 1;
        if (g_api->clReleaseEvent) g_api->clReleaseEvent(ev);
    }
}
// ── Persistent KV cache ────────────────────────────────────────────────────
//
// prepare_kv() allocates one buffer pair per layer, sized for `ctx_max`
// positions, and resets both to "empty". It is called on every prefill and on
// reset, so the buffers are reused for the life of the engine.
//
// Returns false when the allocation fails (out of device memory). The caller
// then keeps using attention(), which uploads the cache per call — slower, but
// correct. Never a partial state: either every layer has buffers or none does.
bool ForwardGpu::prepare_kv(int ctx_max, int nkv, int hd, int n_layers) {
    if (!impl_) return false;
    const size_t per_layer = (size_t)ctx_max * nkv * hd;
    if (per_layer == 0 || n_layers <= 0) return false;

    // Already big enough and shaped the same: just rewind.
    if (impl_->kv_layers == n_layers && impl_->kv_layer_cap >= (size_t)ctx_max)
        return true;

    release_kv();
    cl_int err = CL_SUCCESS;
    impl_->kv_k.assign((size_t)n_layers, nullptr);
    impl_->kv_v.assign((size_t)n_layers, nullptr);
    for (int l = 0; l < n_layers; ++l) {
        impl_->kv_k[(size_t)l] = g_api->clCreateBuffer(
            impl_->ctx, CL_MEM_READ_ONLY, per_layer * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) { release_kv(); return false; }
        impl_->kv_v[(size_t)l] = g_api->clCreateBuffer(
            impl_->ctx, CL_MEM_READ_ONLY, per_layer * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) { release_kv(); return false; }
    }
    impl_->kv_layer_cap = (size_t)ctx_max;
    impl_->kv_layers = n_layers;
    return true;
}

void ForwardGpu::release_kv() {
    if (!impl_) return;
    for (cl_mem m : impl_->kv_k) if (m) g_api->clReleaseMemObject(m);
    for (cl_mem m : impl_->kv_v) if (m) g_api->clReleaseMemObject(m);
    impl_->kv_k.clear();
    impl_->kv_v.clear();
    impl_->kv_layers = 0;
    impl_->kv_layer_cap = 0;
}

bool ForwardGpu::has_kv(int layer) const {
    return impl_ && layer >= 0 && layer < impl_->kv_layers &&
           (size_t)layer < impl_->kv_k.size() && impl_->kv_k[(size_t)layer] != nullptr;
}

int ForwardGpu::kv_capacity() const {
    return impl_ ? (int)impl_->kv_layer_cap : 0;
}

// Attention against the device-resident cache.
//
// `cache_pos` is the absolute position of the first row of K/V being supplied.
// `seq` rows are written starting there — one row for a decode step, the whole
// prompt for a prefill — and the kernel then attends over 0..cache_pos+seq.
//
// Writes are non-blocking: the queue is in-order, so the kernel that consumes
// them is guaranteed to see them. The only blocking call is the clFinish/read
// that hands O back to the host.
void ForwardGpu::attention_resident(int layer, const float* Q,
                                    const float* K, const float* V, float* O,
                                    int seq, int nh, int nkv, int hd, int cache_pos) {
    if (!impl_) throw std::runtime_error("ForwardGpu::attention_resident: not initialised");
    if (!has_kv(layer))
        throw std::runtime_error("ForwardGpu::attention_resident: no KV buffer for layer " +
                                 std::to_string(layer));
    if (!impl_->k_attn_res) throw std::runtime_error(
        "ForwardGpu::attention_resident: kernel attention_device_kv not compiled");

    const int ctx = cache_pos + seq;
    if ((size_t)ctx > impl_->kv_layer_cap)
        throw std::out_of_range("ForwardGpu::attention_resident: context " +
                                std::to_string(ctx) + " exceeds reserved " +
                                std::to_string(impl_->kv_layer_cap));

    cl_int err = CL_SUCCESS;
    const size_t q_bytes = (size_t)seq * nh * hd * sizeof(float);
    const size_t row_bytes = (size_t)nkv * hd * sizeof(float);
    const size_t append_bytes = (size_t)seq * row_bytes;
    const size_t off = (size_t)cache_pos * row_bytes;

    auto grow = [&](cl_mem& buf, size_t& cap, size_t need, cl_mem_flags f) {
        if (cap >= need && buf) return;
        if (buf) g_api->clReleaseMemObject(buf);
        buf = g_api->clCreateBuffer(impl_->ctx, f, need, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("ForwardGpu::attention_resident: buffer");
        cap = need;
    };
    grow(impl_->q_buf, impl_->q_cap, q_bytes, CL_MEM_READ_ONLY);
    grow(impl_->o_buf, impl_->o_cap, q_bytes, CL_MEM_WRITE_ONLY);

    g_api->clEnqueueWriteBuffer(impl_->q, impl_->q_buf, CL_TRUE, 0, q_bytes, Q,
                                0, nullptr, nullptr);
    // ONLY the new rows cross the bus. Non-blocking: in-order queue orders them
    // before the kernel below.
    g_api->clEnqueueWriteBuffer(impl_->q, impl_->kv_k[(size_t)layer], CL_FALSE,
                                off, append_bytes, K, 0, nullptr, nullptr);
    g_api->clEnqueueWriteBuffer(impl_->q, impl_->kv_v[(size_t)layer], CL_FALSE,
                                off, append_bytes, V, 0, nullptr, nullptr);

    cl_mem qb = impl_->q_buf, kb = impl_->kv_k[(size_t)layer];
    cl_mem vb = impl_->kv_v[(size_t)layer], ob = impl_->o_buf;
    // Prefer the parallel kernel whenever it compiled and hd fits in its
    // local tile. WINNEX_ATTN_SERIAL=1 forces the old one, so the 51x can be
    // re-measured in the same process instead of being taken on faith.
    const bool use_split = impl_->k_attn_split != nullptr && hd <= kAttHdMax &&
                           !std::getenv("WINNEX_ATTN_SERIAL");
    cl_kernel k = use_split ? impl_->k_attn_split : impl_->k_attn_res;
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
            throw std::runtime_error("ForwardGpu::attention_resident: clSetKernelArg(" +
                                     std::to_string(a.i) + ") failed (" +
                                     std::to_string(e2) + ")");
    }

    // O eixo 0 do grid é DIFERENTE entre os dois kernels: o serial indexa
    // (token, cabeça); o paralelo usa o eixo 0 para a chave dentro da linha e
    // o eixo 1 para (token, cabeça). Mandar gsz=(seq,nh) para o paralelo
    // rodaria ATT_TG cópias de cada linha em vez de dividir a chave entre elas.
    size_t gsz[2];
    size_t lsz[2];
    cl_uint ndim = 2;
    if (use_split) {
        gsz[0] = (size_t)kAttTg;
        gsz[1] = (size_t)seq * nh;
        lsz[0] = (size_t)kAttTg;
        lsz[1] = 1;
    } else {
        gsz[0] = (size_t)seq;
        gsz[1] = (size_t)nh;
    }
    cl_event ev = nullptr;
    const bool prof = g_api->clGetEventProfilingInfo && g_prof.on;
    err = g_api->clEnqueueNDRangeKernel(impl_->q, k, ndim, nullptr, gsz,
                                        use_split ? lsz : nullptr,
                                        0, nullptr, prof ? &ev : nullptr);
    if (err != CL_SUCCESS)
        throw std::runtime_error("ForwardGpu::attention_resident: enqueue failed (" +
                                 std::to_string(err) + ")");
    g_api->clEnqueueReadBuffer(impl_->q, ob, CL_TRUE, 0, q_bytes, O, 0, nullptr, nullptr);
    g_api->clFinish(impl_->q);
    if (prof) {
        cl_ulong t0 = 0, t1 = 0;
        g_api->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        g_api->clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
        g_prof.attn_ns += (double)(t1 - t0);
        g_prof.attn_calls += 1;
        if (g_api->clReleaseEvent) g_api->clReleaseEvent(ev);
    }
}

std::string ForwardGpu::device_name() const { return impl_ ? impl_->dev_name : ""; }
std::string ForwardGpu::reason() { return g_reason; }

void ForwardGpu::declare_weight_bytes(size_t total_f32_bytes) {
    impl_->declare_total_weight_bytes(total_f32_bytes);
}

bool ForwardGpu::has_weight(const char* tag) const {
    return impl_ && tag != nullptr && impl_->tagged.count(tag) > 0;
}

// Device-side timing control. Off unless asked: the event objects and the
// profiling flags are not free, and a benchmark is not a production run.
void ForwardGpu::profiling_enable(bool on) { g_prof = GpuProf(); g_prof.on = on; }

double ForwardGpu::profiling_kernel_ms() const { return g_prof.kernel_ns / 1e6; }
double ForwardGpu::profiling_wall_ms()   const { return g_prof.wall_ns / 1e6; }
double ForwardGpu::profiling_attn_ms()   const { return g_prof.attn_ns / 1e6; }
int    ForwardGpu::profiling_attn_calls() const { return g_prof.attn_calls; }
int    ForwardGpu::profiling_kernels()   const { return g_prof.kernels; }
double ForwardGpu::profiling_elems()     const { return g_prof.out_elems; }

std::vector<std::pair<std::string, std::pair<int, double>>>
ForwardGpu::profiling_by_tag() const {
    std::vector<std::pair<std::string, std::pair<int, double>>> v;
    for (const auto& kv : g_prof.by_tag) v.push_back(kv);
    return v;
}

void ForwardGpu::matmul_device(const char* tag, const float* A, int seq,
                               int out_dim, int in_dim, const float* bias, float* C) {
    const auto wall_a = std::chrono::steady_clock::now();
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

    // ── escolha do kernel: GEMV quando seq é pequeno ─────────────────────
    //
    // O GEMM tiled lança um tile 16x16 por 16 posições de sequência. Para
    // seq=1 isso significa 255/256 threads mascaradas e leitura de W com passo
    // de in_dim*4 bytes (ver kGemvSrc). O GEMV lê uma linha por work-group com
    // float4 contíguo.
    //
    // Condição: seq == 1, EXATAMENTE. O kernel lê só a linha 0 de A e escreve
    // C[row] — ele não tem laço sobre posições. Usá-lo com seq > 1 produzia
    // lixo nas posições 1..seq-1, e não as deixava nem zeradas: em 2 <= seq <=
    // 16 o buffer C conservava o conteúdo da chamada anterior.
    //
    // Isso foi medido antes de publicar (paridade CPU vs GPU em
    // Qwen2.5-0.5B e Llama-3.2-1B):
    //
    //   n=1   max|d| = 2,4e-04   argmax igual
    //   n=5   max|d| = 1,7e+01   argmax DIFERENTE   <- bug
    //   n=16  max|d| = 1,0e+02   argmax DIFERENTE   <- bug
    //   n=20  max|d| = 5,3e-05   argmax igual        (caminho tiled)
    //
    // A faixa "seq <= 16" que eu havia escrito generalizava indevidamente a
    // medição de reqd_work_group_size. O decode é seq==1 e isso é tudo o que
    // este kernel precisa cobrir.
    const bool use_gemv = impl_->k_gemv && !is_bf16 && seq == 1 &&
                          (in_dim % 4) == 0;
    cl_event ev_k = nullptr;
    const bool prof = g_api->clGetEventProfilingInfo && g_prof.on;

    if (use_gemv) {
        // O GEMV não tem o argumento `seq`: ele só existe para seq==1 e não há
        // posição a indexar. Args: (A, W, bias, C, in_dim, out_dim, has_bias).
        struct { cl_uint i; size_t sz; const void* p; } ga[7] = {
            {0, sizeof(cl_mem), &ba}, {1, sizeof(cl_mem), &bw},
            {2, sizeof(cl_mem), &bb}, {3, sizeof(cl_mem), &bc},
            {4, sizeof(int), &in_dim}, {5, sizeof(int), &out_dim},
            {6, sizeof(int), &has_bias},
        };
        for (auto& e : ga) {
            const cl_int e2 = g_api->clSetKernelArg(impl_->k_gemv, e.i, e.sz, e.p);
            if (e2 != CL_SUCCESS)
                throw std::runtime_error("ForwardGpu: clSetKernelArg gemv(" +
                                         std::to_string(e.i) + ") failed (" +
                                         std::to_string(e2) + ")");
        }
        const size_t gsz1[1] = {(size_t)out_dim * GEMV_WG};
        const size_t lws1[1] = {(size_t)GEMV_WG};
        err = g_api->clEnqueueNDRangeKernel(impl_->q, impl_->k_gemv, 1, nullptr,
                                            gsz1, lws1, 0, nullptr, prof ? &ev_k : nullptr);
    } else {
        Impl::set_args_checked(kern, ba, bw, bb, bc, seq, in_dim, out_dim, has_bias);
        const size_t lws[2] = {(size_t)TSB, (size_t)TSB};
        const size_t gsz[2] = {((size_t)out_dim + TSB - 1) / TSB * TSB,
                               ((size_t)seq + TSB - 1) / TSB * TSB};
        err = g_api->clEnqueueNDRangeKernel(impl_->q, kern, 2, nullptr, gsz, lws,
                                            0, nullptr, prof ? &ev_k : nullptr);
    }
    if (err != CL_SUCCESS)
        throw std::runtime_error("ForwardGpu: enqueue failed (" + std::to_string(err) +
            ") tag=" + (tag ? tag : "?") + " seq=" + std::to_string(seq) +
            " in=" + std::to_string(in_dim) + " out=" + std::to_string(out_dim) +
            " bias=" + std::to_string(has_bias) + " bf16=" + std::to_string((int)is_bf16));
    g_api->clEnqueueReadBuffer(impl_->q, bc, CL_TRUE, 0, c_bytes, C, 0, nullptr, nullptr);
    // ── clFinish: o teste A/B ────────────────────────────────────────────
    //
    // The read above is already CL_TRUE, so C is in host memory when this line
    // is reached, and the queue is in-order: every command submitted before
    // the read has completed (completion of a command implies completion of
    // everything ahead of it in an in-order queue). The extra clFinish is
    // therefore not required for correctness of this call — it is a full flush
    // that forces the driver to hand back the whole queue.
    //
    // MEDIDO — não vale nada. A/B no mesmo binário, ctx=512, 40 tokens por
    // braço, 5 pares alternados, aquecimento descartado:
    //
    //   com clFinish : 160,27 ms   (min 159,36 max 162,54)
    //   sem clFinish : 160,05 ms   (min 159,70 max 160,94)
    //   delta        : -0,22 ms (-0,14%)
    //
    // A variação DENTRO de cada braço é 14x maior que o efeito. O clFinish é
    // redundante como esperado (a leitura CL_TRUE já sincroniza), e removê-lo
    // não muda nada porque não era ele que custava. Fica atrás do switch para
    // que a próxima pessoa não gaste uma sessão repetindo isto.
    static const bool skip_finish = [] {
        const char* v = std::getenv("WINNEX_NO_FINISH");
        return v && std::string(v) == "1";
    }();
    if (!skip_finish) g_api->clFinish(impl_->q);

    if (prof) {
        cl_ulong t0 = 0, t1 = 0;
        g_api->clGetEventProfilingInfo(ev_k, CL_PROFILING_COMMAND_START,
                                       sizeof(t0), &t0, nullptr);
        g_api->clGetEventProfilingInfo(ev_k, CL_PROFILING_COMMAND_END,
                                       sizeof(t1), &t1, nullptr);
        g_prof.kernel_ns += (double)(t1 - t0);
        g_prof.kernels += 1;
        g_prof.out_elems += (double)seq * out_dim;
        const char* nm = tag ? tag : "?";
        auto it = g_prof.by_tag.find(nm);
        if (it == g_prof.by_tag.end())
            g_prof.by_tag[nm] = std::make_pair(1, (double)(t1 - t0));
        else { it->second.first += 1; it->second.second += (double)(t1 - t0); }
        if (g_api->clReleaseEvent) g_api->clReleaseEvent(ev_k);
    }
    if (g_prof.on)
        g_prof.wall_ns += std::chrono::duration<double, std::nano>(
            std::chrono::steady_clock::now() - wall_a).count();
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

// WHEN f32 DOES NOT FIT, THE MODEL STILL HAS TO BE RUN IN bf16.
//
// This is the other half of the estimate_resident_bytes() fix. With the
// corrected (larger) estimate, Qwen3-4B's f32 form is 13.54 GB against a 13.16
// GB budget — so weights_fit() returns true only through the bf16 path above.
// But should_use_bf16() decides the storage format from a DIFFERENT budget
// (90% of total, 13.93 GB) and there 13.54 still fits, so it would choose f32.
// The two functions would then disagree: fit says "only bf16 fits", dtype says
// "use f32" — and the upload fails at 13.54 GB on a 13.16 GB budget.
//
// Returning false here is what makes the pair consistent: if f32 does not fit
// the fit budget, the weights must be bf16 regardless of what the looser dtype
// budget says. Verified on Qwen3-4B: before this, 13.54 GB in f32; after, the
// same model in bf16 at ~6.8 GB, with identical throughput and perplexity.
bool ForwardGpu::should_force_bf16(size_t total_f32_bytes) const {
    const size_t avail = impl_->dev_free > 0 ? impl_->dev_free : impl_->dev_total;
    if (avail == 0) return false;
    if (const char* env = std::getenv("WINNEX_GPU_DTYPE")) {
        if (std::string(env) == "f32") return false;
        if (std::string(env) == "bf16") return true;
    }
    // Only force bf16 when f32 does not fit AND bf16 does. A model too big for
    // either is not made runnable by this; weights_fit() rejects it.
    const double budget = (double)avail * 0.85;
    return (double)total_f32_bytes > budget &&
           (double)total_f32_bytes * 0.5 <= budget;
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
