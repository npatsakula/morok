// hipBLASLt bf16->f32 GEMM shim for the svod-tk `vendor` bench (benches/vendor.rs).
//
// Exposes a tiny C ABI (device_ok / create / run_ns / destroy) over hipBLASLt's
// host GEMM, timed with HIP events -> on-device nanoseconds (comparable to
// svod's per-kernel PM4 stamps). The plan owns EVERYTHING: handle, desc,
// layouts, chosen algo, workspace, the A/B/C device buffers, stream and events.
//
// The buffers are allocated here (hipMalloc), NOT passed in: svod's AMD backend
// is KFD-direct and HIP cannot enumerate the GPU in the same process, and
// svod's device VAs are not valid in HIP's context. So the vendor bench runs
// svod-free and the shim is fully HIP-world. Buffer contents are irrelevant to
// GEMM timing (zero-filled).
//
// Layout: svod tensors are row-major, so every matrix layout is HIPBLASLT_ORDER_
// ROW with its natural (rows, cols, ld=cols); the op flags (transA/transB)
// transpose the logical stored matrix, matching svod's `x·cᵀ` (transB=T).
//
// Built by the flake's `hipblasltShim` derivation; loaded via libloading. Not cargo.

#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <cstring> // host memset, before rocPRIM (its iterators call unqualified memset)
#include <hipblaslt/hipblaslt.h>
#include <rocprim/device/device_segmented_radix_sort.hpp> // not the umbrella (pulls texture_cache_iterator)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

extern "C" {
int hbl_device_ok(void);
void* hbl_gemm_create(int64_t m, int64_t n, int64_t k, int transA, int transB, uint64_t max_ws);
double hbl_gemm_run_ns(void* plan, int warmup, int iters);
void hbl_gemm_destroy(void* plan);

// Complete vendor knn floor: hipBLASLt GEMM (x·cᵀ -> [N,M]) + rocPRIM segmented
// radix sort (the top-K step). create/run_ns(median device ns)/destroy.
void* knn_create(int64_t N, int64_t M, int64_t D, uint64_t max_ws);
double knn_run_ns(void* plan, int warmup, int iters);
void knn_destroy(void* plan);
}

namespace {

struct GemmPlan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    void* workspace = nullptr;
    size_t ws_size = 0;
    void* dA = nullptr; // bf16 operand A (physical ar*ac)
    void* dB = nullptr; // bf16 operand B (physical br*bc)
    void* dC = nullptr; // f32 output C (m*n), also the D buffer
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    float alpha = 1.0f, beta = 0.0f;
    int num_tiles = 1; // output-n tiling to dodge the gfx1151 large-N fault (see create)
};

inline bool ok(hipblasStatus_t s) { return s == HIPBLAS_STATUS_SUCCESS; }
inline bool ok(hipError_t s) { return s == hipSuccess; }

// Output-tiling cap for the gfx1151 large-N small-K GEMM fault: 2048 on gfx1151,
// 0 (no tiling — single GEMM) on every other arch (gfx942/MI300X have no such
// fault). Cached device-0 arch probe.
inline int64_t arch_tile_cap() {
    static const int64_t cap = [] {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, 0) != hipSuccess) return int64_t{2048};
        return std::strncmp(p.gcnArchName, "gfx1151", 7) == 0 ? int64_t{2048} : int64_t{0};
    }();
    return cap;
}

inline bool matmul(GemmPlan* g) {
    return ok(hipblasLtMatmul(g->handle, g->desc, &g->alpha, g->dA, g->la, g->dB, g->lb, &g->beta, g->dC, g->lc, g->dC,
                              g->ld, &g->heur.algo, g->workspace, g->ws_size, g->stream));
}

} // namespace

// 1 if HIP enumerates at least one device in this process (svod-free), else 0.
int hbl_device_ok(void) {
    int n = 0;
    return (hipGetDeviceCount(&n) == hipSuccess && n > 0) ? 1 : 0;
}

// Plan + allocate one GEMM `C[m,n] = op(A)·op(B)` (A,B bf16; C f32; f32 compute).
// transA/transB: 0 = N, 1 = T. Returns null on any failure (no device, no valid
// solution, OOM) so the caller self-skips the row.
void* hbl_gemm_create(int64_t m, int64_t n, int64_t k, int transA, int transB, uint64_t max_ws) {
    auto* g = new GemmPlan();

    // hipBLASLt on gfx1151 faults (illegal memory access) on large-output (n ≳
    // 3072) small-K GEMMs, so there the output n is tiled into ≤2048 chunks — each
    // a valid GEMM that run_ns issues num_tiles times per timed iteration (the plan
    // and buffers are sized to one tile; tiles reuse them, data is irrelevant to
    // timing). Other arches (gfx942/MI300X) have no such fault → a single GEMM.
    const int64_t cap = arch_tile_cap();
    const int64_t TILE = cap > 0 ? cap : n;
    g->num_tiles = static_cast<int>((n + TILE - 1) / TILE);
    const int64_t nt = (n + g->num_tiles - 1) / g->num_tiles; // per-tile n (≤ TILE)

    const int64_t ar = transA ? k : m, ac = transA ? m : k;  // physical A dims
    const int64_t br = transB ? nt : k, bc = transB ? k : nt; // physical B dims (per tile)
    const hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    const hipblasLtOrder_t order = HIPBLASLT_ORDER_ROW;

    bool good = ok(hipblasLtCreate(&g->handle))
                && ok(hipblasLtMatrixLayoutCreate(&g->la, HIP_R_16BF, ar, ac, ac))
                && ok(hipblasLtMatrixLayoutCreate(&g->lb, HIP_R_16BF, br, bc, bc))
                && ok(hipblasLtMatrixLayoutCreate(&g->lc, HIP_R_32F, m, nt, nt))
                && ok(hipblasLtMatrixLayoutCreate(&g->ld, HIP_R_32F, m, nt, nt));
    for (hipblasLtMatrixLayout_t* L : {&g->la, &g->lb, &g->lc, &g->ld}) {
        good = good && ok(hipblasLtMatrixLayoutSetAttribute(*L, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    }

    good = good && ok(hipblasLtMatmulDescCreate(&g->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F))
           && ok(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)))
           && ok(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

    hipblasLtMatmulPreference_t pref = nullptr;
    good = good && ok(hipblasLtMatmulPreferenceCreate(&pref));
    good = good
           && ok(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
                                                        sizeof(max_ws)));
    int returned = 0;
    if (good) {
        good = ok(hipblasLtMatmulAlgoGetHeuristic(g->handle, g->desc, g->la, g->lb, g->lc, g->ld, pref, 1, &g->heur,
                                                  &returned))
               && returned > 0;
    }
    if (pref) {
        hipblasLtMatmulPreferenceDestroy(pref);
    }

    if (good) {
        g->ws_size = g->heur.workspaceSize;
        if (g->ws_size > 0) {
            good = ok(hipMalloc(&g->workspace, g->ws_size));
        }
    }

    if (good) {
        const size_t a_bytes = static_cast<size_t>(ar) * static_cast<size_t>(ac) * sizeof(uint16_t);
        const size_t b_bytes = static_cast<size_t>(br) * static_cast<size_t>(bc) * sizeof(uint16_t);
        const size_t c_bytes = static_cast<size_t>(m) * static_cast<size_t>(nt) * sizeof(float);
        good = ok(hipMalloc(&g->dA, a_bytes)) && ok(hipMalloc(&g->dB, b_bytes)) && ok(hipMalloc(&g->dC, c_bytes));
        if (good) {
            good = ok(hipMemset(g->dA, 0, a_bytes)) && ok(hipMemset(g->dB, 0, b_bytes))
                   && ok(hipMemset(g->dC, 0, c_bytes));
        }
    }

    good = good && ok(hipStreamCreate(&g->stream)) && ok(hipEventCreate(&g->start)) && ok(hipEventCreate(&g->stop));

    if (!good) {
        hbl_gemm_destroy(g);
        return nullptr;
    }
    return g;
}

// `warmup` untimed launches, then `iters` HIP-event-timed launches; returns the
// median device time in nanoseconds (-1 on any HIP/hipBLASLt error).
double hbl_gemm_run_ns(void* plan, int warmup, int iters) {
    auto* g = static_cast<GemmPlan*>(plan);
    // One "run" = the full (tiled) GEMM: num_tiles tile-matmuls on the shared
    // buffers. The HIP-event region brackets the whole sequence.
    auto run_full = [&]() {
        for (int t = 0; t < g->num_tiles; ++t)
            if (!matmul(g)) return false;
        return true;
    };
    for (int i = 0; i < warmup; ++i) {
        if (!run_full()) return -1.0;
    }
    if (!ok(hipStreamSynchronize(g->stream))) return -1.0;

    std::vector<double> samples;
    samples.reserve(iters > 0 ? static_cast<size_t>(iters) : 0);
    for (int i = 0; i < iters; ++i) {
        float ms = 0.0f;
        if (!ok(hipEventRecord(g->start, g->stream)) || !run_full() || !ok(hipEventRecord(g->stop, g->stream))
            || !ok(hipEventSynchronize(g->stop)) || !ok(hipEventElapsedTime(&ms, g->start, g->stop))) {
            return -1.0;
        }
        samples.push_back(static_cast<double>(ms) * 1.0e6); // ms -> ns
    }
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void hbl_gemm_destroy(void* plan) {
    auto* g = static_cast<GemmPlan*>(plan);
    if (!g) return;
    if (g->dA) hipFree(g->dA);
    if (g->dB) hipFree(g->dB);
    if (g->dC) hipFree(g->dC);
    if (g->workspace) hipFree(g->workspace);
    if (g->start) hipEventDestroy(g->start);
    if (g->stop) hipEventDestroy(g->stop);
    if (g->stream) hipStreamDestroy(g->stream);
    if (g->la) hipblasLtMatrixLayoutDestroy(g->la);
    if (g->lb) hipblasLtMatrixLayoutDestroy(g->lb);
    if (g->lc) hipblasLtMatrixLayoutDestroy(g->lc);
    if (g->ld) hipblasLtMatrixLayoutDestroy(g->ld);
    if (g->desc) hipblasLtMatmulDescDestroy(g->desc);
    if (g->handle) hipblasLtDestroy(g->handle);
    delete g;
}

// ───────────────────────── complete vendor knn floor ─────────────────────────
// hipBLASLt GEMM (x·cᵀ -> [N,M] score) + rocPRIM segmented radix sort per row
// (the top-K step — AMD has no warp-select primitive, so a full per-row sort is
// the realistic library path). GEMM/sort are timed together = a complete
// hipBLASLt-based knn, comparable to svod_tk::knn. The GEMM is column-tiled
// (≤TILE) to dodge the gfx1151 large-N fault; tiles write disjoint column slices
// of the full [N,M] score (C layout ld = M).

namespace {

struct KnnPlan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    void* ws = nullptr;
    size_t ws_size = 0;
    void *dX = nullptr, *dCent = nullptr, *dScore = nullptr, *dScoreOut = nullptr;
    int* dOffsets = nullptr;
    void* dTemp = nullptr;
    size_t temp_size = 0;
    int64_t N = 0, M = 0, D = 0, nt = 0;
    int num_tiles = 1;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    float alpha = 1.0f, beta = 0.0f;
};

// bf16 = high 16 bits of an f32; fill via a cheap LCG so the scores (and thus
// the radix sort) run on varied data, not all-zeros.
void fill_bf16(std::vector<uint16_t>& v, uint64_t seed) {
    uint64_t s = seed | 1;
    for (auto& e : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        float f = (static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967295.0f) * 2.0f - 1.0f;
        uint32_t bits;
        __builtin_memcpy(&bits, &f, 4);
        e = static_cast<uint16_t>(bits >> 16);
    }
}

// GEMM into the full [N,M] score, one tile (≤TILE cols) at a time.
bool knn_gemm(KnnPlan* g) {
    for (int t = 0; t < g->num_tiles; ++t) {
        const void* B = static_cast<const char*>(g->dCent) + static_cast<size_t>(t) * g->nt * g->D * sizeof(uint16_t);
        void* C = static_cast<char*>(g->dScore) + static_cast<size_t>(t) * g->nt * sizeof(float);
        if (!ok(hipblasLtMatmul(g->handle, g->desc, &g->alpha, g->dX, g->lA, B, g->lB, &g->beta, C, g->lC, C, g->lC,
                                &g->heur.algo, g->ws, g->ws_size, g->stream)))
            return false;
    }
    return true;
}

// rocPRIM segmented (per-row) ascending radix sort; top-K = first K of each row.
bool knn_sort(KnnPlan* g) {
    return rocprim::segmented_radix_sort_keys(g->dTemp, g->temp_size, static_cast<const float*>(g->dScore),
                                              static_cast<float*>(g->dScoreOut),
                                              static_cast<unsigned>(g->N * g->M), static_cast<unsigned>(g->N),
                                              g->dOffsets, g->dOffsets + 1, 0, 32, g->stream)
           == hipSuccess;
}

} // namespace

void* knn_create(int64_t N, int64_t M, int64_t D, uint64_t max_ws) {
    auto* g = new KnnPlan();
    g->N = N;
    g->M = M;
    g->D = D;

    const int64_t cap = arch_tile_cap(); // 0 (no fault) → one tile = full M
    const int64_t TILE = cap > 0 ? cap : M;
    g->num_tiles = static_cast<int>((M + TILE - 1) / TILE);
    if (M % g->num_tiles != 0) { // need evenly-sized column tiles
        knn_destroy(g);
        return nullptr;
    }
    g->nt = M / g->num_tiles; // ≤ TILE

    const hipblasOperation_t opN = HIPBLAS_OP_N, opT = HIPBLAS_OP_T;
    const hipblasLtOrder_t order = HIPBLASLT_ORDER_ROW;
    // A=[N,D] (ld D); B-tile=[nt,D] (ld D, transB=T); C-tile=[N,nt] strided into
    // the full [N,M] score (ld = M).
    bool good = ok(hipblasLtCreate(&g->handle)) && ok(hipblasLtMatrixLayoutCreate(&g->lA, HIP_R_16BF, N, D, D))
                && ok(hipblasLtMatrixLayoutCreate(&g->lB, HIP_R_16BF, g->nt, D, D))
                && ok(hipblasLtMatrixLayoutCreate(&g->lC, HIP_R_32F, N, g->nt, M));
    for (hipblasLtMatrixLayout_t* L : {&g->lA, &g->lB, &g->lC})
        good = good && ok(hipblasLtMatrixLayoutSetAttribute(*L, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    good = good && ok(hipblasLtMatmulDescCreate(&g->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F))
           && ok(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)))
           && ok(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));

    hipblasLtMatmulPreference_t pref = nullptr;
    good = good && ok(hipblasLtMatmulPreferenceCreate(&pref));
    good = good
           && ok(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
                                                        sizeof(max_ws)));
    int returned = 0;
    if (good)
        good = ok(hipblasLtMatmulAlgoGetHeuristic(g->handle, g->desc, g->lA, g->lB, g->lC, g->lC, pref, 1, &g->heur,
                                                  &returned))
               && returned > 0;
    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    if (good && g->heur.workspaceSize > 0) good = ok(hipMalloc(&g->ws, g->heur.workspaceSize));
    if (good) g->ws_size = g->heur.workspaceSize;

    const size_t xn = static_cast<size_t>(N) * D, cn = static_cast<size_t>(M) * D, sn = static_cast<size_t>(N) * M;
    if (good)
        good = ok(hipMalloc(&g->dX, xn * 2)) && ok(hipMalloc(&g->dCent, cn * 2)) && ok(hipMalloc(&g->dScore, sn * 4))
               && ok(hipMalloc(&g->dScoreOut, sn * 4));
    if (good) {
        std::vector<uint16_t> hx(xn), hc(cn);
        fill_bf16(hx, 0x12345);
        fill_bf16(hc, 0x6789a);
        good = ok(hipMemcpy(g->dX, hx.data(), xn * 2, hipMemcpyHostToDevice))
               && ok(hipMemcpy(g->dCent, hc.data(), cn * 2, hipMemcpyHostToDevice))
               && ok(hipMemset(g->dScore, 0, sn * 4));
    }
    if (good) {
        std::vector<int> off(static_cast<size_t>(N) + 1);
        for (int64_t i = 0; i <= N; ++i) off[i] = static_cast<int>(i * M);
        good = ok(hipMalloc(reinterpret_cast<void**>(&g->dOffsets), (N + 1) * sizeof(int)))
               && ok(hipMemcpy(g->dOffsets, off.data(), (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    }
    if (good)
        good = ok(hipStreamCreate(&g->stream)) && ok(hipEventCreate(&g->start)) && ok(hipEventCreate(&g->stop));
    if (good) { // query + allocate rocPRIM temp storage
        good = (rocprim::segmented_radix_sort_keys(nullptr, g->temp_size, static_cast<const float*>(g->dScore),
                                                   static_cast<float*>(g->dScoreOut), static_cast<unsigned>(N * M),
                                                   static_cast<unsigned>(N), g->dOffsets, g->dOffsets + 1, 0, 32,
                                                   g->stream)
                == hipSuccess);
        if (good && g->temp_size > 0) good = ok(hipMalloc(&g->dTemp, g->temp_size));
    }

    if (!good) {
        knn_destroy(g);
        return nullptr;
    }
    return g;
}

double knn_run_ns(void* plan, int warmup, int iters) {
    auto* g = static_cast<KnnPlan*>(plan);
    auto run_full = [&]() { return knn_gemm(g) && knn_sort(g); };
    for (int i = 0; i < warmup; ++i) {
        if (!run_full()) return -1.0;
    }
    if (!ok(hipStreamSynchronize(g->stream))) return -1.0;

    std::vector<double> samples;
    samples.reserve(iters > 0 ? static_cast<size_t>(iters) : 0);
    for (int i = 0; i < iters; ++i) {
        float ms = 0.0f;
        if (!ok(hipEventRecord(g->start, g->stream)) || !run_full() || !ok(hipEventRecord(g->stop, g->stream))
            || !ok(hipEventSynchronize(g->stop)) || !ok(hipEventElapsedTime(&ms, g->start, g->stop))) {
            return -1.0;
        }
        samples.push_back(static_cast<double>(ms) * 1.0e6);
    }
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void knn_destroy(void* plan) {
    auto* g = static_cast<KnnPlan*>(plan);
    if (!g) return;
    if (g->dTemp) hipFree(g->dTemp);
    if (g->dOffsets) hipFree(g->dOffsets);
    if (g->dScoreOut) hipFree(g->dScoreOut);
    if (g->dScore) hipFree(g->dScore);
    if (g->dCent) hipFree(g->dCent);
    if (g->dX) hipFree(g->dX);
    if (g->ws) hipFree(g->ws);
    if (g->start) hipEventDestroy(g->start);
    if (g->stop) hipEventDestroy(g->stop);
    if (g->stream) hipStreamDestroy(g->stream);
    if (g->lA) hipblasLtMatrixLayoutDestroy(g->lA);
    if (g->lB) hipblasLtMatrixLayoutDestroy(g->lB);
    if (g->lC) hipblasLtMatrixLayoutDestroy(g->lC);
    if (g->desc) hipblasLtMatmulDescDestroy(g->desc);
    if (g->handle) hipblasLtDestroy(g->handle);
    delete g;
}
