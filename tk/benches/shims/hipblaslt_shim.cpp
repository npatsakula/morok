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
#include <rocprim/device/device_segmented_reduce.hpp>      // kmeans-assign arg-min over K
#include <rocprim/device/device_transform.hpp>             // knn score = c_sq − 2·cross (before the sort)
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/thread/thread_operators.hpp> // rocprim::arg_min
#include <rocprim/types/key_value_pair.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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

// Complete vendor kmeans-assign floor: hipBLASLt GEMM (x·cᵀ -> [N,K] cross) +
// rocPRIM segmented arg-min over K of score[n,k] = c_sq[k] − 2·cross[n,k] (the
// nearest-centroid assignment). create/run_ns(median device ns)/destroy.
void* kmeans_create(int64_t N, int64_t K, int64_t D, uint64_t max_ws);
double kmeans_run_ns(void* plan, int warmup, int iters);
void kmeans_destroy(void* plan);
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

// Output-tiling cap (dodges a large-output GEMM fault by splitting output-n into
// ≤2048 chunks): 2048 on gfx1151 AND on the MI300X-VF, 0 (no tiling — single GEMM)
// elsewhere. Two distinct faults, same mitigation:
//   • gfx1151: hipBLASLt illegal-access on large-N small-K GEMMs.
//   • MI300X-VF (gfx942, virtualized): hipBLASLt/Tensile host-init SIGSEGVs at
//     `create` (heuristic get) once any GEMM dim exceeds ~2048 — so knn (M=16384)
//     and kmeans (K=4096), whose only large dim is the output n, previously
//     crashed here untiled. Bare-metal MI300X has neither fault → single GEMM (the
//     truest floor). Detected via the marketing name ("AMD Instinct MI300X VF");
//     `gcnArchName` is plain "gfx942" for both bare-metal and VF. Cached device-0 probe.
inline int64_t arch_tile_cap() {
    static const int64_t cap = [] {
        hipDeviceProp_t p{};
        if (hipGetDeviceProperties(&p, 0) != hipSuccess) return int64_t{2048};
        const bool gfx1151 = std::strncmp(p.gcnArchName, "gfx1151", 7) == 0;
        const bool mi300x_vf = std::strstr(p.name, "VF") != nullptr;
        return (gfx1151 || mi300x_vf) ? int64_t{2048} : int64_t{0};
    }();
    return cap;
}

inline bool matmul(GemmPlan* g) {
    return ok(hipblasLtMatmul(g->handle, g->desc, &g->alpha, g->dA, g->la, g->dB, g->lb, &g->beta, g->dC, g->lc, g->dC,
                              g->ld, &g->heur.algo, g->workspace, g->ws_size, g->stream));
}

// One full (tiled) GEMM = num_tiles tile-matmuls on the shared buffers.
inline bool gemm_tiles(GemmPlan* g) {
    for (int t = 0; t < g->num_tiles; ++t)
        if (!matmul(g)) return false;
    return true;
}

// ── library-call optimisation: multi-heuristic + empirical algo autotune ──
// A real tuned vendor deployment does not take hipBLASLt's single top heuristic
// blind — it asks for several candidate algos and picks the one that is actually
// fastest on this device/shape. That selection is one-time setup (untimed), so
// it makes the floor reflect the best the library can do, not a heuristic guess.
constexpr int ALGO_CANDIDATES = 32; // heuristic algos to consider
constexpr int AUTOTUNE_WARM = 2;    // untimed launches per candidate
constexpr int AUTOTUNE_ITERS = 4;   // timed launches per candidate (min taken)

// Request up to ALGO_CANDIDATES heuristic algos (all fitting `max_ws`) into `out`;
// returns the valid count (0 = no solution).
inline int get_heuristics(hipblasLtHandle_t h, hipblasLtMatmulDesc_t d, hipblasLtMatrixLayout_t a,
                          hipblasLtMatrixLayout_t b, hipblasLtMatrixLayout_t c, hipblasLtMatrixLayout_t dd,
                          uint64_t max_ws, std::vector<hipblasLtMatmulHeuristicResult_t>& out) {
    hipblasLtMatmulPreference_t pref = nullptr;
    if (!ok(hipblasLtMatmulPreferenceCreate(&pref))) return 0;
    int returned = 0;
    if (ok(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws,
                                                 sizeof(max_ws)))) {
        out.resize(ALGO_CANDIDATES);
        if (!ok(hipblasLtMatmulAlgoGetHeuristic(h, d, a, b, c, dd, pref, ALGO_CANDIDATES, out.data(), &returned)))
            returned = 0;
    }
    hipblasLtMatmulPreferenceDestroy(pref);
    out.resize(returned < 0 ? 0 : returned);
    return returned;
}

// Time `run` (nullary, issues device work on `st`) over warm+iters HIP-event
// launches; returns the min device ms (-1 on error). min, not median: it is the
// least-perturbed sample, the right signal for ranking algos.
template <class Run>
double time_ms(hipStream_t st, hipEvent_t s, hipEvent_t e, Run&& run, int warm, int iters) {
    for (int i = 0; i < warm; ++i)
        if (!run()) return -1.0;
    if (!ok(hipStreamSynchronize(st))) return -1.0;
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < iters; ++i) {
        float ms = 0.0f;
        if (!ok(hipEventRecord(s, st)) || !run() || !ok(hipEventRecord(e, st)) || !ok(hipEventSynchronize(e))
            || !ok(hipEventElapsedTime(&ms, s, e)))
            return -1.0;
        best = std::min(best, static_cast<double>(ms));
    }
    return best;
}

// Empirically pick the fastest candidate algo by timing `gemm(g)` (the plan's
// tiled GEMM) for each, and write it into g->heur / g->ws_size. Requires the
// plan's buffers, workspace, stream and events to already exist. The algo only
// affects the GEMM, so the reduce/sort tail is excluded from the selection.
template <class Plan, class GemmFn>
void autotune_gemm(Plan* g, std::vector<hipblasLtMatmulHeuristicResult_t>& cands, GemmFn gemm) {
    int best_i = 0;
    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < cands.size(); ++i) {
        g->heur = cands[i];
        g->ws_size = cands[i].workspaceSize;
        const double t = time_ms(
            g->stream, g->start, g->stop, [&] { return gemm(g); }, AUTOTUNE_WARM, AUTOTUNE_ITERS);
        if (t > 0.0 && t < best) {
            best = t;
            best_i = static_cast<int>(i);
        }
    }
    g->heur = cands[best_i];
    g->ws_size = cands[best_i].workspaceSize;
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

    std::vector<hipblasLtMatmulHeuristicResult_t> heurs;
    if (good) good = get_heuristics(g->handle, g->desc, g->la, g->lb, g->lc, g->ld, max_ws, heurs) > 0;

    if (good) { // size the workspace to the largest candidate so any can be picked
        size_t maxws = 0;
        for (const auto& h : heurs) maxws = std::max(maxws, h.workspaceSize);
        g->heur = heurs[0]; // best-by-heuristic default until autotune runs
        g->ws_size = heurs[0].workspaceSize;
        if (maxws > 0) good = ok(hipMalloc(&g->workspace, maxws));
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

    if (good) autotune_gemm(g, heurs, gemm_tiles); // pick the empirically fastest algo

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
    auto run_full = [&]() { return gemm_tiles(g); };
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
// Three vendor stages, all timed together = a complete hipBLASLt-based knn,
// comparable to svod_tk::knn:
//   1. hipBLASLt GEMM      cross[n,m] = x·cᵀ -> materialised [N,M] f32
//   2. rocPRIM transform   score[n,m] = c_sq[m] − 2·cross[n,m] (the x²-free
//                          squared-L2 order; ‖x‖² is constant per row → drops out).
//                          This combine is REQUIRED for a correct top-K: the
//                          nearest corpus row minimises the distance, NOT the raw
//                          cross (‖c‖² varies per corpus row, so argsort(cross) ≠
//                          nearest). A radix sort needs materialised keys, so the
//                          score is a separate elementwise pass (unlike kmeans,
//                          whose arg-min folds the combine into the reduce iterator).
//   3. rocPRIM segmented radix sort PAIRS per row — keys = score, values = corpus
//                          index. top-K = the first K of each sorted row (key+index).
//                          AMD has no warp-select primitive, so a full per-row sort
//                          carrying the index payload is the realistic library path.
// c_sq is a precomputed f32 input (not timed), matching svod_tk's separately-held
// norms. The GEMM is column-tiled (≤TILE) to dodge the gfx1151 large-N fault;
// tiles write disjoint column slices of the full [N,M] cross (C layout ld = M).

namespace {

struct KnnPlan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    void* ws = nullptr;
    size_t ws_size = 0;
    void *dX = nullptr, *dCent = nullptr, *dScore = nullptr, *dScoreOut = nullptr;
    float* dCsq = nullptr;   // f32 c_sq[M] precomputed corpus norms (the combine input)
    int* dIdxOut = nullptr;  // sorted corpus-index payload [N*M]
    int* dOffsets = nullptr; // N+1 segment offsets (row starts)
    void* dTemp = nullptr;
    size_t temp_size = 0;
    int64_t N = 0, M = 0, D = 0, nt = 0;
    int num_tiles = 1;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    float alpha = 1.0f, beta = 0.0f;
};

// score[i] = c_sq[i mod M] − 2·cross[i] over the flat [N,M] cross — the x²-free
// squared-L2 order (combine step 2). Evaluated on device by the transform pass.
struct KnnScoreOp {
    const float* cross;
    const float* c_sq;
    int M;
    __host__ __device__ float operator()(int i) const { return c_sq[i % M] - 2.0f * cross[i]; }
};

// Within-row corpus index for the flat [N,M] position i — the sort's value payload.
struct IdxModOp {
    int M;
    __host__ __device__ int operator()(int i) const { return i % M; }
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

// Step 1: GEMM into the full [N,M] cross, one tile (≤TILE cols) at a time.
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

// Step 2: combine cross -> score = c_sq − 2·cross, in place over the [N,M] buffer
// (elementwise: reads and writes position i, safe in place). The sort needs
// materialised keys, so unlike kmeans' fused reduce this is a standalone pass.
bool knn_score(KnnPlan* g) {
    auto* s = static_cast<float*>(g->dScore);
    return rocprim::transform(rocprim::make_counting_iterator(0), s, static_cast<size_t>(g->N) * g->M,
                              KnnScoreOp{s, g->dCsq, static_cast<int>(g->M)}, g->stream)
           == hipSuccess;
}

// Step 3: rocPRIM segmented (per-row) ascending radix sort of pairs — keys =
// score, values = corpus index (generated on the fly, i mod M). top-K = the
// first K of each sorted row's keys+indices.
bool knn_sort(KnnPlan* g) {
    auto idx = rocprim::make_transform_iterator(rocprim::make_counting_iterator(0),
                                                IdxModOp{static_cast<int>(g->M)});
    return rocprim::segmented_radix_sort_pairs(g->dTemp, g->temp_size, static_cast<const float*>(g->dScore),
                                               static_cast<float*>(g->dScoreOut), idx, g->dIdxOut,
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

    std::vector<hipblasLtMatmulHeuristicResult_t> heurs;
    if (good) good = get_heuristics(g->handle, g->desc, g->lA, g->lB, g->lC, g->lC, max_ws, heurs) > 0;
    if (good) {
        size_t maxws = 0;
        for (const auto& h : heurs) maxws = std::max(maxws, h.workspaceSize);
        g->heur = heurs[0];
        g->ws_size = heurs[0].workspaceSize;
        if (maxws > 0) good = ok(hipMalloc(&g->ws, maxws));
    }

    const size_t xn = static_cast<size_t>(N) * D, cn = static_cast<size_t>(M) * D, sn = static_cast<size_t>(N) * M;
    if (good)
        good = ok(hipMalloc(&g->dX, xn * 2)) && ok(hipMalloc(&g->dCent, cn * 2)) && ok(hipMalloc(&g->dScore, sn * 4))
               && ok(hipMalloc(&g->dScoreOut, sn * 4))
               && ok(hipMalloc(reinterpret_cast<void**>(&g->dIdxOut), sn * sizeof(int)))
               && ok(hipMalloc(reinterpret_cast<void**>(&g->dCsq), static_cast<size_t>(M) * sizeof(float)));
    if (good) {
        std::vector<uint16_t> hx(xn), hc(cn);
        fill_bf16(hx, 0x12345);
        fill_bf16(hc, 0x6789a);
        std::vector<float> hcsq(static_cast<size_t>(M)); // precomputed ‖c‖², values timing-irrelevant
        uint64_t s = 0xc5c5;
        for (auto& e : hcsq) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            e = static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967295.0f; // [0,1)
        }
        good = ok(hipMemcpy(g->dX, hx.data(), xn * 2, hipMemcpyHostToDevice))
               && ok(hipMemcpy(g->dCent, hc.data(), cn * 2, hipMemcpyHostToDevice))
               && ok(hipMemcpy(g->dCsq, hcsq.data(), static_cast<size_t>(M) * sizeof(float), hipMemcpyHostToDevice))
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
    if (good) { // query + allocate rocPRIM temp storage for the sort_pairs (its largest stage)
        auto idx =
            rocprim::make_transform_iterator(rocprim::make_counting_iterator(0), IdxModOp{static_cast<int>(M)});
        good = (rocprim::segmented_radix_sort_pairs(nullptr, g->temp_size, static_cast<const float*>(g->dScore),
                                                    static_cast<float*>(g->dScoreOut), idx, g->dIdxOut,
                                                    static_cast<unsigned>(N * M), static_cast<unsigned>(N),
                                                    g->dOffsets, g->dOffsets + 1, 0, 32, g->stream)
                == hipSuccess);
        if (good && g->temp_size > 0) good = ok(hipMalloc(&g->dTemp, g->temp_size));
    }
    if (good) autotune_gemm(g, heurs, knn_gemm); // pick the empirically fastest GEMM algo

    if (!good) {
        knn_destroy(g);
        return nullptr;
    }
    return g;
}

double knn_run_ns(void* plan, int warmup, int iters) {
    auto* g = static_cast<KnnPlan*>(plan);
    auto run_full = [&]() { return knn_gemm(g) && knn_score(g) && knn_sort(g); };
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
    if (g->dCsq) hipFree(g->dCsq);
    if (g->dIdxOut) hipFree(g->dIdxOut);
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

// ─────────────────────── complete vendor kmeans-assign floor ──────────────────────
// hipBLASLt GEMM (x·cᵀ -> [N,K] cross) + rocPRIM segmented arg-min over K of the
// x²-free score score[n,k] = c_sq[k] − 2·cross[n,k] (the nearest-centroid
// assignment; the ‖x‖² term is constant across k and drops out). GEMM + arg-min
// are timed together = a complete hipBLASLt-based kmeans-assign, comparable to
// svod_tk::kmeans_assign. c_sq is a precomputed input (not timed), like the knn
// floor's norms. The GEMM is column-tiled (≤TILE) to dodge the gfx1151 large-N
// fault; tiles write disjoint column slices of the [N,K] cross.
//
// The assignment is the .key of each output key_value_pair (the standard rocPRIM
// ArgMin idiom) — left implicit just as the knn floor leaves top-K implicit in
// the sorted rows; no separate int32-extract kernel is added to the timed region.

namespace {

using ArgPair = rocprim::key_value_pair<int, float>;

// Maps a flat row-major index i over [N,K] cross to the arg-min input pair
// {k, score} with k = i mod K (within-row centroid index) and
// score = c_sq[k] − 2·cross[i]. Evaluated on device inside the segmented reduce.
struct ScoreOp {
    const float* cross;
    const float* c_sq;
    int K;
    __host__ __device__ ArgPair operator()(int i) const {
        const int k = i % K;
        return ArgPair(k, c_sq[k] - 2.0f * cross[i]);
    }
};

struct KmeansPlan {
    hipblasLtHandle_t handle = nullptr;
    hipblasLtMatmulDesc_t desc = nullptr;
    hipblasLtMatrixLayout_t lA = nullptr, lB = nullptr, lC = nullptr;
    hipblasLtMatmulHeuristicResult_t heur{};
    void* ws = nullptr;
    size_t ws_size = 0;
    void *dX = nullptr, *dCent = nullptr, *dCross = nullptr; // bf16 x[N,D], bf16 c[K,D], f32 cross[N,K]
    float* dCsq = nullptr;                                   // f32 c_sq[K] precomputed input
    ArgPair* dAssign = nullptr;                              // arg-min output [N] (.key = assignment)
    int* dOffsets = nullptr;                                 // N+1 segment offsets (row starts)
    void* dTemp = nullptr;
    size_t temp_size = 0;
    int64_t N = 0, K = 0, D = 0, nt = 0;
    int num_tiles = 1;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    float alpha = 1.0f, beta = 0.0f;
};

// GEMM into the full [N,K] cross, one column tile (≤TILE) at a time.
bool kmeans_gemm(KmeansPlan* g) {
    for (int t = 0; t < g->num_tiles; ++t) {
        const void* B = static_cast<const char*>(g->dCent) + static_cast<size_t>(t) * g->nt * g->D * sizeof(uint16_t);
        void* C = static_cast<char*>(g->dCross) + static_cast<size_t>(t) * g->nt * sizeof(float);
        if (!ok(hipblasLtMatmul(g->handle, g->desc, &g->alpha, g->dX, g->lA, B, g->lB, &g->beta, C, g->lC, C, g->lC,
                                &g->heur.algo, g->ws, g->ws_size, g->stream)))
            return false;
    }
    return true;
}

// rocPRIM segmented arg-min: N rows of length K, score computed on the fly from
// the cross GEMM + c_sq via ScoreOp; output[n].key = nearest centroid for row n.
bool kmeans_argmin(KmeansPlan* g) {
    auto in = rocprim::make_transform_iterator(
        rocprim::make_counting_iterator(0),
        ScoreOp{static_cast<const float*>(g->dCross), g->dCsq, static_cast<int>(g->K)});
    return rocprim::segmented_reduce(g->dTemp, g->temp_size, in, g->dAssign, static_cast<unsigned>(g->N), g->dOffsets,
                                     g->dOffsets + 1, rocprim::arg_min(),
                                     ArgPair(0, std::numeric_limits<float>::max()), g->stream)
           == hipSuccess;
}

} // namespace

void* kmeans_create(int64_t N, int64_t K, int64_t D, uint64_t max_ws) {
    auto* g = new KmeansPlan();
    g->N = N;
    g->K = K;
    g->D = D;

    const int64_t cap = arch_tile_cap(); // 0 (no fault) → one tile = full K
    const int64_t TILE = cap > 0 ? cap : K;
    g->num_tiles = static_cast<int>((K + TILE - 1) / TILE);
    if (K % g->num_tiles != 0) { // need evenly-sized column tiles
        kmeans_destroy(g);
        return nullptr;
    }
    g->nt = K / g->num_tiles; // ≤ TILE

    const hipblasOperation_t opN = HIPBLAS_OP_N, opT = HIPBLAS_OP_T;
    const hipblasLtOrder_t order = HIPBLASLT_ORDER_ROW;
    // A=x[N,D] (ld D); B-tile=c[nt,D] (ld D, transB=T); C-tile=[N,nt] strided into
    // the full [N,K] cross (ld = K).
    bool good = ok(hipblasLtCreate(&g->handle)) && ok(hipblasLtMatrixLayoutCreate(&g->lA, HIP_R_16BF, N, D, D))
                && ok(hipblasLtMatrixLayoutCreate(&g->lB, HIP_R_16BF, g->nt, D, D))
                && ok(hipblasLtMatrixLayoutCreate(&g->lC, HIP_R_32F, N, g->nt, K));
    for (hipblasLtMatrixLayout_t* L : {&g->lA, &g->lB, &g->lC})
        good = good && ok(hipblasLtMatrixLayoutSetAttribute(*L, HIPBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    good = good && ok(hipblasLtMatmulDescCreate(&g->desc, HIPBLAS_COMPUTE_32F, HIP_R_32F))
           && ok(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)))
           && ok(hipblasLtMatmulDescSetAttribute(g->desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT)));

    std::vector<hipblasLtMatmulHeuristicResult_t> heurs;
    if (good) good = get_heuristics(g->handle, g->desc, g->lA, g->lB, g->lC, g->lC, max_ws, heurs) > 0;
    if (good) {
        size_t maxws = 0;
        for (const auto& h : heurs) maxws = std::max(maxws, h.workspaceSize);
        g->heur = heurs[0];
        g->ws_size = heurs[0].workspaceSize;
        if (maxws > 0) good = ok(hipMalloc(&g->ws, maxws));
    }

    const size_t xn = static_cast<size_t>(N) * D, cn = static_cast<size_t>(K) * D, sn = static_cast<size_t>(N) * K;
    if (good)
        good = ok(hipMalloc(&g->dX, xn * 2)) && ok(hipMalloc(&g->dCent, cn * 2)) && ok(hipMalloc(&g->dCross, sn * 4))
               && ok(hipMalloc(reinterpret_cast<void**>(&g->dCsq), static_cast<size_t>(K) * sizeof(float)))
               && ok(hipMalloc(reinterpret_cast<void**>(&g->dAssign), static_cast<size_t>(N) * sizeof(ArgPair)));
    if (good) {
        std::vector<uint16_t> hx(xn), hc(cn);
        fill_bf16(hx, 0x12345);
        fill_bf16(hc, 0x6789a);
        std::vector<float> hcsq(static_cast<size_t>(K)); // precomputed ‖c‖², values timing-irrelevant
        uint64_t s = 0xc5c5;
        for (auto& e : hcsq) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            e = static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967295.0f; // [0,1)
        }
        good = ok(hipMemcpy(g->dX, hx.data(), xn * 2, hipMemcpyHostToDevice))
               && ok(hipMemcpy(g->dCent, hc.data(), cn * 2, hipMemcpyHostToDevice))
               && ok(hipMemcpy(g->dCsq, hcsq.data(), static_cast<size_t>(K) * sizeof(float), hipMemcpyHostToDevice))
               && ok(hipMemset(g->dCross, 0, sn * 4));
    }
    if (good) {
        std::vector<int> off(static_cast<size_t>(N) + 1);
        for (int64_t i = 0; i <= N; ++i) off[i] = static_cast<int>(i * K);
        good = ok(hipMalloc(reinterpret_cast<void**>(&g->dOffsets), (N + 1) * sizeof(int)))
               && ok(hipMemcpy(g->dOffsets, off.data(), (N + 1) * sizeof(int), hipMemcpyHostToDevice));
    }
    if (good)
        good = ok(hipStreamCreate(&g->stream)) && ok(hipEventCreate(&g->start)) && ok(hipEventCreate(&g->stop));
    if (good) { // query + allocate rocPRIM temp storage for the segmented arg-min
        auto in = rocprim::make_transform_iterator(
            rocprim::make_counting_iterator(0),
            ScoreOp{static_cast<const float*>(g->dCross), g->dCsq, static_cast<int>(g->K)});
        good = (rocprim::segmented_reduce(nullptr, g->temp_size, in, g->dAssign, static_cast<unsigned>(N), g->dOffsets,
                                          g->dOffsets + 1, rocprim::arg_min(),
                                          ArgPair(0, std::numeric_limits<float>::max()), g->stream)
                == hipSuccess);
        if (good && g->temp_size > 0) good = ok(hipMalloc(&g->dTemp, g->temp_size));
    }
    if (good) autotune_gemm(g, heurs, kmeans_gemm); // pick the empirically fastest GEMM algo

    if (!good) {
        kmeans_destroy(g);
        return nullptr;
    }
    return g;
}

double kmeans_run_ns(void* plan, int warmup, int iters) {
    auto* g = static_cast<KmeansPlan*>(plan);
    auto run_full = [&]() { return kmeans_gemm(g) && kmeans_argmin(g); };
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

void kmeans_destroy(void* plan) {
    auto* g = static_cast<KmeansPlan*>(plan);
    if (!g) return;
    if (g->dTemp) hipFree(g->dTemp);
    if (g->dOffsets) hipFree(g->dOffsets);
    if (g->dAssign) hipFree(g->dAssign);
    if (g->dCsq) hipFree(g->dCsq);
    if (g->dCross) hipFree(g->dCross);
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
