// Flat-kernarg wrapper around HipKittens' `micro_tk` 8192³ bf16 GEMM so it can be
// loaded as a standalone gfx942 code object into svod's KFD-direct launcher
// (`AmdProgram::load_external` + `execute_timed`) — no HIP host runtime.
//
// The kernel BODY is copied verbatim from HipKittens'
//   kernels/gemm/bf16fp32/cdna3/8192_256_256_64_16/standalone_launcher.cpp
// with only two changes required by svod's loader:
//   1. Flat kernarg: `extern "C" micro_tk_flat(const bf16* A, const bf16* B,
//      bf16* C)` instead of the 152-byte by-value `micro_globals` struct — the
//      metadata note then exposes three clean `global_buffer` pointer args.
//   2. Static LDS: a `__shared__` 64KB array instead of `extern __shared__`, so
//      the descriptor reports `group_segment_fixed_size = 65536` (svod dispatches
//      the fixed group segment; there is no dynamic-LDS launch argument).
//   3. gridDim: hardcoded from the fixed shape — `gridDim` in the COv5 ABI is read
//      from hidden kernarg the svod loader leaves zero (blockIdx stays hardware).
// Dims are fixed 8192³ (as HK's kernel hardcodes); C is F32 (from the f32
// accumulator). A[M,K], B[N,K], C[M,N] row-major, no B pre-swizzle. Launch is
// 1024 workgroups × 512 threads (8 warps × 64-wide waves).
//
// Build (produces a raw gfx942 code object):
//   hipcc --genco -DKITTENS_CDNA3 --offload-arch=gfx942 -std=c++20 \
//     -I<HipKittens>/include -I/opt/rocm/include/hip \
//     hk_bf16_gemm.cpp -o hk_bf16_gemm_gfx942.co

#include "kittens.cuh"
using namespace kittens;

constexpr int BLOCK_SIZE = 256;
constexpr int K_STEP = 64;
constexpr int REG_BLOCK = BLOCK_SIZE / 4;
constexpr int DOT_SLICE = 16;

#define NUM_WARPS 8
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define M 8192
#define K 8192
#define N 8192

// Runtime dims (exactly as HK's launcher uses), so the load/store helpers see
// identical gl semantics. F32 output (matches aiter/tk2 for a same-dtype
// comparison; the f32 accumulator stores without a bf16 down-cast).
using _gl_A = gl<bf16, -1, -1, -1, -1>;
using _gl_B = gl<bf16, -1, -1, -1, -1>;
using _gl_C = gl<float, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

// `gl` has only a __host__ constructor and no default/device ctor, so fabricate
// one device-side carrying the pointer + the four runtime dims (batch, depth,
// rows, cols), exactly what the host constructor would set. The copy-out uses
// gl's __host__ __device__ copy ctor.
template <typename GL>
__device__ inline GL dev_gl(typename GL::T *p, int rows, int cols) {
    alignas(GL) unsigned char buf[sizeof(GL)];
    GL *g = reinterpret_cast<GL *>(buf);
    g->raw_ptr = p;
    g->batch_internal.v = 1;
    g->depth_internal.v = 1;
    g->rows_internal.v = rows;
    g->cols_internal.v = cols;
    return *g;
}

extern "C" __global__ __launch_bounds__(NUM_THREADS, 2) void micro_tk_flat(const bf16 *A, const bf16 *B, float *C) {
    __shared__ __attribute__((aligned(16))) unsigned char __shm[65536];
    shared_allocator al((int *)&__shm[0]);
    st_bf<BLOCK_SIZE, K_STEP>(&As) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();
    st_bf<BLOCK_SIZE, K_STEP>(&Bs) = al.allocate<st_bf<BLOCK_SIZE, K_STEP>>();

    _gl_A ga = dev_gl<_gl_A>(const_cast<bf16 *>(A), M, K);
    _gl_B gb = dev_gl<_gl_B>(const_cast<bf16 *>(B), N, K);
    _gl_C gc = dev_gl<_gl_C>(C, M, N);

    rt_bf<REG_BLOCK, DOT_SLICE> tiles[8];
    rt_fl<REG_BLOCK, REG_BLOCK, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) {
        zero(C_accum[i]);
    }

    // The grid is a 1-D launch of (M/256)*(N/256)=1024 workgroups; `blockIdx` is a
    // hardware register, but `gridDim` in the COv5 ABI comes from hidden kernarg
    // the svod loader leaves zero — so use the known constants (blockIdx.y == 0).
    int wgid = blockIdx.x;
    const int NUM_WGS = (M / BLOCK_SIZE) * (N / BLOCK_SIZE);
    constexpr int WGM = 4;
    wgid = chiplet_transform_chunked(wgid, NUM_WGS, NUM_XCDS, WGM * WGM);
    const int num_pid_m = ceil_div(M, BLOCK_SIZE);
    const int num_pid_n = ceil_div(N, BLOCK_SIZE);
    int num_wgid_in_group = WGM * num_pid_n;
    int group_id = wgid / num_wgid_in_group;
    int first_pid_m = group_id * WGM;
    int group_size_m = min(num_pid_m - first_pid_m, WGM);
    int pid_m = first_pid_m + ((wgid % num_wgid_in_group) % group_size_m);
    int pid_n = (wgid % num_wgid_in_group) / group_size_m;
    const int row = pid_m;
    const int col = pid_n;

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 4;
    const int warp_col = warp_id % 4;

    const int num_tiles = K / K_STEP;

    G::load(As, ga, {0, 0, row, 0});
    G::load(Bs, gb, {0, 0, col, 0});
    __builtin_amdgcn_s_barrier();

    if (warp_row == 1) {
        __builtin_amdgcn_s_barrier();
    }

#pragma unroll
    for (int tile = 0; tile < num_tiles - 1; ++tile) {
        constexpr int BUFFER_SIZE = (BLOCK_SIZE * K_STEP) / NUM_THREADS;
        float4 a_buffer_next[BUFFER_SIZE * sizeof(bf16) / sizeof(float4)];
        float4 b_buffer_next[BUFFER_SIZE * sizeof(bf16) / sizeof(float4)];

        load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, BUFFER_SIZE, ga, {0, 0, row, tile + 1}, As);
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
        load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
        load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
        load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, BUFFER_SIZE, gb, {0, 0, col, tile + 1}, Bs);
        load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
        load(tiles[6], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
        load(tiles[7], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
        load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
        mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
        store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(C_accum[0], tiles[7], tiles[6], C_accum[0]);
        mma_ABt(C_accum[1], tiles[5], tiles[6], C_accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }

    // Epilogue
    __builtin_amdgcn_sched_barrier(0);
    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 0}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 0}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 1}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 1}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 1}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    load(tiles[0], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 2}));
    load(tiles[1], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 2}));
    load(tiles[2], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 2}));
    load(tiles[3], subtile_inplace<REG_BLOCK, DOT_SLICE>(Bs, {warp_col, 3}));
    load(tiles[4], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row, 3}));
    load(tiles[5], subtile_inplace<REG_BLOCK, DOT_SLICE>(As, {warp_row + 2, 3}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[1], tiles[0], C_accum[0]);
    mma_ABt(C_accum[1], tiles[2], tiles[0], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0], tiles[4], tiles[3], C_accum[0]);
    mma_ABt(C_accum[1], tiles[5], tiles[3], C_accum[1]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_row == 0) {
        __builtin_amdgcn_s_barrier();
    }

    store(gc, C_accum[0], {0, 0, row * 4 + warp_row, col * 4 + warp_col});
    store(gc, C_accum[1], {0, 0, row * 4 + warp_row + 2, col * 4 + warp_col});
}
