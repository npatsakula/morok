//! HipKittens' `micro_tk` BF16→FP32 GEMM (`256_256_64_16.cpp`, lines 34-227) assembled from the
//! verified [`crate::hk`] leaf helpers — a name-faithful, line-to-line port whose rendered gfx942 IR
//! matches HK's oracle (`hk-micro_tk.ll`): the rolled K-loop's 8 clusters C0-C7, HK's ping-pong, and
//! the truncating fp32→bf16 C store.
//!
//! The loop **skeleton** (Range/End + the 32 `<4×f32>` accumulator loop-carries + the 8-wave ping-pong
//! seed/balance) rides [`crate::schedule::pipeline`] — the thinnest proven driver for the parts that
//! are genuinely intractable to hand-roll (the loop-carry that keeps the accumulators phis, and the
//! `eq=0`/`eq=1` wave-barrier balance that would otherwise deadlock). Everything *inside* the loop is
//! authored directly from the `.cpp` with the HK helpers — NO `MatmulHooks`/`SharedTile`/`TilePool`
//! machinery — so the body reads like the source: `load_global_to_register_buffer`@C0/C4, the
//! `load`/`subtile_inplace_dyn` gathers, `store_register_buffer_to_shared`@C6, the 8 `mma_ABt`, the
//! per-cluster `s_barrier`(+baked `sched.barrier(0)`), the `s_setprio(1/0)` MFMA brackets, and the 3
//! `s_waitcnt lgkmcnt(0)` at C1/C3/C6.

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use crate::build::{BF16, Buf, Builder, Effect, F32, Frag, Idx, Val};
use crate::hk::memory::{G_load, load, load_global_to_register_buffer, store, store_register_buffer_to_shared};
use crate::hk::mma::{mma_ABt, zero};
use crate::hk::sync::{self, s_barrier, s_setprio, s_waitcnt_lgkmcnt, wave_phase_barrier};
use crate::hk::types::{rt_bf, rt_fl, st_bf};
use crate::hk::{BLOCK_SIZE, DOT_SLICE, K_STEP, NUM_THREADS, REG_BLOCK};
use crate::ir::TileId;
use crate::kernels::{Program, l2_swizzle};
use crate::schedule::{PipelineCx, SteadyOut, pipeline};

/// The gfx942 16×16 MFMA edge — one accumulator fragment / K-slice edge.
const EDGE: usize = 16;
/// `rt_fl<64,64>` accumulator = `(64/16)²` = 16 fragments; `C_accum[2]` = 32 loop-carried phis.
const ACC_FRAGS: usize = (REG_BLOCK / EDGE) * (REG_BLOCK / EDGE);
/// `BUFFER_SIZE·sizeof(bf16)/sizeof(float4)` = 32·2/16 = 4 — `raw.buffer.load.i128` chunks per tile.
const NCHUNKS: usize = 4;

/// HK's `micro_globals` (`256_256_64_16.cpp:24`) — the three tensors, bound outputs-first (`c` is the
/// launch ABI slot 0). `a`/`b` are `[M,K]`/`[N,K]` (both K-contiguous — the pre-transposed `A·Bᵀ`).
pub struct micro_globals {
    pub c: Buf<BF16>,
    pub a: Buf<BF16>,
    pub b: Buf<BF16>,
}

/// The per-tile DRAM-prefetch addressing for HK's `load_global_to_register_buffer` (`{0,0,origin,tile}`):
/// the SRD base (fixed at buffer element 0, so the whole-buffer range covers every access — the
/// device-race-safe `fixed base` proven bit-exact by `movement::LdsStage::prefetch`, vs HK's advancing
/// base) and the `NCHUNKS` per-lane `i128` byte offsets. The 512-thread collaborative fill copies the
/// `BLOCK_SIZE × K_STEP` tile straight into LDS: thread `tid`'s chunk `cg` is element `tid·32 + cg·8`,
/// row `r = flat/K_STEP`, col `c = flat%K_STEP`, global element `(origin + r)·K + k_base + c`.
fn buffer_load_addr(b: &mut Builder, tid: Idx, origin: Idx, k_base: Idx, k: i64) -> (Idx, Vec<Idx>) {
    let epl = (BLOCK_SIZE * K_STEP) / NUM_THREADS; // 32 bf16/thread
    let k_c = b.idx_const(k);
    // Advancing base (HK's `make_srsrc(base_ptr = &src[{0,0,origin,tile}])`, global_to_shared.cuh:114):
    // the workgroup-uniform `origin·K + k_base` rides in the SRD descriptor (a scalar `s_add` per
    // K-tile), leaving each per-lane voffset loop-invariant — so no per-iteration `v_add` lands on the
    // load-address critical path. Arithmetically identical to the fixed-base form (bit-exact).
    let orig_k = b.idx_mul(origin, k_c);
    let base_off = b.idx_add(orig_k, k_base);
    let epl_c = b.idx_const(epl as i64);
    let lane_epl = b.idx_mul(tid, epl_c);
    let cols_c = b.idx_const(K_STEP as i64);
    let two = b.idx_const(2);
    let voffsets = (0..NCHUNKS)
        .map(|cg| {
            let ec = b.idx_const((cg * 8) as i64); // i128 chunk = 8 bf16
            let flat = b.idx_add(lane_epl, ec);
            let r = b.idx_div(flat, cols_c);
            let c = b.idx_mod(flat, cols_c);
            let rk = b.idx_mul(r, k_c); // r·K — per-lane, loop-invariant
            let goff = b.idx_add(rk, c); // r·K + c
            b.idx_mul(goff, two) // → byte offset
        })
        .collect();
    (base_off, voffsets)
}

/// One HK `load(rt_bf, subtile_inplace<64,16>(lds, {blk_row, slice}))` gather — a fresh operand tile
/// (4 fragments = 4 `ds_read_b64`) at the warp's runtime row-block `blk_row·64` + K-slice `slice·16`.
fn gather_tile(b: &mut Builder, lds: st_bf, blk_row: Idx, slice: usize, wlane: Idx, raw: &[TileId]) -> Vec<Val<BF16>> {
    let sub = lds.subtile_inplace_dyn(REG_BLOCK, DOT_SLICE, blk_row, slice);
    let mut rt = rt_bf::new(b, REG_BLOCK, DOT_SLICE);
    load(b, &mut rt, sub, wlane, raw)
}

/// Seal a memory cluster with HK's `s_barrier()`+`sched_barrier(0)` (one bare `s_barrier`) over every
/// gathered operand value, so the cluster's `ds_read_b64`s stay bracketed inside it. Returns the seal.
fn seal_gathers(b: &mut Builder, gathers: &[&[Val<BF16>]]) -> TileId {
    let ids: Vec<TileId> = gathers.iter().flat_map(|g| g.iter().map(|v| v.id)).collect();
    s_barrier(b, Effect(ids[0]), &ids[1..]).dep()
}

/// One HK compute cluster (C1/C3/C5/C7): the optional `s_waitcnt lgkmcnt(0)` LDS drain (C1/C3), the
/// `s_setprio(1)` bracket, `mma_ABt(C_accum[0], a0, bsh)` + `mma_ABt(C_accum[1], a1, bsh)` (32 MFMAs
/// over the two 4×4 accumulator halves), `s_setprio(0)`, then the `s_barrier` seal. `acc_prev` is the
/// per-accumulator carry (the loop carry on C1, the prior cluster's stores after); `entry` the prior
/// cluster's seal. Returns the 32 accumulator stores + the seal.
#[allow(clippy::too_many_arguments)]
fn compute_cluster(
    b: &mut Builder,
    accs: &[Frag<F32>],
    a0: &[Val<BF16>],
    a1: &[Val<BF16>],
    bsh: &[Val<BF16>],
    acc_prev: &[Vec<TileId>],
    entry: &[TileId],
    with_lgkm: bool,
) -> (Vec<Effect>, TileId) {
    let ri = REG_BLOCK / EDGE; // 4
    let mut pre = entry.to_vec();
    if with_lgkm {
        // C1/C3: drain the prior mem cluster's gather `ds_read`s before the MFMAs consume them.
        pre.push(s_waitcnt_lgkmcnt(b, entry[0]).dep());
    }
    // Anchor `s_setprio(1)` AFTER the cluster-entry barrier + the lgkmcnt drain (`pre`), like HK
    // (cpp:99-100: `s_waitcnt lgkmcnt(0)` then `s_setprio(1)`). Anchoring on the first gather value
    // instead let it float BEFORE the barrier — the wave then held raised priority during the barrier
    // wait, starving its ping-pong partner's memory phase (a priority inversion that defeats the
    // ping-pong). `pre` = [entry barrier, lgkmcnt], so setprio(1) brackets ONLY the MFMA burst.
    let prio1 = s_setprio(b, 1, &pre).dep();
    let reads: Vec<Val<F32>> = (0..ACC_FRAGS * 2)
        .map(|i| {
            let mut deps = acc_prev[i].clone();
            deps.extend(&pre);
            deps.push(prio1);
            b.load_frag_vec_after(accs[i], &deps)
        })
        .collect();
    let mut new = mma_ABt(b, a0, bsh, &reads[..ACC_FRAGS], ri, ri);
    new.extend(mma_ABt(b, a1, bsh, &reads[ACC_FRAGS..], ri, ri));
    let new_ids: Vec<TileId> = new.iter().map(|v| v.id).collect();
    let stores: Vec<Effect> = accs.iter().zip(&new).map(|(&a, &v)| b.store_frag_vec(a, v)).collect();
    let prio0 = s_setprio(b, 0, &new_ids).dep();
    let last = stores.len() - 1;
    let mut deps: Vec<TileId> = stores[..last].iter().map(|e| e.dep()).collect();
    deps.push(prio0);
    let bar = s_barrier(b, stores[last], &deps).dep();
    (stores, bar)
}

/// `pub fn micro_tk(m, n, k)` — HipKittens' `micro_tk` (`256_256_64_16.cpp:35`), the BF16→FP32
/// `A·Bᵀ` GEMM. The tiling is hard-coded to HK's `256_256_64_16` reference (`BLOCK_SIZE=256`,
/// `K_STEP=64`, `REG_BLOCK=64`, `DOT_SLICE=16`, 8 warps / 512 threads); `m/n/k` only size the buffers
/// + trip count (`m,n` multiples of 256, `k` a multiple of 64 with `k/64 ≥ 2`).
pub fn micro_tk(m: usize, n: usize, k: usize) -> Program {
    assert!(m.is_multiple_of(BLOCK_SIZE) && n.is_multiple_of(BLOCK_SIZE), "m/n multiples of BLOCK_SIZE(256)");
    assert!(k.is_multiple_of(K_STEP) && k / K_STEP >= 2, "k multiple of K_STEP(64), ≥2 tiles");
    let mut b = Builder::new("micro_tk");

    // ── micro_globals: outputs-first ABI (c, then a, then b). C is bf16 (truncated). ──
    let g = micro_globals { c: b.global::<BF16>(m * n), a: b.global::<BF16>(m * k), b: b.global::<BF16>(n * k) };

    // ── grid swizzle (chiplet_transform_chunked + grouped-M): wgid → (row, col) tile coords. ──
    let (grid_m, grid_n) = ((m / BLOCK_SIZE) as i64, (n / BLOCK_SIZE) as i64);
    let (row, col) = if grid_m % 4 == 0 {
        let wgid = b.grid_axis(0, grid_m * grid_n);
        l2_swizzle(&mut b, wgid, grid_m, grid_n)
    } else {
        (b.grid_axis(0, grid_m), b.grid_axis(1, grid_n))
    };
    let tid = b.block_axis(NUM_THREADS as i64);
    let warp_row = sync::warp_row(&mut b, tid); // warpid()/4 ∈ {0,1}
    let warp_col = sync::warp_col(&mut b, tid); // warpid()%4 ∈ {0,1,2,3}
    let wlane = {
        let w = b.idx_const(64);
        b.idx_mod(tid, w) // laneid() = tid % 64
    };
    let block = b.idx_const(BLOCK_SIZE as i64);
    let origin_a = b.idx_mul(row, block); // A M-row base (elements)
    let origin_b = b.idx_mul(col, block); // B N-row base (elements)
    let k_i = k as i64;

    // ── LDS: `st_bf<256,64> As, Bs` bump-allocated ONCE (32 KiB each = the 64 KiB dynamic shared
    //    limit) and single-buffered across the whole pipeline — the SAME two tiles the prologue fills,
    //    the steady loop re-fills (C6) + gathers, and the epilogue gathers. ──
    let As = st_bf::new(&mut b, BLOCK_SIZE, K_STEP);
    let Bs = st_bf::new(&mut b, BLOCK_SIZE, K_STEP);

    // ── accumulators: C_accum[0..2] : rt_fl<64,64,col>, each zeroed (the 32 loop-carried phis). ──
    let c0 = rt_fl::new(&mut b, REG_BLOCK, REG_BLOCK);
    let c1 = rt_fl::new(&mut b, REG_BLOCK, REG_BLOCK);
    let accs: Vec<Frag<F32>> = c0.frags.iter().chain(&c1.frags).copied().collect();
    let inited: Vec<Effect> = zero(&mut b, &c0).into_iter().chain(zero(&mut b, &c1)).collect();
    let accs_steady = accs.clone();

    let two = b.idx_const(2);

    // HK's 8-wave ping-pong is authored HERE, directly from `256_256_64_16.cpp` (the prologue's
    // `if(warp_row==1) s_barrier()` and the epilogue's `if(warp_row==0) s_barrier()`) via the hk
    // `wave_phase_barrier` util — NOT delegated to the pipeline driver (so this stays a faithful HK
    // reproduction, not a pipe2 borrow). The driver is handed `None`, so it emits no phase barrier of
    // its own; it owns only the generic Range/End loop skeleton + the 32-accumulator carry-fold.
    // `SVOD_NO_PINGPONG` (diagnostic) drops the pair to isolate HK's single-buffer async-LDS race.
    let pingpong = std::env::var("SVOD_NO_PINGPONG").is_err();

    let out = pipeline(
        &mut b,
        k / K_STEP,
        K_STEP,
        &accs,
        &inited,
        None,
        // ── prologue: G::load(As)/G::load(Bs); s_barrier(); if(warp_row==1) s_barrier();  (cpp:73-79) ──
        move |b| {
            let zero = b.idx_const(0);
            let (base_a, voffs_a) = buffer_load_addr(b, tid, origin_a, zero, k_i);
            let drain_a = G_load(b, As, g.a, base_a, &voffs_a, tid, &[]);
            let (base_b, voffs_b) = buffer_load_addr(b, tid, origin_b, zero, k_i);
            let drain_b = G_load(b, Bs, g.b, base_b, &voffs_b, tid, &[]);
            let bar = s_barrier(b, drain_b, &[drain_a.dep()]);
            match pingpong {
                true => wave_phase_barrier(b, warp_row, 1, &[bar.dep()]).dep(),
                false => bar.dep(),
            }
        },
        // ── steady: the rolled loop body — 8 clusters C0-C7 (line-to-line with the .cpp). ──
        move |b, cx: &PipelineCx| {
            let raw = cx.raw.deps().to_vec(); // LDS-RAW carry (last iter's C6 seal)
            let carry: Vec<Vec<TileId>> = cx.accs.iter().map(|c| c.deps().to_vec()).collect();
            let k_next = cx.next_base; // (tile+1)·K_STEP
            let wr2 = b.idx_add(warp_row, two);

            // C0: prefetch A(k+1) → a_buffer_next; gather slice 0 (t1,t2 = A rows; t0 = B cols).
            let (base_a, voffs_a) = buffer_load_addr(b, tid, origin_a, k_next, k_i);
            let a_buf = load_global_to_register_buffer(b, g.a, base_a, &voffs_a, &raw);
            let t1 = gather_tile(b, As, warp_row, 0, wlane, &raw);
            let t2 = gather_tile(b, As, wr2, 0, wlane, &raw);
            let t0 = gather_tile(b, Bs, warp_col, 0, wlane, &raw);
            // Seal `a_buf` (the A prefetch) into C0's barrier too — HK pins its C0 load with
            // `sched_barrier(0)` (cpp:96) so it issues here and overlaps C1–C5, instead of the tk2
            // scheduler sinking it to its C6 consumer. Keeps the 4 A-chunks live C0→C6 (HK's live range).
            let bar0 = seal_gathers(b, &[t1.as_slice(), t2.as_slice(), t0.as_slice(), a_buf.as_slice()]);

            // C1: mma slice 0.
            let (st, bar1) = compute_cluster(b, &accs_steady, &t1, &t2, &t0, &carry, &[bar0], true);
            let mut prev: Vec<Vec<TileId>> = st.iter().map(|e| vec![e.dep()]).collect();

            // C2: gather slice 1 (t3,t4,t5) + slice 2 lead (t0,t1).
            let t3 = gather_tile(b, Bs, warp_col, 1, wlane, &[bar1]);
            let t4 = gather_tile(b, As, warp_row, 1, wlane, &[bar1]);
            let t5 = gather_tile(b, As, wr2, 1, wlane, &[bar1]);
            let t0 = gather_tile(b, Bs, warp_col, 2, wlane, &[bar1]);
            let t1 = gather_tile(b, As, warp_row, 2, wlane, &[bar1]);
            let bar2 = seal_gathers(b, &[t3.as_slice(), t4.as_slice(), t5.as_slice(), t0.as_slice(), t1.as_slice()]);

            // C3: mma slice 1.
            let (st, bar3) = compute_cluster(b, &accs_steady, &t4, &t5, &t3, &prev, &[bar2], true);
            prev = st.iter().map(|e| vec![e.dep()]).collect();

            // C4: prefetch B(k+1) → b_buffer_next; gather slice 2 tail (t2) + slice 3 (t6,t7,t5).
            let (base_b, voffs_b) = buffer_load_addr(b, tid, origin_b, k_next, k_i);
            let b_buf = load_global_to_register_buffer(b, g.b, base_b, &voffs_b, &[bar3]);
            let t2 = gather_tile(b, As, wr2, 2, wlane, &[bar3]);
            let t6 = gather_tile(b, Bs, warp_col, 3, wlane, &[bar3]);
            let t7 = gather_tile(b, As, warp_row, 3, wlane, &[bar3]);
            let t5 = gather_tile(b, As, wr2, 3, wlane, &[bar3]);
            // Seal `b_buf` (the B prefetch) into C4's barrier — HK pins its C4 load with `sched_barrier(0)`
            // (cpp:132) so it issues here and overlaps C5–C6, not sunk to C6. Keeps 4 B-chunks live C4→C6.
            let bar4 = seal_gathers(b, &[t2.as_slice(), t6.as_slice(), t7.as_slice(), t5.as_slice(), b_buf.as_slice()]);

            // C5: mma slice 2 (no lgkmcnt — HK).
            let (st, bar5) = compute_cluster(b, &accs_steady, &t1, &t2, &t0, &prev, &[bar4], false);
            prev = st.iter().map(|e| vec![e.dep()]).collect();

            // C6: s_waitcnt lgkmcnt(0); store_register_buffer_to_shared(As) + (Bs) (the reg→LDS commit).
            let lgkm = s_waitcnt_lgkmcnt(b, bar5);
            let a_st = store_register_buffer_to_shared(b, As, &a_buf, tid, Some(lgkm.dep()));
            let a_last = a_st.last().map(|e| e.dep());
            let b_st = store_register_buffer_to_shared(b, Bs, &b_buf, tid, a_last);
            let writes: Vec<TileId> = a_st.iter().chain(&b_st).map(|e| e.dep()).collect();
            let bar6 = s_barrier(b, Effect(writes[0]), &writes[1..]);

            // C7: mma slice 3. Fold C7's seal into `raw_next` (the LDS carry the End keeps live) so it
            // is NOT DCE'd — else the loop body loses its 8th `s_barrier`/`sched.barrier`/`setprio(0)`.
            let (st, bar7) = compute_cluster(b, &accs_steady, &t7, &t5, &t6, &prev, &[bar6.dep()], false);
            let raw_next = b.combine(bar6, &[bar7]).dep();
            SteadyOut { acc_stores: st, raw_next }
        },
        // ── epilogue: last tile's clusters C0,C1,C2,C3,C4,C5,C7 (NO C6 commit); then HK's
        //    `if(warp_row==0) s_barrier()` (cpp:221). Mem clusters drain lgkmcnt at their tail. The
        //    driver passes `None` (we own the ping-pong), so its `warp_row` param is unused here. ──
        move |b, ended, acc_loop, _warp_row_pp| {
            let wr2 = b.idx_add(warp_row, two);
            let none: Vec<Vec<TileId>> = (0..ACC_FRAGS * 2).map(|_| Vec::new()).collect();

            // A memory cluster: gather the listed tiles off `ended`, drain lgkmcnt, seal.
            let mem = |b: &mut Builder, gs: &[Vec<Val<BF16>>]| -> TileId {
                let ids: Vec<TileId> = gs.iter().flat_map(|g| g.iter().map(|v| v.id)).collect();
                let lgkm = s_waitcnt_lgkmcnt(b, ids[ids.len() - 1]);
                s_barrier(b, lgkm, &ids).dep()
            };

            // C0 + C1 (slice 0).
            let t0 = gather_tile(b, Bs, warp_col, 0, wlane, &[ended]);
            let t1 = gather_tile(b, As, warp_row, 0, wlane, &[ended]);
            let t2 = gather_tile(b, As, wr2, 0, wlane, &[ended]);
            let bar0 = mem(b, &[t0.clone(), t1.clone(), t2.clone()]);
            let (st, bar1) = compute_cluster(b, acc_loop, &t1, &t2, &t0, &none, &[bar0], false);
            let mut prev: Vec<Vec<TileId>> = st.iter().map(|e| vec![e.dep()]).collect();

            // C2 + C3 (slice 1).
            let t3 = gather_tile(b, Bs, warp_col, 1, wlane, &[bar1]);
            let t4 = gather_tile(b, As, warp_row, 1, wlane, &[bar1]);
            let t5 = gather_tile(b, As, wr2, 1, wlane, &[bar1]);
            let bar2 = mem(b, &[t3.clone(), t4.clone(), t5.clone()]);
            let (st, bar3) = compute_cluster(b, acc_loop, &t4, &t5, &t3, &prev, &[bar2], false);
            prev = st.iter().map(|e| vec![e.dep()]).collect();

            // C4 (slices 2+3) + C5 (slice 2) + C7 (slice 3).
            let t0 = gather_tile(b, Bs, warp_col, 2, wlane, &[bar3]);
            let t1 = gather_tile(b, As, warp_row, 2, wlane, &[bar3]);
            let t2 = gather_tile(b, As, wr2, 2, wlane, &[bar3]);
            let t3 = gather_tile(b, Bs, warp_col, 3, wlane, &[bar3]);
            let t4 = gather_tile(b, As, warp_row, 3, wlane, &[bar3]);
            let t5 = gather_tile(b, As, wr2, 3, wlane, &[bar3]);
            let bar4 = mem(b, &[t0.clone(), t1.clone(), t2.clone(), t3.clone(), t4.clone(), t5.clone()]);
            let (st, bar5) = compute_cluster(b, acc_loop, &t1, &t2, &t0, &prev, &[bar4], false);
            prev = st.iter().map(|e| vec![e.dep()]).collect();
            let (st7, bar7) = compute_cluster(b, acc_loop, &t4, &t5, &t3, &prev, &[bar5], false);

            // HK's `if(warp_row==0) s_barrier()` (cpp:221) — the eq=0 phase rebalance that closes the
            // ping-pong pair, authored from HK's code + routed into the final accumulators (kept live).
            let scatter_seed = pingpong.then(|| wave_phase_barrier(b, warp_row, 0, &[bar7]).dep());
            acc_loop
                .iter()
                .zip(st7)
                .map(|(&a, s)| {
                    let mut deps = vec![s.dep()];
                    deps.extend(scatter_seed);
                    b.frag_after(a, &deps)
                })
                .collect()
        },
    );

    // ── C store: fp32→bf16 truncate. C_accum[0] → {row·4+warp_row, col·4+warp_col};
    //    C_accum[1] → {row·4+warp_row+2, …} (HK's interleaved 64-row-block map). ──
    let n_c = b.idx_const(n as i64);
    let four = b.idx_const(4);
    let e64 = b.idx_const(REG_BLOCK as i64);
    let row4 = b.idx_mul(row, four);
    let rt0 = b.idx_add(row4, warp_row);
    let rt1 = b.idx_add(rt0, two);
    let col4 = b.idx_mul(col, four);
    let ct = b.idx_add(col4, warp_col);
    let c_elem = b.idx_mul(ct, e64);
    let r_elem0 = b.idx_mul(rt0, e64);
    let r_elem1 = b.idx_mul(rt1, e64);
    let rn0 = b.idx_mul(r_elem0, n_c);
    let base0 = b.idx_add(rn0, c_elem);
    let rn1 = b.idx_mul(r_elem1, n_c);
    let base1 = b.idx_add(rn1, c_elem);
    let c_accum0 = rt_fl { frags: out[..ACC_FRAGS].to_vec(), rows: REG_BLOCK, cols: REG_BLOCK };
    let c_accum1 = rt_fl { frags: out[ACC_FRAGS..].to_vec(), rows: REG_BLOCK, cols: REG_BLOCK };
    let mut roots = store(&mut b, g.c, &c_accum0, base0, n as i64, wlane);
    roots.extend(store(&mut b, g.c, &c_accum1, base1, n as i64, wlane));

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "micro_tk".into() }
}
