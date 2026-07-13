//! **Flash-Attention forward on the ClusterCx declarative pipeline** — an EXPERIMENT probing
//! whether [`crate::pipeline`] (built for the HK GEMM) generalises to a second, differently-shaped
//! kernel. Streams K/V blocks: QKᵀ → online-softmax → P·V accumulate → normalize → write O.
//!
//! ## What is real vs stubbed (read before trusting this)
//! - **Compiles + lowers spec-valid** (`lower::verify`): the whole streaming skeleton — prologue
//!   stage/commit, the KV loop as `Mem`(gather K,V + prefetch/commit next) → `Compute`(QKᵀ) →
//!   `Compute`(softmax) → `Compute`(PV), the End-fold, epilogue, normalize, scatter.
//! - **Genuinely correct primitives**: `exp2`, `recip`, the `ds_bpermute` cross-lane row reductions
//!   (running max + norm sum), the online rescale, QKᵀ (= K·Qᵀ, a native tk2 `A·Bᵀ`).
//! - **KNOWN-WRONG numerics (documented gaps, device-correctness was the stretch goal)**: the P·V
//!   matmul. FA's PV is `Vᵀ·att` (contraction over kv), which needs `mma_atb` / a register transpose;
//!   tk2 has ONLY `mma` = `A·Bᵀ` (both operands K-inner), so PV is wired as a plain `mma(att,V)` — the
//!   wrong contraction orientation. See the finding catalogue. The minimal shape is single-fragment
//!   (`q_blk=kv_blk=d=16`, one warp, b=h=1): enough to exercise every DSL seam, not a production tile.
//!
//! The point of this module is the **friction it exposed**, catalogued in the experiment report.

use crate::build::{BF16, Builder, Effect, F32, Frag, Idx, Val};
use crate::ir::{FragMap, TileId};
use crate::kernels::{EDGE, Program};
use crate::movement::{Drain, LdsStage, LdsView, SharedTile};
use crate::pipeline::{CommitDrain, Compute, Hooks, Mem, pipeline};

const WARP: usize = 64;

/// One K-slice's gathered fragments — the [`Hooks::Op`]. Slice 0 = K (QKᵀ operand),
/// slice 1 = V (PV operand). Both are `Vec<Val<BF16>>` (one per outer fragment).
type FaOp = Vec<Val<BF16>>;

/// The register-staged fill carried prefetch→commit: block k+1's K and V chunks in VGPRs.
struct FaFill {
    k: Vec<Val<BF16>>,
    v: Vec<Val<BF16>>,
}

/// FA's [`Hooks`] — the ONLY kernel-specific movement. It rides the SAME [`crate::movement`]
/// handles matmul does (proving the movement layer is not GEMM-bound); the compute math rides the
/// [`Compute`] bodies below, so `Hooks` grows no per-cluster compute method. `PREFETCH_TILES = 2`
/// (K, V) as in matmul (A, B), but the two tiles are the two *operands of two different matmuls*
/// (K feeds QKᵀ, V feeds PV) rather than the A/B of one — the first shape strain (see report).
struct FaHooks {
    k_view: LdsView<BF16>,
    v_view: LdsView<BF16>,
    k_stage: LdsStage<BF16>,
    v_stage: LdsStage<BF16>,
}

impl Hooks for FaHooks {
    type Op = FaOp;
    type Reg = FaFill;
    const PREFETCH_TILES: usize = 2; // 0 = K, 1 = V

    fn prefetch(
        &mut self,
        b: &mut Builder,
        k_base: Idx,
        tile: usize,
        prev: Option<FaFill>,
        order: &[TileId],
    ) -> (FaFill, Vec<TileId>) {
        let mut reg = prev.unwrap_or(FaFill { k: Vec::new(), v: Vec::new() });
        let loaded = match tile {
            0 => {
                reg.k = self.k_stage.prefetch(b, k_base, order);
                &reg.k
            }
            1 => {
                reg.v = self.v_stage.prefetch(b, k_base, order);
                &reg.v
            }
            _ => panic!("FA prefetch: tile ∈ {{0=K, 1=V}}, got {tile}"),
        };
        let anchors = loaded.iter().map(|v| v.id).collect();
        (reg, anchors)
    }

    fn commit(&mut self, b: &mut Builder, _k_base: Idx, reg: &FaFill, war: &[TileId]) -> Vec<Effect> {
        // Intrinsic commit (an `s_barrier` auto-drains the `ds_write` `lgkmcnt(0)`) — the simplest
        // policy; the asm/deferred-drain machinery is a matmul-perf concern, orthogonal to FA's shape.
        let fk = self.k_stage.commit(b, &reg.k, war);
        let fv = self.v_stage.commit(b, &reg.v, war);
        fk.into_iter().chain(fv).collect()
    }

    fn gather(&mut self, b: &mut Builder, slice: usize, raw: &[TileId]) -> (FaOp, Vec<TileId>, TileId) {
        let (vecs, g) = match slice {
            0 => self.k_view.slice(0).gather(b, raw),
            1 => self.v_view.slice(0).gather(b, raw),
            _ => panic!("FA gather: slice ∈ {{0=K, 1=V}}, got {slice}"),
        };
        let anchor = vecs[0].id;
        (vecs, g, anchor)
    }
}

/// Init an accumulator fragment to a constant (the online-softmax `max = −∞` seed — the pipeline's
/// `inited` hook only had `zero_init_frag`, so a non-zero init needed adding; a MINOR gap).
fn const_init_frag(b: &mut Builder, f: Frag<F32>, v: f32) -> Effect {
    let cs: Vec<Val<F32>> = (0..f.map.ept).map(|_| b.f32(v)).collect();
    let cv = b.vec_build(&cs);
    b.store_frag_vec(f, cv)
}

/// **Cross-lane column reduction** of a `col`-map att-shaped fragment vector `val` (kv = rows,
/// q = cols): fold all 16 kv-rows per q-column, combining with the running `init`, and broadcast
/// the per-column result back to every `ept` slot (so the caller can subtract it row-wise). Barrier-
/// FREE via [`Builder::shuffle_lane`] (`ds_bpermute`), so the softmax compute body stays edge-free.
///
/// This entire function is vocabulary tk2 lacked: there is NO reduction primitive, so the in-register
/// partial fold AND the stride-{16,32,48} lane tree are hand-rolled here (verbatim tk1's `reduce.rs`
/// structure, but tk1 has it as a first-class `Group::col_reduce`). See the report — #1 missing op.
fn frag_col_reduce(b: &mut Builder, val: Val<F32>, lane: Idx, init: Val<F32>, is_max: bool) -> Val<F32> {
    let comb = |b: &mut Builder, a: Val<F32>, c: Val<F32>| if is_max { b.max(a, c) } else { b.add(a, c) };
    // (a) in-register fold of this lane's `ept` rows of its column.
    let mut partial = b.vec_extract(val, 0);
    for e in 1..EPT {
        let x = b.vec_extract(val, e);
        partial = comb(b, partial, x);
    }
    // (b) the wave64 lane tree {L, L+16, L+32, L+48} — every lane in a column-group ends identical.
    let mut acc = partial;
    let g = b.idx_const(WARP as i64);
    for d in [EDGE as i64, 2 * EDGE as i64, 3 * EDGE as i64] {
        let dc = b.idx_const(d);
        let sl = b.idx_add(lane, dc);
        let sl = b.idx_mod(sl, g);
        let sh = b.shuffle_lane(partial, sl);
        acc = comb(b, acc, sh);
    }
    // (c) fold the running accumulator, then broadcast to every `ept` slot.
    let init0 = b.vec_extract(init, 0);
    acc = comb(b, acc, init0);
    let copies: Vec<Val<F32>> = (0..EPT).map(|_| acc).collect();
    b.vec_build(&copies)
}

/// Cast an `ept`-wide f32 softmax-weight vector to a bf16 MMA operand (per-element `bf16_trunc`
/// then re-pack). The f32→bf16 relayout FA needs between softmax and PV; tk2 had `bf16_trunc`
/// (an HK-port leaf) but no vector cast helper — a MINOR gap.
fn cast_f32_vec_to_bf16(b: &mut Builder, v: Val<F32>, ept: usize) -> Val<BF16> {
    let els: Vec<Val<BF16>> = (0..ept)
        .map(|e| {
            let s = b.vec_extract(v, e);
            b.bf16_trunc(s)
        })
        .collect();
    b.vec_build(&els)
}

/// gfx942 elements-per-thread for the 16×16 fragment.
const EPT: usize = 4;

/// **Minimal streaming FA-forward** (`b = h = 1`, one warp, `q_blk = kv_blk = d = 16`) authored on
/// the [`crate::pipeline`] ClusterCx combinator. `n` = sequence length (KV blocks = `n/16 ≥ 2`).
/// Returns a lowerable [`Program`]; see the module docs for what is device-correct vs structural.
pub fn flash_attention_fwd(n: usize, d: usize) -> Program {
    assert!(d == EDGE, "minimal FA supports a single d-fragment (d = 16); multi-d is a documented gap");
    assert!(n.is_multiple_of(EDGE) && n / EDGE >= 2, "N must be a multiple of 16 with ≥2 KV blocks");
    let (q_blk, kv_blk) = (EDGE, EDGE);

    let mut b = Builder::new("tk2_fa_fwd");
    // ABI: output O first, then inputs Q, K, V (all flat [n, d]).
    let o = b.global::<F32>(n * d);
    let q = b.global::<BF16>(n * d);
    let k = b.global::<BF16>(n * d);
    let v = b.global::<BF16>(n * d);

    // One workgroup per Q-block; single 64-lane warp.
    let qwg = b.grid_axis(0, (n / q_blk) as i64);
    let lane = b.block_axis(WARP as i64);

    // ── LDS tiles (single-buffered [16,16] each) ──
    let k_smem = b.define_local::<BF16>(kv_blk * d);
    let v_smem = b.define_local::<BF16>(kv_blk * d);
    let q_smem = b.define_local::<BF16>(q_blk * d);

    let row_map = FragMap::gfx942_16x16(false); // A operand (contraction on the inner axis)
    let col_map = FragMap::gfx942_16x16(true); // B / C operands

    // Movement handles — the SAME `SharedTile`→view/stage machinery matmul uses. K is QKᵀ's A
    // (row map, contraction over d); Q is QKᵀ's B (col map, contraction over d); V is PV's operand.
    let k_tile = SharedTile::new(k_smem, d);
    let v_tile = SharedTile::new(v_smem, d);
    let q_tile = SharedTile::new(q_smem, d);
    let k_view = k_tile.gather_view(row_map, kv_blk / EDGE, None, lane, false);
    let v_view = v_tile.gather_view(col_map, kv_blk / EDGE, None, lane, false);
    let q_view = q_tile.gather_view(col_map, q_blk / EDGE, None, lane, false);

    let epl = kv_blk * d / WARP; // 4 — vectorised fill elements per lane
    let zero = b.idx_const(0);
    // K/V stream: origin row 0 (single head/batch); the per-block row advance rides `k_base` (below).
    let k_stage = k_tile.stage_view(k, epl, lane, zero, d as i64, Drain::Intrinsic);
    let v_stage = v_tile.stage_view(v, epl, lane, zero, d as i64, Drain::Intrinsic);
    // Q origin row = qwg·q_blk (this workgroup's query rows); staged ONCE (loop-invariant).
    let qblk_c = b.idx_const(q_blk as i64);
    let q_origin = b.idx_mul(qwg, qblk_c);
    let q_stage = q_tile.stage_view(q, epl, lane, q_origin, d as i64, Drain::Intrinsic);

    // ── prologue: stage + commit Q once, gather it as the loop-invariant QKᵀ operand ──
    let q_loaded = q_stage.prefetch(&mut b, zero, &[]);
    let q_fill = q_stage.commit(&mut b, &q_loaded, &[]);
    let q_fill_deps: Vec<TileId> = q_fill[1..].iter().map(|e| e.dep()).collect();
    let q_bar = b.barrier(q_fill[0], &q_fill_deps);
    let (q_vecs, _q_g) = q_view.slice(0).gather(&mut b, &[q_bar.dep()]);
    let q0 = q_vecs[0]; // Val<BF16>, loop-invariant — captured by the QKᵀ body

    // ── accumulators carried across the KV loop: o, att, max, norm (all Frag<F32>, col map) ──
    // GEMM-shape leak: the pipeline carries a FIXED `Frag<F32>` accumulator set round-tripped by EVERY
    // compute cluster. FA's `att` is a per-iteration TEMPORARY (QKᵀ re-zeros it, the carry is wasted)
    // and `max`/`norm` are logically per-Q VECTORS, not q×kv tiles — both are shoehorned as Frag<F32>.
    let o_acc = b.define_frag::<F32>(col_map);
    let att_acc = b.define_frag::<F32>(col_map);
    let max_acc = b.define_frag::<F32>(col_map);
    let norm_acc = b.define_frag::<F32>(col_map);
    let accs = [o_acc, att_acc, max_acc, norm_acc];
    let inited = [
        b.zero_init_frag(o_acc),
        b.zero_init_frag(att_acc),
        const_init_frag(&mut b, max_acc, f32::NEG_INFINITY), // online-softmax running max seed
        b.zero_init_frag(norm_acc),
    ];

    // Softmax scale folded into the QKᵀ scores: exp2(score·log2(e)/√d) == exp(score/√d).
    let scale = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let scale_bcast = {
        let s = b.f32(scale);
        let cs: Vec<Val<F32>> = (0..EPT).map(|_| s).collect();
        b.vec_build(&cs)
    };
    let zero_c = {
        // QKᵀ's C = 0 (re-zero att every iteration; the carried att read is ignored — carry waste).
        let zs: Vec<Val<F32>> = (0..EPT).map(|_| b.f32(0.0)).collect();
        b.vec_build(&zs)
    };

    // ── the three compute bodies (the "pluggable compute body" claim under test) ──
    // QKᵀ: att = K·Qᵀ (a native tk2 `A·Bᵀ`, contraction over d). Operand = gathered K (slice 0).
    let qk = Compute::<FaHooks>::new(0, move |b: &mut Builder, op: Option<&FaOp>, reads: &[Val<F32>]| {
        let k = op.expect("QKᵀ consumes gathered K")[0];
        let att = b.mma(k, q0, zero_c, EPT); // Val<F32>, ept-wide
        vec![reads[0], att, reads[2], reads[3]] // o, att', max, norm  (only att changes)
    });

    // Online softmax: consumes ONLY the accumulator carry (operand = None). Emits `exp2` + the two
    // `ds_bpermute` row reductions + the running rescale — the whole novel FA math lives HERE, and it
    // needs NO barrier of its own (ds_bpermute is barrier-free), so the edge-free-body claim HOLDS.
    let softmax = Compute::<FaHooks>::new(None, move |b: &mut Builder, _op: Option<&FaOp>, reads: &[Val<F32>]| {
        let (o, att, max_old, norm_old) = (reads[0], reads[1], reads[2], reads[3]);
        let s = b.mul(att, scale_bcast); // scaled scores
        let m = frag_col_reduce(b, s, lane, max_old, true); // running max (broadcast per q)
        let corr = b.sub(max_old, m);
        let scale_f = b.exp2(corr); // exp2(max_old − max_new)
        let o2 = b.mul(o, scale_f); // rescale running output
        let norm2 = b.mul(norm_old, scale_f); // rescale running norm
        let sm = b.sub(s, m);
        let p = b.exp2(sm); // softmax weights P (still f32)
        let norm3 = frag_col_reduce(b, p, lane, norm2, false); // norm += Σ P
        vec![o2, p, m, norm3]
    });

    // P·V accumulate. Operand = gathered V (slice 1). att-read is P (softmaxed f32), cast to bf16.
    // NOTE: correct FA is `o += Vᵀ·att` (contraction over kv) = tk1's `mma_atb`; tk2 has only
    // `A·Bᵀ`, so this `mma(att,V)` is the WRONG contraction orientation — a documented device-numerics
    // gap (structural completeness holds; the report ranks this the #2 GEMM-shape leak).
    let pv = Compute::<FaHooks>::new(1, move |b: &mut Builder, op: Option<&FaOp>, reads: &[Val<F32>]| {
        let vv = op.expect("PV consumes gathered V")[0];
        let (o, p) = (reads[0], reads[1]);
        let att_mma = cast_f32_vec_to_bf16(b, p, EPT);
        let o2 = b.mma(att_mma, vv, o, EPT);
        vec![o2, reads[1], reads[2], reads[3]] // only o changes
    });

    let hooks = FaHooks { k_view, v_view, k_stage, v_stage };
    let acc_final = pipeline(
        &mut b,
        n / kv_blk, // nblocks (streaming over KV)
        kv_blk * d, // k_step: the FLAT per-block advance (kv_blk rows · d) — see the k_step-conflation finding
        2,          // ksteps: gather slices (K, V) — NOT a contraction count, unlike matmul
        &accs,
        &inited,
        None,  // warp_row: no wave-phase ping-pong (single warp)
        false, // asm_gather
        false, // resident
        CommitDrain::IntrinsicAuto,
        false, // bare_seals
        false, // pin_mfma
        hooks,
    )
    .cluster(Mem::builder().prefetch([0, 1]).gathers([0, 1]).commit(true).build())
    .cluster(qk)
    .cluster(softmax)
    .cluster(pv)
    .build();

    // ── post-loop: normalize O = o / norm, scatter to global O ──
    let o_final = acc_final[0];
    let norm_final = acc_final[3];
    let norm_vec = b.load_frag_vec(norm_final);
    let recip_norm = b.recip(norm_vec);
    let o_vec = b.load_frag_vec(o_final);
    let o_norm = b.mul(o_vec, recip_norm);

    let d_c = b.idx_const(d as i64);
    let o_row_base = b.idx_mul(q_origin, d_c); // this workgroup's O flat row origin
    let mut roots = Vec::new();
    for inner in 0..o_final.map.ept {
        let inner_c = b.idx_const(inner as i64);
        let (row, col) = b.lane_rc(o_final.map, lane, inner_c);
        let row_off = b.idx_mul(row, d_c);
        let off = b.idx_add(o_row_base, row_off);
        let off = b.idx_add(off, col);
        let val = b.vec_extract(o_norm, inner);
        roots.push(b.store(o, off, val));
    }

    let (ir, sink) = b.finish(&roots);
    Program { ir, sink, name: "tk2_fa_fwd".into() }
}
