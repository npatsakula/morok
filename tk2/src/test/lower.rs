//! Host tests: the ADT interns/disambiguates correctly, and the verified lowering
//! produces spec-valid device-UOp for BOTH proof kernels.

use svod_dtype::DType;

use crate::ir::{Node, RegClass, Residency, TileIr};
use crate::lower;

// ── the ADT: interning, disambiguators, residency/reg-class fields ───────────

#[test]
fn structurally_identical_nodes_hash_cons() {
    let mut ir = TileIr::new();
    let a = ir.intern(Node::Global { slot: 0, dtype: DType::Float32, len: 4 });
    let b = ir.intern(Node::Global { slot: 0, dtype: DType::Float32, len: 4 });
    assert_eq!(a, b, "identical Global nodes must collapse to one id");
}

#[test]
fn slot_disambiguator_keeps_distinct_globals_apart() {
    let mut ir = TileIr::new();
    let s0 = ir.fresh_slot();
    let s1 = ir.fresh_slot();
    let a = ir.intern(Node::Global { slot: s0, dtype: DType::Float32, len: 4 });
    let b = ir.intern(Node::Global { slot: s1, dtype: DType::Float32, len: 4 });
    assert_ne!(a, b, "different ABI slots must NOT hash-cons together (miscompile guard)");
}

#[test]
fn range_id_disambiguator_keeps_distinct_loops_apart() {
    let mut ir = TileIr::new();
    let r0 = ir.fresh_range_id();
    let r1 = ir.fresh_range_id();
    // Same trip count, different loop — must stay distinct or two loops collapse.
    let a = ir.intern(Node::Range { id: r0, trips: 16 });
    let b = ir.intern(Node::Range { id: r1, trips: 16 });
    assert_ne!(a, b, "same-trip distinct loops must not collapse");
}

#[test]
fn residency_and_reg_class_fields_present() {
    let mut ir = TileIr::new();
    let g = ir.intern(Node::Global { slot: 0, dtype: DType::Float32, len: 8 });
    let r = ir.intern(Node::DefineReg { id: 0, dtype: DType::Float32, len: 1 });
    assert_eq!(ir.meta(g).residency, Residency::Global);
    assert_eq!(ir.meta(r).residency, Residency::Reg);
    // Reg-class channel exists now (the AGPR pass flips this field in Step 3).
    assert_eq!(ir.meta(r).reg_class, RegClass::Vgpr);
}

// ── the verified lowering ────────────────────────────────────────────────────

#[test]
fn matmul_lds_kblock_clustered_lowers_and_balances_the_wave_phase() {
    // The §5c clustered HK replica (2 warp-rows → 128², 4 K-blocks, ksteps=4): the interpreter walks
    // the 8-cluster schedule placing per-cluster barriers + set_prio brackets + the warp-phase
    // ping-pong. It must lower spec-valid, carry the SetPrio brackets, and the wave barriers must be
    // balanced (eq=0 count == eq=1 count == 1) — else `matmul_..._clustered` would have panicked in
    // `verify_warp_phase_balance` at construction. The clustered kernel now gathers via the asm
    // `ds_read_b64 offset:N` primitive (LdsPtrAs3 + DsReadB64) and re-enables the positional
    // `wall_after_barriers` lattice (SchedWallMarker) — both must lower spec-valid.
    let count = |p: &crate::Program, pred: &dyn Fn(&Node) -> bool| {
        (0..p.ir.len()).filter(|&i| pred(p.ir.node(crate::ir::TileId(i as u32)))).count()
    };
    let p = crate::kernels::matmul_lds_kblock_mw_clustered(128, 128, 256, 64, 64, 2, 2, 64);
    lower::verify(&p).expect("clustered HK replica must lower to spec-valid UOp");
    assert!(count(&p, &|n| matches!(n, Node::SetPrio { .. })) > 0, "compute clusters ⇒ SetPrio nodes");
    assert_eq!(count(&p, &|n| matches!(n, Node::WaveBarrier { eq: 1, .. })), 1, "one eq=1 prologue wave barrier");
    assert_eq!(count(&p, &|n| matches!(n, Node::WaveBarrier { eq: 0, .. })), 1, "one eq=0 epilogue wave barrier");
    // Composes with the refinement passes: VectorizePass is a no-op on the asm gather (no fusible
    // scalar run), and SwizzlePass folds the (fragment-invariant) XOR delta into the asm base offset's
    // `lds_col` — so the swizzled clustered kernel still lowers spec-valid.
    let sw = crate::kernels::matmul_lds_kblock_mw_clustered(128, 128, 256, 64, 64, 2, 2, 64)
        .apply(crate::passes::VectorizePass)
        .apply(crate::passes::SwizzlePass);
    lower::verify(&sw).expect("clustered.apply(Vectorize).apply(Swizzle) must lower spec-valid");
}

// ── the FA-forward experiment: the ClusterCx pipeline generalised to a second kernel shape ─────

/// The minimal streaming Flash-Attention forward ([`crate::kernels_fa::flash_attention_fwd`])
/// authored on the SAME `pipeline` combinator as the GEMM: `Mem`(gather K,V + prefetch/commit) →
/// `Compute`(QKᵀ) → `Compute`(online softmax, operand-less) → `Compute`(PV). It must lower to
/// spec-valid device-UOp — proving the new vocabulary (`exp2`/`recip`/`ds_bpermute` row reductions)
/// and the operand-less softmax cluster survive lowering + `type_verify`. (Device NUMERICS are a
/// separate, partly-stubbed matter — see the module docs; this asserts the STRUCTURE compiles.)
#[test]
fn fa_forward_on_clustercx_lowers_spec_valid() {
    let count = |p: &crate::Program, pred: &dyn Fn(&Node) -> bool| {
        (0..p.ir.len()).filter(|&i| pred(p.ir.node(crate::ir::TileId(i as u32)))).count()
    };
    // bh=2, 8-warp split-Q, kv_blk=32 (2 KV-frags); Vectorize.then(Swizzle) = the production form (K
    // gather fused to ds_read_b64 + the bank-conflict swizzle fold).
    let p = crate::kernels_fa::flash_attention_fwd(2, 128, 64)
        .apply(crate::passes::VectorizePass)
        .apply(crate::passes::SwizzlePass);
    lower::verify(&p).expect("FA-forward on ClusterCx must lower to spec-valid UOp");
    // The novel FA vocabulary is present: exp2 (2×/iter), recip (normalize), and the ds_bpermute
    // cross-lane reduction tree (3 shuffles × 2 reductions/iter).
    assert!(count(&p, &|n| matches!(n, Node::Unary { .. })) >= 3, "softmax ⇒ exp2/recip Unary nodes");
    assert!(count(&p, &|n| matches!(n, Node::DsBpermute { .. })) >= 6, "row reductions ⇒ ds_bpermute nodes");
    // The operand-less softmax cluster + the two matmul clusters ⇒ SetPrio brackets, one commit RAW.
    assert!(count(&p, &|n| matches!(n, Node::Mma { .. })) >= 2, "QKᵀ + PV ⇒ ≥2 MMAs");
    // Ping-pong-gated compute seals: FA's disjoint-Q warps (`warp_row = None`) exchange only per-warp
    // registers between QKᵀ/softmax/PV, so those steady compute-cluster seals emit NO workgroup
    // `s_barrier` (a pure `Node::After` ordering combine instead) — only the load-bearing Mem WAR/RAW
    // pair survives in the hot loop. Count only barriers REACHABLE from the sink (`count` over the raw
    // arena double-counts the dead pre-swizzle offset nodes SwizzlePass supersedes). The reachable
    // `Node::Barrier`s are exactly 4: block-0 K/V commit (prologue) + steady (WAR + RAW = 2) + epilogue
    // (Mem gather seal). It was 5 before commit 8ec2afe6 (Probe B) dropped the Q-LDS staging, which
    // removed the prologue Q-commit barrier — this assertion was left stale at 5 by that commit and is
    // corrected to 4 here. Vectorize.then(Swizzle) does not change the count (VectorizePass only fuses
    // the K gather's scalar loads, adding/removing no workgroup barrier).
    let live: std::collections::HashSet<crate::ir::TileId> =
        crate::passes::reachable(&p.ir, p.sink).into_iter().collect();
    let nbar = live.iter().filter(|&&id| matches!(p.ir.node(id), Node::Barrier { .. })).count();
    assert_eq!(
        nbar, 4,
        "FA keeps ONLY the Mem WAR/RAW + prologue-commit/epilogue barriers (compute seals ping-pong-gated off)"
    );
}

// ── the SchedGroupBarrier interleave primitive (FA-redesign step 2) ──────────────────────────────

/// The declarative interleave directive ([`crate::build::Builder::interleave_valu`]) must (a) intern
/// `SchedGroupBarrier` nodes, (b) lower spec-valid, and (c) RENDER to the `@llvm.amdgcn.sched.group.
/// barrier` builtin the AMDGPU backend emits as the `; sched_group_barrier` interleave comment. A tiny
/// 2-slice 32×32×8 MFMA burst (intrinsic accumulator) + a VALU scale carries an `interleave_valu<2,5>`
/// hint threaded live into the store — the minimal proof the primitive emits before FA depends on it.
#[test]
fn sched_group_barrier_lowers_and_renders_the_builtin() {
    use crate::build::{BF16, Builder, F32};
    use crate::shape::{Mfma32x32x8Bf16 as S, MfmaShape};
    let mut b = Builder::new("tk2_sched_group_probe");
    let c = b.global::<F32>(S::M * S::N);
    let a = b.global::<BF16>(S::M * 2 * S::K);
    let bmat = b.global::<BF16>(S::N * 2 * S::K);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(64);
    let (a_map, b_map, dist) = (S::a_map(), S::b_map(), S::acc_dist());
    // 2-slice K-loop into one intrinsic-MFMA accumulator.
    let mut acc = {
        let zs: Vec<_> = (0..S::EPT_C).map(|_| b.f32(0.0)).collect();
        b.vec_build(&zs)
    };
    for ki in 0..2 {
        let af = crate::kernels::load_op_frag(&mut b, a, a_map, 0, ki * S::K, 2 * S::K, lane);
        let bf = crate::kernels::load_op_frag(&mut b, bmat, b_map, 0, ki * S::K, 2 * S::K, lane);
        acc = b.mma_of::<S>(af, bf, acc);
    }
    // A VALU op the interleave can pull under the MFMAs (the softmax-rescale analog).
    let two = b.f32(2.0);
    let mut scaled = Vec::with_capacity(S::EPT_C);
    for i in 0..S::EPT_C {
        let e = b.vec_extract(acc, i);
        scaled.push(b.mul(e, two));
    }
    let acc = b.vec_build(&scaled);
    // Scatter, then thread an interleave_valu<pairs=2, valu=5> hint anchored on the last store — live
    // via the roots, so it survives DCE and reaches the renderer.
    let n_c = b.idx_const(S::N as i64);
    let mut roots = Vec::new();
    for i in 0..S::EPT_C {
        let (row, col) = b.acc_rc(dist, lane, i);
        let rn = b.idx_mul(row, n_c);
        let off = b.idx_add(rn, col);
        let v = b.vec_extract(acc, i);
        roots.push(b.store(c, off, v));
    }
    let anchor = roots.last().expect("stores").dep();
    let hint = b.interleave_valu(2, 5, 1, &[anchor]).expect("pairs>0");
    roots.push(hint);
    let (ir, sink) = b.finish(&roots);
    let p = crate::Program { ir, sink, name: "tk2_sched_group_probe".into() };

    // (a) the nodes are interned — 2 pairs × 2 hints = 4 SchedGroupBarrier.
    let n_sgb = (0..p.ir.len())
        .filter(|&i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::SchedGroupBarrier { .. }))
        .count();
    assert_eq!(n_sgb, 4, "interleave_valu<2,_> ⇒ 2×(MFMA+VALU) = 4 SchedGroupBarrier nodes");
    // (b) spec-valid lowering.
    lower::verify(&p).expect("sched_group probe must lower spec-valid");
    // (c) renders the builtin (→ the `; sched_group_barrier` ASM comment).
    let llvm = crate::launch::render_amd_ir(&p, svod_dtype::AmdArch::Gfx942).expect("render");
    assert!(llvm.contains("llvm.amdgcn.sched.group.barrier"), "must render the sched.group.barrier builtin");
    assert!(llvm.contains("i32 8, i32 1"), "MFMA-mask(0x8) size-1 group present");
    assert!(llvm.contains("i32 2, i32 5"), "VALU-mask(0x2) size-5 group present");
}

/// The 32×32×8 MFMA isolation probe ([`crate::kernels::mfma_32x32x8_probe`]) must lower to spec-valid
/// device-UOp — proving the `Node::Mma` accumulator-width dispatch (`ept 16 → 32×32×8`), the wide
/// `v_mfma_f32_32x32x8_bf16` intrinsic selection, and the 16-VGPR `acc_rc` scatter survive lowering +
/// `type_verify` BEFORE the device gate. Covers one MFMA (32×32×8), a K-loop (32×32×16), and a tiled
/// output (64×64×8) so the accumulation chain + the M/N tiling are all exercised in the linearizer.
#[test]
fn mfma_32x32x8_probe_lowers_spec_valid() {
    for (m, n, k) in [(32usize, 32usize, 8usize), (32, 32, 16), (64, 64, 8)] {
        let p = crate::kernels::mfma_32x32x8_probe(m, n, k);
        // Exactly the tiled MFMA count: (m/32)·(n/32)·(k/8).
        let n_mma =
            (0..p.ir.len()).filter(|&i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::Mma { .. })).count();
        assert_eq!(n_mma, (m / 32) * (n / 32) * (k / 8), "probe {m}×{n}×{k} MFMA count");
        lower::verify(&p).expect("32×32×8 probe must lower to spec-valid UOp");
    }
}

/// A **ragged-`n` FA-32** ([`crate::kernels_fa::flash_attention_fwd_32`]) must lower spec-valid: `n=80`
/// is not a KV-block (32) multiple, so the last KV block is partial and the online softmax carries the
/// per-element ragged-tail mask (`global_kv < n ? score : −∞`). Proves the new `Node::SelectLt` →
/// `WHERE(LT,…)` lowering + `type_verify` accept the mask BEFORE the device gate, at `d=64` (2 KV
/// fragments) and `d=128`, both base and `SwizzlePass` forms. (Constructing it also runs the build-time
/// `verify_v2` scheduling-coherence + pipeline completeness checks over the masked schedule.)
#[test]
fn fa32_ragged_tail_lowers_spec_valid() {
    for d in [64usize, 128] {
        let p = crate::kernels_fa::flash_attention_fwd_32(1, 80, d);
        let n_sel = (0..p.ir.len())
            .filter(|&i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::SelectLt { .. }))
            .count();
        assert!(n_sel > 0, "ragged FA-32 (n=80, d={d}) must emit the SelectLt ragged-tail mask");
        lower::verify(&p).expect("ragged FA-32 must lower spec-valid (base)");
        let ps = crate::kernels_fa::flash_attention_fwd_32(1, 80, d).apply(crate::SwizzlePass);
        lower::verify(&ps).expect("ragged FA-32 must lower spec-valid (swizzled)");
    }
}
