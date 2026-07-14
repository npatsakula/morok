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

// ── the 32×32×8 wide-core probe: a second MFMA shape lowers spec-valid ──────────────────────────

/// The 32×32×8 MFMA isolation probe ([`crate::kernels::mfma_32x32x8_probe`]) must lower to spec-valid
/// device-UOp — proving the `Node::Mma` accumulator-width dispatch (`ept 16 → 32×32×8`), the wide
/// `v_mfma_f32_32x32x8_bf16` intrinsic selection, and the 16-VGPR `acc_rc` scatter survive lowering +
/// `type_verify` BEFORE the device gate. Covers one MFMA (32×32×8), a K-loop (32×32×16), and a tiled
/// output (64×64×8) so the accumulation chain + the M/N tiling are all exercised in the linearizer.
#[test]
fn mfma_32x32x8_probe_lowers_spec_valid() {
    for (m, n, k) in [(32usize, 32usize, 8usize), (32, 32, 16), (64, 64, 8)] {
        for asm in [false, true] {
            let p = crate::kernels::mfma_32x32x8_probe(m, n, k, asm);
            // Exactly the tiled MFMA count: (m/32)·(n/32)·(k/8).
            let n_mma =
                (0..p.ir.len()).filter(|&i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::Mma { .. })).count();
            assert_eq!(n_mma, (m / 32) * (n / 32) * (k / 8), "probe {m}×{n}×{k} MFMA count");
            lower::verify(&p).expect("32×32×8 probe must lower to spec-valid UOp");
        }
    }
}
