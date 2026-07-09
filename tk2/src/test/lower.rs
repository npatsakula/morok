//! Host tests: the ADT interns/disambiguates correctly, and the verified lowering
//! produces spec-valid device-UOp for BOTH proof kernels.

use svod_dtype::DType;
use svod_ir::Op;

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

#[test]
fn clustered_asm_commit_emits_ds_write_b64_and_the_manual_drain() {
    // Phase C-a: the clustered kernel's commit is HK's waitcnt-opaque `asm ds_write_b64` + an EXPOSED
    // manual `s_waitcnt lgkmcnt(0)` drain (host render, no GPU). Contrast: the intrinsic `pipe2` kernel
    // (Drain::Intrinsic) carries NEITHER — its LDS fill is the compiler-visible vector store, and its
    // fenced-barrier waitcnt is backend-inserted (not an asm-sideeffect string in the LLVM IR text).
    use svod_dtype::AmdArch;
    let render = |prog: &crate::Program| -> String {
        let linearized = lower::verify(prog).expect("kernel lowers spec-valid");
        let Op::Program { linear: Some(lin), .. } = linearized.op() else { panic!("no linear stage") };
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(AmdArch::Gfx942);
        svod_codegen::traits::Renderer::render(&renderer, lin, Some(&prog.name)).expect("render").code
    };

    let clustered = crate::kernels::matmul_lds_kblock_mw_clustered(128, 128, 256, 64, 64, 2, 2, 64)
        .apply(crate::passes::VectorizePass)
        .apply(crate::passes::SwizzlePass);
    let code = render(&clustered);
    assert!(code.contains("ds_write_b64"), "clustered commit must emit the asm ds_write_b64");
    assert!(code.contains("s_waitcnt lgkmcnt(0)"), "clustered commit must emit the exposed manual drain");

    // The intrinsic path renders neither the asm write nor the manual-drain string (its commit is
    // `store … addrspace(3)`, and its barrier waitcnt is compiler-inserted, not an asm sideeffect).
    let pipe = crate::kernels::matmul_lds_kblock_mw_pipe2(128, 128, 256, 64, 64, 2, 2, 64)
        .apply(crate::passes::VectorizePass)
        .apply(crate::passes::SwizzlePass);
    let pipe_code = render(&pipe);
    assert!(!pipe_code.contains("ds_write_b64"), "intrinsic commit must NOT emit the asm ds_write_b64");
    assert!(!pipe_code.contains("s_waitcnt lgkmcnt(0)"), "intrinsic commit must NOT emit a manual drain");
}
