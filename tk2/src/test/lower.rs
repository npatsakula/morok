//! Host tests: the ADT interns/disambiguates correctly, and the verified lowering
//! produces spec-valid device-UOp for BOTH proof kernels.

use svod_dtype::DType;
use svod_ir::Op;

use crate::ir::{Node, RegClass, Residency, TileIr};
use crate::kernels::{
    elementwise_add, lds_roundtrip, matmul, matmul_lds, matmul_lds_kblock, matmul_lds_tiled, sum_reduce,
};
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
fn elementwise_add_lowers_to_a_sink() {
    let p = elementwise_add(64, 4);
    let sink = lower::lower(&p.ir, p.sink, &p.name);
    assert!(matches!(sink.op(), Op::Sink { .. }), "lowered root must be a SINK");
}

#[test]
fn elementwise_add_lowering_is_spec_valid() {
    let p = elementwise_add(64, 4);
    lower::verify(&p).expect("tiled elementwise add must lower to spec-valid UOp");
}

#[test]
fn sum_reduce_lowering_is_spec_valid() {
    // The loop-carry proof: Range/End + After edges must produce spec-valid UOp
    // (this is where all prior loop-carry pain lived).
    let p = sum_reduce(256);
    lower::verify(&p).expect("loop-carried sum reduction must lower to spec-valid UOp");
}

#[test]
fn matmul_lowering_is_spec_valid() {
    // The naive matmul: fragment gather + 16×16×16 WMMA + loop-carried f32
    // accumulator must lower to spec-valid UOp (integer addresses, matched ALU
    // dtypes, one RANGE per END, movement lowered away).
    let p = matmul(64, 64, 64);
    lower::verify(&p).expect("naive matmul must lower to spec-valid UOp");
}

#[test]
fn matmul_carries_wmma_and_loop_edges() {
    // Structural check: the WMMA op plus the loop-carry ordering edges (After) and
    // the K-loop RANGE/END scoping are present in the lowered graph.
    let p = matmul(32, 32, 32);
    let sink = lower::lower(&p.ir, p.sink, &p.name);
    let topo = sink.toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "matmul needs a WMMA");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::After { .. })), "loop-carry needs After edges");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Range { .. })), "the K reduction needs a RANGE");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::End { .. })), "the K RANGE must be closed by an END");
}

#[test]
fn matmul_lds_kblock_sw_lowering_is_spec_valid() {
    // The bank-swizzled K-blocked kernel (XOR/shift index ops in the LDS addressing)
    // must lower to spec-valid UOp — the swizzle is a bijection, numerically transparent.
    let p = crate::kernels::matmul_lds_kblock_sw(64, 64, 64, 64, 64, 16);
    lower::verify(&p).expect("swizzled K-blocked matmul must lower to spec-valid UOp");
}

#[test]
fn swizzle_pass_materialises_the_layout() {
    // The `.apply` model: the BASE kernel carries `LdsCol` layout points (flat until
    // materialised); `.apply(SwizzlePass)` replaces every one with the bank XOR (none
    // survive), and the result stays spec-valid — the swizzle is a composable refinement.
    use crate::ir::TileIr;
    use crate::passes::SwizzlePass;
    let base = crate::kernels::matmul_lds_kblock_ks(64, 64, 64, 64, 64, 16);
    let base_has_ldscol =
        (0..base.ir.len()).any(|i| matches!(base.ir.node(crate::ir::TileId(i as u32)), Node::LdsCol { .. }));
    assert!(base_has_ldscol, "the base kernel must carry LdsCol layout points");

    let sw = base.apply(SwizzlePass);
    let count_ldscol = |ir: &TileIr| {
        crate::passes::reachable(ir, sw.sink)
            .into_iter()
            .filter(|&id| matches!(ir.node(id), Node::LdsCol { .. }))
            .count()
    };
    assert_eq!(count_ldscol(&sw.ir), 0, "SwizzlePass must materialise every LdsCol (none may survive)");
    lower::verify(&sw).expect("swizzled program must still lower to spec-valid UOp");
}

#[test]
fn matmul_lds_kblock_ks64_lowering_is_spec_valid() {
    // K_STEP=64: a k_step-wide fill + an inner chain of k_step/16 MFMAs per accumulator.
    let p = crate::kernels::matmul_lds_kblock_ks(64, 64, 128, 64, 64, 64);
    lower::verify(&p).expect("K_STEP=64 matmul must lower to spec-valid UOp");
}

#[test]
fn vectorize_pass_fuses_the_scalar_gathers() {
    // The base kernel emits SCALAR gathers: for a 64×64 tile at K_STEP=16 (ri=cj=4, ksteps=1)
    // the 8 fragment gathers are 8·ept = 32 scalar `store_frag_elem` (StoreGlobal into a bf16
    // frag), and the only LoadVecAt are the fill vector loads. With B taken [N,K] (HK), A and B
    // BOTH use the trivial b128 fill: A-fill 2 + B-fill 2 (epl=16 → 2 `dwordx4` each) = 4.
    // `.apply(VectorizePass)` fuses each ept run → +8 gather LoadVecAt (12 total) and turns the
    // 32 scalar frag stores into 8 StoreRegVec — none survive. Fills stay builder-structural.
    use crate::ir::Node as N;
    use crate::passes::{VectorizePass, reachable};
    let base = crate::kernels::matmul_lds_kblock_ks(64, 64, 64, 64, 64, 16);
    let scalar_frag_stores = |ir: &crate::ir::TileIr, root| {
        reachable(ir, root)
            .into_iter()
            .filter(|&id| {
                matches!(ir.node(id), N::StoreGlobal { buf, .. } if matches!(ir.node(*buf), N::DefineFrag { dtype, .. } if *dtype == svod_dtype::DType::BFloat16))
            })
            .count()
    };
    assert_eq!(scalar_frag_stores(&base.ir, base.sink), 32, "base: 8 gathers × ept=4 scalar frag stores");

    let vec = base.apply(VectorizePass);
    assert_eq!(scalar_frag_stores(&vec.ir, vec.sink), 0, "VectorizePass fuses every scalar gather run");
    let count = |pred: &dyn Fn(&N) -> bool| {
        reachable(&vec.ir, vec.sink).into_iter().filter(|&id| pred(vec.ir.node(id))).count()
    };
    assert_eq!(
        count(&|n| matches!(n, N::LoadVecAt { .. })),
        12,
        "8 fused gather + 4 fill vector loads (A-fill 2 + B-fill 2, both b128)"
    );
    lower::verify(&vec).expect("vectorised matmul must lower to spec-valid UOp");
}

#[test]
fn matmul_lds_kblock_lowering_is_spec_valid() {
    // The K-blocked kernel: per-K-block fill + TWO barriers (RAW + WAR) + the reused
    // 2×2 accumulator grid, all inside one K-loop, must lower to spec-valid UOp.
    let p = matmul_lds_kblock(64, 64, 64, 32, 32);
    lower::verify(&p).expect("K-blocked LDS matmul must lower to spec-valid UOp");
}

#[test]
fn matmul_lds_kblock_multiwarp_lowers_and_derives_warps() {
    // Multi-warp (2×2 → 128×128 tile): block size = 4·64 = 256 (one lidx0 SPECIAL of
    // bound 256), the warp split adds idx_div/idx_mod, and it lowers to spec-valid UOp.
    let p = crate::kernels::matmul_lds_kblock_mw(128, 128, 64, 64, 64, 2, 2, 64);
    let has_block_256 = (0..p.ir.len()).any(|i| {
        matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::Axis { axis: crate::ir::ScopeAxis::Block, bound: 256 })
    });
    assert!(has_block_256, "multi-warp workgroup must be a single 256-thread block axis");
    lower::verify(&p).expect("multi-warp K-blocked matmul must lower to spec-valid UOp");
    // Single-warp stays a 64-thread block (no warp-split div/mod overhead).
    let sw = crate::kernels::matmul_lds_kblock_ks(64, 64, 64, 64, 64, 64);
    let has_block_64 = (0..sw.ir.len()).any(|i| {
        matches!(sw.ir.node(crate::ir::TileId(i as u32)), Node::Axis { axis: crate::ir::ScopeAxis::Block, bound: 64 })
    });
    assert!(has_block_64, "single-warp workgroup stays a 64-thread block");
}

#[test]
fn matmul_lds_kblock_pipe_lowers_and_splits_prologue_steady_epilogue() {
    // stages=2 register-staged pipeline (2×2 → 128×128 tile, 4 K-blocks). The prologue commit +
    // steady range(nblocks-1) + epilogue gather each emit their own gather/commit/barrier cluster,
    // so it must still lower spec-valid — the carried-RAW + WAR + register carry composed correctly.
    let p = crate::kernels::matmul_lds_kblock_mw_pipe(128, 128, 256, 64, 64, 2, 2, 64);
    lower::verify(&p).expect("pipelined (stages=2) K-blocked matmul must lower to spec-valid UOp");
    // The steady loop is one block shorter than the K-block count: range(nblocks-1) = range(3).
    let has_short_loop =
        (0..p.ir.len()).any(|i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::Range { trips: 3, .. }));
    assert!(has_short_loop, "stages=2 steady loop must be range(nblocks-1)");
    // Single-block K (nblocks=1) falls back to stages=1 — no range(0), the full-K loop is range(1).
    let one = crate::kernels::matmul_lds_kblock_mw_pipe(128, 128, 64, 64, 64, 2, 2, 64);
    lower::verify(&one).expect("single-block pipelined matmul must fall back and lower spec-valid");
    let has_zero_loop =
        (0..one.ir.len()).any(|i| matches!(one.ir.node(crate::ir::TileId(i as u32)), Node::Range { trips: 0, .. }));
    assert!(!has_zero_loop, "single-block K must fall back to stages=1, not emit a range(0) steady loop");
}

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
    // manual `s_waitcnt lgkmcnt(0)` drain (host render, no GPU). Contrast: the intrinsic `_pipe` kernel
    // (asm_commit=false) carries NEITHER — its LDS fill is the compiler-visible vector store.
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

    // The intrinsic path is untouched: `_pipe` renders neither the asm write nor the manual drain (its
    // commit is `store … addrspace(3)`, which the compiler-visible waitcnt/barrier handles implicitly).
    let pipe = crate::kernels::matmul_lds_kblock_mw_pipe(128, 128, 256, 64, 64, 2, 2, 64)
        .apply(crate::passes::VectorizePass)
        .apply(crate::passes::SwizzlePass);
    let pipe_code = render(&pipe);
    assert!(!pipe_code.contains("ds_write_b64"), "intrinsic commit must NOT emit the asm ds_write_b64");
    assert!(!pipe_code.contains("s_waitcnt lgkmcnt(0)"), "intrinsic commit must NOT emit a manual drain");
}

#[test]
fn matmul_lds_kblock_carries_two_barriers_per_kstep() {
    // Structural: the single-buffer WAR needs a RAW fence (after fill) AND a WAR fence
    // (after the LDS reads) — at least two Barriers in the K-loop body.
    let p = matmul_lds_kblock(32, 32, 16, 32, 32);
    let sink = lower::lower(&p.ir, p.sink, &p.name);
    let topo = sink.toposort();
    let bars = topo.iter().filter(|u| matches!(u.op(), Op::Barrier { .. })).count();
    assert!(bars >= 2, "single-buffer K-blocking needs RAW + WAR barriers, got {bars}");
}

#[test]
fn matmul_lds_tiled_lowering_is_spec_valid() {
    // The multi-accumulator reuse kernel: a 2×2 fragment grid (4 loop-carried
    // accumulators closed by ONE End via combine) + LDS staging must lower spec-valid.
    let p = matmul_lds_tiled(64, 64, 32, 32, 32);
    lower::verify(&p).expect("block-tiled LDS matmul must lower to spec-valid UOp");
}

#[test]
fn matmul_lds_tiled_carries_four_wmma() {
    // Structural: a 32×32 tile = 4 accumulators ⇒ 4 WMMAs per K-step, one Barrier.
    let p = matmul_lds_tiled(32, 32, 16, 32, 32);
    let sink = lower::lower(&p.ir, p.sink, &p.name);
    let topo = sink.toposort();
    let wmmas = topo.iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).count();
    assert_eq!(wmmas, 4, "a 2×2 fragment grid over a single K-fragment needs 4 WMMAs, got {wmmas}");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "the fill needs a barrier");
}

#[test]
fn matmul_lds_lowering_is_spec_valid() {
    // The LDS-staged matmul: fill loops + a fill barrier + K-loop fragment gathers
    // from LDS + the single-accumulator carry must lower to spec-valid UOp.
    let p = matmul_lds(32, 32, 32);
    lower::verify(&p).expect("LDS-staged matmul must lower to spec-valid UOp");
}

#[test]
fn matmul_lds_carries_lds_barrier_and_wmma() {
    let p = matmul_lds(32, 32, 32);
    let sink = lower::lower(&p.ir, p.sink, &p.name);
    let topo = sink.toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "staged matmul needs LDS buffers");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "the fill needs a barrier before the gathers");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "matmul needs a WMMA");
}

#[test]
fn lds_roundtrip_lowering_is_spec_valid() {
    // The LDS proof: DefineLocal + a cross-lane Barrier + LDS load/store must lower to
    // spec-valid UOp (this is where the store→barrier→load ordering pain would live).
    let p = lds_roundtrip(64);
    lower::verify(&p).expect("cross-lane LDS round-trip must lower to spec-valid UOp");
}

#[test]
fn lds_roundtrip_carries_local_and_barrier() {
    // Structural check: the lowered graph carries the shared-memory allocation
    // (DefineLocal), the workgroup fence (Barrier), and the cross-lane read's After edge.
    let p = lds_roundtrip(64);
    let sink = lower::lower(&p.ir, p.sink, &p.name);
    let topo = sink.toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "LDS stage needs a DefineLocal");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "cross-lane read needs a Barrier fence");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::After { .. })), "the post-barrier read routes through an After");
}

#[test]
fn sum_reduce_carries_the_ordering_edges() {
    // Structural check: the lowered graph carries the first-class ordering edges
    // (`After`) the loop-carry needs, plus RANGE/END loop scoping.
    let p = sum_reduce(64);
    let sink = lower::lower(&p.ir, p.sink, &p.name);
    let topo = sink.toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::After { .. })), "loop-carry needs After edges");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Range { .. })), "reduction needs a RANGE");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::End { .. })), "the RANGE must be closed by an END");
}
