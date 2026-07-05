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
    let p = crate::kernels::matmul_lds_kblock_sw(64, 64, 64, 64, 64);
    lower::verify(&p).expect("swizzled K-blocked matmul must lower to spec-valid UOp");
}

#[test]
fn matmul_lds_kblock_lowering_is_spec_valid() {
    // The K-blocked kernel: per-K-block fill + TWO barriers (RAW + WAR) + the reused
    // 2×2 accumulator grid, all inside one K-loop, must lower to spec-valid UOp.
    let p = matmul_lds_kblock(64, 64, 64, 32, 32);
    lower::verify(&p).expect("K-blocked LDS matmul must lower to spec-valid UOp");
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
