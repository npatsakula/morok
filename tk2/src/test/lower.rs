//! Host tests: the ADT interns/disambiguates correctly, and the verified lowering
//! produces spec-valid device-UOp for BOTH proof kernels.

use svod_dtype::DType;
use svod_ir::Op;

use crate::ir::{Node, RegClass, Residency, TileIr};
use crate::kernels::{elementwise_add, matmul, sum_reduce};
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
