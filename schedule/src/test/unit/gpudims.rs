//! `in_scope_ranges` is what gpudims store masking uses to find the LOCAL ranges
//! that are still open at a store. A `toposort().filter(Range)` would instead
//! return every range the graph ever opened, including ended ones.

use std::collections::HashSet;
use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::types::{AxisId, AxisType};
use svod_ir::{Op, UOp};

fn local_range(end_value: i64, axis_id: usize) -> Arc<UOp> {
    let end = UOp::const_(DType::Index, svod_ir::types::ConstValue::Int(end_value));
    UOp::range_axis(end, AxisId::Renumbered(axis_id), AxisType::Local)
}

fn in_scope_ids(uop: &Arc<UOp>) -> HashSet<u64> {
    uop.in_scope_ranges().iter().copied().collect()
}

#[test]
fn ranges_stay_in_scope_until_they_are_ended() {
    let ended_range = local_range(16, 0);
    let open_range = local_range(32, 1);
    let ended = ended_range.add(&open_range).end(smallvec::smallvec![ended_range.clone()]);
    // AFTER sequences against the END, which is Void and cannot feed an ALU.
    let downstream = open_range.add(&UOp::index_const(5)).after(smallvec::smallvec![ended]);

    let in_scope = in_scope_ids(&downstream);
    assert!(!in_scope.contains(&ended_range.id), "ended range must leave scope");
    assert!(in_scope.contains(&open_range.id), "un-ended range must stay in scope");
    assert!(
        downstream.toposort().iter().any(|u| matches!(u.op(), Op::Range(..)) && u.id == ended_range.id),
        "the ended range is still reachable by toposort — that is why the mask cannot use it",
    );
}

#[test]
fn an_index_only_holds_the_ranges_it_addresses_with() {
    let addressed = local_range(16, 0);
    let unused = local_range(16, 1);
    let buffer = UOp::param(0, 1024, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![addressed.clone()]).call().expect("index");

    let in_scope = in_scope_ids(&index);
    assert!(in_scope.contains(&addressed.id));
    assert!(!in_scope.contains(&unused.id));
}
