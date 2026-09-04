use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{AxisType, Op, ReduceOp, UOp};

use crate::rangeify::{SimplifyRangesContext, pm_simplify_ranges};
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

fn buffer() -> std::sync::Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, 16, DType::Float32)
}

/// `INDEX(buffer, [range gated by `range < bound`])` — the shape a padded or
/// shrunk access takes after rangeify.
fn gated_index(range: &std::sync::Arc<UOp>, bound: i64) -> std::sync::Arc<UOp> {
    let gate = range.try_cmplt(&UOp::index_const(bound)).expect("cmplt");
    UOp::index().buffer(buffer()).indices(vec![range.valid(gate)]).call().expect("index")
}

fn gated_load(range: &std::sync::Arc<UOp>, bound: i64) -> std::sync::Arc<UOp> {
    UOp::load().index(gated_index(range, bound)).call()
}

fn loop_range(axis: usize) -> std::sync::Arc<UOp> {
    UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(axis), AxisType::Loop)
}

fn narrowed_end(root: &std::sync::Arc<UOp>, axis: usize) -> i64 {
    root.ranges()
        .iter()
        .find_map(|range| match range.op() {
            Op::Range(ops::Range { end, axis_id: svod_ir::AxisId::Renumbered(id), .. }) if *id == axis => {
                end.vmax().try_int()
            }
            _ => None,
        })
        .expect("range must remain in rewritten graph")
}

fn simplify(sink: std::sync::Arc<UOp>) -> std::sync::Arc<UOp> {
    graph_rewrite(&pm_simplify_ranges(), sink, &mut SimplifyRangesContext::default())
}

#[test]
fn bounded_load_narrows_range() {
    let result = simplify(UOp::sink(vec![gated_load(&loop_range(0), 7)]));
    assert_eq!(narrowed_end(&result, 0), 7);
}

#[test]
fn bounded_store_narrows_range() {
    let store = gated_index(&loop_range(1), 5).store(UOp::native_const(1.0f32));
    let result = simplify(UOp::sink(vec![store]));
    assert_eq!(narrowed_end(&result, 1), 5);
}

#[test]
fn conflicting_gates_choose_largest_bound() {
    let range = loop_range(2);
    let result = simplify(UOp::sink([4, 9].map(|bound| gated_load(&range, bound)).to_vec()));
    assert_eq!(narrowed_end(&result, 2), 9);
}

#[test]
fn reduce_range_is_protected() {
    let range = UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(3), AxisType::Reduce);
    let reduce = gated_load(&range, 6).reduce(smallvec![range], ReduceOp::Add);
    let result = simplify(UOp::sink(vec![reduce]));
    assert_eq!(narrowed_end(&result, 3), 16);
}

#[test]
fn ungated_and_noncanonical_gates_are_noops() {
    let ungated = loop_range(4);
    let ungated_index = UOp::index().buffer(buffer()).indices(vec![ungated]).call().expect("index");

    // The gate bounds `r + 1`, not `r` — not the canonical form the pass reads.
    let indirect = loop_range(5);
    let gate = indirect.add(&UOp::index_const(1)).try_cmplt(&UOp::index_const(8)).expect("cmplt");
    let indirect_index = UOp::index().buffer(buffer()).indices(vec![indirect.valid(gate)]).call().expect("index");

    let loads = [ungated_index, indirect_index].map(|index| UOp::load().index(index).call());
    let result = simplify(UOp::sink(loads.to_vec()));
    assert_eq!(narrowed_end(&result, 4), 16);
    assert_eq!(narrowed_end(&result, 5), 16);
}

#[test]
fn ungated_trailing_index_protects_its_range() {
    // A range narrowed by one access must not be shrunk when another access uses
    // it in a later, ungated index position.
    let (r, q) = (loop_range(6), loop_range(7));

    let narrow = gated_index(&r, 4);

    let matrix = UOp::new_buffer(DeviceSpec::Cpu, 256, DType::Float32)
        .try_reshape(&smallvec![svod_ir::SInt::Const(16), svod_ir::SInt::Const(16)])
        .expect("reshape");
    let wide_gate = q.try_cmplt(&UOp::index_const(2)).unwrap();
    let wide = UOp::index().buffer(matrix).indices(vec![q.valid(wide_gate), r.clone()]).call().unwrap();

    let loads = [narrow, wide].map(|index| UOp::load().index(index).call());
    let result = simplify(UOp::sink(loads.to_vec()));

    assert_eq!(narrowed_end(&result, 6), 16, "r is used ungated in the second index");
    assert_eq!(narrowed_end(&result, 7), 2, "q is gated everywhere it is used");
}
