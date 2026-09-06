//! Device buffer-limit enforcement: when a kernel would bind more buffers than
//! the device allows, elementwise sub-expressions are materialised.
//!
//! The limit is passed to `buffer_limit_patterns` explicitly, so every row runs
//! on CPU without a device feature gate.

use std::collections::HashSet;
use std::sync::Arc;

use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{AddrSpace, AxisId, AxisType, BufferizeOpts, Op, ReduceOp, SInt, UOp, UOpKey};
use test_case::test_case;

use crate::rangeify::indexing::IndexingContext;
use crate::rangeify::patterns::{buffer_limit_patterns, extract_device_from_graph, is_elementwise};
use crate::rewrite::graph_rewrite;
use svod_ir::ops;

fn test_buffer(size: usize, slot: usize) -> Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![SInt::Const(size)]);
    let arg = svod_ir::ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu));
    UOp::new(Op::Buffer(ops::Buffer { shape, arg: arg.into() }), DType::Float32)
}

fn read(slot: usize, range: &Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(test_buffer(40, slot)).indices(vec![range.clone()]).call().expect("index")
}

/// `(((b0 + b1) + b2) + ...)` over `count` distinct buffers.
fn add_chain(count: usize, range: &Arc<UOp>) -> Arc<UOp> {
    (1..count).fold(read(0, range), |acc, slot| acc.try_add(&read(slot, range)).expect("add"))
}

fn count_stages(uop: &Arc<UOp>) -> usize {
    let (mut stack, mut visited, mut count) = (vec![uop.clone()], HashSet::new(), 0);
    while let Some(current) = stack.pop() {
        if !visited.insert(UOpKey(current.clone())) {
            continue;
        }
        count += usize::from(matches!(current.op(), Op::Stage(..)));
        stack.extend(current.op().sources());
    }
    count
}

/// The output buffer takes one slot, so `max` buffers means at most `max - 1`
/// inputs before an elementwise operand has to be materialised.
#[test_case(30, false ; "one below the limit")]
#[test_case(31, true ; "at the limit")]
#[test_case(35, true ; "well over the limit")]
fn the_output_buffer_costs_one_slot(inputs: usize, materializes: bool) {
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let computation = add_chain(inputs, &range);

    let result = graph_rewrite(&buffer_limit_patterns(31), computation.clone(), &mut ctx);

    assert_eq!(count_stages(&result) > count_stages(&computation), materializes);
    if !materializes {
        assert!(Arc::ptr_eq(&result, &computation));
    }
}

/// A WHERE's condition, true and false arms all count toward the same limit.
#[test]
fn a_ternary_operand_tree_is_materialized_too() {
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    let mut cond = read(1, &range).try_cmplt(&read(0, &range)).expect("cmplt");
    for slot in (2..10).step_by(2) {
        let cmp = read(slot + 1, &range).try_cmplt(&read(slot, &range)).expect("cmplt");
        cond = cond.try_and_op(&cmp).expect("and");
    }
    let on_false = read(11, &range).try_add(&read(12, &range)).expect("add");
    let where_op = UOp::try_where(cond, read(10, &range), on_false).expect("where");

    let result = graph_rewrite(&buffer_limit_patterns(10), where_op.clone(), &mut ctx);
    assert!(count_stages(&result) > count_stages(&where_op));
}

/// Already-materialised operands are not re-staged.
#[test]
fn an_existing_stage_is_not_materialized_again() {
    let mut ctx = IndexingContext::new();
    let ranges = vec![ctx.new_range(&SInt::Const(10), AxisType::Loop)];

    let opts = BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Global, removable: true };
    let staged = UOp::stage(read(0, &ranges[0]), ranges.clone(), opts);
    let indexed = UOp::index().buffer(staged).indices(ranges).call().expect("index");

    let result = graph_rewrite(&buffer_limit_patterns(31), indexed.clone(), &mut ctx);
    assert_eq!(count_stages(&result), count_stages(&indexed));
}

// ===== range-id allocation for the ranges the new STAGE carries =====

/// A range created during indexing and then collapsed by dead-axis cleanup still
/// consumed its id: the STAGE must use the next id, not reuse the collapsed one.
#[test]
fn a_collapsed_range_still_consumes_its_axis_id() {
    let mut ctx = IndexingContext::new();
    let visible = ctx.new_range(&SInt::Const(10), AxisType::Weak);
    let _collapsed = ctx.new_range_from_uop(&UOp::index_const(1), AxisType::Weak);

    let root = add_chain(3, &visible);
    let result = graph_rewrite(&buffer_limit_patterns(3), root, &mut ctx);

    let stage_ids: Vec<_> = result
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Stage(ops::Stage { ranges, .. }) => ranges.iter().find_map(|r| match r.op() {
                Op::Range(ops::Range { axis_id, .. }) => Some(axis_id.clone()),
                _ => None,
            }),
            _ => None,
        })
        .collect();

    assert_eq!(stage_ids, vec![AxisId::Unrenumbered(2)]);
    assert_eq!(ctx.range_counter(), 3);
}

/// A DEVICE range is a launch lane, not an allocated axis: the STAGE keeps it
/// verbatim alongside the freshly numbered WEAK range.
#[test]
fn a_device_range_is_carried_through_without_renumbering() {
    let mut ctx = IndexingContext::new();
    let weak = ctx.new_range(&SInt::Const(10), AxisType::Weak);
    let _collapsed = ctx.new_range_from_uop(&UOp::index_const(1), AxisType::Weak);
    let launched = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(7), AxisType::Device);

    let mixed = read(0, &weak).try_add(&read(1, &launched)).expect("add");
    let root = mixed.try_add(&read(2, &weak)).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(3), root, &mut ctx);
    let stage_ranges = result
        .toposort()
        .into_iter()
        .find_map(|u| match u.op() {
            Op::Stage(ops::Stage { ranges, .. }) => Some(ranges.clone()),
            _ => None,
        })
        .expect("buffer limit should materialize the mixed source");

    assert!(stage_ranges.iter().any(|r| Arc::ptr_eq(r, &launched)));
    assert!(stage_ranges.iter().any(|r| {
        matches!(r.op(), Op::Range(ops::Range { axis_id: AxisId::Unrenumbered(2), axis_type: AxisType::Weak, .. }))
    }));
    assert_eq!(ctx.range_counter(), 3);
}

// ===== helpers the pattern is built on =====

/// Only elementwise nodes are candidates for materialisation — a leaf has
/// nothing to materialise into. The set is tinygrad's `GroupOp.Elementwise`:
/// unary, binary and ternary ALU plus CAST and BITCAST.
#[test]
fn alu_and_cast_nodes_are_elementwise() {
    let (a, b) = (UOp::native_const(1.0f32), UOp::native_const(2.0f32));
    assert!(is_elementwise(&a.try_add(&b).expect("add")));
    assert!(is_elementwise(&UOp::try_where(UOp::native_const(true), a.clone(), b.clone()).expect("where")));
    assert!(is_elementwise(&a.try_sqrt().expect("sqrt")));
    assert!(is_elementwise(&a.cast(DType::Float64)));
    assert!(!is_elementwise(&a));
    assert!(!is_elementwise(&test_buffer(100, 1)));
}

#[test]
fn the_device_comes_from_a_buffer_or_a_copy_target() {
    assert_eq!(extract_device_from_graph(&test_buffer(100, 1)), Some(DeviceSpec::Cpu));
    assert_eq!(
        extract_device_from_graph(&UOp::native_const(1.0f32).copy_to_device(DeviceSpec::Cpu)),
        Some(DeviceSpec::Cpu)
    );
    assert_eq!(extract_device_from_graph(&UOp::native_const(1.0f32)), None);
}

// ===== what counts as a kernel argument =====

fn test_param(slot: usize) -> Arc<UOp> {
    UOp::param(slot, 40, DType::Float32, Some(DeviceSpec::Cpu))
}

fn read_from(buffer: Arc<UOp>, range: &Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![range.clone()]).call().expect("index")
}

/// `(((s0 + s1) + s2) + ...)` over `count` distinct storages built by `storage`.
fn chain_over(count: usize, range: &Arc<UOp>, storage: impl Fn(usize) -> Arc<UOp>) -> Arc<UOp> {
    (1..count)
        .fold(read_from(storage(0), range), |acc, slot| acc.try_add(&read_from(storage(slot), range)).expect("add"))
}

/// Model weights reach rangeify as PARAMs, not BUFFERs. Leaving them out of the
/// count let an inception concat fuse 32 inputs into one kernel, which Metal
/// then refused to bind (`no 'buffer' resource location available for 'data31'`).
#[test_case(30, false ; "one below the limit")]
#[test_case(31, true ; "at the limit")]
#[test_case(35, true ; "well over the limit")]
fn a_param_costs_an_argument_slot_like_a_buffer(inputs: usize, materializes: bool) {
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let computation = chain_over(inputs, &range, test_param);

    let result = graph_rewrite(&buffer_limit_patterns(31), computation.clone(), &mut ctx);

    assert_eq!(count_stages(&result) > count_stages(&computation), materializes);
    if !materializes {
        assert!(Arc::ptr_eq(&result, &computation));
    }
}

/// LOCAL storage is compiler-managed: it lives inside the kernel and never binds
/// an argument, so no amount of it can trip the limit.
#[test]
fn local_storage_does_not_consume_an_argument_slot() {
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let computation = chain_over(40, &range, |slot| UOp::buffer(slot, 40, DType::Float32, AddrSpace::Local, None));

    let result = graph_rewrite(&buffer_limit_patterns(31), computation.clone(), &mut ctx);

    assert!(Arc::ptr_eq(&result, &computation));
}

/// AFTER is a buffer identity: the kernels it orders against write the buffer,
/// they are not read by this one. Walking into its dependencies counted a whole
/// producer cone against a kernel that binds a single argument.
#[test]
fn an_after_costs_one_argument_not_its_producer_cone() {
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    let deps: smallvec::SmallVec<[Arc<UOp>; 4]> = (2..42).map(|slot| read(slot, &range)).collect();
    let ordered = read_from(test_buffer(40, 0).after(deps), &range);

    let fused = ordered.try_add(&read(1, &range)).expect("add");
    let root = fused.try_add(&ordered).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(31), root.clone(), &mut ctx);

    assert!(Arc::ptr_eq(&result, &root));
}

/// `GroupOp.Elementwise` is ALU plus the casts (tinygrad `uop/__init__.py:112`),
/// so a CAST operand is a materialisation candidate like any binary one.
#[test]
fn a_cast_operand_is_materialized() {
    let mut ctx = IndexingContext::new();
    let range = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    let wide = add_chain(30, &range).cast(DType::Float64);
    let root = wide.try_add(&read(30, &range).cast(DType::Float64)).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(31), root.clone(), &mut ctx);

    assert!(count_stages(&result) > count_stages(&root));
}

/// The new STAGE's axes are the ranges still open at the operand. A REDUCE
/// closes its own, and putting one on the STAGE made the range substitution
/// rebuild every producer that binds it — copies that compound through a chain
/// of bufferized kernels until the graph explodes.
#[test]
fn a_closed_reduce_range_is_not_an_axis_of_the_new_stage() {
    let mut ctx = IndexingContext::new();
    let outer = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let inner = ctx.new_range(&SInt::Const(4), AxisType::Reduce);

    let reduced = add_chain(30, &inner).reduce(smallvec::smallvec![inner.clone()], ReduceOp::Add);
    let over_limit = reduced.try_add(&read(30, &outer)).expect("add");
    let root = over_limit.try_add(&read(31, &outer)).expect("add");

    let result = graph_rewrite(&buffer_limit_patterns(31), root.clone(), &mut ctx);

    let stage_ranges = result
        .toposort()
        .into_iter()
        .find_map(|u| match u.op() {
            Op::Stage(ops::Stage { ranges, .. }) => Some(ranges.clone()),
            _ => None,
        })
        .expect("buffer limit should materialize the reduced operand");

    assert_eq!(stage_ranges.len(), 1, "only the open outer range is an axis, got {stage_ranges:?}");
    assert!(stage_ranges.iter().all(|r| !Arc::ptr_eq(r, &inner)));
    // The REDUCE keeps the axis it closes: it was never substituted.
    assert!(result.toposort().iter().any(|u| Arc::ptr_eq(u, &inner)));
}
