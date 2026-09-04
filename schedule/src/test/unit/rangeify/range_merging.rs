//! `merge_consumer_ranges`: one range per dimension across every consumer, and
//! the realize decision that falls out of it.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisType, BinaryOp, Op, SInt, TernaryOp, UOp, UOpKey};

use crate::rangeify::indexing::{IndexingContext, all_ranges_same};
use crate::rangeify::merge_consumer_ranges;
use svod_ir::ops;

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, size, DType::Float32)
}

fn reshaped_2d(dims: &[usize]) -> Arc<UOp> {
    let src = buffer(dims.iter().product());
    let new_shape = UOp::stack(dims.iter().map(|&d| UOp::index_const(d as i64)).collect());
    UOp::new(Op::Reshape(ops::Reshape { src, new_shape }), DType::Float32)
}

/// A range gated by `i < bound`, as a consumer with padding produces.
fn gated(idx: &Arc<UOp>, bound: i64) -> (Arc<UOp>, Arc<UOp>) {
    let valid = idx.try_cmplt(&UOp::index_const(bound)).expect("cmplt");
    let wrapped = UOp::try_where(valid.clone(), idx.clone(), UOp::invalid_marker()).expect("where");
    (wrapped, valid)
}

fn realize_axes(ctx: &IndexingContext, uop: &Arc<UOp>) -> Option<Option<Vec<usize>>> {
    ctx.realize_map.get(&UOpKey(uop.clone())).cloned()
}

/// `all_ranges_same` is the merge's decision procedure. Vacuously true for zero
/// or one entry (tinygrad `helpers.py:31`), and pointer-based beyond that.
#[test]
fn ranges_are_the_same_only_when_they_are_the_same_node() {
    let mut ctx = IndexingContext::new();
    let r0 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    assert!(all_ranges_same(&[]));
    assert!(all_ranges_same(std::slice::from_ref(&r0)));
    assert!(all_ranges_same(&[r0.clone(), r0.clone()]));
    assert!(!all_ranges_same(&[r0.get_idx(), r1.get_idx()]));
}

/// A plain range is its own index and is unconditionally valid; a gated one
/// splits back into the two.
#[test]
fn a_gated_range_decomposes_into_its_index_and_its_condition() {
    let mut ctx = IndexingContext::new();
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);

    assert!(Arc::ptr_eq(&idx.get_idx(), &idx));
    assert!(matches!(idx.get_valid().op(), Op::Const(c) if c.0 == svod_ir::ConstValue::Bool(true)));

    let (wrapped, valid) = gated(&idx, 5);
    assert!(Arc::ptr_eq(&wrapped.get_idx(), &idx));
    assert!(Arc::ptr_eq(&wrapped.get_valid(), &valid));
    let Op::Ternary(TernaryOp::Where, _, _, otherwise) = wrapped.op() else { panic!("expected WHERE") };
    assert!(UOp::is_invalid_marker(otherwise));
}

#[test]
fn identical_consumer_ranges_merge_without_realizing() {
    let mut ctx = IndexingContext::new();
    let buffer = buffer(100);
    let r0 = ctx.new_range(&SInt::Const(100), AxisType::Loop);

    let merged = merge_consumer_ranges(&buffer, &[vec![r0.clone()], vec![r0.clone()]], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1);
    assert!(Arc::ptr_eq(&merged[0], &r0), "nothing to reconcile, so the range is passed through");
    assert!(realize_axes(&ctx, &buffer).is_none());
}

#[test]
fn differing_consumer_ranges_force_a_fresh_range_and_a_realize() {
    let mut ctx = IndexingContext::new();
    let buffer = buffer(100);
    let a = ctx.new_range(&SInt::Const(100), AxisType::Loop);
    let b = ctx.new_range(&SInt::Const(100), AxisType::Loop);

    let merged = merge_consumer_ranges(&buffer, &[vec![a.clone()], vec![b.clone()]], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1);
    assert!(!Arc::ptr_eq(&merged[0], &a) && !Arc::ptr_eq(&merged[0], &b));
    assert_eq!(realize_axes(&ctx, &buffer), Some(Some(vec![0])));
}

/// With PCONTIG=0 a single disagreeing dim realizes them all — tinygrad
/// `indexing.py:217` only consults `all_all_same`.
#[test]
fn one_disagreeing_dimension_realizes_every_dimension() {
    let mut ctx = IndexingContext::new();
    let reshaped = reshaped_2d(&[10, 20]);
    let shared = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let a = ctx.new_range(&SInt::Const(20), AxisType::Loop);
    let b = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    let consumers = [vec![shared.clone(), a.clone()], vec![shared.clone(), b]];
    let merged = merge_consumer_ranges(&reshaped, &consumers, &mut ctx).expect("merge");

    assert_eq!(merged.len(), 2);
    assert!(!Arc::ptr_eq(&merged[0], &shared), "the agreeing dim is realized too");
    assert!(!Arc::ptr_eq(&merged[1], &a));
    assert_eq!(realize_axes(&ctx, &reshaped), Some(Some(vec![0, 1])));
}

/// Consumers reading the same index under different guards merge to that index
/// under the disjunction of the guards.
#[test]
fn differing_validity_masks_are_merged_with_or() {
    let mut ctx = IndexingContext::new();
    let buffer = buffer(10);
    let idx = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let (narrow, _) = gated(&idx, 5);
    let (wide, _) = gated(&idx, 8);

    let merged = merge_consumer_ranges(&buffer, &[vec![narrow], vec![wide]], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1);
    let Op::Ternary(TernaryOp::Where, valid, merged_idx, _) = merged[0].op() else {
        panic!("expected a gated range, got {}", merged[0].tree())
    };
    assert!(Arc::ptr_eq(merged_idx, &idx));
    assert!(matches!(valid.op(), Op::Binary(BinaryOp::Or, _, _)), "got {}", valid.tree());
}

/// `all_same([])` is true upstream, so a dim with no consumer ranges does not
/// drag the other dims into a realize — but it has nothing to inherit either, so
/// it gets a fresh range and is realized on its own.
#[test]
fn a_dimension_with_no_consumers_is_realized_on_its_own() {
    let mut ctx = IndexingContext::new();
    let buffer = buffer(10);

    let merged = merge_consumer_ranges(&buffer, &[], &mut ctx).expect("merge");

    assert_eq!(merged.len(), 1);
    assert_eq!(realize_axes(&ctx, &buffer), Some(Some(vec![0])));
}
