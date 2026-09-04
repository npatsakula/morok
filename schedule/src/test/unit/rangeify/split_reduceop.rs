//! `split_reduceop`: two-stage reduction when the reduced extent is large enough
//! to be worth a materialised intermediate.

use std::sync::Arc;

use smallvec::SmallVec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{Op, ReduceOp, SInt, UOp};
use test_case::test_case;

use crate::rangeify::kernel::{SplitReduceOpConfig, collect_range_ids, split_reduceop};
use svod_ir::ops;

fn tensor(shape: &[usize]) -> Arc<UOp> {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, shape.iter().product(), DType::Float32);
    match shape {
        [_] => buffer,
        _ => buffer.try_reshape(&shape.iter().map(|&s| SInt::Const(s)).collect()).expect("reshape"),
    }
}

fn expanded(base: &[usize], to: &[usize]) -> Arc<UOp> {
    let new_shape = UOp::stack(to.iter().map(|&d| UOp::index_const(d as i64)).collect());
    UOp::new(Op::Expand(ops::Expand { src: tensor(base), new_shape }), DType::Float32)
}

fn has_contiguous(uop: &Arc<UOp>) -> bool {
    uop.toposort().iter().any(|node| matches!(node.op(), Op::Contiguous(..)))
}

/// Ratio of total elements to output elements decides the split; the default
/// threshold is 32768. A broadcast (EXPAND) axis is never a split candidate —
/// splitting it would materialise the same value repeatedly.
#[test_case(tensor(&[1_000]), 0, false ; "1d below threshold")]
#[test_case(tensor(&[100_000]), 0, true ; "1d above threshold")]
#[test_case(tensor(&[1_000, 1_000]), 1, false ; "2d ratio 1000 is below threshold")]
#[test_case(tensor(&[1_000, 100_000]), 1, true ; "2d ratio 100000 is above threshold")]
#[test_case(expanded(&[100, 1, 1_000], &[100, 500, 1_000]), 1, false ; "the reduced axis is the broadcast one")]
#[test_case(expanded(&[100, 1, 100_000], &[100, 50, 100_000]), 2, true ; "another axis is broadcast")]
fn a_reduction_splits_once_its_ratio_clears_the_threshold(source: Arc<UOp>, axis: usize, splits: bool) {
    let reduce = source.try_reduce_axis(ReduceOp::Add, vec![axis]).expect("reduce axis");

    match split_reduceop(&reduce, &SplitReduceOpConfig::default()) {
        Some(transformed) => {
            assert!(splits, "unexpected split: {}", transformed.tree());
            assert!(has_contiguous(&transformed), "the split must materialise its intermediate");
            assert_eq!(
                transformed.shape().expect("shape").expect("static").len(),
                reduce.shape().expect("shape").expect("static").len(),
                "the split must not change the output rank"
            );
        }
        None => assert!(!splits, "expected a split"),
    }
}

/// `RESHAPE(EXPAND(RESHAPE(buffer)))` flattened to one axis: the movement chain
/// has to be pushed through before the extent can be judged.
#[test]
fn a_reduction_behind_a_movement_chain_still_splits() {
    let flattened =
        expanded(&[50, 1], &[50, 1_000]).try_reshape(&smallvec::smallvec![SInt::Const(50_000)]).expect("reshape");
    let reduce = flattened.try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis");

    let transformed = split_reduceop(&reduce, &SplitReduceOpConfig::default()).expect("50000 clears the threshold");
    assert!(has_contiguous(&transformed));
}

#[test_case(ReduceOp::Add ; "add")]
#[test_case(ReduceOp::Mul ; "mul")]
#[test_case(ReduceOp::Max ; "max")]
#[test_case(ReduceOp::Min ; "min")]
fn the_split_keeps_the_original_reduce_op(reduce_op: ReduceOp) {
    let reduce = tensor(&[100_000]).try_reduce_axis(reduce_op, vec![0]).expect("reduce axis");

    let transformed = split_reduceop(&reduce, &SplitReduceOpConfig::default()).expect("split");
    assert!(
        transformed
            .toposort()
            .iter()
            .any(|node| matches!(node.op(), Op::Reduce(ops::Reduce { reduce_op: op, .. }) if *op == reduce_op)),
        "{reduce_op:?} must survive the split"
    );
}

#[test]
fn the_split_can_be_turned_off() {
    let config = SplitReduceOpConfig { enabled: false, ..Default::default() };
    let reduce = tensor(&[100_000]).try_reduce_axis(ReduceOp::Add, vec![0]).expect("reduce axis");

    assert!(split_reduceop(&reduce, &config).is_none());
}

#[test]
fn the_output_size_cap_follows_the_configured_bit_width() {
    let default = SplitReduceOpConfig::default();
    assert_eq!((default.split_threshold, default.max_divisor, default.min_divisor), (32768, 256, 8));
    assert_eq!(default.max_output_size(), 1 << default.output_size_bits);

    let narrower = SplitReduceOpConfig { output_size_bits: 20, ..Default::default() };
    assert_eq!(narrower.max_output_size(), 1 << 20);
}

fn range_ids(ranges: &[(i64, usize)]) -> Vec<usize> {
    let uops: SmallVec<[Arc<UOp>; 4]> = ranges.iter().map(|&(end, id)| UOp::range_const(end, id)).collect();
    let expr = uops.iter().skip(1).fold(uops[0].clone(), |acc, r| acc.try_add(r).expect("add"));
    collect_range_ids(&expr)
}

/// `collect_range_ids` returns every RANGE axis in the expression, sorted.
#[test]
fn range_ids_come_back_sorted() {
    assert_eq!(collect_range_ids(&UOp::native_const(1.0f32)), Vec::<usize>::new());
    assert_eq!(range_ids(&[(10, 0)]), vec![0]);
    assert_eq!(range_ids(&[(10, 0), (5, 1), (3, 2)]), vec![0, 1, 2]);
    assert_eq!(range_ids(&[(3, 2), (10, 0), (5, 1)]), vec![0, 1, 2], "source order does not matter");
}
