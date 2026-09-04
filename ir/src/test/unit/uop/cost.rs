use super::*;
use crate::{AxisId, AxisType};

fn loop_range(axis: usize, size: i64) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(axis), AxisType::Loop)
}

#[test]
fn alu_weight_is_the_product_of_enclosing_ranges() {
    let outer = loop_range(0, 8);
    let inner = loop_range(1, 4);
    let under_both = outer.try_add(&inner).unwrap();
    let unenclosed = UOp::define_var("a".to_string(), 1, 16).try_add(&UOp::define_var("b".to_string(), 1, 16)).unwrap();

    assert_eq!(compute_ops_estimate(&under_both), 8 * 4);
    assert_eq!(compute_ops_estimate(&unenclosed), 1);
    assert_eq!(compute_ops_estimate(&UOp::sink(vec![under_both, unenclosed])), 8 * 4 + 1);
}

#[test]
fn more_than_64_ranges_span_multiple_bitset_words() {
    // Only the last range iterates more than once, so every partial sum but
    // the outermost weighs 1 — a wrong bit-to-size mapping across the word
    // boundary would not land on 3.
    let ranges: Vec<Arc<UOp>> = (0..70).map(|i| loop_range(i, if i == 69 { 3 } else { 1 })).collect();
    let sum = ranges.iter().skip(1).fold(ranges[0].clone(), |acc, r| acc.try_add(r).unwrap());

    assert_eq!(compute_ops_estimate(&sum), 68 + 3);
}
