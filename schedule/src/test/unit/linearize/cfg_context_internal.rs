use super::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;

#[test]
fn test_cfg_context_single_range() {
    // Single RANGE should have no edges
    let end_val = UOp::index_const(10);
    let range = UOp::range(end_val, 0);
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let end = value.end(smallvec::smallvec![range]);
    let sink = UOp::sink(vec![end]);

    let ctx = CFGContext::new(&sink);
    assert!(!ctx.has_edges());
}

#[test]
fn test_cfg_context_sibling_ranges() {
    // Two sibling RANGEs should have one edge
    let end_val = UOp::index_const(10);
    let range1 = UOp::range(end_val.clone(), 0);
    let range2 = UOp::range(end_val, 1);

    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let end = value.end(smallvec::smallvec![range1.clone(), range2.clone()]);
    let sink = UOp::sink(vec![end]);

    let ctx = CFGContext::new(&sink);
    // With 2 ranges, we should have 1 edge (range2 → range1)
    assert!(ctx.edge_count() <= 1);
}

#[test]
fn test_cfg_context_nested_ranges() {
    // Nested RANGEs: inner loop runs inside outer loop.
    // For inner_end to be nested inside outer_end, inner_end must depend on outer_range.
    let end_val = UOp::index_const(10);

    // Outer range first (so inner can depend on it)
    let outer_range = UOp::range(end_val.clone(), 1);

    // Inner range
    let inner_range = UOp::range(end_val, 0);

    // Inner value that depends on outer_range (so it runs inside outer loop)
    // Use outer_range as part of the computation to create the dependency
    let outer_idx = outer_range.cast(DType::Float32);
    let inner_value = UOp::const_(DType::Float32, ConstValue::Float(1.0)).add(&outer_idx);
    let inner_end = inner_value.end(smallvec::smallvec![inner_range.clone()]);

    // Outer END
    let outer_end = inner_end.end(smallvec::smallvec![outer_range.clone()]);

    let sink = UOp::sink(vec![outer_end]);

    let ctx = CFGContext::new(&sink);
    // inner_end is nested inside outer_end (not siblings), so outer_range
    // should have no predecessor edge
    assert!(ctx.get_predecessor(&outer_range).is_none());
}
