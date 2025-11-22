pub mod advanced_edge_cases;
pub mod buffer_folding;
pub mod bufferize_to_store;
pub mod codegen_integration;
pub mod codegen_patterns;
pub mod context;
pub mod cost_based;
pub mod cycle_detection;
pub mod dead_axis;
pub mod edge_cases;
pub mod flatten_range;
pub mod helpers;
pub mod indexing;
pub mod kernel_context;
pub mod kernel_count;
pub mod patterns;
pub mod pipeline;
pub mod pipeline_integration;
pub mod split_kernel;
pub mod split_patterns;
pub mod transform;

use morok_dtype::DType;
use morok_ir::{ConstValue, UOp};

use crate::rangeify::RangeifyContext;
use crate::rangeify::patterns as rangeify_patterns;

#[test]
fn test_rangeify_context_new() {
    let ctx = RangeifyContext::new();
    assert_eq!(ctx.range_counter, 0);
    assert_eq!(ctx.range_map.len(), 0);
}

#[test]
fn test_rangeify_context_next_range_id() {
    let mut ctx = RangeifyContext::new();

    assert_eq!(ctx.next_range_id(), 0);
    assert_eq!(ctx.next_range_id(), 1);
    assert_eq!(ctx.next_range_id(), 2);
    assert_eq!(ctx.range_counter, 3);
}

#[test]
fn test_rangeify_context_record_transform() {
    let mut ctx = RangeifyContext::new();

    let original = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let rangeified = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    ctx.record_transform(original.clone(), rangeified.clone());

    let retrieved = ctx.get_rangeified(&original);
    assert!(retrieved.is_some());
    assert!(std::rc::Rc::ptr_eq(retrieved.unwrap(), &rangeified));
}

#[test]
fn test_rangeify_context_get_missing() {
    let ctx = RangeifyContext::new();

    let uop = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    assert!(ctx.get_rangeified(&uop).is_none());
}

#[test]
fn test_pattern_matchers_stub() {
    // Test that stub pattern matchers return empty matchers
    let m3 = rangeify_patterns::buffer_folding();
    let m4 = rangeify_patterns::buffer_removal();
    let m5 = rangeify_patterns::kernel_splitting();

    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Should all return NoMatch since they're empty
    use crate::pattern::matcher::RewriteResult;
    assert!(matches!(m3.rewrite(&x), RewriteResult::NoMatch));
    assert!(matches!(m4.rewrite(&x), RewriteResult::NoMatch));
    assert!(matches!(m5.rewrite(&x), RewriteResult::NoMatch));
}

#[test]
fn test_early_rewrites_detach_removal() {
    use crate::pattern::matcher::RewriteResult;
    use morok_ir::Op;

    let matcher = rangeify_patterns::early_rewrites();

    // Test: DETACH(x) -> x
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let detach = UOp::new(Op::Detach { src: x.clone() }, DType::Float32);

    let result = matcher.rewrite(&detach);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_early_rewrites_contiguous_backward_removal() {
    use crate::pattern::matcher::RewriteResult;
    use morok_ir::Op;

    let matcher = rangeify_patterns::early_rewrites();

    // Test: CONTIGUOUS_BACKWARD(x) -> x
    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let contiguous = UOp::new(Op::ContiguousBackward { src: x.clone() }, DType::Float32);

    let result = matcher.rewrite(&contiguous);
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::rc::Rc::ptr_eq(&rewritten, &x));
    }
}
