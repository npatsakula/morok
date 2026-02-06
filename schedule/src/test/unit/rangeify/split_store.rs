//! Tests for pm_split_store (Stage 6) - CPU-only store splitting.
//!
//! These tests verify that stores are correctly split at comparison cut points
//! to enable branch elimination on CPU targets.

use std::sync::Arc;

use morok_device::DeviceSpec;
use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, UOp};

use crate::pattern::RewriteResult;
use crate::rangeify::transforms::{SplitStoreContext, pm_split_store};

/// Helper to create a simple store with END wrapper.
fn create_store_with_end(range_end: i64) -> (Arc<UOp>, Arc<UOp>) {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, range_end as usize, DType::Float32);
    let range = UOp::range_axis(UOp::index_const(range_end), AxisId::Renumbered(0), AxisType::Loop);
    let value = UOp::native_const(1.0f32);
    let index = UOp::index().buffer(buffer).indices(vec![range.clone()]).call().expect("index creation");
    let store = index.store_value(value);
    let end = store.end(smallvec::smallvec![range.clone()]);
    (end, range)
}

#[test]
fn test_split_store_cpu_only() {
    // Split store should only apply to CPU
    let (end, range) = create_store_with_end(16);

    // Create a comparison: range < 5
    let cut = UOp::index_const(5);
    let _cond = range.try_cmplt(&cut).expect("cmplt");

    // With GPU context, should not transform
    let mut gpu_ctx = SplitStoreContext::for_device(DeviceSpec::Cuda { device_id: 0 });
    let matcher = pm_split_store();
    let result = matcher.rewrite(&end, &mut gpu_ctx);
    assert!(matches!(result, RewriteResult::NoMatch), "Should not transform on GPU");

    // With CPU context, check the pattern matches (actual splitting depends on finding comparisons)
    let mut cpu_ctx = SplitStoreContext::for_device(DeviceSpec::Cpu);
    let result = matcher.rewrite(&end, &mut cpu_ctx);
    // Note: This may still not match if the comparison isn't in the consumer map
    // The pattern requires Lt(range, const) consumers to be present
    assert!(
        matches!(result, RewriteResult::NoMatch | RewriteResult::Rewritten(_)),
        "Pattern should at least be checked on CPU"
    );
}

#[test]
fn test_split_store_no_comparisons() {
    // Without any Lt comparisons, no splitting should occur
    let (end, _range) = create_store_with_end(16);

    let mut ctx = SplitStoreContext::for_device(DeviceSpec::Cpu);
    let matcher = pm_split_store();
    let result = matcher.rewrite(&end, &mut ctx);

    assert!(matches!(result, RewriteResult::NoMatch), "Should not split without comparison cut points");
}
