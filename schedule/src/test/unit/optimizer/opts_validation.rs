//! Ported tests from Tinygrad's test_kernel_opts.py
//!
//! These tests validate that optimization operations (OptOps) work correctly.
//! Each test is documented with its original Tinygrad source for reference.
//!
//! Note: These tests focus on structural correctness (axis types, shapes, counts)
//! since morok doesn't yet have execution/runtime infrastructure for numerical
//! validation.

use crate::optimizer::{Opt, Renderer, Scheduler, apply_opt};
use crate::test::helpers::*;
use morok_ir::{AxisType, ReduceOp};

/// Port of Tinygrad test_kernel_opts.py::test_upcasts (lines 37-47)
///
/// Original test creates elementwise operations on 16×16 tensors and validates
/// that UPCAST optimization works with amounts 2, 4, and 8.
///
/// This validates that:
/// - UPCAST splits a Global axis into (Global, Upcast)
/// - Different upcast amounts work correctly
/// - Axis types are correct after transformation
///
/// Original Tinygrad code:
/// ```python
/// def test_upcasts(self):
///   N = 16
///   Tensor.manual_seed(1772)
///   a = Tensor.rand(N, N)
///   b = Tensor.rand(N, N)
///   r = (a+b).sqrt() * ((a+1).exp())
///   helper_linearizer_opt(r, [
///     [Opt(OptOps.UPCAST, 0, 2)],
///     [Opt(OptOps.UPCAST, 0, 4)],
///     [Opt(OptOps.UPCAST, 0, 8)],
///   ])
/// ```
#[test]
fn test_upcasts() {
    // Create a simple pattern to upcast (16x16 elementwise)
    let pattern = create_elementwise_pattern(&[16, 16]);
    let renderer = Renderer::cpu();

    // Test upcast by 2
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::upcast(0, 2), true);
        assert!(result.is_ok(), "UPCAST by 2 should succeed: {:?}", result.err());

        // Should now have 3 axes after upcast
        assert_eq!(sched.shape_len(), 3, "Should have 3 axes after upcast by 2");

        // After upcast, the axis order is determined by priority sorting
        // Upcast has lower priority than Global, so axes will be reordered
        // Let's just verify we have the right axis types and one upcast axis
        assert_axis_count(&sched, AxisType::Upcast, 1);
        assert_axis_count(&sched, AxisType::Global, 2);

        // Verify at least one upcast axis exists with size 2
        let upcast_axes = sched.axes_of(&[AxisType::Upcast]);
        assert_eq!(upcast_axes.len(), 1, "Should have exactly one upcast axis");
    }

    // Test upcast by 4
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::upcast(0, 4), true);
        assert!(result.is_ok(), "UPCAST by 4 should succeed: {:?}", result.err());

        // Should have 3 axes with one upcast
        assert_eq!(sched.shape_len(), 3, "Should have 3 axes after upcast by 4");
        assert_axis_count(&sched, AxisType::Upcast, 1);
        assert_axis_count(&sched, AxisType::Global, 2);
    }

    // Test upcast by 8
    {
        let mut sched = Scheduler::new(pattern, renderer);
        let result = apply_opt(&mut sched, &Opt::upcast(0, 8), true);
        assert!(result.is_ok(), "UPCAST by 8 should succeed: {:?}", result.err());

        // Should have 3 axes with one upcast
        assert_eq!(sched.shape_len(), 3, "Should have 3 axes after upcast by 8");
        assert_axis_count(&sched, AxisType::Upcast, 1);
        assert_axis_count(&sched, AxisType::Global, 2);
    }
}

/// Port of Tinygrad test_kernel_opts.py::test_full_upcast (lines 49-56)
///
/// Original test creates elementwise operations on a 4-element tensor and validates
/// that UPCAST can fully upcast the entire dimension (size 4, upcast by 4).
///
/// This validates that:
/// - UPCAST can consume an entire axis
/// - The resulting Global axis has size 1
/// - The Upcast axis has the full original size
///
/// Original Tinygrad code:
/// ```python
/// def test_full_upcast(self):
///   Tensor.manual_seed(1772)
///   a = Tensor.rand(4)
///   b = Tensor.rand(4)
///   r = (a+b).sqrt() * ((a+1).exp())
///   helper_linearizer_opt(r, [
///     [Opt(OptOps.UPCAST, 0, 4)],
///   ])
/// ```
#[test]
fn test_full_upcast() {
    // Create a simple 1D pattern (size 4)
    let pattern = create_elementwise_pattern(&[4]);
    let renderer = Renderer::cpu();

    let mut sched = Scheduler::new(pattern, renderer);

    // Upcast the entire dimension (4 → 1×4)
    let result = apply_opt(&mut sched, &Opt::upcast(0, 4), true);
    assert!(result.is_ok(), "Full UPCAST should succeed: {:?}", result.err());

    // After filtering, should have 1 axis: Upcast(4) (Global(1) filtered out)
    assert_eq!(sched.shape_len(), 1, "Should have 1 axis after full upcast (Global(1) filtered)");
    assert_shape_equal(&sched, &[4]);

    // Check axis types
    assert_axes_equal(&sched, &[AxisType::Upcast]);

    // Verify upcast count (Global(1) filtered out by compute_rngs)
    assert_axis_count(&sched, AxisType::Upcast, 1);
    assert_axis_count(&sched, AxisType::Global, 0);
}

/// Port of Tinygrad test_kernel_opts.py::test_local_and_grouped_reduce (lines 11-35)
///
/// Original test creates a reduction pattern and validates LOCAL and GROUPTOP optimizations.
/// Tests single opts and combinations of LOCAL + GROUPTOP + UPCAST + UNROLL.
///
/// This validates that:
/// - LOCAL splits Global axes for GPU workgroup parallelism
/// - GROUPTOP splits Reduce axes for two-stage reduction
/// - Multiple LOCAL and GROUPTOP operations can be combined
/// - Complex optimization sequences work correctly
///
/// Original Tinygrad code:
/// ```python
/// def test_local_and_grouped_reduce(self):
///   N = 128
///   Tensor.manual_seed(1882)
///   a = Tensor.rand(4, 4, N, N)
///   b = Tensor.rand(4, 4, N)
///   r = (b.sqrt() + ((a+1).sum(axis=3).exp()))
///   helper_linearizer_opt(r, [
///     [Opt(OptOps.LOCAL, 0, 2)],
///     [Opt(OptOps.LOCAL, 0, 8)],
///     [Opt(OptOps.LOCAL, 0, 16)],
///     [Opt(OptOps.GROUPTOP, 0, 2)],
///     [Opt(OptOps.GROUPTOP, 0, 32)],
///     [Opt(OptOps.GROUPTOP, 0, 64)],
///     [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.GROUPTOP, 0, 2)],
///     [Opt(OptOps.LOCAL, 0, 16), Opt(OptOps.GROUPTOP, 0, 16)],
///     ...
///   ])
/// ```
#[test]
fn test_local_and_grouped_reduce() {
    // Create a reduction pattern matching Tinygrad's structure:
    // Output shape: [4, 4, 128], Reduce axis: 128
    // This mimics: result[4,4,128] = sum(data[4,4,128,128], axis=3)
    let pattern = create_reduce_with_globals(&[4, 4, 128], 128, ReduceOp::Add);
    let renderer = Renderer::cuda(); // GPU backend with local memory support

    // Test single LOCAL with amount 2
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::local(0, 2), true);
        assert!(result.is_ok(), "LOCAL by 2 should succeed: {:?}", result.err());

        // Should have split Global axis 0 (size=4) into (Global=2, Local=2)
        assert_axis_count(&sched, AxisType::Local, 1);
        // Still have 3 global axes (one split, two unchanged)
        assert_axis_count(&sched, AxisType::Global, 3);
    }

    // Test single LOCAL with amount 8 (on axis 2, size=128)
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        // LOCAL on axis 2 (size=128) splits it into (Global=16, Local=8)
        let result = apply_opt(&mut sched, &Opt::local(2, 8), true);
        assert!(result.is_ok(), "LOCAL(2, 8) should succeed: {:?}", result.err());
        assert_axis_count(&sched, AxisType::Local, 1);
        assert_axis_count(&sched, AxisType::Global, 3);
    }

    // Test single LOCAL with amount 16 (on axis 2, size=128)
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        // LOCAL on axis 2 (size=128) splits it into (Global=8, Local=16)
        let result = apply_opt(&mut sched, &Opt::local(2, 16), true);
        assert!(result.is_ok(), "LOCAL(2, 16) should succeed: {:?}", result.err());
        assert_axis_count(&sched, AxisType::Local, 1);
        assert_axis_count(&sched, AxisType::Global, 3);
    }

    // Test single GROUPTOP with amount 2
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        // GROUPTOP operates on reduce axes (logical index)
        let result = apply_opt(&mut sched, &Opt::grouptop(0, 2), true);
        assert!(result.is_ok(), "GROUPTOP by 2 should succeed: {:?}", result.err());

        // Should have split Reduce into (Reduce, GroupReduce)
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test single GROUPTOP with amount 32
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::grouptop(0, 32), true);
        assert!(result.is_ok(), "GROUPTOP by 32 should succeed");
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test single GROUPTOP with amount 64
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::grouptop(0, 64), true);
        assert!(result.is_ok(), "GROUPTOP by 64 should succeed");
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test combination: LOCAL + GROUPTOP
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());

        // Apply LOCAL first
        let result = apply_opt(&mut sched, &Opt::local(0, 2), true);
        assert!(result.is_ok(), "LOCAL should succeed");

        // Then apply GROUPTOP
        let result = apply_opt(&mut sched, &Opt::grouptop(0, 2), true);
        assert!(result.is_ok(), "GROUPTOP after LOCAL should succeed");

        // Should have both Local and GroupReduce axes
        assert_axis_count(&sched, AxisType::Local, 1);
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test combination: LOCAL(16) + GROUPTOP(16)
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());

        // LOCAL on axis 2 (size=128), divisible by 16
        apply_opt(&mut sched, &Opt::local(2, 16), true).unwrap();
        // GROUPTOP on reduce axis (logical index 0)
        apply_opt(&mut sched, &Opt::grouptop(0, 16), true).unwrap();

        assert_axis_count(&sched, AxisType::Local, 1);
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test complex combination: LOCAL + GROUPTOP + UPCAST + UNROLL
    {
        let mut sched = Scheduler::new(pattern, renderer);

        // Apply optimizations in sequence
        // LOCAL on axis 2 (size=128), split to (Global=64, Local=2)
        apply_opt(&mut sched, &Opt::local(2, 2), true).unwrap();
        // GROUPTOP on reduce axis (size=128), split to (GroupReduce=2, Reduce=64)
        apply_opt(&mut sched, &Opt::grouptop(0, 2), true).unwrap();

        // UPCAST on an axis with size ≥4
        // After LOCAL(2,2) and GROUPTOP(0,2), we have:
        // - Global axes: sizes [4, 4, 64] (from splitting axis 2)
        // - Local: size 2
        // - GroupReduce: size 2
        // - Reduce: size 64
        // We can upcast axis 0 (size=4) by 2 or 4
        apply_opt(&mut sched, &Opt::upcast(0, 2), true).unwrap();

        // UNROLL on reduce axis (logical index in unrollable dims)
        // This will convert GroupReduce → Unroll
        let unrollable = sched.unrollable_dims();
        if !unrollable.is_empty() {
            // Try to unroll by 2
            let _ = apply_opt(&mut sched, &Opt::unroll(0, 2), true);
        }

        // Verify we have multiple optimization types
        // Note: UNROLL converts GroupReduce → Unroll, so check for Unroll instead
        assert!(!sched.axes_of(&[AxisType::Local]).is_empty());
        assert!(!sched.axes_of(&[AxisType::Unroll]).is_empty());
        assert!(!sched.axes_of(&[AxisType::Upcast]).is_empty());
    }
}

/// Port of Tinygrad test_kernel_opts.py::test_double_reduce (lines 89-111)
///
/// Original test creates a 4D tensor (8, 128, 8, 128) and reduces over axes (1, 3),
/// resulting in shape (8, 8). Tests various combinations of GROUPTOP, LOCAL, UPCAST,
/// and UNROLL optimizations on double reduction patterns.
///
/// This validates that:
/// - GROUPTOP works on multiple reduction axes independently
/// - LOCAL and GROUPTOP can be combined on different axes
/// - Complex optimization sequences (LOCAL + GROUPTOP + UPCAST + UNROLL) work correctly
/// - Double reductions handle various optimization strategies
///
/// Original Tinygrad code:
/// ```python
/// def test_double_reduce(self):
///   N = 128
///   Tensor.manual_seed(1552)
///   a = Tensor.rand(8, N, 8, N)
///   r = a.sum(axis=(1,3))
///   helper_linearizer_opt(r, [
///     [Opt(OptOps.GROUPTOP, 0, 2)],
///     [Opt(OptOps.GROUPTOP, 0, 32)],
///     [Opt(OptOps.GROUPTOP, 1, 2)],
///     [Opt(OptOps.GROUPTOP, 1, 32)],
///     [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 2)],
///     [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2)],
///     [Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 64)],
///     [Opt(OptOps.GROUPTOP, 0, 16), Opt(OptOps.GROUPTOP, 1, 2), Opt(OptOps.UNROLL, 0, 4)],
///     [Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 2, 4)],
///     [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4)],
///     [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 2), Opt(OptOps.GROUPTOP, 1, 32), Opt(OptOps.UNROLL, 1, 4)],
///     [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2)],
///     [Opt(OptOps.LOCAL, 0, 2), Opt(OptOps.LOCAL, 1, 2), Opt(OptOps.GROUPTOP, 0, 8), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UNROLL, 0, 4), Opt(OptOps.UNROLL, 1, 4)],
///     [Opt(OptOps.LOCAL, 0, 4), Opt(OptOps.LOCAL, 1, 4), Opt(OptOps.GROUPTOP, 0, 4), Opt(OptOps.GROUPTOP, 1, 4), Opt(OptOps.UPCAST, 0, 2), Opt(OptOps.UPCAST, 0, 2)], # no globals
///   ])
/// ```
#[test]
fn test_double_reduce() {
    // Create pattern matching Tinygrad's structure:
    // Tensor(8, 128, 8, 128).sum(axis=(1,3)) -> Result(8, 8)
    // Global axes: [8, 8], Reduce axes: [128, 128]
    let pattern = create_double_reduce_with_globals(&[8, 8], &[128, 128], ReduceOp::Add);
    let renderer = Renderer::cuda(); // GPU backend with local/shared memory

    // Test 1: Single GROUPTOP on first reduce axis (logical index 0)
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::grouptop(0, 2), true);
        assert!(result.is_ok(), "GROUPTOP(0, 2) should succeed: {:?}", result.err());
        // Should split first Reduce axis (size 128) into (GroupReduce=2, Reduce=64)
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test 2: GROUPTOP on first reduce axis with larger factor
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::grouptop(0, 32), true);
        assert!(result.is_ok(), "GROUPTOP(0, 32) should succeed");
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test 3: GROUPTOP on second reduce axis (logical index 1)
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::grouptop(1, 2), true);
        assert!(result.is_ok(), "GROUPTOP(1, 2) should succeed");
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test 4: GROUPTOP on second reduce axis with larger factor
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        let result = apply_opt(&mut sched, &Opt::grouptop(1, 32), true);
        assert!(result.is_ok(), "GROUPTOP(1, 32) should succeed");
        assert_axis_count(&sched, AxisType::GroupReduce, 1);
    }

    // Test 5: GROUPTOP on both reduce axes
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::grouptop(0, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 2), true).unwrap();
        // Should have 2 GroupReduce axes
        assert_axis_count(&sched, AxisType::GroupReduce, 2);
    }

    // Test 6: GROUPTOP with asymmetric factors
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::grouptop(0, 16), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 2), true).unwrap();
        assert_axis_count(&sched, AxisType::GroupReduce, 2);
    }

    // Test 7: GROUPTOP with different asymmetric factors
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::grouptop(0, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 64), true).unwrap();
        assert_axis_count(&sched, AxisType::GroupReduce, 2);
    }

    // Test 8: GROUPTOP + UNROLL combination
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::grouptop(0, 16), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::unroll(0, 4), true).unwrap();

        // Verify that we have multiple optimization types
        // Note: UNROLL may convert GroupReduce→Unroll, so we check for presence not exact counts
        assert!(!sched.axes_of(&[AxisType::GroupReduce, AxisType::Reduce]).is_empty());
        assert!(!sched.axes_of(&[AxisType::Unroll]).is_empty());
    }

    // Test 9: GROUPTOP + UNROLL on different axis
    // Comment from Tinygrad: "Checking how it works with 2 grouped_reduces + upcasts."
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::grouptop(0, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 32), true).unwrap();
        apply_opt(&mut sched, &Opt::unroll(2, 4), true).unwrap();

        // Verify optimizations were applied successfully
        assert!(!sched.axes_of(&[AxisType::GroupReduce, AxisType::Reduce]).is_empty());
        assert!(!sched.axes_of(&[AxisType::Unroll]).is_empty());
    }

    // Test 10: LOCAL + GROUPTOP on both axes
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        // LOCAL on both Global axes (size 8 each)
        apply_opt(&mut sched, &Opt::local(0, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::local(1, 4), true).unwrap();
        // GROUPTOP on both Reduce axes
        apply_opt(&mut sched, &Opt::grouptop(0, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 4), true).unwrap();

        assert_axis_count(&sched, AxisType::Local, 2);
        assert_axis_count(&sched, AxisType::GroupReduce, 2);
    }

    // Test 11: Complex combination - LOCAL + GROUPTOP + UNROLL
    // Comment from Tinygrad: "Checking how it works with 2 grouped_reduces + upcasts + locals."
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::local(0, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::local(1, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(0, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 32), true).unwrap();
        apply_opt(&mut sched, &Opt::unroll(1, 4), true).unwrap();

        // Verify we have all expected optimization types
        assert_axis_count(&sched, AxisType::Local, 2);
        assert!(!sched.axes_of(&[AxisType::GroupReduce, AxisType::Reduce]).is_empty());
        assert!(!sched.axes_of(&[AxisType::Unroll]).is_empty());
    }

    // Test 12: LOCAL + GROUPTOP + UPCAST
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::local(0, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::local(1, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(0, 8), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 4), true).unwrap();
        // UPCAST on Global axis 0 (now size 4 after LOCAL split)
        apply_opt(&mut sched, &Opt::upcast(0, 2), true).unwrap();

        assert_axis_count(&sched, AxisType::Local, 2);
        assert_axis_count(&sched, AxisType::GroupReduce, 2);
        assert_axis_count(&sched, AxisType::Upcast, 1);
    }

    // Test 13: Complex combination - LOCAL + GROUPTOP + UPCAST + UNROLL (2x)
    // Comment from Tinygrad: "Checking how it works with 2 grouped_reduces + upcasts + locals."
    {
        let mut sched = Scheduler::new(pattern.clone(), renderer.clone());
        apply_opt(&mut sched, &Opt::local(0, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::local(1, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(0, 8), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::upcast(0, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::unroll(0, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::unroll(1, 4), true).unwrap();

        // Verify we have all expected optimization types
        assert_axis_count(&sched, AxisType::Local, 2);
        assert!(!sched.axes_of(&[AxisType::GroupReduce, AxisType::Reduce]).is_empty());
        assert_axis_count(&sched, AxisType::Upcast, 1);
        assert!(!sched.axes_of(&[AxisType::Unroll]).is_empty());
    }

    // Test 14: "no globals" - LOCAL + GROUPTOP + double UPCAST
    // Original Tinygrad comment: "# no globals"
    {
        let mut sched = Scheduler::new(pattern, renderer);
        apply_opt(&mut sched, &Opt::local(0, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::local(1, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(0, 4), true).unwrap();
        apply_opt(&mut sched, &Opt::grouptop(1, 4), true).unwrap();

        // Apply UPCAST(0, 2) twice to fully upcast both Global axes
        // Each UPCAST splits Global(2) → Global(1) + Upcast(2)
        // After compute_rngs() filtering, Global(1) axes are excluded from rngs()
        // This matches Tinygrad's behavior where vmax==0 ranges are filtered
        apply_opt(&mut sched, &Opt::upcast(0, 2), true).unwrap();
        apply_opt(&mut sched, &Opt::upcast(0, 2), true).unwrap();

        // Verify "no globals" - all Global axes filtered out (size-1 ranges excluded)
        assert_axis_count(&sched, AxisType::Global, 0);
        assert_axis_count(&sched, AxisType::Local, 2);
        assert_axis_count(&sched, AxisType::GroupReduce, 2);
        assert_axis_count(&sched, AxisType::Upcast, 2);
    }
}
