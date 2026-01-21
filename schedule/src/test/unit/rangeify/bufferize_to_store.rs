use std::sync::Arc;

use morok_ir::{Op, UOp};

use crate::rangeify::kernel::split_store;
use crate::rangeify::{KernelContext, bufferize_to_store};
#[allow(unused_imports)]
use crate::test::unit::rangeify::helpers::extract_kernel;

/// Helper to call split_store with the new signature
fn call_split_store(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    let mut uop_list = Vec::new();
    split_store(&mut uop_list, x)
}

#[test]
fn test_bufferize_to_store_global() {
    let mut ctx = KernelContext::new();

    // Create a simple BUFFERIZE with one range
    let compute = UOp::native_const(42.0f32);
    let range = UOp::range_const(10, 0);

    let bufferize = UOp::bufferize_global(compute.clone(), vec![range.clone()]);

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // bufferize_to_store returns AFTER(passthrough=BUFFER, deps=[END(STORE)])
    // Following Tinygrad's architecture: .store().end(*rngs)
    // BUFFER → DEFINE_GLOBAL conversion happens later in split_store
    let Op::After { passthrough, deps } = result.op() else {
        panic!("Expected AFTER operation, got {:?}", result.op());
    };

    // Passthrough should be BUFFER (not DEFINE_GLOBAL - that conversion happens in split_store)
    assert!(matches!(passthrough.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", passthrough.op());
    assert_eq!(deps.len(), 1);

    // Deps should contain END wrapping STORE
    let Op::End { computation, ranges: end_ranges } = deps[0].op() else {
        panic!("Expected END operation in deps, got {:?}", deps[0].op());
    };

    // END should have 1 range
    assert_eq!(end_ranges.len(), 1);
    assert!(std::sync::Arc::ptr_eq(&end_ranges[0], &range));

    // Unwrap END to get STORE
    let Op::Store { index, value, ranges } = computation.op() else {
        panic!("Expected STORE operation inside END, got {:?}", computation.op());
    };

    // STORE should have empty ranges (iteration space on END)
    assert!(ranges.is_empty());

    // Index should contain the buffer reference
    let Op::Index { buffer, .. } = index.op() else {
        panic!("Expected INDEX operation, got {:?}", index.op());
    };
    assert!(matches!(buffer.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", buffer.op());
    assert!(std::sync::Arc::ptr_eq(buffer, &passthrough));

    // Value should be the compute
    assert!(std::sync::Arc::ptr_eq(value, &compute));

    // Index should be INDEX operation with the range
    assert!(matches!(index.op(), Op::Index { .. }));

    // Verify context state - global_counter is NOT incremented for BUFFER ops
    // (it's only used for DEFINE_GLOBAL/DEFINE_LOCAL counters)
    assert_eq!(ctx.local_counter, 0);
    assert!(ctx.has_buffer(&bufferize));
}

#[test]
fn test_bufferize_to_store_local_with_barrier() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with LOCAL addrspace
    let compute = UOp::native_const(1.0f32);
    let range = UOp::range_const(5, 0);

    let bufferize = UOp::bufferize_local(compute.clone(), vec![range.clone()]);

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // bufferize_to_store returns AFTER(passthrough=DEFINE_LOCAL, deps=[BARRIER(END(STORE))])
    let Op::After { passthrough, deps } = result.op() else {
        panic!("Expected AFTER operation, got {:?}", result.op());
    };

    // Passthrough should be DEFINE_LOCAL
    assert!(matches!(passthrough.op(), Op::DefineLocal(0)));
    assert_eq!(deps.len(), 1);

    // Unwrap AFTER to get BARRIER
    let Op::Barrier { src, .. } = deps[0].op() else {
        panic!("Expected BARRIER operation in deps");
    };

    // BARRIER wraps END(STORE)
    let Op::End { computation, ranges: end_ranges } = src.op() else {
        panic!("Expected END operation inside BARRIER, got {:?}", src.op());
    };

    // END should have 1 range
    assert_eq!(end_ranges.len(), 1);
    assert!(std::sync::Arc::ptr_eq(&end_ranges[0], &range));

    // Unwrap END to get STORE
    let Op::Store { index, value, ranges } = computation.op() else {
        panic!("Expected STORE operation inside END, got {:?}", computation.op());
    };

    // STORE should have empty ranges
    assert!(ranges.is_empty());

    // Index should contain the buffer reference
    let Op::Index { buffer, .. } = index.op() else {
        panic!("Expected INDEX operation, got {:?}", index.op());
    };
    assert!(matches!(buffer.op(), Op::DefineLocal(0)));
    assert!(std::sync::Arc::ptr_eq(value, &compute));

    // Verify context state
    assert_eq!(ctx.global_counter, 0);
    assert_eq!(ctx.local_counter, 1);
    assert!(ctx.has_buffer(&bufferize));
}

#[test]
fn test_bufferize_to_store_multiple_ranges() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with multiple ranges
    let compute = UOp::native_const(100i32);
    let range1 = UOp::range_const(4, 0);
    let range2 = UOp::range_const(8, 1);

    let bufferize = UOp::bufferize_global(compute.clone(), vec![range1.clone(), range2.clone()]);

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // bufferize_to_store returns AFTER(passthrough=BUFFER, deps=[END(STORE)])
    // BUFFER → DEFINE_GLOBAL conversion happens later in split_store
    let Op::After { passthrough, deps } = result.op() else {
        panic!("Expected AFTER operation, got {:?}", result.op());
    };

    // Passthrough should be BUFFER (not DEFINE_GLOBAL)
    assert!(matches!(passthrough.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", passthrough.op());
    assert_eq!(deps.len(), 1);

    // Deps should contain END wrapping STORE
    let Op::End { computation, ranges: end_ranges } = deps[0].op() else {
        panic!("Expected END operation in deps, got {:?}", deps[0].op());
    };

    // END should have 2 ranges in the same order
    assert_eq!(end_ranges.len(), 2);
    assert!(std::sync::Arc::ptr_eq(&end_ranges[0], &range1));
    assert!(std::sync::Arc::ptr_eq(&end_ranges[1], &range2));

    // Unwrap END to get STORE
    let Op::Store { index, value, ranges } = computation.op() else {
        panic!("Expected STORE operation inside END, got {:?}", computation.op());
    };

    // STORE should have empty ranges
    assert!(ranges.is_empty());

    // Value should be the compute
    assert!(std::sync::Arc::ptr_eq(value, &compute));

    // Index should be INDEX with linearized index (2 ranges → 1 linear index)
    // For dims [4, 8], strides are [8, 1], so linear = range1 * 8 + range2
    let Op::Index { buffer: idx_buffer, indices, .. } = index.op() else {
        panic!("Expected INDEX operation");
    };

    // Buffer should be BUFFER (same as passthrough)
    assert!(matches!(idx_buffer.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", idx_buffer.op());
    assert!(std::sync::Arc::ptr_eq(idx_buffer, &passthrough));

    // Should have 1 linearized index (not 2 separate ranges)
    assert_eq!(indices.len(), 1, "Multi-index should be linearized to single index");

    // Verify context
    assert!(ctx.has_buffer(&bufferize));
}

#[test]
fn test_non_bufferize_returns_none() {
    let mut ctx = KernelContext::new();

    // Create a non-BUFFERIZE operation
    let const_op = UOp::native_const(1.0f32);

    // Should return None
    let result = bufferize_to_store(&const_op, &mut ctx);
    assert!(result.is_none());
}

#[test]
fn test_buffer_tracked_in_context() {
    let mut ctx = KernelContext::new();

    let compute = UOp::native_const(1.0f32);
    let bufferize = UOp::bufferize_global(compute, vec![]);

    // Before conversion, buffer should not be tracked
    assert!(!ctx.has_buffer(&bufferize));

    // Convert to STORE
    bufferize_to_store(&bufferize, &mut ctx);

    // After conversion, buffer should be tracked
    assert!(ctx.has_buffer(&bufferize));

    // Should be able to get the AFTER wrapping BUFFER
    // bufferize_to_store stores AFTER(buffer, [STORE]) in context
    let replacement = ctx.get_buffer(&bufferize).unwrap();

    // Unwrap AFTER to get the actual BUFFER (not DEFINE_GLOBAL)
    let Op::After { passthrough, .. } = replacement.op() else {
        panic!("Expected AFTER operation, got {:?}", replacement.op());
    };
    assert!(matches!(passthrough.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", passthrough.op());
}

#[test]
fn test_bufferize_to_store_sequential_global_ids() {
    let mut ctx = KernelContext::new();

    // Create three BUFFERIZE operations
    for i in 0..3 {
        let compute = UOp::native_const((i as f64) as f32);
        let bufferize = UOp::bufferize_global(compute, vec![]);

        let result = bufferize_to_store(&bufferize, &mut ctx);
        assert!(result.is_some());

        // For BUFFER ops, global_counter is NOT incremented (it's only for DEFINE_GLOBAL)
        // But each BUFFERIZE should be tracked
        assert!(ctx.has_buffer(&bufferize));
        assert_eq!(ctx.local_counter, 0);
    }
}

#[test]
fn test_bufferize_to_store_sequential_local_ids() {
    let mut ctx = KernelContext::new();

    // Create three BUFFERIZE operations with LOCAL addrspace
    for i in 0..3 {
        let compute = UOp::native_const((i as f64) as f32);
        let bufferize = UOp::bufferize_local(compute, vec![]);

        bufferize_to_store(&bufferize, &mut ctx);

        // Counter should increment
        assert_eq!(ctx.global_counter, 0);
        assert_eq!(ctx.local_counter, (i + 1) as usize);
    }
}

#[test]
fn test_bufferize_to_store_mixed_global_local() {
    let mut ctx = KernelContext::new();

    let global_compute = UOp::native_const(1.0f32);
    let local_compute = UOp::native_const(2.0f32);

    // Global buffer
    let global_bufferize = UOp::bufferize_global(global_compute.clone(), vec![]);

    // Local buffer
    let local_bufferize = UOp::bufferize_local(local_compute.clone(), vec![]);

    // Convert both
    let global_result = bufferize_to_store(&global_bufferize, &mut ctx);
    let local_result = bufferize_to_store(&local_bufferize, &mut ctx);

    // For global (BUFFER), global_counter is NOT incremented
    // For local (DEFINE_LOCAL), local_counter IS incremented
    assert_eq!(ctx.local_counter, 1);

    // Global: AFTER(passthrough=BUFFER, deps=[STORE])
    // No ranges means no END wrapper, but still has AFTER
    let global_result = global_result.unwrap();
    let Op::After { passthrough: global_buf, deps: global_deps } = global_result.op() else {
        panic!("Expected AFTER operation for global, got {:?}", global_result.op());
    };
    assert!(matches!(global_buf.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", global_buf.op());
    assert_eq!(global_deps.len(), 1);

    // deps[0] should be STORE (no ranges = no END)
    let Op::Store { index, value, .. } = global_deps[0].op() else {
        panic!("Expected STORE in global AFTER deps, got {:?}", global_deps[0].op());
    };
    // Index should contain the buffer reference
    let Op::Index { buffer, .. } = index.op() else {
        panic!("Expected INDEX operation, got {:?}", index.op());
    };
    assert!(std::sync::Arc::ptr_eq(buffer, global_buf));
    assert!(std::sync::Arc::ptr_eq(value, &global_compute));

    // Local: AFTER(passthrough=DEFINE_LOCAL, deps=[BARRIER(STORE)])
    // No ranges means no END wrapper, but still has AFTER and BARRIER
    let local_result = local_result.unwrap();
    let Op::After { passthrough: local_buf, deps: local_deps } = local_result.op() else {
        panic!("Expected AFTER operation for local, got {:?}", local_result.op());
    };
    assert!(matches!(local_buf.op(), Op::DefineLocal(0)));
    assert_eq!(local_deps.len(), 1);

    // deps[0] should be BARRIER wrapping STORE
    let Op::Barrier { src, .. } = local_deps[0].op() else {
        panic!("Expected BARRIER in local AFTER deps, got {:?}", local_deps[0].op());
    };
    let Op::Store { index, value, .. } = src.op() else {
        panic!("Expected STORE inside BARRIER, got {:?}", src.op());
    };
    // Index should contain the buffer reference
    let Op::Index { buffer, .. } = index.op() else {
        panic!("Expected INDEX operation, got {:?}", index.op());
    };
    assert!(std::sync::Arc::ptr_eq(buffer, local_buf));
    assert!(std::sync::Arc::ptr_eq(value, &local_compute));

    // Verify both are tracked in context
    assert!(ctx.has_buffer(&global_bufferize));
    assert!(ctx.has_buffer(&local_bufferize));
}

#[test]
fn test_bufferize_to_store_integration_with_split_kernel() {
    let mut ctx = KernelContext::new();

    // Create a BUFFERIZE operation with non-OUTER range
    // split_store skips END operations with OUTER ranges (control flow markers)
    // so we use range_const which creates a non-OUTER range
    let compute = UOp::native_const(42.0f32);
    let range = UOp::range_const(10, 0);

    let bufferize = UOp::bufferize_global(compute.clone(), vec![range]);

    // Stage 1: BUFFERIZE → AFTER(BUFFER, [END(STORE)])
    let store_result = bufferize_to_store(&bufferize, &mut ctx).unwrap();

    // Verify buffer was tracked
    assert!(ctx.has_buffer(&bufferize));

    // Extract structure: AFTER(passthrough=BUFFER, deps=[END(STORE)])
    let Op::After { passthrough: buffer_node, deps } = store_result.op() else {
        panic!("Expected AFTER operation, got {:?}", store_result.op());
    };
    assert!(matches!(buffer_node.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", buffer_node.op());
    assert_eq!(deps.len(), 1);

    // deps[0] should be END wrapping STORE
    let end_op = &deps[0];
    let Op::End { computation, ranges: end_ranges } = end_op.op() else {
        panic!("Expected END in AFTER deps, got {:?}", end_op.op());
    };
    assert_eq!(end_ranges.len(), 1);
    assert!(matches!(computation.op(), Op::Store { .. }), "Expected STORE inside END");

    // Stage 2: split_store transforms END(STORE) to KERNEL
    // The BUFFER node will be converted to DEFINE_GLOBAL inside the KERNEL
    let kernel = call_split_store(end_op).expect("split_store should create a KERNEL");

    // Verify KERNEL structure
    let Op::Kernel { sources, ast } = kernel.op() else {
        panic!("Expected KERNEL operation, got {:?}", kernel.op());
    };

    // KERNEL sources should contain the BUFFER (mapped to itself by local_to_define_global_patterns)
    assert!(!sources.is_empty(), "KERNEL should have at least one source");

    // AST should be SINK wrapping the transformed computation
    let Op::Sink { sources: sink_sources } = ast.op() else {
        panic!("Expected SINK operation in kernel AST, got {:?}", ast.op());
    };
    assert_eq!(sink_sources.len(), 1, "SINK should have 1 source");

    // Verify the context stores AFTER with BUFFER passthrough
    let ctx_buffer = ctx.get_buffer(&bufferize).unwrap();
    let Op::After { passthrough, .. } = ctx_buffer.op() else {
        panic!("Expected AFTER in context buffer mapping");
    };
    assert!(matches!(passthrough.op(), Op::Buffer { .. }), "Expected BUFFER, got {:?}", passthrough.op());
}
