use morok_ir::{Op, UOp};

use crate::rangeify::{KernelContext, bufferize_to_store};
use crate::test::unit::rangeify::helpers::extract_kernel;

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

    // bufferize_to_store returns AFTER(passthrough=DEFINE_GLOBAL, deps=[END(STORE)])
    // Following Tinygrad's pattern: return buf.after(do_store)
    let Op::After { passthrough, deps } = result.op() else {
        panic!("Expected AFTER operation, got {:?}", result.op());
    };

    // Passthrough should be DEFINE_GLOBAL
    assert!(matches!(passthrough.op(), Op::DefineGlobal(0)));
    assert_eq!(deps.len(), 1);

    // Unwrap AFTER to get END
    let Op::End { computation, ranges } = deps[0].op() else {
        panic!("Expected END operation in deps");
    };

    // Should have 1 range
    assert_eq!(ranges.len(), 1);
    assert!(std::sync::Arc::ptr_eq(&ranges[0], &range));

    // Computation should be STORE
    let Op::Store { buffer, index, value } = computation.op() else {
        panic!("Expected STORE operation");
    };

    // Buffer should be DEFINE_GLOBAL with ID 0
    assert!(matches!(buffer.op(), Op::DefineGlobal(0)));

    // Value should be the compute
    assert!(std::sync::Arc::ptr_eq(value, &compute));

    // Index should be INDEX operation with the range
    assert!(matches!(index.op(), Op::Index { .. }));

    // Verify context state
    assert_eq!(ctx.global_counter, 1);
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

    // Unwrap BARRIER to check the END inside
    let Op::End { computation, ranges } = src.op() else {
        panic!("Expected END operation inside BARRIER");
    };

    // Should have 1 range
    assert_eq!(ranges.len(), 1);
    assert!(std::sync::Arc::ptr_eq(&ranges[0], &range));

    // Computation should be STORE with DEFINE_LOCAL(0)
    let Op::Store { buffer, value, .. } = computation.op() else {
        panic!("Expected STORE operation");
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

    // bufferize_to_store returns AFTER(passthrough=DEFINE_GLOBAL, deps=[END(STORE)])
    let Op::After { passthrough, deps } = result.op() else {
        panic!("Expected AFTER operation, got {:?}", result.op());
    };

    // Passthrough should be DEFINE_GLOBAL
    assert!(matches!(passthrough.op(), Op::DefineGlobal(0)));
    assert_eq!(deps.len(), 1);

    // Unwrap AFTER to get END
    let Op::End { computation, ranges } = deps[0].op() else {
        panic!("Expected END operation in deps");
    };

    // Should have 2 ranges in the same order
    assert_eq!(ranges.len(), 2);
    assert!(std::sync::Arc::ptr_eq(&ranges[0], &range1));
    assert!(std::sync::Arc::ptr_eq(&ranges[1], &range2));

    // Verify STORE structure
    let Op::Store { buffer, index, value } = computation.op() else {
        panic!("Expected STORE operation");
    };

    // Buffer should be DEFINE_GLOBAL(0)
    assert!(matches!(buffer.op(), Op::DefineGlobal(0)));

    // Value should be the compute
    assert!(std::sync::Arc::ptr_eq(value, &compute));

    // Index should be INDEX with both ranges
    let Op::Index { buffer: idx_buffer, indices, .. } = index.op() else {
        panic!("Expected INDEX operation");
    };

    // Buffer should match the STORE buffer
    assert!(std::sync::Arc::ptr_eq(idx_buffer, buffer));

    // Should have 2 indices
    assert_eq!(indices.len(), 2);

    // Verify context
    assert_eq!(ctx.global_counter, 1);
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

    // Should be able to get the AFTER wrapping DEFINE_GLOBAL
    // bufferize_to_store stores AFTER(buffer, [STORE]) in context
    let replacement = ctx.get_buffer(&bufferize).unwrap();

    // Unwrap AFTER to get the actual DEFINE_GLOBAL
    let Op::After { passthrough, .. } = replacement.op() else {
        panic!("Expected AFTER operation, got {:?}", replacement.op());
    };
    assert!(matches!(passthrough.op(), Op::DefineGlobal(_)));
}

#[test]
fn test_bufferize_to_store_sequential_global_ids() {
    let mut ctx = KernelContext::new();

    // Create three BUFFERIZE operations
    for i in 0..3 {
        let compute = UOp::native_const((i as f64) as f32);
        let bufferize = UOp::bufferize_global(compute, vec![]);

        bufferize_to_store(&bufferize, &mut ctx);

        // Counter should increment
        assert_eq!(ctx.global_counter, (i + 1) as usize);
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

    // Verify independent counters
    assert_eq!(ctx.global_counter, 1);
    assert_eq!(ctx.local_counter, 1);

    // Global: AFTER(passthrough=DEFINE_GLOBAL, deps=[STORE])
    // No ranges means no END wrapper, but still has AFTER
    let global_result = global_result.unwrap();
    let Op::After { passthrough: global_buf, deps: global_deps } = global_result.op() else {
        panic!("Expected AFTER operation for global, got {:?}", global_result.op());
    };
    assert!(matches!(global_buf.op(), Op::DefineGlobal(0)));
    assert_eq!(global_deps.len(), 1);

    // deps[0] should be STORE (no ranges = no END)
    let Op::Store { buffer, value, .. } = global_deps[0].op() else {
        panic!("Expected STORE in global AFTER deps, got {:?}", global_deps[0].op());
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
    let Op::Store { buffer, value, .. } = src.op() else {
        panic!("Expected STORE inside BARRIER, got {:?}", src.op());
    };
    assert!(std::sync::Arc::ptr_eq(buffer, local_buf));
    assert!(std::sync::Arc::ptr_eq(value, &local_compute));

    // Verify both are tracked in context
    assert!(ctx.has_buffer(&global_bufferize));
    assert!(ctx.has_buffer(&local_bufferize));
}

#[test]
fn test_bufferize_to_store_integration_with_split_kernel() {
    use crate::rangeify::kernel::split_store;

    let mut ctx = KernelContext::new();

    // Create a BUFFERIZE operation with non-OUTER range
    // split_store skips END operations with OUTER ranges (control flow markers)
    // so we use range_const which creates a non-OUTER range
    let compute = UOp::native_const(42.0f32);
    let range = UOp::range_const(10, 0);

    let bufferize = UOp::bufferize_global(compute.clone(), vec![range]);

    // Stage 1: BUFFERIZE → AFTER(DEFINE_GLOBAL, [END(STORE)])
    let store_result = bufferize_to_store(&bufferize, &mut ctx).unwrap();

    // Verify buffer was tracked
    assert_eq!(ctx.global_counter, 1);
    assert!(ctx.has_buffer(&bufferize));

    // Extract structure: AFTER(passthrough=DEFINE_GLOBAL, deps=[END(STORE)])
    let Op::After { passthrough: define_global, deps } = store_result.op() else {
        panic!("Expected AFTER operation, got {:?}", store_result.op());
    };
    assert!(matches!(define_global.op(), Op::DefineGlobal(0)));
    assert_eq!(deps.len(), 1);

    // deps[0] should be END wrapping STORE
    let Op::End { computation: store, .. } = deps[0].op() else {
        panic!("Expected END in AFTER deps, got {:?}", deps[0].op());
    };
    assert!(matches!(store.op(), Op::Store { .. }));

    // Stage 2: AFTER → AFTER(DEFINE_GLOBAL, [END(KERNEL)])
    // split_store transforms END(STORE) to END(KERNEL) but preserves AFTER wrapper
    let split_result = split_store(&store_result, &mut ctx).unwrap();

    // Extract KERNEL from result (wrapped in AFTER structure)
    let kernel = extract_kernel(&split_result).expect("split_store should create a KERNEL");

    // Verify KERNEL structure
    let Op::Kernel { sources, ast } = kernel.op() else {
        panic!("Expected KERNEL operation, got {:?}", kernel.op());
    };

    // KERNEL should have at least 1 source (the DEFINE_GLOBAL buffer)
    assert!(!sources.is_empty(), "KERNEL should have at least one source");
    assert!(
        sources.iter().any(|s| matches!(s.op(), Op::DefineGlobal(0))),
        "KERNEL sources should include DEFINE_GLOBAL(0)"
    );

    // AST should be SINK wrapping the computation
    let Op::Sink { sources: sink_sources } = ast.op() else {
        panic!("Expected SINK operation in kernel AST, got {:?}", ast.op());
    };
    assert_eq!(sink_sources.len(), 1, "SINK should have 1 source");

    // Verify the context stores AFTER (not bare DEFINE_GLOBAL)
    let ctx_buffer = ctx.get_buffer(&bufferize).unwrap();
    let Op::After { passthrough, .. } = ctx_buffer.op() else {
        panic!("Expected AFTER in context buffer mapping");
    };
    assert!(matches!(passthrough.op(), Op::DefineGlobal(0)));
}
