use std::sync::Arc;

use morok_dtype::{DType, DeviceSpec};
use morok_ir::{AxisId, AxisType, ConstValue, Op, UOp};
use smallvec::smallvec;

use crate::rangeify::kernel::split_store;

/// Helper to call split_store with the new signature
fn call_split_store(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    let mut uop_list = Vec::new();
    split_store(&mut uop_list, x)
}

#[test]
fn test_split_store_basic() {
    // Create a simple STORE operation with a proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, value);

    // Try to split
    let result = call_split_store(&store);

    // Should return a KERNEL
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));

    // BUFFER should be tracked in kernel sources
    if let Op::Kernel { sources, .. } = kernel.op() {
        // With proper BUFFER ops, sources should contain the buffer
        assert!(!sources.is_empty(), "Kernel sources should contain the buffer");
    }
}

#[test]
fn test_split_store_non_store_returns_none() {
    // Create a non-STORE operation
    let const_op = UOp::native_const(1.0f32);

    // Try to split
    let result = call_split_store(&const_op);

    // Should return None
    assert!(result.is_none());
}

#[test]
fn test_split_store_end_operation() {
    // Create an END operation wrapping a STORE with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, value);
    let range = UOp::range_const(10, 0);
    let end = UOp::end(store.clone(), smallvec![range.clone()]);

    // Try to split
    let result = call_split_store(&end);

    // Should process END wrapping STORE
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));

    // Verify KERNEL structure: ast should be SINK wrapping transformed END
    if let Op::Kernel { sources, ast } = kernel.op() {
        // With proper BUFFER, sources should contain the buffer mapping
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");

        if let Op::Sink { sources: sink_sources } = ast.op() {
            // SINK should wrap the transformed computation
            assert_eq!(sink_sources.len(), 1);
            // Note: The END may be transformed, so we just check it's an END with ranges
            if let Op::End { ranges, .. } = sink_sources[0].op() {
                assert_eq!(ranges.len(), 1);
            } else {
                panic!("Expected END operation in SINK");
            }
        } else {
            panic!("Expected SINK operation");
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_end_non_store_returns_none() {
    // Create an END operation wrapping non-STORE (control flow marker)
    let noop = UOp::noop();
    let range = UOp::range_const(10, 0);
    let end = UOp::end(noop, smallvec![range]);

    // Try to split
    let result = call_split_store(&end);

    // Should return None (skip control flow markers)
    assert!(result.is_none());
}

#[test]
fn test_split_store_creates_sink() {
    // Create a STORE operation with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, value.clone());

    let result = call_split_store(&store).unwrap();

    // Extract the AST from the KERNEL
    if let Op::Kernel { sources, ast } = result.op() {
        // The AST should be a SINK operation
        if let Op::Sink { sources: sink_sources } = ast.op() {
            // SINK should wrap the transformed STORE
            assert_eq!(sink_sources.len(), 1);

            // Verify the STORE structure has DEFINE_GLOBAL (buffer converted)
            if let Op::Store { index: store_index, value: store_val, .. } = sink_sources[0].op() {
                // Index should contain the buffer reference
                let Op::Index { buffer: store_buf, .. } = store_index.op() else {
                    panic!("Expected INDEX operation in STORE, got {:?}", store_index.op());
                };
                // Buffer should be converted to DEFINE_GLOBAL
                assert!(
                    matches!(store_buf.op(), Op::DefineGlobal(_)),
                    "Expected DEFINE_GLOBAL, got {:?}",
                    store_buf.op()
                );
                // Value should be preserved
                assert!(std::sync::Arc::ptr_eq(store_val, &value));
            } else {
                panic!("Expected STORE in SINK sources");
            }
        } else {
            panic!("Expected SINK operation");
        }

        // With proper BUFFER ops, sources should contain the buffer mapping
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_preserves_computation() {
    // Create STOREs with different value dtypes
    let test_cases = [
        (DType::Float32, ConstValue::Float(1.0)),
        (DType::Int32, ConstValue::Int(1)),
        (DType::Bool, ConstValue::Bool(true)),
    ];

    for (_dtype_idx, (dtype, _const_val)) in test_cases.iter().enumerate() {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, dtype.clone());
        let const_idx = UOp::index_const(0);
        let value = match _dtype_idx {
            0 => UOp::native_const(1.0f32),
            1 => UOp::native_const(1i32),
            2 => UOp::native_const(true),
            _ => panic!("Unsupported dtype index"),
        };
        let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
        let store = UOp::store(store_idx, value.clone());

        let result = call_split_store(&store);

        assert!(result.is_some());
        let kernel = result.unwrap();

        // Verify KERNEL structure
        if let Op::Kernel { ast, .. } = kernel.op()
            && let Op::Sink { sources } = ast.op()
        {
            // Verify the stored value dtype is preserved
            if let Op::Store { value: stored_val, .. } = sources[0].op() {
                assert_eq!(stored_val.dtype(), *dtype);
                assert!(std::sync::Arc::ptr_eq(stored_val, &value));
            }
        }
    }
}

#[test]
fn test_split_store_multiple_calls_independent() {
    // Create two different STORE operations with proper BUFFER ops
    let buffer1 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let idx_offset1 = UOp::index_const(0);
    let value1 = UOp::native_const(1.0f32);
    let index1 = UOp::index().buffer(buffer1).indices(vec![idx_offset1]).call().unwrap();
    let store1 = UOp::store(index1, value1);

    let buffer2 = UOp::new_buffer(DeviceSpec::Cpu, 200, DType::Float32);
    let idx_offset2 = UOp::index_const(0);
    let value2 = UOp::native_const(2.0f32);
    let index2 = UOp::index().buffer(buffer2).indices(vec![idx_offset2]).call().unwrap();
    let store2 = UOp::store(index2, value2);

    // Split both
    let kernel1 = call_split_store(&store1).unwrap();
    let kernel2 = call_split_store(&store2).unwrap();

    // Both should be valid kernels
    assert!(matches!(kernel1.op(), Op::Kernel { .. }));
    assert!(matches!(kernel2.op(), Op::Kernel { .. }));

    // They should be different kernels (different UOps)
    assert!(!std::sync::Arc::ptr_eq(&kernel1, &kernel2));
}

#[test]
fn test_split_store_end_with_multiple_ranges() {
    // Create END with multiple ranges wrapping a STORE with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, value);
    let range1 = UOp::range_const(4, 0);
    let range2 = UOp::range_const(8, 1);
    let end = UOp::end(store.clone(), smallvec![range1.clone(), range2.clone()]);

    let result = call_split_store(&end);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify KERNEL wraps the END (transformed) - matches Tinygrad behavior
    if let Op::Kernel { sources, ast } = kernel.op() {
        // With proper BUFFER, sources should contain buffer mappings
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");

        if let Op::Sink { sources: sink_sources } = ast.op() {
            // SINK should wrap the END (possibly transformed)
            assert_eq!(sink_sources.len(), 1);

            // Verify END structure with multiple ranges is preserved
            if let Op::End { ranges, .. } = sink_sources[0].op() {
                assert_eq!(ranges.len(), 2);
            } else {
                panic!("Expected END operation in SINK");
            }
        } else {
            panic!("Expected SINK operation");
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_end_with_outer_range() {
    // Create END with OUTER range wrapping a STORE with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, value);
    let range_outer = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Outer);
    let end = UOp::end(store, smallvec![range_outer]);

    let result = call_split_store(&end);

    // Should skip END with OUTER ranges (control flow marker)
    // Tinygrad line 485: if x.op is Ops.END and x.src[1].arg[-1] == AxisType.OUTER: return None
    assert!(result.is_none());
}

#[test]
fn test_split_store_end_with_mixed_ranges() {
    // Create END with mix of LOOP and OUTER ranges wrapping a STORE with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, value);
    let range_loop = UOp::range_const(4, 0);
    let range_outer = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(1), AxisType::Outer);
    let end = UOp::end(store, smallvec![range_loop, range_outer]);

    let result = call_split_store(&end);

    // Should skip if ANY range is OUTER (our implementation checks all ranges)
    assert!(result.is_none());
}

// ============================================================================
// COPY/BUFFER_VIEW Support Tests
// ============================================================================

#[test]
fn test_split_store_with_copy() {
    // Create a COPY operation with proper BUFFER
    let src_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let copy = src_buffer.copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    // Create STORE using the COPY result with proper BUFFER
    let output_buffer = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(output_buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, copy.clone());

    let result = call_split_store(&store);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is COPY, not SINK
    if let Op::Kernel { ast, .. } = kernel.op() {
        assert!(matches!(ast.op(), Op::Copy { .. }), "Expected COPY operation as kernel AST, got: {:?}", ast.op());
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_with_buffer_view() {
    // Create a BUFFER_VIEW operation with proper BUFFER
    let base_buffer = UOp::new_buffer(DeviceSpec::Cpu, 512, DType::Float32);
    let buffer_view = UOp::buffer_view(base_buffer, 256, 128);

    // Create STORE using the BUFFER_VIEW result with proper BUFFER
    let output_buffer = UOp::new_buffer(DeviceSpec::Cpu, 256, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(output_buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, buffer_view.clone());

    let result = call_split_store(&store);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is BUFFER_VIEW, not SINK
    if let Op::Kernel { ast, .. } = kernel.op() {
        if let Op::BufferView { size, offset, .. } = ast.op() {
            assert_eq!(*size, 256);
            assert_eq!(*offset, 128);
        } else {
            panic!("Expected BUFFER_VIEW operation as kernel AST, got: {:?}", ast.op());
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_normal_computation_uses_sink() {
    // Create normal arithmetic computation (no COPY/BUFFER_VIEW)
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let value = a.try_add(&b).unwrap();

    // Create STORE with normal computation using proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, value.clone());

    let result = call_split_store(&store);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is SINK (normal case)
    if let Op::Kernel { sources, ast } = kernel.op() {
        // With proper BUFFER, sources should contain buffer mappings
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");

        if let Op::Sink { sources: sink_sources } = ast.op() {
            // SINK should wrap the transformed STORE
            assert_eq!(sink_sources.len(), 1);
        } else {
            panic!("Expected SINK operation for normal computation, got: {:?}", ast.op());
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_nested_copy_in_store() {
    // Create nested structure: END(STORE(COPY)) with proper BUFFER
    let src_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let copy = src_buffer.copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    let output_buffer = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(output_buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, copy.clone());

    let range = UOp::range_const(10, 0);
    let end = UOp::end(store, smallvec![range]);

    let result = call_split_store(&end);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is COPY (found via toposort in END→STORE→COPY)
    if let Op::Kernel { ast, .. } = kernel.op() {
        assert!(
            matches!(ast.op(), Op::Copy { .. }),
            "Expected COPY operation as kernel AST even when nested, got: {:?}",
            ast.op()
        );
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_copy_precedence_documented() {
    // This test documents the precedence behavior when multiple COPY/BUFFER_VIEW
    // operations exist in a computation graph:
    //
    // **Behavior:** The first COPY or BUFFER_VIEW found during toposort() traversal
    // becomes the kernel AST. This is deterministic based on graph structure.
    //
    // **Rationale:** Scheduler needs direct AST access to detect cross-device
    // transfers (COPY) or extract view parameters (BUFFER_VIEW). Using the first
    // special op found ensures we don't miss important operations.
    //
    // For this test, we create a structure with nested COPY operations to verify
    // that find_copy_or_buffer_view() correctly identifies them.

    // Create nested COPY: COPY(COPY(buffer)) with proper BUFFER
    let base_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let copy1 = base_buffer.copy_to_device(DeviceSpec::Cuda { device_id: 0 });
    let copy2 = copy1.clone().copy_to_device(DeviceSpec::Cpu);

    let output_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(output_buffer).indices(vec![const_idx]).call().unwrap();
    let store = UOp::store(store_idx, copy2.clone());

    let result = call_split_store(&store);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is one of the COPY operations (toposort order determines which)
    if let Op::Kernel { ast, .. } = kernel.op() {
        assert!(matches!(ast.op(), Op::Copy { .. }), "Expected COPY operation as kernel AST, got: {:?}", ast.op());
    } else {
        panic!("Expected KERNEL operation");
    }
}
