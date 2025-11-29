use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, ConstValue, Op, UOp};
use smallvec::smallvec;

use crate::rangeify::{KernelContext, split_kernel::split_store};

#[test]
fn test_split_store_basic() {
    let mut ctx = KernelContext::new();

    // Create a simple STORE operation
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer: buffer.clone(), index, value }, DType::Void);

    // Try to split
    let result = split_store(&store, &mut ctx);

    // Should return a KERNEL
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));
}

#[test]
fn test_split_store_non_store_returns_none() {
    let mut ctx = KernelContext::new();

    // Create a non-STORE operation
    let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Try to split
    let result = split_store(&const_op, &mut ctx);

    // Should return None
    assert!(result.is_none());
}

#[test]
fn test_split_store_end_operation() {
    let mut ctx = KernelContext::new();

    // Create an END operation wrapping a STORE
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);
    let range = UOp::range_const(10, 0);
    let end = UOp::end(store.clone(), smallvec![range.clone()]);

    // Try to split
    let result = split_store(&end, &mut ctx);

    // Should process END wrapping STORE
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));

    // Verify KERNEL structure matches Tinygrad: KERNEL(..., ast=SINK(END(STORE, RANGE)))
    if let Op::Kernel { ast, .. } = kernel.op() {
        if let Op::Sink { sources } = ast.op() {
            // SINK should wrap the END (not extract STORE)
            assert_eq!(sources.len(), 1);
            assert!(std::rc::Rc::ptr_eq(&sources[0], &end));

            // Verify END structure is preserved
            if let Op::End { computation, ranges } = sources[0].op() {
                assert!(std::rc::Rc::ptr_eq(computation, &store));
                assert_eq!(ranges.len(), 1);
                assert!(std::rc::Rc::ptr_eq(&ranges[0], &range));
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
    let mut ctx = KernelContext::new();

    // Create an END operation wrapping non-STORE (control flow marker)
    let noop = UOp::noop();
    let range = UOp::range_const(10, 0);
    let end = UOp::end(noop, smallvec![range]);

    // Try to split
    let result = split_store(&end, &mut ctx);

    // Should return None (skip control flow markers)
    assert!(result.is_none());
}

#[test]
fn test_split_store_gated() {
    let mut ctx = KernelContext::new();

    // Create a STORE_GATED operation
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let store_gated = UOp::new(Op::StoreGated { buffer, index, value, gate }, DType::Void);

    // Try to split
    let result = split_store(&store_gated, &mut ctx);

    // Should return a KERNEL
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Kernel { .. }));
}

#[test]
fn test_split_store_creates_sink() {
    let mut ctx = KernelContext::new();

    // Create a STORE operation
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer: buffer.clone(), index, value: value.clone() }, DType::Void);

    let result = split_store(&store, &mut ctx).unwrap();

    // Extract the AST from the KERNEL
    if let Op::Kernel { sources, ast } = result.op() {
        // The AST should be a SINK operation
        if let Op::Sink { sources: sink_sources } = ast.op() {
            // SINK should wrap the original STORE
            assert_eq!(sink_sources.len(), 1);
            assert!(std::rc::Rc::ptr_eq(&sink_sources[0], &store));

            // Verify the STORE structure is preserved
            if let Op::Store { buffer: store_buf, value: store_val, .. } = sink_sources[0].op() {
                assert!(std::rc::Rc::ptr_eq(store_buf, &buffer));
                assert!(std::rc::Rc::ptr_eq(store_val, &value));
            } else {
                panic!("Expected STORE in SINK sources");
            }
        } else {
            panic!("Expected SINK operation");
        }

        // Sources should be empty since no buffers were tracked in context
        assert!(sources.is_empty());
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_preserves_computation() {
    let mut ctx = KernelContext::new();

    // Create STOREs with different value dtypes
    let test_cases = [
        (DType::Float32, ConstValue::Float(1.0)),
        (DType::Int32, ConstValue::Int(1)),
        (DType::Bool, ConstValue::Bool(true)),
    ];

    for (dtype, const_val) in test_cases {
        let buffer = UOp::unique(Some(0));
        let index = UOp::const_(DType::Index, ConstValue::Int(0));
        let value = UOp::const_(dtype.clone(), const_val);
        let store = UOp::new(Op::Store { buffer, index, value: value.clone() }, DType::Void);

        let result = split_store(&store, &mut ctx);

        assert!(result.is_some());
        let kernel = result.unwrap();

        // Verify KERNEL structure
        if let Op::Kernel { ast, .. } = kernel.op()
            && let Op::Sink { sources } = ast.op()
        {
            // SINK should wrap the STORE
            assert!(std::rc::Rc::ptr_eq(&sources[0], &store));

            // Verify the stored value dtype is preserved
            if let Op::Store { value: stored_val, .. } = sources[0].op() {
                assert_eq!(stored_val.dtype(), dtype);
                assert!(std::rc::Rc::ptr_eq(stored_val, &value));
            }
        }
    }
}

#[test]
fn test_split_store_multiple_calls_independent() {
    let mut ctx = KernelContext::new();

    // Create two different STORE operations
    let buffer1 = UOp::unique(Some(1));
    let index1 = UOp::const_(DType::Index, ConstValue::Int(0));
    let value1 = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store1 = UOp::new(Op::Store { buffer: buffer1, index: index1, value: value1 }, DType::Void);

    let buffer2 = UOp::unique(Some(2));
    let index2 = UOp::const_(DType::Index, ConstValue::Int(0));
    let value2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let store2 = UOp::new(Op::Store { buffer: buffer2, index: index2, value: value2 }, DType::Void);

    // Split both
    let kernel1 = split_store(&store1, &mut ctx).unwrap();
    let kernel2 = split_store(&store2, &mut ctx).unwrap();

    // Both should be valid kernels
    assert!(matches!(kernel1.op(), Op::Kernel { .. }));
    assert!(matches!(kernel2.op(), Op::Kernel { .. }));

    // They should be different kernels (different UOps)
    assert!(!std::rc::Rc::ptr_eq(&kernel1, &kernel2));
}

#[test]
fn test_split_store_end_with_multiple_ranges() {
    let mut ctx = KernelContext::new();

    // Create END with multiple ranges wrapping a STORE
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);
    let range1 = UOp::range_const(4, 0);
    let range2 = UOp::range_const(8, 1);
    let end = UOp::end(store.clone(), smallvec![range1.clone(), range2.clone()]);

    let result = split_store(&end, &mut ctx);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify KERNEL wraps the END (not extracted STORE) - matches Tinygrad behavior
    if let Op::Kernel { ast, .. } = kernel.op() {
        if let Op::Sink { sources } = ast.op() {
            // SINK should wrap the END (preserving full structure)
            assert_eq!(sources.len(), 1);
            assert!(std::rc::Rc::ptr_eq(&sources[0], &end));

            // Verify END structure with multiple ranges is preserved
            if let Op::End { computation, ranges } = sources[0].op() {
                assert!(std::rc::Rc::ptr_eq(computation, &store));
                assert_eq!(ranges.len(), 2);
                assert!(std::rc::Rc::ptr_eq(&ranges[0], &range1));
                assert!(std::rc::Rc::ptr_eq(&ranges[1], &range2));
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
    let mut ctx = KernelContext::new();

    // Create END with OUTER range wrapping a STORE
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);
    let range_outer =
        UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), AxisId::Renumbered(0), AxisType::Outer);
    let end = UOp::end(store, smallvec![range_outer]);

    let result = split_store(&end, &mut ctx);

    // Should skip END with OUTER ranges (control flow marker)
    // Tinygrad line 485: if x.op is Ops.END and x.src[1].arg[-1] == AxisType.OUTER: return None
    assert!(result.is_none());
}

#[test]
fn test_split_store_end_with_mixed_ranges() {
    let mut ctx = KernelContext::new();

    // Create END with mix of LOOP and OUTER ranges wrapping a STORE
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { buffer, index, value }, DType::Void);
    let range_loop = UOp::range_const(4, 0);
    let range_outer =
        UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(8)), AxisId::Renumbered(1), AxisType::Outer);
    let end = UOp::end(store, smallvec![range_loop, range_outer]);

    let result = split_store(&end, &mut ctx);

    // Should skip if ANY range is OUTER (our implementation checks all ranges)
    assert!(result.is_none());
}

// ============================================================================
// COPY/BUFFER_VIEW Support Tests
// ============================================================================

#[test]
fn test_split_store_with_copy() {
    use morok_ir::DeviceSpec;

    let mut ctx = KernelContext::new();

    // Create a COPY operation
    let src_buffer = UOp::unique(Some(1));
    let copy = src_buffer.copy_to_device(DeviceSpec::Cpu);

    // Create STORE using the COPY result
    let output_buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let store = UOp::new(Op::Store { buffer: output_buffer, index, value: copy.clone() }, DType::Void);

    let result = split_store(&store, &mut ctx);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is COPY, not SINK
    if let Op::Kernel { ast, .. } = kernel.op() {
        assert!(matches!(ast.op(), Op::Copy { .. }), "Expected COPY operation as kernel AST, got: {:?}", ast.op());

        // Verify it's the same COPY we created
        assert!(std::rc::Rc::ptr_eq(ast, &copy));
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_with_buffer_view() {
    let mut ctx = KernelContext::new();

    // Create a BUFFER_VIEW operation
    let base_buffer = UOp::unique(Some(1));
    let buffer_view = UOp::buffer_view(base_buffer, 256, 128);

    // Create STORE using the BUFFER_VIEW result
    let output_buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let store = UOp::new(Op::Store { buffer: output_buffer, index, value: buffer_view.clone() }, DType::Void);

    let result = split_store(&store, &mut ctx);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is BUFFER_VIEW, not SINK
    if let Op::Kernel { ast, .. } = kernel.op() {
        if let Op::BufferView { size, offset, .. } = ast.op() {
            assert_eq!(*size, 256);
            assert_eq!(*offset, 128);
            // Verify it's the same BUFFER_VIEW we created
            assert!(std::rc::Rc::ptr_eq(ast, &buffer_view));
        } else {
            panic!("Expected BUFFER_VIEW operation as kernel AST, got: {:?}", ast.op());
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_normal_computation_uses_sink() {
    let mut ctx = KernelContext::new();

    // Create normal arithmetic computation (no COPY/BUFFER_VIEW)
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let value = a.try_add_op(&b).unwrap();

    // Create STORE with normal computation
    let buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let store = UOp::new(Op::Store { buffer: buffer.clone(), index, value: value.clone() }, DType::Void);

    let result = split_store(&store, &mut ctx);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is SINK (normal case)
    if let Op::Kernel { ast, .. } = kernel.op() {
        if let Op::Sink { sources } = ast.op() {
            // SINK should wrap the STORE
            assert_eq!(sources.len(), 1);
            assert!(std::rc::Rc::ptr_eq(&sources[0], &store));
        } else {
            panic!("Expected SINK operation for normal computation, got: {:?}", ast.op());
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_nested_copy_in_store() {
    use morok_ir::DeviceSpec;

    let mut ctx = KernelContext::new();

    // Create nested structure: END(STORE(COPY))
    let src_buffer = UOp::unique(Some(1));
    let copy = src_buffer.copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    let output_buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let store = UOp::new(Op::Store { buffer: output_buffer, index, value: copy.clone() }, DType::Void);

    let range = UOp::range_const(10, 0);
    let end = UOp::end(store, smallvec![range]);

    let result = split_store(&end, &mut ctx);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is COPY (found via toposort in END→STORE→COPY)
    if let Op::Kernel { ast, .. } = kernel.op() {
        assert!(
            matches!(ast.op(), Op::Copy { .. }),
            "Expected COPY operation as kernel AST even when nested, got: {:?}",
            ast.op()
        );

        // Verify it's the same COPY we created
        assert!(std::rc::Rc::ptr_eq(ast, &copy));
    } else {
        panic!("Expected KERNEL operation");
    }
}

#[test]
fn test_split_store_copy_precedence_documented() {
    use morok_ir::DeviceSpec;

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

    let mut ctx = KernelContext::new();

    // Create nested COPY: COPY(COPY(buffer))
    let base_buffer = UOp::unique(Some(1));
    let copy1 = base_buffer.copy_to_device(DeviceSpec::Cpu);
    let copy2 = copy1.clone().copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    let output_buffer = UOp::unique(Some(0));
    let index = UOp::const_(DType::Index, ConstValue::Int(0));
    let store = UOp::new(Op::Store { buffer: output_buffer, index, value: copy2.clone() }, DType::Void);

    let result = split_store(&store, &mut ctx);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is one of the COPY operations (toposort order determines which)
    if let Op::Kernel { ast, .. } = kernel.op() {
        assert!(matches!(ast.op(), Op::Copy { .. }), "Expected COPY operation as kernel AST, got: {:?}", ast.op());

        // Should be either copy1 or copy2 (both are valid COPY operations)
        let is_copy1 = std::rc::Rc::ptr_eq(ast, &copy1);
        let is_copy2 = std::rc::Rc::ptr_eq(ast, &copy2);

        assert!(is_copy1 || is_copy2, "Kernel AST should be one of the COPY operations in the graph");
    } else {
        panic!("Expected KERNEL operation");
    }
}
