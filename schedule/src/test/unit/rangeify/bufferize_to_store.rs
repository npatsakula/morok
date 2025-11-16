use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, Op, UOp};
use smallvec::SmallVec;

use crate::rangeify::{KernelContext, bufferize_to_store::bufferize_to_store};

#[test]
fn test_bufferize_to_store_global() {
    let mut ctx = KernelContext::new();

    // Create a simple BUFFERIZE with one range
    let compute = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let range = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(10)), axis_id: 0, axis_type: AxisType::Loop },
        DType::Index,
    );

    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: smallvec::smallvec![range.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // Verify END structure
    if let Op::End { computation, ranges } = result.op() {
        // Should have 1 range
        assert_eq!(ranges.len(), 1);

        // Range should match the input range
        assert!(std::rc::Rc::ptr_eq(&ranges[0], &range));

        // Computation should be STORE
        if let Op::Store { buffer, index, value } = computation.op() {
            // Buffer should be DEFINE_GLOBAL with ID 0
            assert!(matches!(buffer.op(), Op::DefineGlobal(0)));

            // Value should be the compute
            assert!(std::rc::Rc::ptr_eq(value, &compute));

            // Index should be INDEX operation with the range
            assert!(matches!(index.op(), Op::Index { .. }));
        } else {
            panic!("Expected STORE operation");
        }
    } else {
        panic!("Expected END operation");
    }

    // Verify context state
    assert_eq!(ctx.global_counter, 1);
    assert_eq!(ctx.local_counter, 0);
    assert!(ctx.has_buffer(&bufferize));
}

#[test]
fn test_bufferize_to_store_local_with_barrier() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with LOCAL addrspace
    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let range = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(5)), axis_id: 0, axis_type: AxisType::Loop },
        DType::Index,
    );

    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: smallvec::smallvec![range.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
        },
        DType::Float32,
    );

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // Should be wrapped in BARRIER
    if let Op::Barrier { src, .. } = result.op() {
        // Unwrap BARRIER to check the END inside
        if let Op::End { computation, ranges } = src.op() {
            // Should have 1 range
            assert_eq!(ranges.len(), 1);
            assert!(std::rc::Rc::ptr_eq(&ranges[0], &range));

            // Computation should be STORE with DEFINE_LOCAL(0)
            if let Op::Store { buffer, value, .. } = computation.op() {
                assert!(matches!(buffer.op(), Op::DefineLocal(0)));
                assert!(std::rc::Rc::ptr_eq(value, &compute));
            } else {
                panic!("Expected STORE operation");
            }
        } else {
            panic!("Expected END operation inside BARRIER");
        }
    } else {
        panic!("Expected BARRIER operation");
    }

    // Verify context state
    assert_eq!(ctx.global_counter, 0);
    assert_eq!(ctx.local_counter, 1);
    assert!(ctx.has_buffer(&bufferize));
}

#[test]
fn test_bufferize_to_store_multiple_ranges() {
    let mut ctx = KernelContext::new();

    // Create BUFFERIZE with multiple ranges
    let compute = UOp::const_(DType::Int32, ConstValue::Int(100));
    let range1 = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(4)), axis_id: 0, axis_type: AxisType::Loop },
        DType::Index,
    );
    let range2 = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(8)), axis_id: 1, axis_type: AxisType::Loop },
        DType::Index,
    );

    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: smallvec::smallvec![range1.clone(), range2.clone()],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Int32,
    );

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // Verify END structure with multiple ranges
    if let Op::End { computation, ranges } = result.op() {
        // Should have 2 ranges in the same order
        assert_eq!(ranges.len(), 2);
        assert!(std::rc::Rc::ptr_eq(&ranges[0], &range1));
        assert!(std::rc::Rc::ptr_eq(&ranges[1], &range2));

        // Verify STORE structure
        if let Op::Store { buffer, index, value } = computation.op() {
            // Buffer should be DEFINE_GLOBAL(0)
            assert!(matches!(buffer.op(), Op::DefineGlobal(0)));

            // Value should be the compute
            assert!(std::rc::Rc::ptr_eq(value, &compute));

            // Index should be INDEX with both ranges
            if let Op::Index { buffer: idx_buffer, indices, .. } = index.op() {
                // Buffer should match the STORE buffer
                assert!(std::rc::Rc::ptr_eq(idx_buffer, buffer));

                // Should have 2 indices
                assert_eq!(indices.len(), 2);
            } else {
                panic!("Expected INDEX operation");
            }
        } else {
            panic!("Expected STORE operation");
        }
    } else {
        panic!("Expected END operation");
    }

    // Verify context
    assert_eq!(ctx.global_counter, 1);
    assert!(ctx.has_buffer(&bufferize));
}

#[test]
fn test_non_bufferize_returns_none() {
    let mut ctx = KernelContext::new();

    // Create a non-BUFFERIZE operation
    let const_op = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // Should return None
    let result = bufferize_to_store(&const_op, &mut ctx);
    assert!(result.is_none());
}

#[test]
fn test_buffer_tracked_in_context() {
    let mut ctx = KernelContext::new();

    let compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let bufferize = UOp::new(
        Op::Bufferize {
            compute,
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Before conversion, buffer should not be tracked
    assert!(!ctx.has_buffer(&bufferize));

    // Convert to STORE
    bufferize_to_store(&bufferize, &mut ctx);

    // After conversion, buffer should be tracked
    assert!(ctx.has_buffer(&bufferize));

    // Should be able to get the DEFINE_GLOBAL
    let replacement = ctx.get_buffer(&bufferize).unwrap();
    assert!(matches!(replacement.op(), Op::DefineGlobal(_)));
}

#[test]
fn test_bufferize_to_store_sequential_global_ids() {
    let mut ctx = KernelContext::new();

    // Create three BUFFERIZE operations
    for i in 0..3 {
        let compute = UOp::const_(DType::Float32, ConstValue::Float(i as f64));
        let bufferize = UOp::new(
            Op::Bufferize {
                compute,
                ranges: SmallVec::new(),
                opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
            },
            DType::Float32,
        );

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
        let compute = UOp::const_(DType::Float32, ConstValue::Float(i as f64));
        let bufferize = UOp::new(
            Op::Bufferize {
                compute,
                ranges: SmallVec::new(),
                opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
            },
            DType::Float32,
        );

        bufferize_to_store(&bufferize, &mut ctx);

        // Counter should increment
        assert_eq!(ctx.global_counter, 0);
        assert_eq!(ctx.local_counter, (i + 1) as usize);
    }
}

#[test]
fn test_bufferize_to_store_mixed_global_local() {
    let mut ctx = KernelContext::new();

    let global_compute = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let local_compute = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    // Global buffer
    let global_bufferize = UOp::new(
        Op::Bufferize {
            compute: global_compute.clone(),
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Local buffer
    let local_bufferize = UOp::new(
        Op::Bufferize {
            compute: local_compute.clone(),
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
        },
        DType::Float32,
    );

    // Convert both
    let global_result = bufferize_to_store(&global_bufferize, &mut ctx);
    let local_result = bufferize_to_store(&local_bufferize, &mut ctx);

    // Verify independent counters
    assert_eq!(ctx.global_counter, 1);
    assert_eq!(ctx.local_counter, 1);

    // Global should be STORE (no ranges, so no END wrapper)
    if let Op::Store { buffer, value, .. } = global_result.unwrap().op() {
        assert!(matches!(buffer.op(), Op::DefineGlobal(0)));
        assert!(std::rc::Rc::ptr_eq(value, &global_compute));
    } else {
        panic!("Expected STORE operation for global");
    }

    // Local should be BARRIER wrapping STORE
    if let Op::Barrier { src, .. } = local_result.unwrap().op() {
        if let Op::Store { buffer, value, .. } = src.op() {
            assert!(matches!(buffer.op(), Op::DefineLocal(0)));
            assert!(std::rc::Rc::ptr_eq(value, &local_compute));
        } else {
            panic!("Expected STORE operation inside BARRIER");
        }
    } else {
        panic!("Expected BARRIER operation for local");
    }

    // Verify both are tracked in context
    assert!(ctx.has_buffer(&global_bufferize));
    assert!(ctx.has_buffer(&local_bufferize));
}

#[test]
fn test_bufferize_to_store_integration_with_split_kernel() {
    use crate::rangeify::split_kernel::split_store;

    let mut ctx = KernelContext::new();

    // Create a BUFFERIZE operation
    let compute = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let range = UOp::new(
        Op::Range { end: UOp::const_(DType::Index, ConstValue::Int(10)), axis_id: 0, axis_type: AxisType::Loop },
        DType::Index,
    );

    let bufferize = UOp::new(
        Op::Bufferize {
            compute: compute.clone(),
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Stage 1: BUFFERIZE → STORE
    let store_result = bufferize_to_store(&bufferize, &mut ctx).unwrap();

    // Verify buffer was tracked
    assert_eq!(ctx.global_counter, 1);
    assert!(ctx.has_buffer(&bufferize));

    // Extract the STORE from END
    let store = if let Op::End { computation, .. } = store_result.op() {
        computation.clone()
    } else {
        panic!("Expected END operation");
    };

    // Stage 2: STORE → KERNEL (using split_store)
    let kernel_result = split_store(&store, &mut ctx).unwrap();

    // Verify KERNEL structure
    if let Op::Kernel { sources, ast } = kernel_result.op() {
        // KERNEL should have 1 source (the DEFINE_GLOBAL buffer)
        assert_eq!(sources.len(), 1);
        assert!(matches!(sources[0].op(), Op::DefineGlobal(0)));

        // AST should be SINK wrapping the STORE
        if let Op::Sink { sources: sink_sources } = ast.op() {
            assert_eq!(sink_sources.len(), 1);
            assert!(std::rc::Rc::ptr_eq(&sink_sources[0], &store));

            // Verify STORE references the buffer from context
            if let Op::Store { buffer, value, .. } = sink_sources[0].op() {
                // Buffer should be the DEFINE_GLOBAL from context
                let ctx_buffer = ctx.get_buffer(&bufferize).unwrap();
                assert!(std::rc::Rc::ptr_eq(buffer, ctx_buffer));

                // Value should be the original compute
                assert!(std::rc::Rc::ptr_eq(value, &compute));
            } else {
                panic!("Expected STORE in SINK");
            }
        } else {
            panic!("Expected SINK operation");
        }
    } else {
        panic!("Expected KERNEL operation");
    }
}
