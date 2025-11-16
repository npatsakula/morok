use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisType, BufferizeOpts, ConstValue, Op, UOp};
use smallvec::SmallVec;

use crate::rangeify::{bufferize_to_store::bufferize_to_store, KernelContext};

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

    // Should be an END operation
    assert!(matches!(result.op(), Op::End { .. }));

    // Should have created a DEFINE_GLOBAL
    assert_eq!(ctx.global_counter, 1);
    assert_eq!(ctx.local_counter, 0);

    // Should NOT have a BARRIER (global buffer)
    assert!(!matches!(result.op(), Op::Barrier { .. }));
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
            ranges: smallvec::smallvec![range],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
        },
        DType::Float32,
    );

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // Should be wrapped in BARRIER
    assert!(matches!(result.op(), Op::Barrier { .. }));

    // Should have created a DEFINE_LOCAL
    assert_eq!(ctx.global_counter, 0);
    assert_eq!(ctx.local_counter, 1);
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
            ranges: smallvec::smallvec![range1, range2],
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Int32,
    );

    // Convert to STORE
    let result = bufferize_to_store(&bufferize, &mut ctx);

    assert!(result.is_some());
    let result = result.unwrap();

    // Should be wrapped in END
    assert!(matches!(result.op(), Op::End { .. }));

    // Should have STORE as computation and 2 ranges
    if let Op::End { computation, ranges } = result.op() {
        // Computation should be STORE
        assert!(matches!(computation.op(), Op::Store { .. }));
        // Should have 2 ranges
        assert_eq!(ranges.len(), 2);
    } else {
        panic!("Expected END operation");
    }
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
            Op::Bufferize { compute, ranges: SmallVec::new(), opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global } },
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
            Op::Bufferize { compute, ranges: SmallVec::new(), opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local } },
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

    // Global buffer
    let global_bufferize = UOp::new(
        Op::Bufferize {
            compute: UOp::const_(DType::Float32, ConstValue::Float(1.0)),
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Global },
        },
        DType::Float32,
    );

    // Local buffer
    let local_bufferize = UOp::new(
        Op::Bufferize {
            compute: UOp::const_(DType::Float32, ConstValue::Float(2.0)),
            ranges: SmallVec::new(),
            opts: BufferizeOpts { device: None, addrspace: AddrSpace::Local },
        },
        DType::Float32,
    );

    // Convert both
    let global_result = bufferize_to_store(&global_bufferize, &mut ctx);
    let local_result = bufferize_to_store(&local_bufferize, &mut ctx);

    // Check counters
    assert_eq!(ctx.global_counter, 1);
    assert_eq!(ctx.local_counter, 1);

    // Global should be STORE without BARRIER
    assert!(matches!(global_result.unwrap().op(), Op::Store { .. }));

    // Local should be BARRIER wrapping STORE
    assert!(matches!(local_result.unwrap().op(), Op::Barrier { .. }));
}

