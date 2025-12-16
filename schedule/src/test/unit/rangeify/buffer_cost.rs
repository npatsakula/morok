use std::sync::Arc;

use crate::rangeify::kernel::*;
use morok_dtype::DType;
use morok_ir::{AddrSpace, AxisId, AxisType, BufferizeOpts, Op, UOp};
use smallvec::SmallVec;

/// Helper: Create a BUFFER operation for testing.
fn create_test_buffer(size: usize, dtype: DType, id: usize) -> Arc<UOp> {
    let unique = UOp::buffer_id(Some(id));
    let device = UOp::device(morok_device::DeviceSpec::Cpu);
    UOp::new(Op::Buffer { unique, device, size }, dtype)
}

#[test]
fn test_pcontig_config_default() {
    let config = PcontigConfig::default();
    assert_eq!(config.level, 2);
    assert_eq!(config.max_buffers_threshold, 3);
    assert_eq!(config.out_in_ratio_threshold, 10.0);
}

#[test]
fn test_collect_accessed_buffers_empty() {
    // Constant has no buffers
    let const_val = UOp::native_const(1.0f32);
    let buffers = collect_accessed_buffers(&const_val);
    assert_eq!(buffers.len(), 0);
}

#[test]
fn test_collect_accessed_buffers_single_buffer() {
    // Single BUFFER operation
    let buffer = create_test_buffer(100, DType::Float32, 0);
    let buffers = collect_accessed_buffers(&buffer);
    assert_eq!(buffers.len(), 1);
    assert!(Arc::ptr_eq(&buffers[0], &buffer));
}

#[test]
fn test_collect_accessed_buffers_multiple() {
    // Two buffers in ADD operation
    let buf1 = create_test_buffer(100, DType::Float32, 0);
    let buf2 = create_test_buffer(100, DType::Float32, 1);
    let add = buf1.try_add(&buf2).unwrap();

    let buffers = collect_accessed_buffers(&add);
    assert_eq!(buffers.len(), 2);
}

#[test]
fn test_collect_accessed_buffers_stops_at_global_bufferize() {
    // Create inner buffer
    let inner_buf = create_test_buffer(100, DType::Float32, 0);

    // Wrap in GLOBAL bufferize
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Loop);
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Global };
    let bufferize = UOp::bufferize(inner_buf, vec![range], opts);

    // Should only see the bufferize, not the inner buffer
    let buffers = collect_accessed_buffers(&bufferize);
    assert_eq!(buffers.len(), 1);
    assert!(Arc::ptr_eq(&buffers[0], &bufferize));
}

#[test]
fn test_collect_accessed_buffers_deduplication() {
    // Use same buffer twice (b + b)
    let buf = create_test_buffer(100, DType::Float32, 0);
    let add = buf.try_add(&buf).unwrap();

    let buffers = collect_accessed_buffers(&add);
    assert_eq!(buffers.len(), 1); // Deduplicated
}

#[test]
fn test_collect_reduces_empty() {
    // Constant has no reduces
    let const_val = UOp::native_const(1.0f32);
    let reduces = collect_reduces(&const_val);
    assert_eq!(reduces.len(), 0);
}

#[test]
fn test_collect_reduces_single() {
    // Create simple reduce
    let const_val = UOp::native_const(1.0f32);
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);

    use morok_ir::ReduceOp;
    let reduce = UOp::reduce(const_val, SmallVec::from_iter([range]), ReduceOp::Add);

    let reduces = collect_reduces(&reduce);
    assert_eq!(reduces.len(), 1);
    assert!(Arc::ptr_eq(&reduces[0], &reduce));
}

#[test]
fn test_collect_indexes_empty() {
    // Constant has no indexes
    let const_val = UOp::native_const(1.0f32);
    let indexes = collect_indexes(&const_val);
    assert_eq!(indexes.len(), 0);
}

#[test]
fn test_collect_indexes_single() {
    // Create INDEX operation
    let buffer = create_test_buffer(100, DType::Float32, 0);
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Loop);

    let index = UOp::index(buffer, vec![range]).unwrap();

    let indexes = collect_indexes(&index);
    assert_eq!(indexes.len(), 1);
    assert!(Arc::ptr_eq(&indexes[0], &index));
}

#[test]
fn test_calculate_buffer_size_concrete_ranges() {
    // BUFFERIZE with concrete ranges [10, 20, 30] → 10*20*30=6000 elements → 6000*4=24000 bytes (Float32)
    let range1 = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Loop);
    let range2 = UOp::range_axis(UOp::index_const(20), AxisId::Renumbered(1), AxisType::Loop);
    let range3 = UOp::range_axis(UOp::index_const(30), AxisId::Renumbered(2), AxisType::Loop);

    let compute = UOp::native_const(1.0f32);
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
    let bufferize = UOp::bufferize(compute, vec![range1, range2, range3], opts);

    let size = calculate_buffer_size(&bufferize);
    assert_eq!(size, Some(24000)); // 6000 elements * 4 bytes = 24000 bytes
}

#[test]
fn test_calculate_buffer_size_buffer_op() {
    // BUFFER with explicit size
    let buffer = create_test_buffer(12345, DType::Float32, 0);

    let size = calculate_buffer_size(&buffer);
    assert_eq!(size, Some(12345));
}

#[test]
fn test_calculate_buffer_size_symbolic() {
    // Symbolic range → None
    let batch_size = UOp::define_var("batch".to_string(), 0, 128);
    let range = UOp::range_axis(batch_size, AxisId::Renumbered(0), AxisType::Loop);

    let compute = UOp::native_const(1.0f32);
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
    let bufferize = UOp::bufferize(compute, vec![range], opts);

    let size = calculate_buffer_size(&bufferize);
    assert_eq!(size, None);
}

#[test]
#[ignore] // TODO: Implement mstack test when UOp::mstack constructor is available
fn test_calculate_buffer_size_mstack() {
    // MSTACK should return 1
    let _buf1 = create_test_buffer(100, DType::Float32, 0);
    let _buf2 = create_test_buffer(100, DType::Float32, 1);
    // let mstack = UOp::mstack(vec![buf1, buf2]);

    // let size = calculate_buffer_size(&mstack);
    // assert_eq!(size, Some(1));
}

#[test]
fn test_calculate_out_in_ratio_efficient() {
    // output=100, inputs=[90, 5, 5] → ratio = 101/101 = 1.0 < 10
    let output_size = 100;
    let input_buffers = vec![
        create_test_buffer(90, DType::Float32, 0),
        create_test_buffer(5, DType::Float32, 1),
        create_test_buffer(5, DType::Float32, 2),
    ];

    let ratio = calculate_out_in_ratio(output_size, &input_buffers).unwrap();
    assert!(ratio < 10.0);
    assert!((ratio - 1.0).abs() < 0.01); // Should be ~1.0
}

#[test]
fn test_calculate_out_in_ratio_wasteful() {
    // output=10000, inputs=[10, 10, 10] → ratio = 10001/31 ≈ 322 > 10
    let output_size = 10000;
    let input_buffers = vec![
        create_test_buffer(10, DType::Float32, 0),
        create_test_buffer(10, DType::Float32, 1),
        create_test_buffer(10, DType::Float32, 2),
    ];

    let ratio = calculate_out_in_ratio(output_size, &input_buffers).unwrap();
    assert!(ratio > 10.0);
    assert!(ratio > 300.0); // Should be ~322
}

#[test]
fn test_calculate_out_in_ratio_with_symbolic_bufferize() {
    // Bufferize with symbolic range → None
    let output_size = 100;

    let batch_size = UOp::define_var("batch".to_string(), 0, 128);
    let range = UOp::range_axis(batch_size, AxisId::Renumbered(0), AxisType::Loop);
    let compute = UOp::native_const(1.0f32);
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
    let symbolic_bufferize = UOp::bufferize(compute, vec![range], opts);

    let input_buffers = vec![symbolic_bufferize];

    let ratio = calculate_out_in_ratio(output_size, &input_buffers);
    assert_eq!(ratio, None);
}

#[test]
fn test_has_buffer_in_reduce_positive() {
    // REDUCE(LOAD(BUFFER)) → true
    use morok_ir::ReduceOp;

    let buffer = create_test_buffer(100, DType::Float32, 0);
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = UOp::reduce(buffer, SmallVec::from_iter([range]), ReduceOp::Add);

    let reduces = vec![reduce];
    assert!(has_buffer_in_reduce(&reduces));
}

#[test]
fn test_has_buffer_in_reduce_negative() {
    // REDUCE(CONST) → false
    use morok_ir::ReduceOp;

    let const_val = UOp::native_const(1.0f32);
    let range = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = UOp::reduce(const_val, SmallVec::from_iter([range]), ReduceOp::Add);

    let reduces = vec![reduce];
    assert!(!has_buffer_in_reduce(&reduces));
}

#[test]
fn test_has_buffer_in_reduce_empty() {
    // Empty reduces → false
    let reduces: Vec<Arc<UOp>> = vec![];
    assert!(!has_buffer_in_reduce(&reduces));
}

#[test]
fn test_has_buffer_in_reduce_nested_bufferize() {
    // REDUCE(BUFFERIZE(...)) → true
    use morok_ir::ReduceOp;

    let compute = UOp::native_const(1.0f32);
    let range1 = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Loop);
    let opts = BufferizeOpts { device: None, addrspace: AddrSpace::Local };
    let bufferize = UOp::bufferize(compute, vec![range1.clone()], opts);

    let range2 = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(1), AxisType::Reduce);
    let reduce = UOp::reduce(bufferize, SmallVec::from_iter([range2]), ReduceOp::Add);

    let reduces = vec![reduce];
    assert!(has_buffer_in_reduce(&reduces));
}
