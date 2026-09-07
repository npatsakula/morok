use super::*;
use svod_dtype::DeviceSpec;

use crate::{AxisId, AxisType, DType, KernelInfo, RendererDevice, WmmaMetadata};

fn loop_range(axis: usize, size: i64) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(size), AxisId::Renumbered(axis), AxisType::Loop)
}

#[test]
fn alu_weight_is_the_product_of_enclosing_ranges() {
    let outer = loop_range(0, 8);
    let inner = loop_range(1, 4);
    let under_both = outer.try_add(&inner).unwrap();
    let unenclosed = UOp::define_var("a".to_string(), 1, 16).try_add(&UOp::define_var("b".to_string(), 1, 16)).unwrap();

    assert_eq!(compute_ops_estimate(&under_both), 8 * 4);
    assert_eq!(compute_ops_estimate(&unenclosed), 1);
    assert_eq!(compute_ops_estimate(&UOp::sink(vec![under_both, unenclosed])), 8 * 4 + 1);
}

#[test]
fn more_than_64_ranges_span_multiple_bitset_words() {
    // Only the last range iterates more than once, so every partial sum but
    // the outermost weighs 1 — a wrong bit-to-size mapping across the word
    // boundary would not land on 3.
    let ranges: Vec<Arc<UOp>> = (0..70).map(|i| loop_range(i, if i == 69 { 3 } else { 1 })).collect();
    let sum = ranges.iter().skip(1).fold(ranges[0].clone(), |acc, r| acc.try_add(r).unwrap());

    assert_eq!(compute_ops_estimate(&sum), 68 + 3);
}

/// Address arithmetic is not compute. An INDEX's operands are how a kernel
/// finds its data, and counting them made a hand-lowered matmul — whose body
/// is almost entirely index math — report tens of times the hardware's peak.
#[test]
fn index_arithmetic_is_not_counted() {
    let outer = loop_range(0, 8);
    let inner = loop_range(1, 4);
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 64, DType::Float32);
    // `outer * 4 + inner` is addressing, and the INDEX it feeds is not an ALU op.
    let address = outer.try_mul(&UOp::index_const(4)).unwrap().try_add(&inner).unwrap();
    let indexed = UOp::index().buffer(buffer.clone()).indices(vec![address]).call().unwrap();
    assert_eq!(compute_ops_estimate(&indexed), 0, "an address computation is not flops");

    // The same arithmetic used as a value still counts, and a value that also
    // addresses stays counted — the conservative direction.
    let value = outer.try_add(&inner).unwrap();
    assert_eq!(compute_ops_estimate(&value), 8 * 4);
    let both = UOp::index().buffer(buffer).indices(vec![value.clone()]).call().unwrap();
    assert_eq!(compute_ops_estimate(&UOp::sink(vec![both, value])), 8 * 4);
}

/// A WMMA is one instruction but a whole tile of MACs, so it cannot weigh the
/// same as a scalar add.
#[test]
fn wmma_weighs_its_macs() {
    let dims = (8usize, 16usize, 16usize);
    let (n, m, k) = dims;
    let macs = (2 * n * m * k) as u64;
    let metadata = || WmmaMetadata {
        name: "test".into(),
        dims,
        dtype_in: DType::BFloat16,
        dtype_out: DType::Float32,
        device: RendererDevice::CudaSm80,
        threads: 32,
        upcast_axes: None,
        reduce_axes: Vec::new(),
    };

    let c = UOp::native_const(0.0f32);
    let mma = UOp::wmma(c.clone(), c.clone(), c, metadata());
    assert_eq!(compute_ops_estimate(&mma), macs, "m16n8k16 is 4096 MACs, not one op");

    // An accumulator carried by a loop puts the mma inside it, and the tile is
    // then worth its MACs on every iteration.
    let outer = loop_range(0, 8);
    let acc = outer.cast(DType::Float32);
    let looped = UOp::wmma(acc.clone(), acc.clone(), acc, metadata());
    assert_eq!(compute_ops_estimate(&looped), macs * 8);
}

/// A hand-lowered tile-DSL AST (`opts_to_apply == Some([])`) carries its own
/// addressing, so range nesting is no longer recoverable from operand
/// dependence. Report no count rather than a wrong one; the profiler renders
/// `-` for the roofline instead of a fabricated GFLOP/s.
#[test]
fn hand_lowered_kernels_report_no_estimate() {
    let body = loop_range(0, 8).try_add(&loop_range(1, 4)).unwrap();
    let scheduled = UOp::sink_with_info(vec![body.clone()], KernelInfo::default());
    assert_eq!(compute_ops_estimate(&scheduled), 8 * 4, "an optimizer-scheduled AST still counts");

    let hand = UOp::sink_with_info(vec![body], KernelInfo { opts_to_apply: Some(Vec::new()), ..KernelInfo::default() });
    assert_eq!(compute_ops_estimate(&hand), u64::MAX, "no reliable count for a hand-lowered body");
}
