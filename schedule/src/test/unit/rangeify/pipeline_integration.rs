//! Structure of the graph `try_get_kernel_graph` hands back.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{Op, UOp};

use crate::rangeify::try_get_kernel_graph;

use super::helpers::extract_kernel;
use svod_ir::ops;

/// Two `[3,4]` views of distinct buffers, added and materialised — the graph
/// `Tensor::from_slice(a) + Tensor::from_slice(b)` lowers to.
fn added_reshaped_buffers() -> Arc<UOp> {
    let view = || {
        let buffer = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 12, DType::Float32);
        let new_shape = UOp::stack(vec![UOp::index_const(3), UOp::index_const(4)].into());
        UOp::new(Op::Reshape(ops::Reshape { src: buffer, new_shape }), DType::Float32)
    };
    UOp::sink(vec![view().try_add(&view()).expect("add").contiguous()])
}

/// Only GLOBAL stages become STORE/BUFFER; a LOCAL stage left at the top level
/// is not a legal kernel graph and must be reported, not silently accepted.
#[test]
fn an_outer_local_stage_fails_the_kernel_graph_boundary() {
    let range = UOp::range_const(16, 0);
    let root = UOp::sink(vec![
        UOp::stage_global(UOp::native_const(1.0f32), vec![range.clone()]),
        UOp::stage_local(UOp::native_const(2.0f32), vec![range]),
    ]);

    let Err(err) = try_get_kernel_graph(root) else { panic!("outer local STAGE must be rejected") };
    assert!(err.to_string().contains("kernel graph specification"), "unexpected error: {err}");
}

#[test]
fn a_stage_lowers_to_a_call_over_its_own_buffer() {
    let stage = UOp::stage_global(UOp::native_const(std::f32::consts::PI), vec![UOp::range_const(20, 0)]);

    let (result, _ctx) = try_get_kernel_graph(stage).expect("kernel split");
    let kernel = extract_kernel(&result).expect("CALL");

    let Op::Call(ops::Call { body, args, .. }) = kernel.op() else { panic!("expected CALL, got {}", kernel.tree()) };
    assert!(matches!(body.op(), Op::Sink(..)), "a compute kernel body is a SINK");
    let [buffer] = args.as_slice() else { panic!("expected the single staged buffer, got {args:?}") };
    assert!(matches!(buffer.op(), Op::Buffer(..)), "CALL args stay BUFFERs; PARAMs live in the body");
}

#[test]
fn rangeify_lowers_every_reshape_and_the_split_indexes_the_input_buffers() {
    let (rangeified, _ctx) = crate::rangeify::rangeify(added_reshaped_buffers()).expect("rangeify");
    assert!(
        !rangeified.toposort().iter().any(|node| matches!(node.op(), Op::Reshape(..))),
        "RESHAPE must be gone after rangeify:\n{}",
        rangeified.tree()
    );

    let (kernel_graph, _ctx) = try_get_kernel_graph(rangeified).expect("kernel split");
    assert!(
        kernel_graph.toposort().iter().any(
            |node| matches!(node.op(), Op::Index(ops::Index { buffer, .. }) if matches!(buffer.op(), Op::Buffer(..) | Op::Param(..)))
        ),
        "input buffers must be reached through INDEX:\n{}",
        kernel_graph.tree()
    );
}
