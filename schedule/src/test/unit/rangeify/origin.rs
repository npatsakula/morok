//! Origin propagation from the tensor graph to the kernel cut, and the harvest,
//! strip and stamp that `split_store` performs there.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::origin::{self, OriginId, OriginScope};
use svod_ir::{Op, ReduceOp, SInt, UOp, ops};
use test_case::test_case;

use crate::rangeify::{kernel_graph_pre_cut, rangeify, try_get_kernel_graph};

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

fn matrix(rows: usize, cols: usize) -> Arc<UOp> {
    buffer(rows * cols).try_reshape(&smallvec![SInt::Const(rows), SInt::Const(cols)]).expect("reshape")
}

/// Build a sink under a fresh module scope, returning it with that scope's id.
fn under_module(name: &str, build: impl FnOnce() -> Arc<UOp>) -> (Arc<UOp>, OriginId) {
    let _scope = OriginScope::module(name);
    let id = origin::current().expect("a module scope is active while capture is on");
    (UOp::sink(vec![build()]), id)
}

fn calls(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    root.toposort().into_iter().filter(|node| matches!(node.op(), Op::Call(..))).collect()
}

fn call_info(call: &Arc<UOp>) -> &svod_ir::CallInfo {
    match call.op() {
        Op::Call(ops::Call { info, .. }) => info,
        op => panic!("expected CALL, got {op:?}"),
    }
}

fn body_of(call: &Arc<UOp>) -> Arc<UOp> {
    match call.op() {
        Op::Call(ops::Call { body, .. }) => body.clone(),
        op => panic!("expected CALL, got {op:?}"),
    }
}

// =========================================================================
// Propagation up to the cut
// =========================================================================

fn graph_elementwise() -> Arc<UOp> {
    buffer(8).try_mul(&buffer(8)).expect("mul").contiguous()
}

fn graph_reduce() -> Arc<UOp> {
    matrix(2, 3).try_reduce_axis(ReduceOp::Add, vec![1]).expect("reduce").contiguous()
}

fn graph_permute() -> Arc<UOp> {
    matrix(2, 3).try_permute(vec![1, 0]).expect("permute").contiguous()
}

fn graph_reshape() -> Arc<UOp> {
    // A reshape of a realized buffer is a pure view, so it needs a computation
    // under it to produce a kernel at all.
    buffer(6)
        .try_mul(&buffer(6))
        .expect("mul")
        .try_reshape(&smallvec![SInt::Const(3), SInt::Const(2)])
        .expect("reshape")
        .contiguous()
}

fn graph_expand() -> Arc<UOp> {
    matrix(3, 1).try_expand(&smallvec![SInt::Const(3), SInt::Const(4)]).expect("expand").contiguous()
}

fn graph_pad() -> Arc<UOp> {
    matrix(2, 3).try_pad(&[(1.into(), 1.into()), (0.into(), 0.into())]).expect("pad").contiguous()
}

fn graph_shrink() -> Arc<UOp> {
    matrix(4, 4)
        .try_shrink(&[(SInt::Const(1), SInt::Const(3)), (SInt::Const(0), SInt::Const(4))])
        .expect("shrink")
        .contiguous()
}

fn graph_cast() -> Arc<UOp> {
    buffer(8).cast(DType::Float16).contiguous()
}

fn graph_fused_reduce() -> Arc<UOp> {
    matrix(2, 3)
        .try_mul(&matrix(2, 3))
        .expect("mul")
        .try_reduce_axis(ReduceOp::Add, vec![1])
        .expect("reduce")
        .contiguous()
}

#[test_case(graph_elementwise; "elementwise")]
#[test_case(graph_reduce; "reduce")]
#[test_case(graph_permute; "permute")]
#[test_case(graph_reshape; "reshape")]
#[test_case(graph_expand; "expand")]
#[test_case(graph_pad; "pad")]
#[test_case(graph_shrink; "shrink")]
#[test_case(graph_cast; "cast")]
#[test_case(graph_fused_reduce; "fused reduce")]
fn origin_reaches_the_store_chain_before_the_cut(build: fn() -> Arc<UOp>) {
    let _capture = origin::capture_for_thread(true);
    let (sink, id) = under_module("pre-cut", build);

    let (rangeified, _) = rangeify(sink).expect("rangeify");
    let (pre_cut, _) = kernel_graph_pre_cut(rangeified);

    let stores: Vec<_> = pre_cut.toposort().into_iter().filter(|node| matches!(node.op(), Op::Store(..))).collect();
    assert!(!stores.is_empty(), "the pipeline must produce at least one STORE:\n{}", pre_cut.tree());
    assert!(
        stores.iter().any(|store| store.origin() == Some(id)),
        "the STORE chain must still carry the module origin:\n{}",
        pre_cut.tree()
    );
}

#[test_case(graph_elementwise; "elementwise")]
#[test_case(graph_reduce; "reduce")]
#[test_case(graph_permute; "permute")]
#[test_case(graph_reshape; "reshape")]
#[test_case(graph_expand; "expand")]
#[test_case(graph_pad; "pad")]
#[test_case(graph_shrink; "shrink")]
#[test_case(graph_cast; "cast")]
#[test_case(graph_fused_reduce; "fused reduce")]
fn every_kernel_is_attributed_and_its_body_stripped(build: fn() -> Arc<UOp>) {
    let _capture = origin::capture_for_thread(true);
    let (sink, id) = under_module("harvest", build);

    let (rangeified, _) = rangeify(sink).expect("rangeify");
    let (graph, _) = try_get_kernel_graph(rangeified).expect("kernel graph");

    let kernels = calls(&graph);
    assert!(!kernels.is_empty(), "expected at least one kernel:\n{}", graph.tree());
    for call in &kernels {
        let info = call_info(call);
        assert_eq!(info.origin, Some(id), "kernel is charged to the scope it was built under");
        assert_eq!(
            info.origins.iter().copied().collect::<Vec<_>>(),
            [id],
            "single-scope kernel carries exactly one origin"
        );
        assert!(
            body_of(call).toposort().iter().all(|node| node.origin().is_none()),
            "the kernel body must be origin-free so identical kernels share one program"
        );
    }
}

// =========================================================================
// Harvest, strip, stamp
// =========================================================================

/// The load-bearing property: same computation, two scopes ⇒ two CALLs with
/// distinct attribution over one shared body.
#[test]
fn identical_kernels_in_two_scopes_share_a_body_and_differ_in_origin() {
    let _capture = origin::capture_for_thread(true);

    let kernel_of = |name: &str| {
        let (sink, id) = under_module(name, graph_elementwise);
        let (rangeified, _) = rangeify(sink).expect("rangeify");
        let (graph, _) = try_get_kernel_graph(rangeified).expect("kernel graph");
        let call = calls(&graph).first().cloned().expect("one kernel");
        (call, id)
    };
    let (left, left_id) = kernel_of("a");
    let (right, right_id) = kernel_of("b");

    assert_ne!(left_id, right_id);
    assert_eq!(call_info(&left).origin, Some(left_id));
    assert_eq!(call_info(&right).origin, Some(right_id));
    assert!(!Arc::ptr_eq(&left, &right), "distinct origins must not collapse the two dispatches");
    assert!(
        Arc::ptr_eq(&body_of(&left), &body_of(&right)),
        "stripped bodies hash-cons to one node, so the optimizer and every kernel cache see one kernel"
    );
}

/// The strip must be an identity on structure: the stripped body is the very node
/// an origin-free build produces.
#[test]
fn a_stripped_body_hash_conses_to_the_origin_free_build() {
    let bodies = |capture: bool| {
        let _capture = origin::capture_for_thread(capture);
        let sink = if capture {
            under_module("stripped", graph_fused_reduce).0
        } else {
            UOp::sink(vec![graph_fused_reduce()])
        };
        let (rangeified, _) = rangeify(sink).expect("rangeify");
        let (graph, _) = try_get_kernel_graph(rangeified).expect("kernel graph");
        calls(&graph).iter().map(body_of).collect::<Vec<_>>()
    };
    let scoped = bodies(true);
    let plain = bodies(false);

    assert_eq!(scoped.len(), plain.len());
    for (with_origin, without) in scoped.iter().zip(&plain) {
        assert!(
            Arc::ptr_eq(with_origin, without),
            "stripping must reproduce the origin-free node, not merely an equal one"
        );
    }
}

#[test]
fn a_kernel_fusing_two_scopes_carries_both_origins() {
    let _capture = origin::capture_for_thread(true);

    let (left, left_id) = {
        let _scope = OriginScope::module("left");
        (buffer(8).try_add(&buffer(8)).expect("add"), origin::current().expect("scope"))
    };
    let (right, right_id) = {
        let _scope = OriginScope::module("right");
        // The multiplication is the stored value, so `right` is the primary; the
        // addition fuses into the same kernel and joins the set.
        (left.try_mul(&buffer(8)).expect("mul").contiguous(), origin::current().expect("scope"))
    };

    let (rangeified, _) = rangeify(UOp::sink(vec![right])).expect("rangeify");
    let (graph, _) = try_get_kernel_graph(rangeified).expect("kernel graph");

    let call = calls(&graph).first().cloned().expect("one kernel");
    let info = call_info(&call);
    assert_eq!(info.origin, Some(right_id), "the stored value's scope is the primary");
    assert_eq!(info.origins.len(), 2, "a fused kernel lists every scope it consumed: {:?}", info.origins);
    assert!(info.origins.contains(&left_id) && info.origins.contains(&right_id));
}

#[test]
fn a_copy_only_kernel_is_attributed() {
    let _capture = origin::capture_for_thread(true);
    let (sink, id) = under_module("copy", || {
        // Cross-device so the copy is not elided; nothing is executed here.
        buffer(8).copy(DeviceSpec::Amd { device_id: 0 })
    });

    let (rangeified, _) = rangeify(sink).expect("rangeify");
    let (graph, _) = try_get_kernel_graph(rangeified).expect("kernel graph");

    let kernels = calls(&graph);
    assert!(!kernels.is_empty(), "a copy is still a kernel:\n{}", graph.tree());
    assert!(
        kernels.iter().all(|call| call_info(call).origin == Some(id)),
        "copy-only kernels lose their tag at bufferize; the origin must survive it"
    );
    assert!(
        kernels.iter().all(|call| body_of(call).toposort().iter().all(|node| node.origin().is_none())),
        "a direct COPY body is stripped like any other kernel body"
    );
}

#[test]
fn capture_off_leaves_kernels_unattributed() {
    let _capture = origin::capture_for_thread(false);
    let sink = {
        // The scope is a no-op while capture is off, exactly as in a default build.
        let _scope = OriginScope::module("ignored");
        UOp::sink(vec![graph_elementwise()])
    };

    let (rangeified, _) = rangeify(sink).expect("rangeify");
    let (graph, _) = try_get_kernel_graph(rangeified).expect("kernel graph");

    for call in calls(&graph) {
        let info = call_info(&call);
        assert_eq!(info.origin, None);
        assert!(info.origins.is_empty());
    }
}
