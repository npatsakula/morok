//! Kernel attribution from the tensor graph to the execution plan.
//!
//! Scopes are installed with [`origin::install`] rather than built with
//! [`svod_ir::origin::OriginScope`]: installing an already-interned id applies
//! whether or not capture is enabled, so these tests never touch the
//! process-wide flag and can run beside any other test.

use std::sync::Arc;

use svod_ir::origin::{self, Origin, OriginFrame, OriginId};
use svod_runtime::execution_plan::ExecutionPlan;

use crate::Tensor;

fn module(name: &str) -> OriginId {
    origin::intern(Origin { parent: None, frame: OriginFrame::Module { name: Arc::from(name) } })
}

/// Whether an attribution sits under `ancestor`. Op entry points push their own
/// call frames when capture is enabled, so a kernel's origin is a descendant of
/// the installed module scope rather than the scope itself.
fn descends_from(id: Option<OriginId>, ancestor: OriginId) -> bool {
    id.is_some_and(|id| origin::chain(id).contains(&ancestor))
}

/// `a * b + c` on fresh inputs — two kernels' worth of work with a shape the CPU
/// backend compiles quickly.
fn linear(scale: f32) -> Tensor {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([scale, scale, scale, scale]);
    let c = Tensor::from_slice([0.5f32, 0.5, 0.5, 0.5]);
    &(&a * &b) + &c
}

fn kernels(plan: &ExecutionPlan) -> Vec<&svod_runtime::PreparedKernel> {
    plan.prepared_kernels()
}

/// The load-bearing property: identical computations in two scopes compile once
/// and dispatch twice, each dispatch carrying its own attribution.
#[test]
fn two_scopes_share_one_program_and_differ_in_origin() {
    crate::test::helpers::test_setup();
    let (left_id, right_id) = (module("carriage.left"), module("carriage.right"));

    let mut left = {
        let _scope = origin::install(Some(left_id));
        linear(2.0)
    };
    let mut right = {
        let _scope = origin::install(Some(right_id));
        linear(3.0)
    };

    let plan = Tensor::prepare_batch([&mut left, &mut right]).expect("prepare batch");
    let dispatches = kernels(&plan);
    assert!(dispatches.len() >= 2, "each scope contributes at least one dispatch");

    let left_kernels: Vec<_> = dispatches.iter().filter(|k| descends_from(k.origin, left_id)).collect();
    let right_kernels: Vec<_> = dispatches.iter().filter(|k| descends_from(k.origin, right_id)).collect();
    assert!(!left_kernels.is_empty() && !right_kernels.is_empty(), "both scopes must be represented");
    assert!(
        dispatches.iter().all(|kernel| kernel
            .origins
            .iter()
            .all(|&id| descends_from(Some(id), left_id) || descends_from(Some(id), right_id))),
        "every dispatch is attributed to the scope it was built under"
    );

    // Same computation, different inputs: one compiled program, two dispatches.
    let (left_kernel, right_kernel) = (left_kernels[0], right_kernels[0]);
    // The bodies still carry rangeify's per-tensor tags (excluded from the content
    // hash by design), so identity here is the content hash — the key every kernel
    // cache uses. Nothing origin-shaped may remain in them.
    assert_eq!(
        crate::schedule_cache::content_hash(&left_kernel.ast),
        crate::schedule_cache::content_hash(&right_kernel.ast),
        "stripped bodies must hash identically across scopes:\nLEFT\n{}\nRIGHT\n{}",
        left_kernel.ast.tree(),
        right_kernel.ast.tree()
    );
    assert!(
        [left_kernel, right_kernel].iter().all(|kernel| kernel
            .ast
            .toposort()
            .iter()
            .all(|node| node.origin().is_none())),
        "the dispatched body must be origin-free"
    );
    assert!(
        Arc::ptr_eq(&left_kernel.kernel, &right_kernel.kernel),
        "one optimized-kernel cache entry serves both dispatches"
    );
}

/// A schedule-cache hit must reproduce origins rather than inherit the first
/// graph's: the origin is part of the normalized sink's content hash, so the two
/// runs key different entries and each restores its own CALL.
#[test]
fn scheduling_the_same_scoped_graph_twice_reproduces_its_origins() {
    crate::test::helpers::test_setup();
    let id = module("carriage.repeat");

    let origins_of_a_fresh_run = || {
        let mut tensor = {
            let _scope = origin::install(Some(id));
            linear(4.0)
        };
        let plan = tensor.prepare().expect("prepare");
        kernels(&plan).iter().map(|kernel| (kernel.origin, kernel.origins.clone())).collect::<Vec<_>>()
    };

    let first = origins_of_a_fresh_run();
    let second = origins_of_a_fresh_run();
    assert!(!first.is_empty());
    assert_eq!(first, second, "a schedule-cache hit must reproduce the origins, not lose or swap them");
    assert!(first.iter().all(|(origin, _)| descends_from(*origin, id)));
}

/// No installed scope, no module attribution: with capture off nothing is
/// recorded at all, and with capture on only the op entry points' own frames are.
#[test]
fn an_unscoped_graph_carries_no_module_attribution() {
    crate::test::helpers::test_setup();
    let mut tensor = linear(5.0);
    let plan = tensor.prepare().expect("prepare");

    for kernel in kernels(&plan) {
        if !origin::enabled() {
            assert_eq!(kernel.origin, None);
            assert!(kernel.origins.is_empty());
        }
        let frames = kernel.origins.iter().flat_map(|&id| origin::chain(id)).filter_map(origin::get);
        assert!(
            frames.map(|origin| origin.frame).all(|frame| !matches!(frame, OriginFrame::Module { .. })),
            "a module frame can only come from a scope the caller installed"
        );
    }
}

/// Scheduling and compiling inside a live scope must not stamp that scope onto the
/// kernel body: the debug assertions at the cache keys fire if it does, and in
/// release the on-disk beam key would fork on an allocation-ordered id.
#[test]
fn preparing_inside_a_live_scope_keeps_bodies_origin_free() {
    crate::test::helpers::test_setup();
    let id = module("carriage.live");

    // The ONNX importer and the model stages realize with their scope still open.
    let _scope = origin::install(Some(id));
    let mut tensor = linear(7.0);
    let plan = tensor.prepare().expect("prepare inside a live scope");

    let dispatches = kernels(&plan);
    assert!(!dispatches.is_empty());
    for kernel in dispatches {
        assert!(
            kernel.ast.toposort().iter().all(|node| node.origin().is_none()),
            "the pipeline's own nodes must not adopt the caller's scope"
        );
        assert!(descends_from(kernel.origin, id), "the graph built under the scope is still attributed to it");
    }
    plan.execute().expect("execute");
}

/// The profiler reads attribution off the dispatch, not off the shared program.
#[test]
fn profiles_carry_the_dispatch_origins() {
    crate::test::helpers::test_setup();
    let id = module("carriage.profiled");
    let mut tensor = {
        let _scope = origin::install(Some(id));
        linear(6.0)
    };

    let plan = tensor.prepare().expect("prepare");
    let expected: Vec<_> = kernels(&plan).iter().map(|kernel| (kernel.origin, kernel.origins.clone())).collect();
    let profiles = plan.execute_profiled().expect("profiled execution");

    assert_eq!(profiles.len(), expected.len());
    for (profile, (origin, origins)) in profiles.iter().zip(expected) {
        assert_eq!(profile.origin, origin);
        assert_eq!(profile.origins, origins);
    }
}
