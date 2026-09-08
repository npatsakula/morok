//! Kernel attribution from the tensor graph to the execution plan.
//!
//! Scopes are installed with [`origin::install`] rather than built with
//! [`svod_ir::origin::OriginScope`]: installing an already-interned id applies
//! whether or not capture is enabled, so these tests never touch the
//! process-wide flag and can run beside any other test.

use std::sync::Arc;

use svod_ir::origin::{self, Origin, OriginFrame, OriginId, OriginScope};
use svod_runtime::execution_plan::ExecutionPlan;

use crate::Tensor;

fn module(name: &str) -> OriginId {
    origin::intern(Origin { parent: None, frame: OriginFrame::Module { name: name.to_owned() } })
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
    (&(&a * &b).unwrap() + &c).unwrap()
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

    let left = {
        let _scope = origin::install(Some(left_id));
        linear(2.0)
    };
    let right = {
        let _scope = origin::install(Some(right_id));
        linear(3.0)
    };

    let plan = Tensor::prepare_batch([&left, &right]).expect("prepare batch");
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
        let tensor = {
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
    let tensor = linear(5.0);
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
    let tensor = linear(7.0);
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
    let tensor = {
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

/// A hand-lowered kernel never reaches `split_store` — the cut gates on a
/// kernel-marked SINK — so `UOp::custom_kernel` owes it the same contract: an
/// origin-free body, attribution on the callable. Without it every conformer
/// layer's tile-DSL attention kernel compiles separately.
#[test]
fn hand_lowered_kernel_bodies_dedup_across_scopes() {
    use svod_ir::{Op, UOp, ops};

    let (left_id, right_id) = (module("carriage.hand.left"), module("carriage.hand.right"));
    let build = || {
        let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
        input
            .custom_kernel(&[], |placeholders| UOp::sink(vec![placeholders[0].contiguous()]))
            .expect("custom kernel")
            .into_iter()
            .next()
            .expect("one output per source")
    };
    let left = {
        let _scope = origin::install(Some(left_id));
        build()
    };
    let right = {
        let _scope = origin::install(Some(right_id));
        build()
    };

    let callable = |tensor: &Tensor| match tensor.uop().op() {
        Op::After(ops::After { deps, .. }) => deps[0].clone(),
        op => panic!("custom_kernel returns AFTER(CALL), got {op:?}"),
    };
    let parts = |call: &Arc<svod_ir::UOp>| match call.op() {
        Op::Call(ops::Call { body, info, .. }) => (body.clone(), info.clone()),
        op => panic!("an opaque body wraps into CALL, got {op:?}"),
    };
    let (left_body, left_info) = parts(&callable(&left));
    let (right_body, right_info) = parts(&callable(&right));

    for body in [&left_body, &right_body] {
        assert!(
            body.toposort().iter().all(|node| node.origin().is_none()),
            "the body keys every downstream cache and must be origin-free:\n{}",
            body.tree()
        );
    }
    assert!(
        Arc::ptr_eq(&left_body, &right_body),
        "stripped bodies must hash-cons to one node:\nLEFT\n{}\nRIGHT\n{}",
        left_body.tree(),
        right_body.tree()
    );
    assert!(descends_from(left_info.origin, left_id) && descends_from(right_info.origin, right_id));
    assert_eq!(left_info.origins.iter().copied().collect::<Vec<_>>(), vec![left_id]);
    assert_eq!(right_info.origins.iter().copied().collect::<Vec<_>>(), vec![right_id]);
}

/// Materialising an input belongs to its producer: two scopes handing the same
/// unscoped tensor to a hand-lowered kernel share one copy instead of one each.
#[test]
fn hand_lowered_kernel_inputs_are_materialised_once_per_producer() {
    use svod_ir::{Op, UOp, ops};

    let shared = (&Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]) * &Tensor::from_slice([2.0f32; 4])).unwrap();
    let launch = |scope: OriginId| {
        let _scope = origin::install(Some(scope));
        let out = Tensor::from_slice([0.0f32; 4]);
        out.custom_kernel(&[&shared], |placeholders| UOp::sink(vec![placeholders[0].contiguous()]))
            .expect("custom kernel")
            .into_iter()
            .next()
            .expect("one output per source")
    };
    let left = launch(module("carriage.input.left"));
    let right = launch(module("carriage.input.right"));

    let input_arg = |tensor: &Tensor| match tensor.uop().op() {
        Op::After(ops::After { deps, .. }) => match deps[0].op() {
            Op::Call(ops::Call { args, .. }) => args[1].clone(),
            op => panic!("expected CALL, got {op:?}"),
        },
        op => panic!("expected AFTER(CALL), got {op:?}"),
    };
    let (left_input, right_input) = (input_arg(&left), input_arg(&right));
    assert!(matches!(left_input.op(), Op::Contiguous(..)));
    assert_eq!(left_input.origin(), shared.uop().origin(), "the copy takes the producer's origin, not the caller's");
    assert!(Arc::ptr_eq(&left_input, &right_input), "both scopes must share one materialisation");
}

/// A value-producing body is inlined by rangeify and cut like any other graph, so
/// its nodes keep their origins for the cut to harvest instead of being stripped.
#[test]
fn inlined_function_bodies_keep_their_origins() {
    use svod_ir::{Op, ops};

    let scope = module("carriage.function");
    let output = {
        let _scope = origin::install(Some(scope));
        let input = Tensor::from_slice([1.0f32, 2.0]);
        input
            .custom_kernel(&[], |placeholders| placeholders[0].try_mul(&placeholders[0]).expect("mul"))
            .expect("custom kernel")
            .into_iter()
            .next()
            .expect("one output per source")
    };
    let output = output.uop();
    let Op::After(ops::After { deps, .. }) = output.op() else { panic!("expected AFTER") };
    let Op::Function(ops::Function { body, info, .. }) = deps[0].op() else {
        panic!("a value body wraps into FUNCTION, got {}", deps[0].op().as_ref())
    };
    assert!(info.origins.is_empty(), "attribution stays on the nodes, not on the callable");
    assert!(
        body.toposort().iter().any(|node| node.origin() == Some(scope)),
        "the body keeps its origins for the cut:\n{}",
        body.tree()
    );
}

/// Symbolic dimensions are shape algebra, not work: two modules naming the same
/// variable share one node and one binding whatever scope they run under.
#[test]
fn symbolic_dimensions_are_shared_across_scopes() {
    let _capture = origin::capture_for_thread(true);
    let define = |scope: &str| {
        let _scope = OriginScope::module(scope);
        let t = crate::Variable::new("carriage.t", 1, 16);
        let bound = t.bind(8).expect("bind");
        (t.uop().clone(), bound.uop().clone())
    };
    let (left_var, left_bound) = define("symbolic.left");
    let (right_var, right_bound) = define("symbolic.right");
    assert!(Arc::ptr_eq(&left_var, &right_var), "one variable, not one per scope");
    assert!(Arc::ptr_eq(&left_bound, &right_bound), "one binding, not one per scope");
    assert_eq!(left_bound.origin(), None);
}

/// A caller-supplied attribution wins; the harvested set still describes the body.
#[test]
fn hand_lowered_kernel_keeps_an_explicit_origin() {
    use svod_ir::{Op, UOp, ops};

    let (declared, built_under) = (module("carriage.hand.declared"), module("carriage.hand.built"));
    let info = svod_ir::CallInfo { origin: Some(declared), ..svod_ir::CallInfo::default() };
    let output = {
        let _scope = origin::install(Some(built_under));
        let input = Tensor::from_slice([1.0f32, 2.0]);
        input
            .custom_kernel_with(&[], info, |placeholders| UOp::sink(vec![placeholders[0].contiguous()]))
            .expect("custom kernel")
            .into_iter()
            .next()
            .expect("one output per source")
    };
    let output = output.uop();
    let Op::After(ops::After { deps, .. }) = output.op() else { panic!("expected AFTER") };
    let Op::Call(ops::Call { info, .. }) = deps[0].op() else { panic!("expected CALL") };
    assert_eq!(info.origin, Some(declared));
    assert_eq!(info.origins.iter().copied().collect::<Vec<_>>(), vec![built_under]);
}

/// A literal is the one node two scopes build independently yet identically, so an
/// origin on it splits what should be a single constant. The kernel cut re-merges
/// the halves with `without_origins`, which lets a structural rewrite the pre-cut
/// passes could not see — `WHERE(_, t, t) -> t` here — fire *after* the CALL ABI was
/// fixed, leaving the kernel bound to a buffer its compiled program never reads.
///
/// `scatter_reduce(include_self = false)` is the smallest graph with that shape: the
/// `full` target's fill value and the reduction's own identity are the same literal
/// reached through two call frames. Unlike the tests above this needs real call
/// frames, so it enables capture rather than installing an id.
#[test]
fn a_literal_split_by_call_frames_does_not_change_the_kernel_abi() {
    use crate::indexing::ScatterReduction;
    use svod_ir::{ConstValue, DType};

    crate::test::helpers::test_setup();

    let scattered = |capture: bool| -> Vec<f32> {
        let _capture = origin::capture_for_thread(capture);
        let target = Tensor::full(&[4], ConstValue::Float(0.0), DType::Float32);
        let index = Tensor::from_slice([0i32, 0, 2, 2]);
        let source = Tensor::full(&[4], ConstValue::Float(1.0), DType::Float32);
        let out = target
            .scatter_reduce(0, &index, &source, ScatterReduction::Sum, false)
            .expect("scatter_reduce")
            .contiguous();
        out.realize().expect("the CALL must bind exactly the buffers its program reads");
        out.array_view::<f32>().expect("read back").iter().copied().collect()
    };

    assert_eq!(scattered(true), scattered(false), "capture must not change the compiled result");
}
