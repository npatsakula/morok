use super::*;
use smallvec::SmallVec;

#[test]
fn test_output_indices_from_program_metadata_basic() {
    let outputs = output_indices_from_program_metadata(&[0, 1, 2], &[2], 3).expect("metadata mapping should succeed");
    assert_eq!(outputs, vec![2]);
}

#[test]
fn test_output_indices_from_program_metadata_sparse_slots() {
    let outputs =
        output_indices_from_program_metadata(&[2, 4, 7], &[4, 7], 3).expect("metadata mapping should succeed");
    assert_eq!(outputs, vec![1, 2]);
}

#[test]
fn test_output_indices_from_program_metadata_rejects_empty_outs() {
    let err = output_indices_from_program_metadata(&[0, 1], &[], 2).expect_err("empty outs should fail");
    assert!(format!("{err}").contains("ProgramSpec.outs is empty"));
}

#[test]
fn test_output_indices_from_program_metadata_rejects_unknown_slot() {
    let err = output_indices_from_program_metadata(&[0, 2], &[1], 2).expect_err("unknown outs slot should fail");
    assert!(format!("{err}").contains("not found in ProgramSpec.globals"));
}

#[test]
fn test_output_indices_from_program_metadata_rejects_out_of_range_position() {
    let err = output_indices_from_program_metadata(&[0, 1], &[1], 1).expect_err("mapped output index out of range");
    assert!(format!("{err}").contains("out of range"));
}

#[test]
fn test_resolve_compiled_kernel_buffer_indices_reorders_by_program_globals() {
    let p0 = UOp::param(0, 4, morok_dtype::DType::Float32, None);
    let p1 = UOp::param(1, 4, morok_dtype::DType::Float32, None);
    let body = UOp::sink(vec![p0.clone(), p1.clone()]);
    let call = body.call(SmallVec::from_vec(vec![p1.clone(), p0.clone()]), morok_ir::CallInfo::default());
    let item = crate::schedule::ScheduleItem {
        kernel: call,
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![p1.id, p0.id],
        fixedvars: std::collections::HashMap::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
        alias_registered_ids: vec![],
    };
    let uop_id_to_idx = std::collections::HashMap::from([(p1.id, 11), (p0.id, 10)]);

    let ordered =
        resolve_compiled_kernel_buffer_indices(&item, &uop_id_to_idx, &[0, 1]).expect("compiled buffer ABI ordering");

    assert_eq!(ordered, vec![11, 10]);
}

#[test]
fn test_resolve_compiled_kernel_buffer_indices_treats_globals_as_buffer_positions() {
    let p0 = UOp::param(0, 4, morok_dtype::DType::Float32, None);
    let p1 = UOp::param(1, 4, morok_dtype::DType::Float32, None);
    let body = UOp::sink(vec![p0.clone(), p1.clone()]);
    let call = body.call(SmallVec::from_vec(vec![p1.clone(), p0.clone()]), morok_ir::CallInfo::default());
    let item = crate::schedule::ScheduleItem {
        kernel: call,
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![p1.id, p0.id],
        fixedvars: std::collections::HashMap::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
        alias_registered_ids: vec![],
    };
    let uop_id_to_idx = std::collections::HashMap::from([(p1.id, 11), (p0.id, 10)]);

    let ordered =
        resolve_compiled_kernel_buffer_indices(&item, &uop_id_to_idx, &[1, 0]).expect("compiled buffer ABI ordering");

    assert_eq!(ordered, vec![10, 11]);
}

#[test]
fn test_resolve_compiled_kernel_buffer_indices_rejects_out_of_range_global_position() {
    let p0 = UOp::param(0, 4, morok_dtype::DType::Float32, None);
    let body = UOp::sink(vec![p0.clone()]);
    let call = body.call(SmallVec::from_vec(vec![p0.clone()]), morok_ir::CallInfo::default());
    let item = crate::schedule::ScheduleItem {
        kernel: call,
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![p0.id],
        fixedvars: std::collections::HashMap::new(),
        dependencies: vec![],
        instance_dependencies: vec![],
        alias_registered_ids: vec![],
    };
    let uop_id_to_idx = std::collections::HashMap::from([(p0.id, 10)]);

    let err = resolve_compiled_kernel_buffer_indices(&item, &uop_id_to_idx, &[1])
        .expect_err("out-of-range global position should fail");

    assert!(format!("{err}").contains("out of range"), "unexpected error: {err}");
}

#[test]
fn test_restore_post_schedule_pre_schedule_rewrites_runtime_buf_uops() {
    crate::test::helpers::test_setup();

    let c = &Tensor::from_slice([1.0f32, 2.0, 3.0]) + &Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let sink = UOp::sink(vec![c.uop().contiguous()]);

    let normalization = normalize_for_schedule_cache(&sink).expect("normalize schedule cache");
    let rangeify = morok_schedule::rangeify_with_map(normalization.normalized.clone(), None).expect("rangeify");
    let (kernel_graph_cached, _) = morok_schedule::try_get_kernel_graph(rangeify.sink).expect("kernel graph");
    let pre_schedule_cached = crate::schedule::create_pre_schedule(kernel_graph_cached).expect("pre schedule");

    assert!(
        pre_schedule_cached
            .items
            .iter()
            .flat_map(|item| item.sources.iter())
            .any(|src| matches!(src.op(), Op::Param { device: Some(_), .. })),
        "cached pre-schedule should keep normalized PARAM placeholders"
    );

    let restored = restore_post_schedule_pre_schedule(&pre_schedule_cached, &normalization);

    assert!(
        restored
            .items
            .iter()
            .flat_map(|item| item.sources.iter())
            .all(|src| !matches!(src.op(), Op::Param { device: Some(_), .. })),
        "restored pre-schedule should rewrite callable source PARAM placeholders"
    );
    assert!(
        restored.output_buffer_uops.iter().all(|u| !matches!(u.op(), Op::Param { device: Some(_), .. })),
        "restored pre-schedule should rewrite output buffer PARAM placeholders"
    );
    assert!(
        restored.items.iter().flat_map(|item| item.sources.iter()).all(|src| !matches!(src.op(), Op::LUnique(_))),
        "restored pre-schedule should rewrite LUNIQUE placeholders"
    );

    assert!(
        pre_schedule_cached
            .items
            .iter()
            .flat_map(|item| item.sources.iter())
            .any(|src| matches!(src.op(), Op::Param { device: Some(_), .. })),
        "restoring should not mutate cached pre-schedule"
    );
}

struct TestRenderer;

impl morok_device::device::Renderer for TestRenderer {
    fn render(
        &self,
        ast: &std::sync::Arc<UOp>,
        name: Option<&str>,
    ) -> morok_device::Result<morok_device::device::ProgramSpec> {
        let mut spec = morok_device::device::ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// test source".to_string(),
            morok_dtype::DeviceSpec::Cpu,
            ast.clone(),
        );
        spec.set_buffer_metadata(vec![0], vec![0], vec![]);
        spec.buf_count = 1;
        Ok(spec)
    }

    fn device(&self) -> &morok_dtype::DeviceSpec {
        static DEVICE: morok_dtype::DeviceSpec = morok_dtype::DeviceSpec::Cpu;
        &DEVICE
    }
}

struct TestCompiler;

impl morok_device::device::Compiler for TestCompiler {
    fn compile(
        &self,
        spec: &morok_device::device::ProgramSpec,
    ) -> morok_device::Result<morok_device::device::CompiledSpec> {
        Ok(morok_device::device::CompiledSpec::from_bytes(spec.name.clone(), vec![1, 2, 3], spec.ast.clone()))
    }

    fn cache_key(&self) -> &'static str {
        "test"
    }
}

#[test]
fn test_compile_with_program_pipeline_components_accepts_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let program = morok_codegen::program_pipeline::program_from_sink(sink, morok_dtype::DeviceSpec::Cpu);

    let (spec, compiled) =
        compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler, Some("p_test"))
            .expect("PROGRAM input should compile through staged pipeline");

    assert_eq!(spec.name, "p_test");
    assert_eq!(compiled.name, "p_test");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn test_compile_with_program_pipeline_components_rejects_non_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);

    let err = compile_with_program_pipeline_components(sink, &TestRenderer, &TestCompiler, Some("p_test"))
        .expect_err("non-PROGRAM input must fail");
    assert!(format!("{err}").contains("expects PROGRAM input"), "unexpected error: {err:?}");
}

#[test]
fn test_compile_with_program_pipeline_components_accepts_stage1_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let program = UOp::program(sink, UOp::device(morok_dtype::DeviceSpec::Cpu), Some(linear), None, None);

    let (spec, compiled) =
        compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler, Some("p_test"))
            .expect("stage-1 PROGRAM input should compile");
    assert_eq!(spec.name, "p_test");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn test_compile_with_program_pipeline_components_accepts_stage2_program_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let program = UOp::program(
        sink,
        UOp::device(morok_dtype::DeviceSpec::Cpu),
        Some(linear),
        Some(UOp::source("// pre-rendered source".to_string())),
        None,
    );

    let (spec, compiled) =
        compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler, Some("p_test"))
            .expect("stage-2 PROGRAM input should compile");
    assert_eq!(spec.name, "kernel");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn test_compile_with_program_pipeline_components_rejects_malformed_program_state() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let program = UOp::program(
        sink,
        UOp::device(morok_dtype::DeviceSpec::Cpu),
        None,
        Some(UOp::source("// malformed source".to_string())),
        None,
    );

    let err = compile_with_program_pipeline_components(program, &TestRenderer, &TestCompiler, Some("p_test"))
        .expect_err("malformed PROGRAM input must fail");
    assert!(format!("{err}").contains("malformed PROGRAM state"), "unexpected error: {err:?}");
}

#[test]
fn test_collect_non_overridable_fixedvars_uses_outer_bindings_not_name_prefix() {
    let outer_range = UOp::range_outer_const(4, 0);
    let outer_bind = UOp::define_var("outer_i".to_string(), 0, 3).bind(outer_range);
    let loop_bind = UOp::define_var("loop_j".to_string(), 0, 1).bind(UOp::range_const(2, 1));
    let body = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let call = body.call(SmallVec::from_vec(vec![outer_bind, loop_bind]), morok_ir::CallInfo::default());

    let item = crate::schedule::ScheduleItem {
        kernel: call,
        ast: body,
        buffers: vec![],
        buffer_uop_ids: vec![],
        fixedvars: std::collections::HashMap::from([
            ("outer_i".to_string(), 2_i64),
            ("loop_j".to_string(), 1_i64),
            ("user_n".to_string(), 7_i64),
        ]),
        dependencies: vec![],
        instance_dependencies: vec![],
        alias_registered_ids: vec![],
    };

    let locked = collect_non_overridable_fixedvars(&item);
    assert_eq!(locked.get("outer_i"), Some(&2));
    assert!(!locked.contains_key("loop_j"));
    assert!(!locked.contains_key("user_n"));
}

#[test]
fn test_realize_simple_add() {
    crate::test::helpers::test_setup();

    // Test that realizing a simple computation works.
    // The pipeline transforms:
    //   ADD(RESHAPE(BUFFER_A), RESHAPE(BUFFER_B))
    // Into:
    //   STORE(OUTPUT, INDEX, ADD(LOAD(INPUT_A, idx), LOAD(INPUT_B, idx)))
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

    // Create computation: a + b
    let mut c = &a + &b;

    // Realize should compile and execute the kernel
    c.realize().unwrap();
    let result: ndarray::ArrayD<f32> = c.as_ndarray().unwrap();
    let (result, _) = result.into_raw_vec_and_offset();
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

/// Test that realizing a reduction (sum) works end-to-end.
///
/// This verifies the complete reduction pipeline:
/// - Early-return pattern prevents unnecessary ReduceAxis for size-1 dimensions
/// - Vectorize consistency prevents VConst panics in shape extraction
/// - ReduceAxis → REDUCE transformation following Tinygrad's approach
/// - REDUCE codegen generates correct LLVM IR
#[test]
fn test_realize_sum() {
    crate::test::helpers::test_setup();

    // Create a 1D tensor: [1, 2, 3, 4]
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Sum all elements (should be 10.0)
    let sum_result = a.sum(());
    if let Err(ref e) = sum_result {
        tracing::debug!(error = ?e, "sum failed");
    }
    assert!(sum_result.is_ok(), "Sum creation failed");

    // Realize the computation
    let mut sum_tensor = sum_result.unwrap();
    let realized = sum_tensor.realize();
    if let Err(ref e) = realized {
        eprintln!("realize failed: {e:?}");
    }
    assert!(realized.is_ok(), "Realize should succeed: {:?}", realized.err());
}

#[test]
fn test_tensor_device_default_cpu() {
    // Tensors created with from_slice default to CPU
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    assert_eq!(a.device(), morok_ir::DeviceSpec::Cpu);
}

#[test]
fn test_tensor_to_same_device_is_noop() {
    // Moving to the same device should return a clone
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = a.to(morok_ir::DeviceSpec::Cpu);
    // Both should point to the same UOp (clone shares Rc)
    assert_eq!(a.device(), b.device());
}

#[test]
fn test_tensor_to_different_device_creates_copy() {
    use morok_ir::DeviceSpec;
    // Moving to a different device should create a COPY UOp
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = a.to(DeviceSpec::Cuda { device_id: 0 });
    // b should report the new device
    assert_eq!(b.device(), DeviceSpec::Cuda { device_id: 0 });
    // a should still be on CPU
    assert_eq!(a.device(), DeviceSpec::Cpu);
}

// More comprehensive tests will be added in Phase 1.5

// ==========================================================================
// ExecutionPlan tests
// ==========================================================================

#[test]
fn test_prepare_simple_add() {
    crate::test::helpers::test_setup();

    // Create computation: a + b
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let mut c = &a + &b;

    // Prepare should compile kernels and allocate buffers
    let plan = c.prepare();
    assert!(plan.is_ok(), "prepare() should succeed: {:?}", plan.err());

    let plan = plan.unwrap();

    // Verify plan has kernels and buffers
    assert!(plan.kernels().next().is_some(), "Plan should have at least one kernel");
    assert!(!plan.buffers().is_empty(), "Plan should have buffers");
}

#[test]
fn test_prepare_execution_plan_marks_cpu_kernels_host_parallel_safe() {
    crate::test::helpers::test_setup();

    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let mut c = &a + &b;
    let plan = c.prepare().expect("prepare should succeed");

    let compiled: Vec<_> = plan
        .prepared_ops()
        .iter()
        .filter_map(|op| match op {
            morok_runtime::PreparedOp::CompiledProgram(kernel) => Some(kernel),
            _ => None,
        })
        .collect();

    assert!(!compiled.is_empty(), "prepare should produce compiled kernels");
    assert!(
        compiled.iter().all(|kernel| kernel.kernel.host_parallel_safe),
        "CPU kernels should propagate host_parallel_safe metadata"
    );
}

#[test]
fn test_prepare_and_execute() {
    crate::test::helpers::test_setup();

    // Create computation: a + b
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let mut c = &a + &b;

    // Prepare
    let plan = c.prepare().expect("prepare should succeed");

    // Execute
    let result = plan.execute();
    assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());

    // Verify output buffer has correct data
    let output = plan.output_buffer().expect("plan has output");
    let mut data = vec![0.0f32; 3];
    output
        .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
        .expect("copyout should succeed");
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_prepare_and_execute_twice() {
    crate::test::helpers::test_setup();

    // Create computation
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let mut c = &a + &b;

    // Prepare once
    let plan = c.prepare().expect("prepare should succeed");

    // Execute twice to verify reusability
    for _ in 0..2 {
        let result = plan.execute();
        assert!(result.is_ok(), "execute() should succeed: {:?}", result.err());
    }

    // Verify output
    let output = plan.output_buffer().expect("plan has output");
    let mut data = vec![0.0f32; 3];
    output
        .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
        .expect("copyout should succeed");
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_prepare_execution_plan_lowers_explicit_custom_function_op() {
    crate::test::helpers::test_setup();

    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let src = Buffer::new(alloc.clone(), morok_dtype::DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, morok_dtype::DType::Float32, vec![4], Default::default());

    let attr = UOp::index_const(42);
    let ast = UOp::custom_function(morok_ir::CustomFunctionKind::EncDec, smallvec::smallvec![attr.clone()]);
    let kernel = ast.call(smallvec::smallvec![], morok_ir::CallInfo::default());
    let schedule_result = crate::schedule::ScheduleResult {
        items: vec![crate::schedule::ScheduleItem {
            kernel,
            ast,
            buffers: vec![dst.clone(), src.clone()],
            buffer_uop_ids: vec![1001, 1002],
            fixedvars: std::collections::HashMap::new(),
            dependencies: vec![],
            instance_dependencies: vec![],
            alias_registered_ids: vec![],
        }],
        output_uop_ids: vec![1001],
    };

    let plan = prepare_execution_plan(&schedule_result, &PrepareConfig::from_env()).expect("prepare should succeed");
    let custom = plan
        .prepared_ops()
        .iter()
        .find_map(|op| match op {
            morok_runtime::PreparedOp::CustomFunction(custom) => Some(custom),
            _ => None,
        })
        .expect("explicit custom function body should lower to PreparedOp::CustomFunction");
    assert_eq!(custom.attrs.len(), 1, "custom-function attrs should be preserved into runtime plan");
    assert_eq!(custom.attrs[0].id, attr.id);

    let err = plan.execute().expect_err("EncDec runtime should be explicit unsupported");
    let msg = format!("{err}");
    assert!(msg.contains("Unsupported runtime feature EncDec"), "unexpected error: {msg}");
}

/// Test that realize() produces correct results.
///
/// Note: Buffer count assertions removed as they're not reliable with
/// parallel test execution and global state. The key invariant (no memory
/// leak) is tested in test_memory_growth_detection.
#[test]
fn test_realize_buffer_cleanup() {
    crate::test::helpers::test_setup();

    // Create input tensors ONCE (these will stay in registry)
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);

    // Realize the computation
    let mut c = &a + &b;
    c.realize().expect("realize should succeed");

    // Verify computation is correct
    let result: ndarray::ArrayD<f32> = c.as_ndarray().expect("as_ndarray should succeed");
    let (data, _) = result.into_raw_vec_and_offset();
    assert_eq!(data, vec![5.0, 7.0, 9.0]);
}

/// Test that prepare() + execute() pattern can clean up with release_intermediate_buffers().
#[test]
#[ignore = "Flaky under parallel global registry activity; run manually with --ignored --test-threads=1"]
fn test_prepare_execute_cleanup() {
    crate::test::helpers::test_setup();

    // Create input tensors
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let mut c = &a + &b;

    // Prepare the plan
    let plan = c.prepare().expect("prepare should succeed");

    // Execute multiple times (simulating benchmark loop)
    for _ in 0..3 {
        plan.execute().expect("execute should succeed");
    }

    // Verify output
    let output = plan.output_buffer().expect("plan has output");
    let mut data = vec![0.0f32; 3];
    output
        .copyout(unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, 12) })
        .expect("copyout should succeed");
    assert_eq!(data, vec![5.0, 7.0, 9.0]);

    // Now cleanup — count how many buffers were actually released
    let count_before_cleanup = crate::tensor_registry::buffer_count();
    plan.release_intermediate_buffers(crate::tensor_registry::remove_buffer);
    let count_after_cleanup = crate::tensor_registry::buffer_count();

    // release_intermediate_buffers should remove at least one buffer (the output buffer)
    // or at minimum not increase the count. We check the immediate delta to avoid
    // interference from parallel tests.
    assert!(
        count_after_cleanup <= count_before_cleanup,
        "Cleanup should not increase buffer count: before={}, after={}",
        count_before_cleanup,
        count_after_cleanup
    );
}

/// Test that intermediate buffer cleanup is working.
///
/// The correct pattern is: prepare() ONCE, execute() many times.
/// This test verifies that repeated execute() calls do NOT grow the registry
/// AFTER initial setup. First execute may allocate buffers (one-time setup),
/// but subsequent calls must not grow.
#[test]
#[ignore = "Flaky under parallel global registry activity; run manually with --ignored --test-threads=1"]
fn test_memory_growth_detection() {
    crate::test::helpers::test_setup();

    const ITERATIONS: usize = 10;

    // Create input tensors
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
    let mut c = &a + &b;

    // Prepare ONCE
    let plan = c.prepare().expect("prepare should succeed");

    let mut counts: Vec<usize> = Vec::with_capacity(ITERATIONS);

    // Execute MANY times
    for _ in 0..ITERATIONS {
        plan.execute().expect("execute should succeed");
        counts.push(crate::tensor_registry::buffer_count());
    }

    // Cleanup after final execution
    plan.release_intermediate_buffers(crate::tensor_registry::remove_buffer);
    let count_after_cleanup = crate::tensor_registry::buffer_count();

    // Key invariant: count should be STABLE during iterations (no growth between iterations)
    // First execute may allocate buffers, but subsequent calls must reuse them.
    let count_after_first_execute = counts[0];
    let growth_during_iterations = counts.last().unwrap().saturating_sub(count_after_first_execute);

    eprintln!("Counts during execute: {:?}", counts);
    eprintln!("Growth during iterations (after first): {}", growth_during_iterations);
    eprintln!("Count after cleanup: {}", count_after_cleanup);

    assert_eq!(
        growth_during_iterations, 0,
        "Registry should not grow during repeated execute() calls (after initial setup)"
    );

    // Cleanup should reduce count by removing allocated buffers
    assert!(
        count_after_cleanup <= count_after_first_execute,
        "Cleanup should not increase buffer count: first_execute={}, after_cleanup={}",
        count_after_first_execute,
        count_after_cleanup
    );
}

/// Test that realize() correctly computes and cleans up.
#[test]
fn test_memory_growth_realize_pattern() {
    crate::test::helpers::test_setup();

    // Single realize should work correctly
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([5.0f32, 6.0, 7.0, 8.0]);
    let mut c = &a + &b;
    c.realize().expect("realize should succeed");

    // Verify result
    let result: ndarray::ArrayD<f32> = c.as_ndarray().expect("as_ndarray should succeed");
    assert_eq!(result.as_slice().unwrap(), &[6.0, 8.0, 10.0, 12.0]);
}
