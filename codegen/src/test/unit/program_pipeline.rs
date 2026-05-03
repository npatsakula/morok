use morok_device::device::{CompiledSpec, Compiler, ProgramSpec, Renderer};
use morok_dtype::{AddrSpace, DType, DeviceSpec};
use morok_ir::{Op, UOp};

struct MockRenderer {
    device: DeviceSpec,
}

struct LinearOnlyRenderer {
    device: DeviceSpec,
}

impl Renderer for LinearOnlyRenderer {
    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> morok_device::Result<ProgramSpec> {
        assert!(matches!(ast.op(), Op::Linear { .. }), "renderer should receive LINEAR stage");
        let spec = ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// linear source".to_string(),
            self.device.clone(),
            ast.clone(),
        );
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

impl Renderer for MockRenderer {
    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> morok_device::Result<ProgramSpec> {
        let mut spec = ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// mock source".to_string(),
            self.device.clone(),
            ast.clone(),
        );
        spec.set_var_names(vec!["N".to_string()]);
        spec.set_buffer_metadata(vec![0], vec![0], vec![]);
        spec.buf_count = 1;
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

struct MockCompiler;

impl Compiler for MockCompiler {
    fn compile(&self, spec: &ProgramSpec) -> morok_device::Result<CompiledSpec> {
        let mut compiled = CompiledSpec::from_bytes(spec.name.clone(), vec![1, 2, 3], spec.ast.clone());
        compiled.var_names = spec.var_names.clone();
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        compiled.buf_count = spec.buf_count;
        Ok(compiled)
    }

    fn cache_key(&self) -> &'static str {
        "mock"
    }
}

struct PanicCompiler;

impl Compiler for PanicCompiler {
    fn compile(&self, _spec: &ProgramSpec) -> morok_device::Result<CompiledSpec> {
        panic!("compiler should not be invoked when PROGRAM already has BINARY")
    }

    fn cache_key(&self) -> &'static str {
        "panic"
    }
}

#[test]
fn test_program_pipeline_sets_all_stages() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let renderer = MockRenderer { device: DeviceSpec::Cpu };
    let compiler = MockCompiler;

    let program = crate::program_pipeline::program_from_sink(sink.clone(), DeviceSpec::Cpu);
    let (program, rendered_spec) =
        crate::program_pipeline::do_render(&program, &renderer, Some("k_test")).expect("render stage");
    let (program, compiled) = crate::program_pipeline::do_compile(&program, &compiler).expect("compile stage");
    let spec = ProgramSpec::from_uop(&program).expect("ProgramSpec::from_uop");

    match program.op() {
        Op::Program { linear, source, binary, .. } => {
            assert!(linear.is_some(), "LINEAR stage missing");
            assert!(source.is_some(), "SOURCE stage missing");
            assert!(binary.is_some(), "BINARY stage missing");
        }
        other => panic!("expected PROGRAM op, got {other:?}"),
    }

    assert_eq!(rendered_spec.name, "k_test");
    assert_eq!(spec.name, "k_test");
    assert_eq!(spec.src, "// mock source");
    assert_eq!(spec.ast.id, sink.id);
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
    assert_eq!(compiled.name, "k_test");
    assert_eq!(spec.globals, vec![0]);
    assert_eq!(spec.outs, vec![0]);
    assert!(spec.ins.is_empty());
}

#[test]
fn test_do_compile_requires_source_stage() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu);
    let compiler = MockCompiler;

    let err = crate::program_pipeline::do_compile(&program, &compiler).expect_err("compile should fail without source");
    assert!(format!("{err}").contains("PROGRAM has no SOURCE stage"));
}

#[test]
fn test_do_compile_reuses_existing_binary_stage() {
    let sink = UOp::sink(vec![UOp::native_const(2.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let source = UOp::source("// binary source".to_string());
    let mut spec =
        ProgramSpec::new("precompiled".to_string(), "// precompiled source".to_string(), DeviceSpec::Cpu, sink.clone());
    spec.set_var_names(vec!["N".to_string()]);
    spec.buf_count = 3;
    spec.set_work_sizes([4, 1, 1], [1, 1, 1]);

    let program =
        UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(source), Some(UOp::binary(vec![9, 8, 7])))
            .with_metadata(spec.clone());

    let (compiled_program, compiled) =
        crate::program_pipeline::do_compile(&program, &PanicCompiler).expect("binary stage should be reused");

    assert!(std::sync::Arc::ptr_eq(&compiled_program, &program));
    assert_eq!(compiled.name, "precompiled");
    assert_eq!(compiled.bytes, vec![9, 8, 7]);
    assert_eq!(compiled.src.as_deref(), Some("// binary source"));
    assert_eq!(compiled.var_names, vec!["N".to_string()]);
    assert_eq!(compiled.buf_count, 3);
    let vars = std::collections::HashMap::new();
    let launch = ProgramSpec::resolve_launch_dims(&compiled.global_size, compiled.local_size.as_ref(), &vars)
        .expect("resolve launch dims");
    assert_eq!(launch.global_size, [4, 1, 1]);

    let rebuilt =
        ProgramSpec::from_uop(&compiled_program).expect("ProgramSpec::from_uop should support binary+metadata");
    assert_eq!(rebuilt.name, "precompiled");
    assert_eq!(rebuilt.src, "// binary source");
}

#[test]
fn test_do_render_uses_linear_stage_input() {
    let sink = UOp::sink(vec![UOp::native_const(5.0f32)]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu);
    let renderer = LinearOnlyRenderer { device: DeviceSpec::Cpu };

    let (rendered_program, spec) =
        crate::program_pipeline::do_render(&program, &renderer, Some("k_linear")).expect("render stage should succeed");

    assert_eq!(spec.name, "k_linear");
    assert!(matches!(spec.ast.op(), Op::Linear { .. }));
    match rendered_program.op() {
        Op::Program { linear, source, .. } => {
            assert!(linear.is_some(), "LINEAR stage should be present");
            assert!(source.is_some(), "SOURCE stage should be present");
        }
        other => panic!("expected PROGRAM op, got {other:?}"),
    }
}

#[test]
fn test_do_render_rejects_program_with_existing_source_stage() {
    let sink = UOp::sink(vec![UOp::native_const(8.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let program = UOp::program(
        sink,
        UOp::device(DeviceSpec::Cpu),
        Some(linear),
        Some(UOp::source("// stale source".to_string())),
        None,
    );
    let renderer = MockRenderer { device: DeviceSpec::Cpu };

    let err = crate::program_pipeline::do_render(&program, &renderer, Some("k_rerender"))
        .expect_err("render should reject programs that already have SOURCE stage");
    assert!(format!("{err}").contains("LINEAR only"));
}

#[test]
fn test_do_render_rejects_program_with_existing_binary_stage() {
    let sink = UOp::sink(vec![UOp::native_const(9.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let program =
        UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), None, Some(UOp::binary(vec![1, 2, 3])));
    let renderer = MockRenderer { device: DeviceSpec::Cpu };

    let err = crate::program_pipeline::do_render(&program, &renderer, Some("k_rerender_binary"))
        .expect_err("render should reject programs that already have BINARY stage");
    let msg = format!("{err}");
    assert!(msg.contains("LINEAR only") || msg.contains("BINARY requires SOURCE"), "unexpected error: {msg}");
}

#[test]
fn test_do_compile_rejects_malformed_binary_stage() {
    let sink = UOp::sink(vec![UOp::native_const(6.0f32)]);
    let program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), None, None, Some(UOp::native_const(1.0f32)));

    let err = crate::program_pipeline::do_compile(&program, &MockCompiler)
        .expect_err("compile should fail when binary stage is not ProgramBinary");
    assert!(format!("{err}").contains("ProgramBinary"));
}

#[test]
fn test_do_compile_rejects_empty_source_stage() {
    let sink = UOp::sink(vec![UOp::native_const(7.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let mut meta = ProgramSpec::new("empty_source".to_string(), String::new(), DeviceSpec::Cpu, sink.clone());
    meta.set_var_names(vec!["N".to_string()]);
    meta.buf_count = 1;

    let program =
        UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(UOp::source(String::new())), None)
            .with_metadata(meta);

    let err = crate::program_pipeline::do_compile(&program, &MockCompiler)
        .expect_err("compile should fail when SOURCE stage is empty");
    assert!(format!("{err}").contains("empty SOURCE stage"));
}

#[test]
fn test_do_linearize_emits_cleaned_linear_stage() {
    let ptr_dtype = DType::Float32.ptr(None, AddrSpace::Global);
    let out = UOp::param(0, 16, ptr_dtype, None);
    let idx = UOp::index_const(0);
    let gate = UOp::native_const(true);
    let out_index = UOp::index().buffer(out).indices(vec![idx]).gate(gate).call().expect("gated index");
    let store = out_index.store(UOp::native_const(1.0f32));
    let sink = UOp::sink(vec![store]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu);

    let linearized = crate::program_pipeline::do_linearize(&program).expect("linearize stage should succeed");

    let Op::Program { linear: Some(linear), .. } = linearized.op() else {
        panic!("expected PROGRAM with LINEAR stage");
    };
    let Op::Linear { ops } = linear.op() else {
        panic!("expected LINEAR payload");
    };

    assert!(ops.iter().any(|u| matches!(u.op(), Op::If { .. })), "expected IF from cleanup");
    assert!(ops.iter().any(|u| matches!(u.op(), Op::EndIf { .. })), "expected ENDIF from cleanup");
    assert!(ops.iter().any(|u| {
        if let Op::Store { index, .. } = u.op() {
            matches!(index.op(), Op::Index { gate, .. } if gate.is_none())
        } else {
            false
        }
    }));
}

#[test]
fn test_get_program_progresses_from_stage1_to_binary() {
    let sink = UOp::sink(vec![UOp::native_const(3.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), None, None);

    let advanced = crate::program_pipeline::get_program(
        &program,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        Some("k_stage1"),
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect("stage-1 PROGRAM should advance to BINARY");

    match advanced.op() {
        Op::Program { linear, source, binary, .. } => {
            assert!(linear.is_some());
            assert!(source.is_some());
            assert!(binary.is_some());
        }
        other => panic!("expected PROGRAM op, got {other:?}"),
    }
}

#[test]
fn test_get_program_progresses_from_stage2_to_binary() {
    let sink = UOp::sink(vec![UOp::native_const(4.0f32)]);
    let linear = UOp::linear(morok_schedule::linearize_with_cfg(sink.clone()).into());
    let source = UOp::source("// pre-rendered source".to_string());
    let program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(source), None);

    let advanced = crate::program_pipeline::get_program(
        &program,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        Some("k_stage2"),
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect("stage-2 PROGRAM should advance to BINARY");

    let (_, compiled) = crate::program_pipeline::do_compile(&advanced, &PanicCompiler)
        .expect("binary stage should be reusable after get_program");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn test_get_program_rejects_malformed_staged_program() {
    let sink = UOp::sink(vec![UOp::native_const(5.0f32)]);
    let malformed = UOp::program(
        sink,
        UOp::device(DeviceSpec::Cpu),
        None,
        Some(UOp::source("// source without linear".to_string())),
        None,
    );

    let err = crate::program_pipeline::get_program(
        &malformed,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        Some("k_bad"),
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect_err("malformed PROGRAM state must be rejected");

    assert!(format!("{err}").contains("malformed PROGRAM state"));
}

#[test]
fn test_get_program_rejects_sink_input() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);

    let err = crate::program_pipeline::get_program(
        &sink,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        Some("k_sink"),
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect_err("SINK input should be rejected by strict staged PROGRAM pipeline");

    assert!(format!("{err}").contains("expected PROGRAM input"));
}
