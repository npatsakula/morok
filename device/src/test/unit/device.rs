use crate::{DeviceSpec, registry::DeviceSpecExt};
use morok_dtype::DType;
use morok_ir::{Op, UOp};

#[test]
fn test_device_spec_parse() {
    assert_eq!(DeviceSpec::parse("CPU").unwrap(), DeviceSpec::Cpu);
    assert_eq!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu);

    #[cfg(feature = "cuda")]
    {
        assert_eq!(DeviceSpec::parse("CUDA:0").unwrap(), DeviceSpec::Cuda { device_id: 0 });
        assert_eq!(DeviceSpec::parse("cuda").unwrap(), DeviceSpec::Cuda { device_id: 0 });
        assert_eq!(DeviceSpec::parse("GPU:2").unwrap(), DeviceSpec::Cuda { device_id: 2 });
    }
}

#[test]
fn test_device_spec_canonicalize() {
    assert_eq!(DeviceSpec::Cpu.canonicalize(), "CPU");

    #[cfg(feature = "cuda")]
    {
        assert_eq!(DeviceSpec::Cuda { device_id: 1 }.canonicalize(), "CUDA:1");
    }
}

#[test]
fn test_program_spec_from_uop_with_metadata() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let device = UOp::device(DeviceSpec::Cpu);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// test kernel".to_string());
    let program = UOp::program(sink.clone(), device, Some(linear), Some(source), None);

    let mut spec =
        crate::device::ProgramSpec::new("k_test".to_string(), "// old src".to_string(), DeviceSpec::Cpu, sink.clone());
    spec.buf_count = 2;

    let program = program.with_metadata(spec.clone());
    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("program spec from uop");

    assert_eq!(rebuilt.name, "k_test");
    assert_eq!(rebuilt.src, "// test kernel");
    assert_eq!(rebuilt.device, DeviceSpec::Cpu);
    assert_eq!(rebuilt.ast.id, sink.id);
    assert!(rebuilt.var_names.is_empty());
    assert!(rebuilt.globals.is_empty());
    assert!(rebuilt.outs.is_empty());
    assert!(rebuilt.ins.is_empty());
    assert_eq!(rebuilt.buf_count, 2);
}

#[test]
fn test_program_spec_from_uop_preserves_metadata_io() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let device = UOp::device(DeviceSpec::Cpu);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// test kernel".to_string());
    let program = UOp::program(sink.clone(), device, Some(linear), Some(source), None);

    let mut spec =
        crate::device::ProgramSpec::new("k_test".to_string(), "// old src".to_string(), DeviceSpec::Cpu, sink);
    spec.set_buffer_metadata(vec![1, 0], vec![1], vec![0]);
    spec.buf_count = 2;

    let rebuilt = crate::device::ProgramSpec::from_uop(&program.with_metadata(spec)).expect("program spec from uop");
    assert_eq!(rebuilt.globals, vec![1, 0]);
    assert_eq!(rebuilt.outs, vec![1]);
    assert_eq!(rebuilt.ins, vec![0]);
    assert_eq!(rebuilt.buf_count, 2);
}

#[test]
fn test_program_spec_from_uop_without_metadata_derives_name_and_vars() {
    let var = UOp::define_var("N".to_string(), 1, 8);
    let sink = UOp::sink(vec![var]);
    let device = UOp::device(DeviceSpec::Cpu);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void var_kernel(float* data0) {}".to_string());
    let program = UOp::program(sink.clone(), device, Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should succeed");
    assert_eq!(rebuilt.name, "kernel");
    assert_eq!(rebuilt.var_names, vec!["N".to_string()]);
    assert_eq!(rebuilt.vars.len(), 1);
}

#[test]
fn test_program_spec_derives_launch_dims_from_specials() {
    let g = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let l = UOp::special(UOp::index_const(4), "lidx0".to_string());
    let sink = UOp::sink(vec![g, l]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void launch_kernel() {}".to_string());
    let program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(source), None);

    let spec = crate::device::ProgramSpec::from_uop(&program).expect("program spec from specials");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [8, 1, 1]);
    assert_eq!(launch.local_size, Some([4, 1, 1]));
}

#[test]
fn test_program_spec_direct_global_special_disables_local_size() {
    let idx = UOp::special(UOp::index_const(16), "idx0".to_string());
    let sink = UOp::sink(vec![idx]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void direct_global_kernel() {}".to_string());
    let program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(source), None);

    let spec = crate::device::ProgramSpec::from_uop(&program).expect("program spec from idx special");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [16, 1, 1]);
    assert_eq!(launch.local_size, None);
}

#[test]
fn test_program_spec_core_id_sets_cpu_global_size() {
    let core_id = UOp::define_var("core_id".to_string(), 0, 7);
    let sink = UOp::sink(vec![core_id]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void core_kernel(int core_id) {}".to_string());
    let program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(source), None);

    let spec = crate::device::ProgramSpec::from_uop(&program).expect("program spec from core_id");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [8, 1, 1]);
}

#[test]
fn test_program_spec_default_metadata_launch_dims_do_not_hide_derived_core_id() {
    let core_id = UOp::define_var("core_id".to_string(), 0, 3);
    let sink = UOp::sink(vec![core_id]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void core_kernel(int core_id) {}".to_string());
    let program = UOp::program(sink.clone(), UOp::device(DeviceSpec::Cpu), Some(linear), Some(source), None);
    let meta = crate::device::ProgramSpec::new("core".to_string(), "// old".to_string(), DeviceSpec::Cpu, sink);

    let spec = crate::device::ProgramSpec::from_uop(&program.with_metadata(meta)).expect("program spec from metadata");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [4, 1, 1]);
}

#[test]
fn test_program_spec_from_uop_without_metadata_derives_buf_count_and_io() {
    let param = UOp::param(0, 16, DType::Float32, None);
    let idx = UOp::index_const(0);
    let load_idx = UOp::index().buffer(param.clone()).indices(vec![idx.clone()]).call().expect("load index");
    let load = UOp::load().buffer(param.clone()).index(load_idx).call();
    let store_idx = UOp::index().buffer(param).indices(vec![idx]).call().expect("store index");
    let sink = UOp::sink(vec![store_idx.store(load)]);
    let device = UOp::device(DeviceSpec::Cpu);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void io_kernel(float* data0) {}".to_string());
    let program = UOp::program(sink.clone(), device, Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should derive I/O");
    assert_eq!(rebuilt.globals, vec![0]);
    assert_eq!(rebuilt.outs, vec![0]);
    assert_eq!(rebuilt.ins, vec![0]);
    assert_eq!(rebuilt.buf_count, 1);
}

#[test]
fn test_program_spec_from_uop_requires_program_source() {
    let sink = UOp::sink(vec![UOp::native_const(3.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let program_without_source = UOp::program(sink.clone(), UOp::device(DeviceSpec::Cpu), Some(linear), None, None);
    assert!(crate::device::ProgramSpec::from_uop(&program_without_source).is_err());

    let non_program = UOp::native_const(1.0f32);
    assert!(crate::device::ProgramSpec::from_uop(&non_program).is_err());

    let bad_source = UOp::native_const(1.0f32);
    let linear = UOp::linear(sink.toposort().into());
    let bad_program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(bad_source), None);
    assert!(crate::device::ProgramSpec::from_uop(&bad_program).is_err());

    if let Op::Program { .. } = bad_program.op() {
        // ensure we exercised Program path in this test
    } else {
        panic!("expected PROGRAM op");
    }
}

#[test]
fn test_program_spec_from_uop_binary_stage_with_metadata() {
    let sink = UOp::sink(vec![UOp::native_const(4.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// binary source".to_string());
    let program = UOp::program(
        sink.clone(),
        UOp::device(DeviceSpec::Cpu),
        Some(linear),
        Some(source),
        Some(UOp::binary(vec![1, 2, 3])),
    );

    let mut spec =
        crate::device::ProgramSpec::new("precompiled".to_string(), "// cached src".to_string(), DeviceSpec::Cpu, sink);
    spec.set_var_names(vec!["N".to_string()]);
    spec.buf_count = 3;

    let program = program.with_metadata(spec);
    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("program spec from binary+metadata");
    assert_eq!(rebuilt.name, "precompiled");
    assert_eq!(rebuilt.src, "// binary source");
    assert_eq!(rebuilt.var_names, vec!["N".to_string()]);
    assert_eq!(rebuilt.buf_count, 3);
}

#[test]
fn test_program_spec_from_uop_without_metadata_defaults_name_to_kernel() {
    let sink = UOp::sink(vec![UOp::native_const(4.5f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void default_name_kernel() {}".to_string());
    let program = UOp::program(sink, UOp::device(DeviceSpec::Cpu), Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should succeed");
    assert_eq!(rebuilt.name, "kernel");
}

#[test]
fn test_program_spec_apply_derived_metadata_from_ast() {
    let p0 = UOp::param(0, 16, DType::Float32, None);
    let p1 = UOp::param(1, 16, DType::Float32, None);
    let sink = UOp::sink(vec![p1, p0]);

    let mut spec =
        crate::device::ProgramSpec::new("derived_meta".to_string(), "// src".to_string(), DeviceSpec::Cpu, sink);
    spec.apply_derived_metadata_from_ast();

    assert_eq!(spec.globals, vec![0, 1]);
    assert!(spec.outs.is_empty());
    assert!(spec.ins.is_empty());
    assert_eq!(spec.buf_count, 2);
}
