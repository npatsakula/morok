//! CPU device integration tests.
//!
//! Tests the full Device pipeline: render → compile → runtime factory → execute

use svod_device::device::{AbiParamDescriptor, AbiParamKind, ProgramSpec};
use svod_device::registry::DeviceRegistry;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::UOp;

use crate::devices::cpu::{CpuBackend, create_cpu_device_with_backend};

fn storage(slot: usize) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Int32, name: None }
}

fn scalar(slot: usize, name: &str) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some(name.into()) }
}

/// The LLVM backend hands the runtime reusable relocatable object bytes rather
/// than source, and each backend gets its own compiler cache key.
#[test]
fn llvm_jit_emits_reusable_object_bytes() {
    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();
    let spec = ProgramSpec::new(
        "source_only".into(),
        "define void @source_only() { ret void }".into(),
        DeviceSpec::Cpu,
        UOp::sink(vec![]),
    );

    let compiled = device.compiler.compile(&spec).unwrap();
    assert!(compiled.src.is_none(), "LLVM compiler should hand reusable object bytes to the runtime");
    assert!(!compiled.bytes.is_empty());
    crate::clang::validate_relocatable_object(&compiled.bytes, "source_only").unwrap();
    assert_eq!(device.base_device_key(), "CPU");
    let producer = if crate::llvm_inprocess::library().is_ok() { "cpu-llvm-inprocess:" } else { "cpu-llvm-clang:" };
    assert!(device.compiler.cache_key().starts_with(producer), "{}", device.compiler.cache_key());

    let invalid =
        ProgramSpec::new("test".into(), "this is not valid LLVM IR".into(), DeviceSpec::Cpu, UOp::sink(vec![]));
    assert!(device.compiler.compile(&invalid).is_err(), "invalid LLVM IR must fail during object compilation");
}

/// Only compiler output that carries a semantic PROGRAM stage identity may
/// reach the runtime: no identity, a count-only ABI, or tampered bytes must all
/// be refused before the JIT is created.
#[test]
fn cpu_runtime_loads_only_authenticated_compiler_output() {
    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();

    let direct = device
        .compiler
        .compile(&ProgramSpec::new(
            "test_kernel".into(),
            "define void @test_kernel() {\nentry:\n  ret void\n}\n".into(),
            DeviceSpec::Cpu,
            UOp::sink(vec![]),
        ))
        .expect("direct-source compilation remains supported");
    assert_eq!(direct.name, "test_kernel");
    let Err(err) = (device.runtime)(&direct) else { panic!("identity-less output must not reach the runtime") };
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { .. }), "{err:?}");

    let mut count_only = svod_device::device::CompiledSpec::from_source(
        "bad_abi".into(),
        "define void @bad_abi(ptr %data) { ret void }".into(),
        UOp::sink(vec![]),
        vec![],
    )
    .unwrap();
    count_only.buf_count = 1;
    let Err(err) = (device.runtime)(&count_only) else { panic!("count-only ABI must be rejected before JIT creation") };
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { .. }), "{err:?}");

    let staged = svod_codegen::program_pipeline::program_from_sink(UOp::sink(vec![]), DeviceSpec::Cpu).unwrap();
    let (staged, _) = svod_codegen::program_pipeline::do_render(&staged, device.renderer.as_ref()).unwrap();
    let (_, mut compiled) = svod_codegen::program_pipeline::do_compile(&staged, device.compiler.as_ref()).unwrap();
    let program = (device.runtime)(&compiled).expect("validated RuntimeFactory should succeed");
    assert!(!program.name().is_empty());
    unsafe { program.execute(&[], &[], None, None, true).expect("execution should succeed") };

    compiled.bytes.push(0);
    let Err(err) = (device.runtime)(&compiled) else { panic!("tampered output must not reach the runtime") };
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");
}

/// A second process with a warm `SVOD_OBJECT_CACHE_DIR` must serve the object
/// from disk instead of shelling out to clang again.
#[cfg(unix)]
#[test]
fn clang_object_cache_survives_fresh_process_without_invoking_clang() {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    const HELPER: &str = "SVOD_TEST_CLANG_CACHE_CHILD";
    if std::env::var_os(HELPER).is_some() {
        let registry = DeviceRegistry::default();
        let device = create_cpu_device_with_backend(&registry, CpuBackend::Clang).unwrap();
        let spec = ProgramSpec::new(
            "fresh_process_kernel".into(),
            "void fresh_process_kernel(float *out) { out[0] = 7.0f; }\n".into(),
            DeviceSpec::Cpu,
            UOp::sink(vec![]),
        );
        assert!(!device.compiler.compile(&spec).unwrap().bytes.is_empty());
        return;
    }

    let directory = tempfile::tempdir().unwrap();
    let bin = directory.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    let count = directory.path().join("clang-invocations");
    let real_clang = Command::new("sh").args(["-c", "command -v clang"]).output().unwrap();
    assert!(real_clang.status.success());
    let real_clang = std::fs::canonicalize(String::from_utf8(real_clang.stdout).unwrap().trim()).unwrap();
    let wrapper = bin.join("clang");
    std::fs::write(
        &wrapper,
        format!("#!/bin/sh\nprintf 'invoked\\n' >> '{}'\nexec '{}' \"$@\"\n", count.display(), real_clang.display()),
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&wrapper).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&wrapper, permissions).unwrap();

    let test_name = std::thread::current().name().unwrap().to_string();
    let executable = std::env::current_exe().unwrap();
    let mut paths = vec![bin];
    paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap()));
    let path = std::env::join_paths(paths).unwrap();
    let run_child = || {
        Command::new(&executable)
            .args(["--exact", &test_name, "--nocapture"])
            .env(HELPER, "1")
            .env("PATH", &path)
            .env("SVOD_OBJECT_CACHE_DIR", directory.path().join("cache"))
            .env("SVOD_OBJECT_CACHE_MAX_BYTES", "10485760")
            .status()
            .unwrap()
    };

    assert!(run_child().success());
    let cold_count = std::fs::read_to_string(&count).unwrap().lines().count();
    assert!(cold_count >= 3, "cold process must probe version/target and compile");
    assert!(run_child().success());
    let warm_count = std::fs::read_to_string(&count).unwrap().lines().count();
    assert_eq!(warm_count, cold_count, "warm fresh process invoked clang");
}

/// Storage and scalar arguments bind by their declared ABI slot, not by
/// position: interleaved kinds and sparse slot numbers both reach the right
/// formal, and a buffer list that is too short is a typed ABI mismatch.
#[test]
fn cpu_dispatch_binds_arguments_by_abi_slot() {
    let kernel = crate::jit_loader::JitKernel::compile_with_abi(
        "void interleaved(int *data0, int data1, int *data2, int data3) { *data0 = data1; *data2 = data3; }",
        "interleaved",
        vec!["low".into(), "high".into()],
        &[storage(0), scalar(1, "low"), storage(2), scalar(3, "high")],
    )
    .expect("compile interleaved ABI fixture");
    let (mut low, mut high) = (0i32, 0i32);
    let buffers = [(&mut low as *mut i32).cast::<u8>(), (&mut high as *mut i32).cast::<u8>()];
    let err = unsafe { kernel.execute_with_vals(&buffers[..1], &[17, -9]) }
        .expect_err("runtime ABI arity mismatch must be typed");
    assert!(matches!(err, crate::Error::Device { source: svod_device::Error::ProgramAbiMismatch { .. } }), "{err:?}");
    unsafe { kernel.execute_with_vals(&buffers, &[17, -9]).expect("execute interleaved ABI fixture") };
    assert_eq!((low, high), (17, -9));

    let kernel = crate::jit_loader::JitKernel::compile_with_abi(
        "void sparse(int *data0, int *data5) { *data0 = 17; *data5 = -9; }",
        "sparse",
        vec![],
        &[storage(0), storage(5)],
    )
    .expect("compile sparse storage ABI fixture");
    let (mut first, mut second) = (0i32, 0i32);
    let buffers = [(&mut first as *mut i32).cast::<u8>(), (&mut second as *mut i32).cast::<u8>()];
    unsafe { kernel.execute_with_vals(&buffers, &[]).unwrap() };
    assert_eq!((first, second), (17, -9));
}

/// libffi reads a `Type::i32()` argument through a pointer at the declared
/// width. Packing scalars into `u64` slots only worked because a little-endian
/// read takes the low half; the typed slots make it endian-independent.
#[test_case::test_case(1)]
#[test_case::test_case(-7)]
#[test_case::test_case(i32::MAX)]
#[test_case::test_case(i32::MIN)]
fn cpu_dispatch_passes_scalars_at_their_declared_width(value: i32) {
    let kernel = crate::jit_loader::JitKernel::compile_with_abi(
        "void scalar_width(int *out, int n) { out[0] = n; }",
        "scalar_width",
        vec!["n".into()],
        &[storage(0), scalar(1, "n")],
    )
    .expect("compile scalar-width fixture");

    let mut out = 0i32;
    let buffers = [(&mut out as *mut i32).cast::<u8>()];
    unsafe { kernel.execute_with_vals(&buffers, &[value as i64]).expect("execute scalar-width fixture") };
    assert_eq!(out, value);
}

#[test]
fn cpu_device_is_memoized_per_backend() {
    use crate::devices::cpu::cpu_device_with_backend;
    use std::sync::Arc;

    let registry = svod_device::registry::registry();
    let clang = cpu_device_with_backend(registry, CpuBackend::Clang).expect("clang device");
    let clang_again = cpu_device_with_backend(registry, CpuBackend::Clang).expect("clang device");
    assert!(Arc::ptr_eq(&clang, &clang_again), "same backend must reuse one device");

    let llvm = cpu_device_with_backend(registry, CpuBackend::Llvm).expect("llvm device");
    assert!(!Arc::ptr_eq(&clang, &llvm), "distinct backends must get distinct devices");
    assert_ne!(clang.compiler.cache_key(), llvm.compiler.cache_key());

    // A non-global allocator registry bypasses the cache: the cached device
    // holds the allocators of the registry it was built with.
    let local = DeviceRegistry::default();
    let fresh = cpu_device_with_backend(&local, CpuBackend::Clang).expect("local clang device");
    assert!(!Arc::ptr_eq(&clang, &fresh));
}
