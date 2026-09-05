//! Metal device integration tests: factory registration, compile → load
//! through the runtime glue, and compiler identity stability. Hardware tests
//! self-skip on hosts without a Metal device.

use svod_device::Allocator;
use svod_device::device::{AbiParamDescriptor, AbiParamKind, ProgramSpec};
use svod_device::metal::{MetalDevice, has_devices};
use svod_device::registry::DeviceRegistry;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::UOp;

use crate::devices::metal::{create_metal_codegen, create_metal_device, create_metal_program};

const VADD_MSL: &str = "#include <metal_stdlib>\nusing namespace metal;\n\
kernel void vadd(device float* data0, device float* data1, device float* data2, \
uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {\n\
  uint i = gid.x * 32 + lid.x;\n  data0[i] = data1[i] + data2[i];\n}\n";

fn storage(slot: usize) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Float32, name: None }
}

fn vadd_spec() -> ProgramSpec {
    let mut spec =
        ProgramSpec::new("vadd".into(), VADD_MSL.into(), DeviceSpec::Metal { device_id: 0 }, UOp::sink(vec![]));
    spec.abi = vec![storage(0), storage(1), storage(2)];
    spec.buf_count = 3;
    spec
}

fn skip() -> bool {
    if has_devices() {
        return false;
    }
    eprintln!("skipping Metal runtime test: no Metal device on this host");
    true
}

/// The `METAL` factory exists exactly when a device does; `METAL:0` resolves
/// through the process-global registry to a Metal device.
#[test]
fn factory_registered_iff_device_present() {
    let spec = DeviceSpec::Metal { device_id: 0 };
    let resolved = crate::DEVICE_FACTORIES.device(&spec, svod_device::registry::registry());
    match resolved {
        Ok(device) => {
            assert!(has_devices());
            assert_eq!(device.device, spec);
            assert_eq!(device.base_device_key(), "METAL");
        }
        Err(error) => {
            assert!(!has_devices(), "factory missing although a device exists: {error}");
            assert!(matches!(error, crate::Error::UnsupportedDevice { .. }), "{error:?}");
        }
    }
}

#[test]
fn compiler_emits_loadable_bytes_with_launch_dims() {
    if skip() {
        return;
    }
    let device = create_metal_device(&DeviceRegistry::default(), 0).unwrap();
    let compiled = device.compiler.compile(&vadd_spec()).expect("compile MSL");
    assert!(compiled.src.is_none() && !compiled.bytes.is_empty());
    svod_device::metal::compile::validate_metallib(&compiled.bytes, "vadd").unwrap();
    let dev = MetalDevice::open(0).unwrap();
    let program = create_metal_program(&dev, &compiled).expect("load compiled kernel");
    assert_eq!(program.name(), "vadd");

    let mut broken = vadd_spec();
    broken.src = "kernel void vadd() { nonsense }".into();
    assert!(device.compiler.compile(&broken).is_err(), "MSL diagnostics must fail the compile stage");
}

#[test]
fn compiler_identity_is_stable_and_backend_specific() {
    if skip() {
        return;
    }
    let (_, first) = create_metal_codegen(0).unwrap();
    let (_, second) = create_metal_codegen(0).unwrap();
    assert_eq!(first.cache_key(), second.cache_key());
    assert!(first.cache_key().starts_with("metal:"), "{}", first.cache_key());
    let (_, cpu) = crate::devices::cpu::create_cpu_codegen(crate::devices::cpu::CpuBackend::Llvm).unwrap();
    assert_ne!(first.cache_key(), cpu.cache_key());
    // Deterministic output for a deterministic input (what the object cache relies on).
    assert_eq!(first.compile(&vadd_spec()).unwrap().bytes, second.compile(&vadd_spec()).unwrap().bytes);
}

#[test]
fn runtime_factory_rejects_empty_bytes_and_bad_abi() {
    if skip() {
        return;
    }
    let dev = MetalDevice::open(0).unwrap();
    let device = create_metal_device(&DeviceRegistry::default(), 0).unwrap();
    let mut compiled = device.compiler.compile(&vadd_spec()).unwrap();
    compiled.bytes.clear();
    let Err(error) = create_metal_program(&dev, &compiled) else { panic!("empty bytes must be rejected") };
    assert!(format!("{error}").contains("empty metallib bytes"), "{error}");

    let mut compiled = device.compiler.compile(&vadd_spec()).unwrap();
    compiled.buf_count = 2;
    let Err(error) = create_metal_program(&dev, &compiled) else { panic!("ABI projection mismatch must be rejected") };
    assert!(matches!(error, svod_device::Error::ProgramAbiMismatch { .. }), "{error:?}");
}

/// `SVOD_DEVICE=METAL:0` now parses (it used to fall back to CPU silently).
#[test]
fn metal_spec_round_trips_through_registry_parse() {
    use svod_device::registry::DeviceSpecExt;
    assert_eq!(DeviceSpec::parse("METAL:1").unwrap(), DeviceSpec::Metal { device_id: 1 });
    assert_eq!(DeviceSpec::Metal { device_id: 0 }.base_type(), "METAL");
}

/// The device carries a graph factory, and a static chain of loaded kernels
/// is captured by it (the execution plan's replay path).
#[test]
fn device_installs_indirect_command_buffer_graphs() {
    if skip() {
        return;
    }
    let device = create_metal_device(&DeviceRegistry::default(), 0).unwrap();
    let factory = device.graph.clone().expect("Metal devices graph static plans");
    let compiled = device.compiler.compile(&vadd_spec()).unwrap();
    let dev = MetalDevice::open(0).unwrap();
    let program = create_metal_program(&dev, &compiled).unwrap();
    let alloc = svod_device::metal::MetalAllocator::new(0).unwrap();
    let spec = svod_device::BufferSpec::default();
    let buffers: Vec<_> = (0..3).map(|_| alloc._alloc(256, &spec, true).unwrap()).collect();
    let pointers: Vec<*mut u8> = buffers
        .iter()
        .map(|buffer| match buffer {
            svod_device::allocator::RawBuffer::Metal { contents, .. } => contents.as_ptr(),
            _ => unreachable!(),
        })
        .collect();
    let kernel = svod_device::device::GraphKernel {
        program: program.as_ref(),
        buffers: pointers,
        vals: vec![],
        global_size: Some([2, 1, 1]),
        local_size: Some([32, 1, 1]),
        deps: vec![],
    };
    let graph = factory(&[kernel]).unwrap().expect("static Metal chain is graphable");
    graph.replay(&[], &[]).unwrap();
    dev.synchronize().unwrap();
}
