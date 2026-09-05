use std::time::Instant;

use svod_dtype::{AddrSpace, DType};

use super::{SCALE_MSL, VADD_MSL, compile_for_test, metal_alloc_or_skip, metal_device_or_skip};
use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::device::{AbiParamDescriptor, AbiParamKind, Program};
use crate::metal::{MetalAllocator, MetalProgram};

fn storage(slot: usize) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Float32, name: None }
}

fn scalar(slot: usize, name: &str) -> AbiParamDescriptor {
    AbiParamDescriptor { slot, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some(name.to_string()) }
}

pub(crate) fn vadd_abi() -> Vec<AbiParamDescriptor> {
    vec![storage(0), storage(1), storage(2)]
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding; the slice is re-viewed byte-wise.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn upload(alloc: &MetalAllocator, values: &[f32]) -> RawBuffer {
    let buffer = alloc._alloc(values.len() * 4, &BufferSpec::default(), false).unwrap();
    alloc._copyin(&buffer, 0, f32_bytes(values)).unwrap();
    buffer
}

fn download(alloc: &MetalAllocator, buffer: &RawBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; len * 4];
    alloc._copyout(&mut bytes, buffer, 0).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect()
}

fn host_ptr(buffer: &RawBuffer) -> *mut u8 {
    let RawBuffer::Metal { contents, .. } = buffer else { unreachable!() };
    contents.as_ptr()
}

#[test]
fn vector_add_executes_on_the_gpu() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let bytes = compile_for_test(&alloc.dev, VADD_MSL).unwrap();
    let program = MetalProgram::load(alloc.dev.clone(), &bytes, "vadd", &vadd_abi()).expect("load vadd");
    assert_eq!(program.name(), "vadd");
    const N: usize = 1024;
    let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();
    let (out, a_buf, b_buf) = (upload(&alloc, &vec![0.0; N]), upload(&alloc, &a), upload(&alloc, &b));
    unsafe {
        program.execute(
            &[host_ptr(&out), host_ptr(&a_buf), host_ptr(&b_buf)],
            &[],
            Some([N / 32, 1, 1]),
            Some([32, 1, 1]),
            true,
        )
    }
    .expect("dispatch");
    let expected: Vec<f32> = (0..N).map(|i| (3 * i) as f32).collect();
    assert_eq!(download(&alloc, &out, N), expected);
}

#[test]
fn scalar_argument_is_bound_with_set_bytes() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let bytes = compile_for_test(&alloc.dev, SCALE_MSL).unwrap();
    let abi = vec![storage(0), storage(1), scalar(2, "n")];
    let program = MetalProgram::load(alloc.dev.clone(), &bytes, "scale", &abi).expect("load scale");
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let (out, a_buf) = (upload(&alloc, &vec![0.0; 64]), upload(&alloc, &a));
    unsafe { program.execute(&[host_ptr(&out), host_ptr(&a_buf)], &[5], Some([2, 1, 1]), Some([32, 1, 1]), true) }
        .unwrap();
    assert_eq!(download(&alloc, &out, 64), a.iter().map(|x| x * 5.0).collect::<Vec<_>>());

    let error = unsafe {
        program.execute(&[host_ptr(&out), host_ptr(&a_buf)], &[i64::MAX], Some([2, 1, 1]), Some([32, 1, 1]), true)
    }
    .expect_err("scalar must fit i32");
    assert!(format!("{error}").contains("does not fit i32"), "{error}");
    let error = unsafe { program.execute(&[host_ptr(&out)], &[5], Some([2, 1, 1]), Some([32, 1, 1]), true) }
        .expect_err("arity is checked");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
}

/// Two `wait=false` dispatches chained through a buffer, then a copyout: the
/// copy must drain the queue so the second kernel's result is visible.
#[test]
fn async_dispatches_are_drained_by_copyout() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let bytes = compile_for_test(&alloc.dev, VADD_MSL).unwrap();
    let program = MetalProgram::load(alloc.dev.clone(), &bytes, "vadd", &vadd_abi()).unwrap();
    const N: usize = 1 << 16;
    let ones = vec![1.0f32; N];
    let (a, b, mid, out) =
        (upload(&alloc, &ones), upload(&alloc, &ones), upload(&alloc, &vec![0.0; N]), upload(&alloc, &vec![0.0; N]));
    for _ in 0..8 {
        unsafe {
            program
                .execute(
                    &[host_ptr(&mid), host_ptr(&a), host_ptr(&b)],
                    &[],
                    Some([N / 32, 1, 1]),
                    Some([32, 1, 1]),
                    false,
                )
                .unwrap();
            program
                .execute(
                    &[host_ptr(&out), host_ptr(&mid), host_ptr(&b)],
                    &[],
                    Some([N / 32, 1, 1]),
                    Some([32, 1, 1]),
                    false,
                )
                .unwrap();
        }
    }
    assert!(download(&alloc, &out, N).iter().all(|value| *value == 3.0));
}

#[test]
fn wait_blocks_until_completion() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let bytes = compile_for_test(&alloc.dev, VADD_MSL).unwrap();
    let program = MetalProgram::load(alloc.dev.clone(), &bytes, "vadd", &vadd_abi()).unwrap();
    const N: usize = 1 << 20;
    let (out, a, b) = (upload(&alloc, &vec![0.0; N]), upload(&alloc, &vec![1.0; N]), upload(&alloc, &vec![2.0; N]));
    let started = Instant::now();
    unsafe {
        program.execute(
            &[host_ptr(&out), host_ptr(&a), host_ptr(&b)],
            &[],
            Some([N / 32, 1, 1]),
            Some([32, 1, 1]),
            true,
        )
    }
    .unwrap();
    assert!(started.elapsed().as_nanos() > 0);
    // Completed work is visible without any further synchronization.
    let RawBuffer::Metal { contents, .. } = &out else { unreachable!() };
    assert_eq!(unsafe { *(contents.as_ptr() as *const f32).add(N - 1) }, 3.0);
}

#[test]
fn wrong_entry_point_is_rejected_at_load() {
    let Some(dev) = metal_device_or_skip() else { return };
    let bytes = compile_for_test(&dev, VADD_MSL).unwrap();
    let error = MetalProgram::load(dev, &bytes, "not_there", &vadd_abi()).expect_err("missing entry point");
    assert!(format!("{error}").contains("not_there"), "{error}");
}

#[test]
fn oversized_threadgroup_is_rejected_with_limits() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let bytes = compile_for_test(&alloc.dev, VADD_MSL).unwrap();
    let program = MetalProgram::load(alloc.dev.clone(), &bytes, "vadd", &vadd_abi()).unwrap();
    let (out, a, b) = (upload(&alloc, &[0.0; 32]), upload(&alloc, &[0.0; 32]), upload(&alloc, &[0.0; 32]));
    let error = unsafe {
        program.execute(&[host_ptr(&out), host_ptr(&a), host_ptr(&b)], &[], Some([1, 1, 1]), Some([2048, 1, 1]), true)
    }
    .expect_err("2048 threads exceed the pipeline limit");
    let message = format!("{error}");
    for needle in ["maxTotalThreadsPerThreadgroup", "threadExecutionWidth", "staticThreadgroupMemoryLength"] {
        assert!(message.contains(needle), "{message}");
    }
}

#[test_case::test_case(vec![storage(0), storage(2)]; "non-contiguous slots")]
#[test_case::test_case(vec![storage(1), storage(0)]; "unsorted slots")]
#[test_case::test_case((0..32).map(storage).collect(); "more than 31 bindings")]
fn positional_abi_violations_are_rejected(abi: Vec<AbiParamDescriptor>) {
    let Some(dev) = metal_device_or_skip() else { return };
    let bytes = compile_for_test(&dev, VADD_MSL).unwrap();
    let error = MetalProgram::load(dev, &bytes, "vadd", &abi).expect_err("ABI contract");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
}

#[test]
fn unregistered_pointer_is_reported() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let bytes = compile_for_test(&alloc.dev, VADD_MSL).unwrap();
    let program = MetalProgram::load(alloc.dev.clone(), &bytes, "vadd", &vadd_abi()).unwrap();
    let (a, b) = (upload(&alloc, &[0.0; 32]), upload(&alloc, &[0.0; 32]));
    let mut host = vec![0f32; 32];
    let error = unsafe {
        program.execute(
            &[host.as_mut_ptr().cast(), host_ptr(&a), host_ptr(&b)],
            &[],
            Some([1, 1, 1]),
            Some([32, 1, 1]),
            true,
        )
    }
    .expect_err("host memory is not a Metal buffer");
    assert!(format!("{error}").contains("no registered MTLBuffer"), "{error}");
}
