use std::sync::Arc;

use test_case::test_case;

use super::program::{download, host_ptr, upload, vadd_abi};
use super::{SCALE_MSL, VADD_MSL, compile_for_test, metal_alloc_or_skip};
use crate::allocator::{Allocator, RawBuffer};
use crate::device::{AbiParamDescriptor, AbiParamKind, Graph, GraphKernel, Program};
use crate::metal::graph::needs_icb_fix;
use crate::metal::{MetalAllocator, MetalGraph, MetalProgram};

#[test_case("Apple9", false; "m3 and later")]
#[test_case("Apple12", false; "future generation")]
#[test_case("Apple8", true; "m2")]
#[test_case("Apple7", true; "m1")]
#[test_case("Mac2", true; "intel or amd")]
#[test_case("Unknown", true; "unknown family")]
fn icb_fix_is_applied_before_apple9(family: &str, expected: bool) {
    assert_eq!(needs_icb_fix(family), expected);
}

const N: usize = 4096;

struct Chain {
    alloc: MetalAllocator,
    program: MetalProgram,
    a: RawBuffer,
    b: RawBuffer,
    mid1: RawBuffer,
    mid2: RawBuffer,
    out: RawBuffer,
}

impl Chain {
    fn new() -> Option<Self> {
        let alloc = metal_alloc_or_skip()?;
        let bytes = compile_for_test(&alloc.dev, VADD_MSL).unwrap();
        let program = MetalProgram::load(alloc.dev.clone(), &bytes, "vadd", &vadd_abi()).unwrap();
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (2 * i) as f32).collect();
        let zeros = vec![0.0f32; N];
        Some(Self {
            a: upload(&alloc, &a),
            b: upload(&alloc, &b),
            mid1: upload(&alloc, &zeros),
            mid2: upload(&alloc, &zeros),
            out: upload(&alloc, &zeros),
            alloc,
            program,
        })
    }

    /// `mid1 = a + b; mid2 = mid1 + b; out = mid2 + mid1` — every kernel reads
    /// the previous one's output, so the chain is only correct in order.
    fn kernels(&self) -> Vec<GraphKernel<'_>> {
        let launch = |buffers: Vec<*mut u8>| GraphKernel {
            program: &self.program as &dyn Program,
            buffers,
            vals: vec![],
            global_size: Some([N / 32, 1, 1]),
            local_size: Some([32, 1, 1]),
            deps: vec![],
        };
        vec![
            launch(vec![host_ptr(&self.mid1), host_ptr(&self.a), host_ptr(&self.b)]),
            launch(vec![host_ptr(&self.mid2), host_ptr(&self.mid1), host_ptr(&self.b)]),
            launch(vec![host_ptr(&self.out), host_ptr(&self.mid2), host_ptr(&self.mid1)]),
        ]
    }

    fn flattened(&self) -> Vec<u64> {
        self.kernels().iter().flat_map(|kernel| kernel.buffers.iter().map(|pointer| *pointer as u64)).collect()
    }

    /// `out[i] = (a + b) + b + (a + b) = 2a + 3b`.
    fn expected(&self) -> Vec<f32> {
        (0..N).map(|i| (2 * i + 3 * 2 * i) as f32).collect()
    }

    fn capture(&self) -> Box<dyn Graph> {
        MetalGraph::capture(self.alloc.dev.clone(), &self.kernels()).unwrap().expect("static chain is graphable")
    }
}

#[test]
fn captured_chain_replays_in_order() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
    // Replaying with the same bindings reuses the indirect commands untouched;
    // the second replay must wait for the first before re-submitting.
    for _ in 0..3 {
        graph.replay(&chain.flattened(), &[]).unwrap();
    }
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
}

#[test]
fn replay_matches_per_call_dispatch() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    let batched = download(&chain.alloc, &chain.out, N);
    chain.alloc._copyin(&chain.out, 0, &vec![0u8; N * 4]).unwrap();
    for kernel in chain.kernels() {
        unsafe { kernel.program.execute(&kernel.buffers, &[], kernel.global_size, kernel.local_size, false) }.unwrap();
    }
    assert_eq!(download(&chain.alloc, &chain.out, N), batched);
}

#[test]
fn replay_rebinds_changed_buffers() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    graph.replay(&[], &[]).unwrap();
    // Swap the first kernel's inputs for fresh buffers (a' = 10, b' = 1) and
    // route the final output elsewhere: every replay resolves the new bindings.
    let a2 = upload(&chain.alloc, &vec![10.0; N]);
    let b2 = upload(&chain.alloc, &vec![1.0; N]);
    let out2 = upload(&chain.alloc, &vec![0.0; N]);
    let mut buffers = chain.flattened();
    buffers[1] = host_ptr(&a2) as u64;
    buffers[2] = host_ptr(&b2) as u64;
    buffers[5] = host_ptr(&b2) as u64;
    buffers[6] = host_ptr(&out2) as u64;
    graph.replay(&buffers, &[]).unwrap();
    // mid1 = 11, mid2 = 12, out2 = 23; the original output is untouched.
    assert!(download(&chain.alloc, &out2, N).iter().all(|value| *value == 23.0));
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
    // A sub-buffer view binds through its offset.
    buffers[6] = unsafe { host_ptr(&out2).add(64 * 4) } as u64;
    graph.replay(&buffers, &[]).unwrap();
    let shifted = download(&chain.alloc, &out2, N);
    assert!(shifted[64..].iter().all(|value| *value == 23.0));
    assert_eq!(shifted[..64], download(&chain.alloc, &out2, N)[..64]);
}

#[test]
fn replay_arguments_are_validated() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    let error = graph.replay(&chain.flattened()[..4], &[]).expect_err("buffer count");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
    let error = graph.replay(&[], &[1]).expect_err("scalars");
    assert!(matches!(error, crate::Error::ProgramAbiMismatch { .. }), "{error:?}");
    let mut host = vec![0f32; N];
    let mut buffers = chain.flattened();
    buffers[0] = host.as_mut_ptr() as u64;
    let error = graph.replay(&buffers, &[]).expect_err("host memory");
    assert!(format!("{error}").contains("no registered MTLBuffer"), "{error}");
    // The failed rebind left the other slots usable.
    graph.replay(&chain.flattened(), &[]).unwrap();
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
}

#[test]
fn profiled_replay_stamps_every_kernel() {
    let Some(chain) = Chain::new() else { return };
    let graph = chain.capture();
    let handles = graph.replay_profiled(&[], &[]).unwrap().expect("Metal graphs stamp dispatches");
    assert_eq!(handles.len(), 3);
    let mut previous_end = 0;
    for handle in handles {
        let (start, end) = handle.timestamps_ns().expect("completed command buffer has GPU stamps");
        assert!(start > 0 && end >= start, "{start} {end}");
        assert!(start >= previous_end, "kernels retire in order: {start} < {previous_end}");
        previous_end = end;
    }
    assert_eq!(download(&chain.alloc, &chain.out, N), chain.expected());
}

/// Scalar arguments would need a graph-owned argument buffer; the plan never
/// offers them, and a hand-built chain with one is declined, not mis-captured.
#[test]
fn chains_with_scalars_are_declined() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let bytes = compile_for_test(&alloc.dev, SCALE_MSL).unwrap();
    let abi = vec![
        AbiParamDescriptor {
            slot: 0,
            kind: AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        },
        AbiParamDescriptor {
            slot: 1,
            kind: AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        },
        AbiParamDescriptor {
            slot: 2,
            kind: AbiParamKind::Scalar,
            dtype: svod_dtype::DType::Int32,
            name: Some("n".into()),
        },
    ];
    let program = MetalProgram::load(alloc.dev.clone(), &bytes, "scale", &abi).unwrap();
    let (out, a) = (upload(&alloc, &[0.0; 64]), upload(&alloc, &[1.0; 64]));
    let kernel = GraphKernel {
        program: &program,
        buffers: vec![host_ptr(&out), host_ptr(&a)],
        vals: vec![3],
        global_size: Some([2, 1, 1]),
        local_size: Some([32, 1, 1]),
        deps: vec![],
    };
    assert!(MetalGraph::capture(Arc::clone(&alloc.dev), &[kernel]).unwrap().is_none());
    assert!(MetalGraph::capture(alloc.dev.clone(), &[]).unwrap().is_none());
}

struct NotMetal;

impl Program for NotMetal {
    unsafe fn execute(
        &self,
        _: &[*mut u8],
        _: &[i64],
        _: Option<[usize; 3]>,
        _: Option<[usize; 3]>,
        _: bool,
    ) -> crate::Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "not_metal"
    }
}

#[test]
fn foreign_programs_are_declined() {
    let Some(alloc) = metal_alloc_or_skip() else { return };
    let kernel = GraphKernel {
        program: &NotMetal,
        buffers: vec![],
        vals: vec![],
        global_size: None,
        local_size: None,
        deps: vec![],
    };
    assert!(MetalGraph::capture(alloc.dev, &[kernel]).unwrap().is_none());
}
