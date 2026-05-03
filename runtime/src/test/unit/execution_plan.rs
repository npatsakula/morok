use super::*;

use morok_device::device::Program;
use morok_dtype::DType;
use morok_ir::{CustomFunctionKind, UOp};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

fn default_launch_size() -> [Arc<UOp>; 3] {
    [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)]
}

#[test]
fn test_builder_basic() {
    let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let plan = builder.build().expect("build plan");

    assert!(plan.prepared_kernels().is_empty());
    assert!(plan.buffers.is_empty());
    assert_eq!(plan.device, DeviceSpec::Cpu);
}

#[test]
fn test_empty_plan_output_buffer_returns_none() {
    let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let plan = builder.build().expect("build plan");

    assert!(plan.output_buffer().is_none(), "empty plan must not expose an output buffer");
    assert!(plan.output_buffer_at(0).is_none(), "empty plan output_buffer_at must be None");
    assert!(plan.output_buffer_at(7).is_none(), "out-of-range output_buffer_at must be None");
}

#[test]
fn test_builder_map_buffer_alias() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let buf = Buffer::new(alloc, morok_dtype::DType::Float32, vec![8], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let idx = builder.add_buffer(10, buf);
    builder.map_buffer(11, idx);
    builder.set_output_buffer(idx);
    let plan = builder.build().expect("build plan");

    assert_eq!(plan.ast_to_buffer_map().get(&10), Some(&idx));
    assert_eq!(plan.ast_to_buffer_map().get(&11), Some(&idx));
    assert_eq!(plan.buffers().len(), 1);
}

#[test]
fn test_builder_requires_explicit_output_indices() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let buf = Buffer::new(alloc, morok_dtype::DType::Float32, vec![8], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    builder.add_buffer(10, buf);

    let err = builder.build().expect_err("build should fail when outputs are not set");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("output buffers must be set explicitly"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_buffer_copy_op() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(1, dst);
    let src_idx = builder.add_buffer(2, src);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 99,
        buffer_indices: vec![dst_idx, src_idx],
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute copy op");

    let mut output_data = vec![0.0f32; 4];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("dst copyout");

    assert_eq!(output_data, input_data);
}

#[test]
fn test_execute_buffer_view_op() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let mut base = Buffer::new(alloc.clone(), DType::Float32, vec![8], Default::default());
    let output_placeholder = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let base_data = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let base_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(base_data.as_ptr() as *const u8, base_data.len() * std::mem::size_of::<f32>())
    };
    base.copyin(base_bytes).expect("base copyin");

    let element_offset = 2usize;
    let element_count = 4usize;
    let byte_offset = element_offset * std::mem::size_of::<f32>();
    let byte_size = element_count * std::mem::size_of::<f32>();
    let view = base.view(byte_offset, byte_size).expect("create view");

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let out_idx = builder.add_buffer(100, output_placeholder);
    let base_idx = builder.add_buffer(101, base);
    builder.replace_buffer(out_idx, view);
    builder.add_op(PreparedOp::BufferView(PreparedBufferView {
        id: 123,
        buffer_indices: vec![out_idx, base_idx],
        byte_offset,
        byte_size,
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute view op");

    let mut output_data = vec![0.0f32; element_count];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("view copyout");

    assert_eq!(output_data, vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_execute_custom_function_op_returns_unsupported() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(201, dst);
    let src_idx = builder.add_buffer(202, src);
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 200,
        kind: CustomFunctionKind::EncDec,
        attrs: smallvec::smallvec![morok_ir::UOp::index_const(3)],
        buffer_indices: vec![dst_idx, src_idx],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    let err = plan.execute().expect_err("EncDec runtime should be explicit unsupported");
    match err {
        crate::error::Error::Unsupported { kind, reason } => {
            assert_eq!(kind, "EncDec");
            assert!(reason.contains("attrs=1"), "unexpected reason: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[derive(Debug)]
struct Copy4F32Program {
    calls: Arc<AtomicUsize>,
}

impl Program for Copy4F32Program {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> morok_device::Result<()> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let bytes = 4 * std::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(buffers[1], buffers[0], bytes);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "copy4f32"
    }
}

#[test]
fn test_builder_rejects_invalid_compiled_output_index() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let b = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let a_idx = builder.add_buffer(700, a);
    let b_idx = builder.add_buffer(701, b);
    builder.add_kernel(PreparedKernel {
        id: 77,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: Arc::new(AtomicUsize::new(0)) }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![2],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![a_idx, b_idx],
        output_indices: vec![2],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(a_idx);

    let err = builder.build().expect_err("invalid compiled output index should fail build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("output index out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[derive(Debug)]
struct ObserveParallelProgram {
    calls: Arc<AtomicUsize>,
    active: Arc<AtomicUsize>,
    max_active: Arc<AtomicUsize>,
    sleep_ms: u64,
}

impl Program for ObserveParallelProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> morok_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let active_now = self.active.fetch_add(1, Ordering::SeqCst) + 1;

        loop {
            let current_max = self.max_active.load(Ordering::SeqCst);
            if active_now <= current_max {
                break;
            }
            if self.max_active.compare_exchange(current_max, active_now, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                break;
            }
        }

        std::thread::sleep(Duration::from_millis(self.sleep_ms));
        self.active.fetch_sub(1, Ordering::SeqCst);
        Ok(())
    }

    fn name(&self) -> &str {
        "observe_parallel"
    }
}

#[derive(Debug)]
struct RecordLaunchProgram {
    calls: Arc<AtomicUsize>,
    global_x: Arc<AtomicUsize>,
    first_val: Arc<AtomicUsize>,
}

#[derive(Clone)]
struct RecordLaunchCounters {
    calls: Arc<AtomicUsize>,
    global_x: Arc<AtomicUsize>,
    first_val: Arc<AtomicUsize>,
}

impl Program for RecordLaunchProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
    ) -> morok_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.global_x.store(global_size.map(|size| size[0]).unwrap_or(0), Ordering::SeqCst);
        self.first_val.store(vals.first().copied().unwrap_or(0) as usize, Ordering::SeqCst);
        Ok(())
    }

    fn name(&self) -> &str {
        "record_launch"
    }
}

fn add_record_launch_kernel(
    builder: &mut ExecutionPlanBuilder,
    buffer_idx: usize,
    var: Arc<UOp>,
    global_expr: Arc<UOp>,
    counters: RecordLaunchCounters,
    initial_val: i64,
) {
    builder.add_kernel(PreparedKernel {
        id: 8500,
        ast: UOp::sink(vec![var.clone(), global_expr.clone()]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(RecordLaunchProgram {
                calls: counters.calls,
                global_x: counters.global_x,
                first_val: counters.first_val,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "record_launch".to_string(),
            var_names: vec![match var.op() {
                morok_ir::Op::DefineVar { name, .. } => name.clone(),
                _ => "N".to_string(),
            }],
            globals: vec![0],
            outs: vec![0],
            ins: Vec::new(),
            host_parallel_safe: true,
            global_size: [global_expr, UOp::index_const(1), UOp::index_const(1)],
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![buffer_idx],
        output_indices: vec![0],
        vals: vec![initial_val],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
}

#[test]
fn test_execute_mixed_ops_compiled_copy_view_in_order() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut mid = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut copy_dst = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let output_placeholder = Buffer::new(alloc, DType::Float32, vec![3], Default::default());

    let input_data = [1.0f32, 2.0, 3.0, 4.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    mid.copyin(zero_bytes).expect("mid init");
    copy_dst.copyin(zero_bytes).expect("copy_dst init");

    let byte_offset = std::mem::size_of::<f32>();
    let byte_size = 3 * std::mem::size_of::<f32>();
    let view = copy_dst.view(byte_offset, byte_size).expect("create output view");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);

    let src_idx = builder.add_buffer(10, src);
    let mid_idx = builder.add_buffer(11, mid);
    let copy_idx = builder.add_buffer(12, copy_dst);
    let out_idx = builder.add_buffer(13, output_placeholder);

    let prepared_kernel = PreparedKernel {
        id: 1,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![mid_idx, src_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    };
    builder.add_kernel(prepared_kernel);

    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 2,
        buffer_indices: vec![copy_idx, mid_idx],
        dependencies: vec![1],
    }));

    builder.replace_buffer(out_idx, view);
    builder.add_op(PreparedOp::BufferView(PreparedBufferView {
        id: 3,
        buffer_indices: vec![out_idx, copy_idx],
        byte_offset,
        byte_size,
        dependencies: vec![2],
    }));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute mixed op plan");

    assert_eq!(calls.load(Ordering::Relaxed), 1, "compiled op should run exactly once");

    let mut output_data = vec![0.0f32; 3];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("output copyout");
    assert_eq!(output_data, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_execute_mixed_ops_respects_dependencies_not_insertion_order() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut mid = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut out = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let input_data = [9.0f32, 8.0, 7.0, 6.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    mid.copyin(zero_bytes).expect("mid init");
    out.copyin(zero_bytes).expect("out init");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);

    let src_idx = builder.add_buffer(300, src);
    let mid_idx = builder.add_buffer(301, mid);
    let out_idx = builder.add_buffer(302, out);

    // Insert compiled kernel first, but make it depend on copy op id=2.
    // Mixed-op execution must honor deps and run copy before this kernel.
    builder.add_kernel(PreparedKernel {
        id: 3,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32_out".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![out_idx, mid_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: vec![2],
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 2,
        buffer_indices: vec![mid_idx, src_idx],
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute dependency-ordered mixed ops");

    assert_eq!(calls.load(Ordering::Relaxed), 1, "compiled op should run exactly once");

    let mut output_data = vec![0.0f32; 4];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("output copyout");
    assert_eq!(output_data, input_data);
}

#[test]
fn test_execute_mixed_ops_missing_dependency_errors() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(400, dst);
    let src_idx = builder.add_buffer(401, src);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 10,
        buffer_indices: vec![dst_idx, src_idx],
        dependencies: vec![999],
    }));
    builder.set_output_buffer(dst_idx);

    let err = builder.build().expect_err("missing dependency should fail during build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("unknown op id"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_mixed_ops_cycle_errors() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let b = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let a_idx = builder.add_buffer(500, a);
    let b_idx = builder.add_buffer(501, b);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 1,
        buffer_indices: vec![a_idx, b_idx],
        dependencies: vec![2],
    }));
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 2,
        buffer_indices: vec![b_idx, a_idx],
        dependencies: vec![1],
    }));
    builder.set_output_buffer(a_idx);

    let err = builder.build().expect_err("cyclic deps should fail during build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("cycle detected"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_mixed_ops_allows_duplicate_ids_in_expanded_schedule_order() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mid = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let out = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let input_data = [3.0f32, 1.0, 4.0, 1.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let src_idx = builder.add_buffer(800, src);
    let mid_idx = builder.add_buffer(801, mid);
    let out_idx = builder.add_buffer(802, out);

    // Expanded schedules can produce repeated op ids for per-iteration items.
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 42,
        buffer_indices: vec![mid_idx, src_idx],
        dependencies: Vec::new(),
    }));
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 42,
        buffer_indices: vec![out_idx, mid_idx],
        dependencies: vec![42],
    }));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute duplicate-id schedule");

    let mut output_data = vec![0.0f32; 4];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("output copyout");
    assert_eq!(output_data, input_data);
}

#[test]
fn test_execute_copy_invalid_indices_errors() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(600, dst);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 55,
        buffer_indices: vec![dst_idx, dst_idx + 1],
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    let err = plan.execute().expect_err("invalid copy indices should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_buffer_view_missing_indices_errors() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let out = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let out_idx = builder.add_buffer(700, out);
    builder.add_op(PreparedOp::BufferView(PreparedBufferView {
        id: 77,
        buffer_indices: vec![out_idx],
        byte_offset: 0,
        byte_size: 4 * std::mem::size_of::<f32>(),
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    let err = plan.execute().expect_err("invalid view indices should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("requires at least two buffer indices"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_build_compiled_program_invalid_buffer_indices_errors() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = {
        let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
        dst.ensure_allocated().expect("dst allocation");
        dst
    };

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(860, dst);
    builder.add_kernel(PreparedKernel {
        id: 861,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: Arc::new(AtomicUsize::new(0)) }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "invalid_indices".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, dst_idx + 1],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let err = builder.build().expect_err("invalid compiled-program buffer indices should fail during build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("buffer index out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_custom_function_invalid_indices_errors() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(880, dst);
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 881,
        kind: CustomFunctionKind::EncDec,
        attrs: smallvec::smallvec![],
        buffer_indices: vec![dst_idx, dst_idx + 1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    let err = plan.execute().expect_err("invalid custom function indices should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("buffer index out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_with_vars_does_not_override_fixedvars() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(zero_bytes).expect("src init");
    dst.copyin(zero_bytes).expect("dst init");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(900, dst);
    let src_idx = builder.add_buffer(901, src);
    builder.add_kernel(PreparedKernel {
        id: 900,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32_fixedvars".to_string(),
            var_names: vec!["N".to_string()],
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, src_idx],
        output_indices: vec![0],
        vals: vec![7],
        fixedvars: HashMap::from([(String::from("N"), 7)]),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("N", 42)]).expect("execute with vars");

    let kernels = plan.prepared_kernels();
    assert_eq!(kernels[0].vals.as_slice(), &[7], "fixedvars should win over execute_with_vars overrides");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "kernel should execute exactly once");
}

#[test]
fn test_execute_with_vars_updates_non_fixed_vars() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(zero_bytes).expect("src init");
    dst.copyin(zero_bytes).expect("dst init");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(910, dst);
    let src_idx = builder.add_buffer(911, src);
    builder.add_kernel(PreparedKernel {
        id: 910,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32_dynamicvars".to_string(),
            var_names: vec!["N".to_string()],
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, src_idx],
        output_indices: vec![0],
        vals: vec![1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("N", 42)]).expect("execute with vars");

    let kernels = plan.prepared_kernels();
    assert_eq!(kernels[0].vals.as_slice(), &[42], "execute_with_vars should update non-fixed variable values");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "kernel should execute exactly once");
}

#[test]
fn test_execute_with_vars_updates_symbolic_global_size_without_recompile() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(0));
    let counters =
        RecordLaunchCounters { calls: calls.clone(), global_x: global_x.clone(), first_val: first_val.clone() };
    let n = UOp::define_var("N".to_string(), 1, 8);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8500, dst);
    add_record_launch_kernel(&mut builder, dst_idx, n.clone(), n, counters, 1);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("N", 5)]).expect("execute with dynamic launch size");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(global_x.load(Ordering::SeqCst), 5);
    assert_eq!(first_val.load(Ordering::SeqCst), 5);
}

#[test]
fn test_execute_with_vars_rejects_out_of_bounds_launch_var_before_dispatch() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(0));
    let counters = RecordLaunchCounters { calls: calls.clone(), global_x, first_val };
    let n = UOp::define_var("N".to_string(), 1, 4);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8510, dst);
    add_record_launch_kernel(&mut builder, dst_idx, n.clone(), n, counters, 1);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    let err = plan.execute_with_vars(&[("N", 5)]).expect_err("out-of-bounds launch var should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("outside bounds"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0, "kernel must not dispatch after launch-var bounds failure");
}

#[test]
fn test_execute_with_vars_profiled_updates_symbolic_global_size() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(0));
    let counters = RecordLaunchCounters { calls: calls.clone(), global_x: global_x.clone(), first_val };
    let n = UOp::define_var("N".to_string(), 1, 8);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8520, dst);
    add_record_launch_kernel(&mut builder, dst_idx, n.clone(), n, counters, 1);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    let profiles = plan.execute_with_vars_profiled(&[("N", 6)]).expect("execute profiled dynamic launch size");

    assert_eq!(profiles.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(global_x.load(Ordering::SeqCst), 6);
}

#[test]
fn test_execute_with_vars_does_not_override_core_id_runtime_var() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(usize::MAX));
    let counters = RecordLaunchCounters { calls: calls.clone(), global_x, first_val: first_val.clone() };
    let core_id = UOp::define_var("core_id".to_string(), 0, 3);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8530, dst);
    add_record_launch_kernel(&mut builder, dst_idx, core_id, UOp::index_const(1), counters, 0);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("core_id", 2)]).expect("execute with ignored core_id override");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(first_val.load(Ordering::SeqCst), 0, "core_id is a runtime var and must not be user-overridden");
}

#[test]
fn test_execute_parallel_safe_compiled_ops_can_overlap() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let dst_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst_b = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_b = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst_a.ensure_allocated().expect("allocate dst_a");
    src_a.ensure_allocated().expect("allocate src_a");
    dst_b.ensure_allocated().expect("allocate dst_b");
    src_b.ensure_allocated().expect("allocate src_b");

    let calls = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_a_idx = builder.add_buffer(1000, dst_a);
    let src_a_idx = builder.add_buffer(1001, src_a);
    let dst_b_idx = builder.add_buffer(1002, dst_b);
    let src_b_idx = builder.add_buffer(1003, src_b);

    builder.add_kernel(PreparedKernel {
        id: 100,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(ObserveParallelProgram {
                calls: calls.clone(),
                active: active.clone(),
                max_active: max_active.clone(),
                sleep_ms: 30,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "parallel_a".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_a_idx, src_a_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.add_kernel(PreparedKernel {
        id: 101,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(ObserveParallelProgram {
                calls: calls.clone(),
                active,
                max_active: max_active.clone(),
                sleep_ms: 30,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "parallel_b".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_b_idx, src_b_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.set_output_buffer(dst_a_idx);
    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute parallel-safe kernels");

    assert_eq!(calls.load(Ordering::SeqCst), 2, "both kernels should run");
    assert!(max_active.load(Ordering::SeqCst) >= 2, "independent host_parallel_safe kernels should overlap");
}

#[test]
fn test_execute_threaded_cpu_kernels_do_not_use_outer_overlap() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let dst_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst_b = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_b = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst_a.ensure_allocated().expect("allocate dst_a");
    src_a.ensure_allocated().expect("allocate src_a");
    dst_b.ensure_allocated().expect("allocate dst_b");
    src_b.ensure_allocated().expect("allocate src_b");

    let calls = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let threaded_launch = [UOp::index_const(4), UOp::index_const(1), UOp::index_const(1)];

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_a_idx = builder.add_buffer(1300, dst_a);
    let src_a_idx = builder.add_buffer(1301, src_a);
    let dst_b_idx = builder.add_buffer(1302, dst_b);
    let src_b_idx = builder.add_buffer(1303, src_b);

    builder.add_kernel(PreparedKernel {
        id: 130,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(ObserveParallelProgram {
                calls: calls.clone(),
                active: active.clone(),
                max_active: max_active.clone(),
                sleep_ms: 30,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "threaded_outer_a".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: threaded_launch.clone(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_a_idx, src_a_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.add_kernel(PreparedKernel {
        id: 131,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(ObserveParallelProgram {
                calls: calls.clone(),
                active,
                max_active: max_active.clone(),
                sleep_ms: 30,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "threaded_outer_b".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: threaded_launch,
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_b_idx, src_b_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.set_output_buffer(dst_a_idx);
    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute threaded CPU kernels");

    assert_eq!(calls.load(Ordering::SeqCst), 2, "both kernels should run");
    assert_eq!(
        max_active.load(Ordering::SeqCst),
        1,
        "threaded CPU kernels should not also overlap at the outer host level"
    );
}

#[test]
fn test_execute_unsafe_compiled_ops_are_serialized() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let dst_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst_b = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_b = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst_a.ensure_allocated().expect("allocate dst_a");
    src_a.ensure_allocated().expect("allocate src_a");
    dst_b.ensure_allocated().expect("allocate dst_b");
    src_b.ensure_allocated().expect("allocate src_b");

    let calls = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_a_idx = builder.add_buffer(1100, dst_a);
    let src_a_idx = builder.add_buffer(1101, src_a);
    let dst_b_idx = builder.add_buffer(1102, dst_b);
    let src_b_idx = builder.add_buffer(1103, src_b);

    for (id, dst_idx, src_idx) in [(110_u64, dst_a_idx, src_a_idx), (111_u64, dst_b_idx, src_b_idx)] {
        builder.add_kernel(PreparedKernel {
            id,
            ast: UOp::sink(vec![]),
            kernel: Arc::new(CachedKernel {
                program: Box::new(ObserveParallelProgram {
                    calls: calls.clone(),
                    active: active.clone(),
                    max_active: max_active.clone(),
                    sleep_ms: 20,
                }),
                device: "CPU".to_string(),
                code: String::new(),
                entry_point: format!("unsafe_{id}"),
                var_names: Vec::new(),
                globals: vec![0, 1],
                outs: vec![0],
                ins: vec![1],
                host_parallel_safe: false,
                global_size: default_launch_size(),
                local_size: Some(default_launch_size()),
            }),
            device: DeviceSpec::Cpu,
            buffer_indices: vec![dst_idx, src_idx],
            output_indices: vec![0],
            vals: Vec::new(),
            fixedvars: HashMap::new(),
            dependencies: Vec::new(),
            buffer_ptrs: Vec::new(),
            buffer_ids: Vec::new(),
            runtime_vars: Vec::new(),
        });
    }

    builder.set_output_buffer(dst_a_idx);
    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute unsafe kernels");

    assert_eq!(calls.load(Ordering::SeqCst), 2, "both kernels should run");
    assert_eq!(max_active.load(Ordering::SeqCst), 1, "host_parallel_safe=false kernels must remain serialized");
}

#[test]
fn test_execute_mixed_op_types_are_serialized_across_barriers() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let dst_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst_b = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src_b = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let copy_dst = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let copy_src = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst_a.ensure_allocated().expect("allocate dst_a");
    src_a.ensure_allocated().expect("allocate src_a");
    dst_b.ensure_allocated().expect("allocate dst_b");
    src_b.ensure_allocated().expect("allocate src_b");
    copy_dst.ensure_allocated().expect("allocate copy_dst");
    copy_src.ensure_allocated().expect("allocate copy_src");

    let calls = Arc::new(AtomicUsize::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_a_idx = builder.add_buffer(1200, dst_a);
    let src_a_idx = builder.add_buffer(1201, src_a);
    let dst_b_idx = builder.add_buffer(1202, dst_b);
    let src_b_idx = builder.add_buffer(1203, src_b);
    let copy_dst_idx = builder.add_buffer(1204, copy_dst);
    let copy_src_idx = builder.add_buffer(1205, copy_src);

    builder.add_kernel(PreparedKernel {
        id: 120,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(ObserveParallelProgram {
                calls: calls.clone(),
                active: active.clone(),
                max_active: max_active.clone(),
                sleep_ms: 20,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "mixed_a".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_a_idx, src_a_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 121,
        buffer_indices: vec![copy_dst_idx, copy_src_idx],
        dependencies: Vec::new(),
    }));

    builder.add_kernel(PreparedKernel {
        id: 122,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(ObserveParallelProgram {
                calls: calls.clone(),
                active,
                max_active: max_active.clone(),
                sleep_ms: 20,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "mixed_b".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_b_idx, src_b_idx],
        output_indices: vec![0],
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.set_output_buffer(dst_a_idx);
    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute mixed-op plan");

    assert_eq!(calls.load(Ordering::SeqCst), 2, "both kernels should run");
    assert_eq!(max_active.load(Ordering::SeqCst), 1, "safe kernels separated by side-effectful ops must not overlap");
}

#[test]
fn test_compute_execution_levels_duplicate_ids_is_deterministic() {
    let ops = vec![
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![0, 1], dependencies: vec![] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 9, buffer_indices: vec![2, 3], dependencies: vec![42] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![4, 5], dependencies: vec![9] }),
    ];

    let order = compute_mixed_op_order(&ops).expect("dependency order");
    let levels = compute_execution_levels(&ops).expect("dependency levels");
    assert_eq!(order, vec![0, 1, 2]);
    assert_eq!(levels, vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn test_instance_dependencies_target_exact_duplicate_id_instance() {
    let ops = vec![
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![0, 1], dependencies: vec![] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 9, buffer_indices: vec![2, 3], dependencies: vec![] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![4, 5], dependencies: vec![9] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 77, buffer_indices: vec![6, 7], dependencies: vec![] }),
    ];
    let instance_deps = vec![vec![], vec![], vec![], vec![0]];

    let levels = compute_execution_levels_with_instance_dependencies(&ops, &instance_deps).expect("dependency levels");
    assert_eq!(levels, vec![vec![0, 1], vec![2, 3]]);
}

#[test]
fn test_instance_dependencies_reject_unknown_op_index() {
    let ops = vec![PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![0, 1], dependencies: vec![] })];
    let instance_deps = vec![vec![1]];

    let err = compute_execution_levels_with_instance_dependencies(&ops, &instance_deps)
        .expect_err("unknown op-index dependency should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("unknown op index"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_with_vars_profiled_updates_non_fixed_vars() {
    let alloc = morok_device::registry::cpu().expect("cpu allocator");

    let src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    src.ensure_allocated().expect("allocate src");
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(1300, dst);
    let src_idx = builder.add_buffer(1301, src);
    builder.add_kernel(PreparedKernel {
        id: 1300,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "profiled_var_update".to_string(),
            var_names: vec!["N".to_string()],
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            host_parallel_safe: true,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, src_idx],
        output_indices: vec![0],
        vals: vec![1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    let profiles = plan.execute_with_vars_profiled(&[("N", 42)]).expect("execute with vars profiled");

    assert_eq!(profiles.len(), 1, "profile should include the compiled kernel");
    let kernels = plan.prepared_kernels();
    assert_eq!(kernels[0].vals.as_slice(), &[42], "execute_with_vars_profiled should update non-fixed variables");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "kernel should execute exactly once");
}
