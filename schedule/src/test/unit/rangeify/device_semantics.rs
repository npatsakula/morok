//! Device semantics at the kernel boundary.
//!
//! `extract_device_from_graph` is covered by the rows in `buffer_limits.rs`.

use svod_dtype::{DType, DeviceSpec};
use svod_ir::ops;
use svod_ir::{Op, UOp};

#[test]
fn copy_records_its_target_device() {
    let copy = UOp::native_const(1.0f32).copy_to_device(DeviceSpec::Cpu);
    assert!(matches!(copy.op(), Op::Copy(ops::Copy { device: DeviceSpec::Cpu, .. })));
}

/// Only a COPY may straddle devices; a compute kernel must read and write one.
#[test]
fn mixed_device_kernel_is_rejected() {
    let cpu = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let amd = UOp::new_buffer(DeviceSpec::Amd { device_id: 0 }, 8, DType::Float32);
    let sink = UOp::sink(vec![cpu.try_mul(&amd).expect("mul").contiguous()]);

    let (rangeified, _ctx) = crate::rangeify::rangeify(sink).expect("rangeify accepts the mixed graph");
    assert!(crate::rangeify::try_get_kernel_graph(rangeified).is_err());
}
