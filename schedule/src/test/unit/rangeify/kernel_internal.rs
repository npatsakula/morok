use morok_ir::{CallInfo, DType, DeviceSpec, Error, KernelInfo, UOp};
use smallvec::smallvec;

use super::fix_assign;

#[test]
fn test_fix_assign_cycle_returns_typed_error() {
    let b1 = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);
    let b2 = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);

    let call_for_b2 =
        UOp::sink_with_info(vec![], KernelInfo::default()).call(smallvec![b2.clone()], CallInfo::default());
    let after_b2 = b2.after(smallvec![call_for_b2]);

    let call_for_b1 = UOp::sink_with_info(vec![], KernelInfo::default()).call(
        smallvec![b2.clone()],
        CallInfo { grad_tag: None, metadata: vec!["writer".to_string()], ..CallInfo::default() },
    );
    let after_b1 = b1.after(smallvec![after_b2, call_for_b1]);

    let root = UOp::sink(vec![after_b1]);
    let err = fix_assign(&root).expect_err("expected typed cycle error");

    assert!(matches!(
        err,
        Error::KernelSplitDependencyCycle { writer_buffer, read_buffer }
            if writer_buffer == b1.id && read_buffer == b2.id
    ));
}

fn find_after_for_buffer(root: &std::sync::Arc<UOp>, buffer_id: u64) -> std::sync::Arc<UOp> {
    root.toposort()
        .into_iter()
        .find(|u| matches!(u.op(), morok_ir::Op::After { .. }) && u.buf_uop().id == buffer_id)
        .expect("expected AFTER for buffer")
}

#[test]
fn test_fix_assign_adds_war_dep_when_callable_differs_even_with_shared_dep() {
    let read_write_buf = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);
    let output_buf = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);

    let shared_dep = UOp::noop();

    let call_writer =
        UOp::sink_with_info(vec![], KernelInfo::default()).call(smallvec![read_write_buf.clone()], CallInfo::default());
    let writer_after = read_write_buf.after(smallvec![shared_dep.clone(), call_writer]);

    let call_reader = UOp::sink_with_info(vec![], KernelInfo::default())
        .call(smallvec![read_write_buf.clone(), output_buf.clone()], CallInfo::default());
    let reader_after = output_buf.after(smallvec![shared_dep, call_reader]);

    let root = UOp::sink(vec![writer_after.clone(), reader_after.clone()]);
    let fixed = fix_assign(&root).expect("fix_assign should succeed");

    let updated_writer = find_after_for_buffer(&fixed, read_write_buf.id);
    let morok_ir::Op::After { deps, .. } = updated_writer.op() else {
        panic!("expected AFTER op");
    };
    assert!(
        deps.iter().any(|d| matches!(d.op(), morok_ir::Op::After { .. }) && d.buf_uop().id == output_buf.id),
        "writer AFTER should depend on reader AFTER when callable differs"
    );
}

#[test]
fn test_fix_assign_skips_war_dep_for_same_callable_multi_output() {
    let shared_buf = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);
    let output_buf = UOp::new_buffer(DeviceSpec::Cpu, 1, DType::Float32);

    let shared_callable = UOp::sink_with_info(vec![], KernelInfo::default())
        .call(smallvec![shared_buf.clone(), output_buf.clone()], CallInfo::default());

    let writer_after = shared_buf.after(smallvec![shared_callable.clone()]);
    let reader_after = output_buf.after(smallvec![shared_callable]);

    let root = UOp::sink(vec![writer_after.clone(), reader_after]);
    let fixed = fix_assign(&root).expect("fix_assign should succeed");

    let updated_writer = find_after_for_buffer(&fixed, shared_buf.id);
    let morok_ir::Op::After { deps, .. } = updated_writer.op() else {
        panic!("expected AFTER op");
    };
    assert_eq!(deps.len(), 1, "same-callable outputs should not receive extra WAR deps");
}
