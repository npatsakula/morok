//! `reduce_to_acc`: REDUCE -> DEFINE_REG accumulator + loop, and the WMMA-add
//! fusion that runs beside it. Ported from tinygrad `devectorizer.py:291-308`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::types::ConstValue;
use svod_ir::{AxisType, BinaryOp, Op, ReduceOp, RendererDevice, SInt, UOp, WmmaMetadata};
use test_case::test_case;

use super::helpers::*;
use svod_ir::ops;

fn test_wmma(c: Arc<UOp>) -> Arc<UOp> {
    let operand = UOp::stack((0..6).map(|i| UOp::var(format!("operand_{i}"), DType::Float32, -100, 100)).collect());
    UOp::wmma(
        operand.clone(),
        operand,
        c,
        WmmaMetadata {
            name: "test".into(),
            dims: (16, 16, 16),
            dtype_in: DType::Float32,
            dtype_out: DType::Float32,
            device: RendererDevice::Cpu,
            threads: 32,
            upcast_axes: None,
            reduce_axes: vec![],
            tile_grid: (1, 1),
        },
    )
}

fn shaped_values(prefix: &str, shape: &[usize]) -> Arc<UOp> {
    let count = shape.iter().product();
    UOp::stack((0..count).map(|i| UOp::var(format!("{prefix}_{i}"), DType::Float32, -100, 100)).collect())
        .try_reshape(&shape.iter().copied().map(SInt::Const).collect())
        .unwrap()
}

fn fuse_wmma_add(root: Arc<UOp>) -> Arc<UOp> {
    crate::rewrite::graph_rewrite(crate::devectorize::pm_wmma_add(), root, &mut ())
}

#[test]
fn test_wmma_add_direct_moves_into_accumulator() {
    let accumulator = shaped_values("acc", &[6]);
    let add = shaped_values("add", &[6]);
    let result = fuse_wmma_add(test_wmma(accumulator.clone()).add(&add));

    let Op::Wmma(ops::Wmma { c, .. }) = result.op() else { panic!("direct WMMA add must fuse") };
    assert!(matches!(c.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
        if Arc::ptr_eq(lhs, &accumulator) && Arc::ptr_eq(rhs, &add)));
}

/// A non-broadcastable ADD must leave the WMMA unfused rather than abort (tinygrad's
/// `codegen/__init__.py:110` asserts inside `alu`; we decline the rewrite).
#[test]
fn test_wmma_add_with_mismatched_operand_does_not_fuse() {
    let fusable = test_wmma(shaped_values("acc", &[6])).add(&shaped_values("add", &[6]));
    let root = fusable.with_sources(vec![fusable.op().sources()[0].clone(), shaped_values("bad", &[3])]);

    assert!(Arc::ptr_eq(&fuse_wmma_add(root.clone()), &root), "mismatched WMMA add must stay unfused");
}

#[test]
fn test_wmma_add_moves_through_permute() {
    let permuted = test_wmma(shaped_values("acc", &[2, 3])).try_permute(vec![1, 0]).unwrap();
    let result = fuse_wmma_add(permuted.add(&shaped_values("add", &[3, 2])));

    let Op::Permute(ops::Permute { src, axes }) = result.op() else {
        panic!("output permutation must remain outside WMMA")
    };
    assert_eq!(axes, &[1, 0]);
    let Op::Wmma(ops::Wmma { c, .. }) = src.op() else { panic!("permuted add must fuse into WMMA") };
    assert!(
        matches!(c.op(), Op::Binary(BinaryOp::Add, _, moved) if matches!(moved.op(), Op::Permute(ops::Permute { axes, .. }) if axes == &[1, 0]))
    );
}

#[test]
fn test_wmma_add_moves_through_permute_reshape() {
    let reshaped =
        test_wmma(shaped_values("acc", &[6])).try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let result = fuse_wmma_add(reshaped.try_permute(vec![1, 0]).unwrap().add(&shaped_values("add", &[3, 2])));

    let Op::Permute(ops::Permute { src, .. }) = result.op() else { panic!("output permutation must remain") };
    let Op::Reshape(ops::Reshape { src, .. }) = src.op() else { panic!("output reshape must remain") };
    let Op::Wmma(ops::Wmma { c, .. }) = src.op() else { panic!("reshape-permute add must fuse into WMMA") };
    assert!(
        matches!(c.op(), Op::Binary(BinaryOp::Add, _, moved) if matches!(moved.op(), Op::Reshape(ops::Reshape { src, .. })
        if matches!(src.op(), Op::Permute(..))))
    );
}

#[test]
fn test_movement_cleanup_must_precede_reduce_local() {
    let wmma = test_wmma(shaped_values("acc", &[6]));
    let inner = wmma.try_reshape(&smallvec![SInt::Const(3), SInt::Const(2)]).unwrap();
    let outer = inner.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let root = outer.try_permute(vec![1, 0]).unwrap().add(&shaped_values("add", &[3, 2]));

    let mut ctx = crate::devectorize::ReduceContext::default();
    let without_cleanup = crate::rewrite::graph_rewrite(&crate::devectorize::pm_reduce_local(), root.clone(), &mut ctx);
    assert!(matches!(without_cleanup.op(), Op::Binary(BinaryOp::Add, ..)), "counterexample must not match early");

    let matcher = crate::devectorize::movement_cleanup_patterns().with_context::<crate::devectorize::ReduceContext>()
        + crate::devectorize::pm_reduce_local();
    let mut ctx = crate::devectorize::ReduceContext::default();
    let ordered = crate::rewrite::graph_rewrite(&matcher, root, &mut ctx);
    assert!(ordered.toposort().iter().any(|node| matches!(node.op(), Op::Wmma(ops::Wmma { c, .. })
        if matches!(c.op(), Op::Binary(BinaryOp::Add, ..)))));
}

/// Every reduce op lowers to the same accumulator skeleton: a dense REG slot, a
/// loop END and no surviving REDUCE.
#[test_case(ReduceOp::Add, 16; "add")]
#[test_case(ReduceOp::Mul, 8; "mul")]
#[test_case(ReduceOp::Max, 32; "max")]
#[test_case(ReduceOp::Min, 32; "min")]
#[test_case(ReduceOp::Add, 1; "single element range")]
fn reduce_lowers_to_an_accumulator_loop(reduce_op: ReduceOp, extent: i64) {
    let reduce = create_reduce(create_float_const(1.0), vec![create_range_reduce(extent, 0)], reduce_op);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce(..)), "REDUCE must be replaced by the accumulator pattern");
    assert_eq!(result.dtype(), DType::Float32);
    assert!(
        result.toposort().iter().any(|node| matches!(node.op(), Op::Buffer(ops::Buffer { arg, .. })
            if arg.addrspace == Some(svod_ir::AddrSpace::Reg) && arg.slot == 0)),
        "the first accumulator must use dense REG slot 0"
    );
    assert!(count_ends(&result) > 0, "the reduce loop must be closed by an END");
}

#[test]
fn test_reduce_multiple_ranges() {
    let ranges = vec![create_range_reduce(8, 0), create_range_reduce(4, 1)];
    let result = apply_pm_reduce(&create_reduce(create_float_const(1.0), ranges, ReduceOp::Add));

    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(count_define_regs(&result) > 0);
    assert!(count_ends(&result) > 0);
}

/// A REDUCE over a LOAD: the realistic shape, where the reduce range is also the
/// load address.
#[test]
fn test_reduce_over_load_lowers_to_an_accumulator() {
    let range = create_range_reduce(32, 0);
    let index = UOp::index().buffer(UOp::param(0, 1024, DType::Float32, None)).indices(vec![range.clone()]).call();
    let load = UOp::load().index(index.unwrap()).call();

    let result = apply_pm_reduce(&load.reduce(smallvec![range], ReduceOp::Add));

    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(count_define_regs(&result) > 0);
}

#[test]
fn test_invalid_padded_lane_survives_reduction_removal() {
    let cond = UOp::var("valid", DType::Bool, 0, 1);
    let value = UOp::var("value", DType::Float32, 0, 100);
    let src = UOp::try_where(cond, value, UOp::invalid_marker()).unwrap();
    let reduce = create_reduce(src, vec![create_range_reduce(16, 0)], ReduceOp::Max);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(
        result.any_in_subtree(UOp::is_invalid_marker),
        "reduction removal must preserve Invalid for the later gater"
    );
}

#[test]
fn test_reduce_shaped_to_scalar() {
    let src = UOp::stack((0..4).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect());
    let reduce = src.reduce_with_num_axes(smallvec![create_range_reduce(16, 0)], ReduceOp::Add, 1);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(count_define_regs(&result) > 0);
    assert!(result.toposort().iter().any(|node| {
        matches!(node.op(), Op::Index(ops::Index { buffer, indices }) if Arc::ptr_eq(buffer, &src) && indices.len() == 1)
    }));
}

/// Without a range there is no accumulator: the shaped source folds straight into
/// a left fold of scalar INDEXes.
#[test]
fn test_horizontal_reduce_no_ranges() {
    let src = UOp::stack((0..4).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect());

    let result = apply_pm_reduce(&src.reduce_with_num_axes(smallvec![], ReduceOp::Add, 1));

    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert_eq!(count_define_regs(&result), 0, "a horizontal-only reduce needs no DEFINE_REG");
    assert_eq!(result.dtype(), DType::Float32);
    assert_eq!(
        count_ops(
            &result,
            |node| matches!(node.op(), Op::Index(ops::Index { buffer, .. }) if Arc::ptr_eq(buffer, &src))
        ),
        4
    );
}

#[test]
fn test_horizontal_reduce_uses_target_dtype() {
    let source_dtype = DType::BFloat16.vec(16).unwrap();
    let target_dtype = DType::BFloat16.vec(4).unwrap();
    let src = UOp::stack((0..4).map(|i| UOp::const_(source_dtype.clone(), ConstValue::Float(i as f64))).collect());
    let reduce = UOp::new(
        Op::Reduce(ops::Reduce {
            src: src.clone(),
            ranges: smallvec![create_range_reduce(16, 0)],
            reduce_op: ReduceOp::Add,
            num_axes: 1,
        }),
        target_dtype.clone(),
    );

    let result = apply_pm_reduce(&reduce);

    assert_eq!(result.dtype(), target_dtype);
    for node in result.toposort() {
        if let Op::Index(ops::Index { buffer, .. }) = node.op()
            && Arc::ptr_eq(buffer, &src)
        {
            assert_eq!(node.dtype(), target_dtype);
        }
        if let Op::Binary(BinaryOp::Add, lhs, rhs) = node.op() {
            assert_eq!(lhs.dtype(), rhs.dtype());
        }
    }
    assert!(!result.toposort().iter().any(|node| matches!(node.op(), Op::Cast(..))));
}

/// tinygrad puts every RANGE reachable from the source into `input_ranges`, whatever
/// its axis type; only the reduce range itself (and already-ended ranges) drop out.
#[test_case(&[AxisType::Thread]; "thread")]
#[test_case(&[AxisType::Global]; "global")]
#[test_case(&[AxisType::Local]; "local")]
#[test_case(&[AxisType::Loop]; "loop axis")]
#[test_case(&[]; "reduce range is its own only input")]
#[test_case(&[AxisType::Global, AxisType::Thread, AxisType::Loop]; "mixed")]
fn input_ranges_accept_every_axis_type(outer: &[AxisType]) {
    let reduce_range = create_range_reduce(16, 9);
    let src = outer
        .iter()
        .enumerate()
        .map(|(id, &axis_type)| create_range(8 << id, id, axis_type).cast(DType::Float32))
        .reduce(|acc, term| acc.add(&term))
        .unwrap_or_else(|| reduce_range.cast(DType::Float32));
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce(..)));
    assert!(count_define_regs(&result) > 0);
}
