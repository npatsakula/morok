//! White-box tests over `crate::optimizer` internals: local-buffer staging and
//! the index-lowering stage matchers. They reach private helpers, so they live
//! beside the rest of the optimizer unit tests rather than in the source file.

mod stage_local_tests {
    use crate::optimizer::{LocalBufferContext, add_local_buffer};
    use std::sync::Arc;
    use svod_dtype::{AddrSpace, DType};
    use svod_ir::{AxisId, Op, UOp};

    #[test]
    fn add_local_buffer_matches_stage_mapping_and_numbering() {
        let r0 = UOp::range_const(2, 0);
        let r1 = UOp::range_const(3, 1);
        let compute = UOp::native_const(7.0f32);
        let stage = UOp::stage_local(compute.clone(), vec![r0.clone(), r1.clone()]);
        assert_eq!(
            stage.shape().unwrap().unwrap().iter().map(|x| x.as_const()).collect::<Vec<_>>(),
            [Some(2), Some(3)]
        );

        let mut ctx = LocalBufferContext::default();
        let lowered = add_local_buffer(&stage, &mut ctx).unwrap();
        assert_eq!(lowered.dtype(), DType::Float32);

        let Op::After(svod_ir::ops::After { passthrough, deps }) = lowered.op() else { panic!("expected AFTER") };
        let storage = passthrough.base();
        assert!(matches!(storage.op(), Op::Buffer(svod_ir::ops::Buffer { arg, .. })
            if arg.slot == 0 && arg.dtype == DType::Float32 && arg.addrspace == Some(AddrSpace::Local)));
        let [end] = deps.as_slice() else { panic!("expected one dependency") };
        let Op::End(svod_ir::ops::End { computation, ranges }) = end.op() else { panic!("expected END") };
        assert!(ranges.iter().zip([&r0, &r1]).all(|(actual, expected)| Arc::ptr_eq(actual, expected)));
        let Op::Store(svod_ir::ops::Store { index, value, gate: None }) = computation.op() else {
            panic!("expected STORE")
        };
        assert!(Arc::ptr_eq(value, &compute));
        let Op::Index(svod_ir::ops::Index { buffer, indices }) = index.op() else { panic!("expected INDEX") };
        assert!(Arc::ptr_eq(buffer, passthrough));
        assert!(indices.iter().zip([&r0, &r1]).all(|(actual, expected)| Arc::ptr_eq(actual, expected)));

        let second = UOp::stage_local(UOp::native_const(8.0f32), vec![]);
        let second = add_local_buffer(&second, &mut ctx).unwrap();
        assert!(matches!(second.buf_uop().op(), Op::Buffer(svod_ir::ops::Buffer { arg, .. }) if arg.slot == 1));
    }

    #[test]
    fn grouped_local_axis_drives_slot_without_colliding_with_nested_axes() {
        let scalar_axis = AxisId::Renumbered(7);
        let nested_axis = scalar_axis.child(0);
        let scalar =
            UOp::stage(UOp::native_const(1.0f32), vec![], svod_ir::BufferizeOpts::local_for_axis(scalar_axis.clone()));
        let nested =
            UOp::stage(UOp::native_const(2.0f32), vec![], svod_ir::BufferizeOpts::local_for_axis(nested_axis.clone()));
        assert!(!Arc::ptr_eq(&scalar, &nested));

        let mut ctx = LocalBufferContext::default();
        let scalar = add_local_buffer(&scalar, &mut ctx).unwrap();
        let nested = add_local_buffer(&nested, &mut ctx).unwrap();
        let scalar_slot = match scalar.buf_uop().op() {
            Op::Buffer(svod_ir::ops::Buffer { arg, .. }) => arg.slot,
            _ => unreachable!(),
        };
        let nested_slot = match nested.buf_uop().op() {
            Op::Buffer(svod_ir::ops::Buffer { arg, .. }) => arg.slot,
            _ => unreachable!(),
        };
        assert_eq!(scalar_slot, 7);
        assert_ne!(nested_slot, scalar_slot);
        assert_eq!(nested_slot, LocalBufferContext::axis_slot(&nested_axis));
    }

    #[test]
    fn grouped_local_slots_repeat_across_kernel_rewrites() {
        fn lower() -> Vec<usize> {
            let stages = [AxisId::Renumbered(3), AxisId::Renumbered(3).child(1)]
                .into_iter()
                .enumerate()
                .map(|(value, axis)| {
                    UOp::stage(UOp::native_const(value as f32), vec![], svod_ir::BufferizeOpts::local_for_axis(axis))
                })
                .collect::<Vec<_>>();
            let mut ctx = LocalBufferContext::default();
            stages
                .iter()
                .map(|stage| match add_local_buffer(stage, &mut ctx).unwrap().buf_uop().op() {
                    Op::Buffer(svod_ir::ops::Buffer { arg, .. }) => arg.slot,
                    _ => unreachable!(),
                })
                .collect()
        }
        assert_eq!(lower(), lower());
    }
}

mod lower_index_stage_tests {
    use crate::optimizer::{
        Renderer, apply_post_optimization_with_renderer, extra_symbolic_patterns, lower_index_patterns,
    };
    use crate::rewrite::graph_rewrite;
    use crate::symbolic::patterns::symbolic;
    use std::sync::Arc;
    use svod_dtype::DType;
    use svod_ir::UOp;
    use svod_ir::{BinaryOp, ConstValue, Op};

    fn weak(value: i64) -> Arc<UOp> {
        UOp::const_(DType::WeakInt, ConstValue::Int(value))
    }

    fn weak_float(value: f64) -> Arc<UOp> {
        UOp::const_(DType::WeakFloat, ConstValue::Float(value))
    }

    fn production_value(value: Arc<UOp>) -> Arc<UOp> {
        let root = graph_rewrite(extra_symbolic_patterns(), UOp::sink(vec![value]), &mut ());
        let root = graph_rewrite(lower_index_patterns(), root, &mut crate::symbolic::WeakMemo::default());
        let lowered = graph_rewrite(symbolic(), root, &mut ());
        let Op::Sink(svod_ir::ops::Sink { sources, .. }) = lowered.op() else { panic!("expected SINK") };
        sources[0].clone()
    }

    fn lower_value(value: Arc<UOp>) -> Arc<UOp> {
        let lowered =
            graph_rewrite(lower_index_patterns(), UOp::sink(vec![value]), &mut crate::symbolic::WeakMemo::default());
        let Op::Sink(svod_ir::ops::Sink { sources, .. }) = lowered.op() else { panic!("expected SINK") };
        sources[0].clone()
    }

    #[test]
    fn lower_index_composition_pushes_long_cast_through_invalid() {
        let buffer = UOp::param(0, 16, DType::Float32, None);
        let x = UOp::variable("x".into(), 0, 15, DType::WeakInt);
        let valid = x.lt(&weak(8));
        let index = UOp::index().buffer(buffer).indices(vec![x.valid(valid).cast(DType::Int64)]).call().unwrap();

        let lowered = graph_rewrite(lower_index_patterns(), index, &mut crate::symbolic::WeakMemo::default());
        let Op::Index(svod_ir::ops::Index { indices, .. }) = lowered.op() else { panic!("expected INDEX") };
        let Op::Ternary(_, _, value, invalid) = indices[0].op() else { panic!("expected gated index") };
        assert_eq!(value.dtype(), DType::Int32, "{}", lowered.tree());
        assert!(UOp::is_invalid_marker(invalid));
        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
    }

    #[test]
    fn post_optimization_propagates_stale_index_before_decomposition() {
        let stale = UOp::param(0, 1, DType::Index, None);
        let renderer = Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
        let err = apply_post_optimization_with_renderer(UOp::sink(vec![stale]), &renderer)
            .expect_err("legacy Index must fail at the post-index-lowering invariant");
        assert!(err.to_string().contains("post-index-lowering"), "unexpected error: {err}");
        assert!(err.to_string().contains("legacy Index dtype"), "unexpected error: {err}");
    }

    #[test]
    fn extra_symbolic_distributes_weak_index_before_lowering() {
        let buffer = UOp::param(0, 64, DType::Float32, None);
        let x = UOp::variable("x".into(), 0, 7, DType::WeakInt);
        let index_expr = x.add(&weak(2)).mul(&weak(4));
        let index = UOp::index().buffer(buffer).indices(vec![index_expr]).call().unwrap();

        let distributed = graph_rewrite(extra_symbolic_patterns(), index, &mut ());
        let Op::Index(svod_ir::ops::Index { indices, .. }) = distributed.op() else { panic!("expected INDEX") };
        assert!(matches!(indices[0].op(), Op::Binary(BinaryOp::Add, ..)), "{}", distributed.tree());

        let lowered = graph_rewrite(lower_index_patterns(), distributed, &mut crate::symbolic::WeakMemo::default());
        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
    }

    /// The f32 midpoint: two weak floats a `f64` tells apart but an `f32` does not.
    /// Folding must happen after the commitment to f32, or the two disagree.
    fn midpoint() -> f64 {
        1.0 + 2f64.powi(-24)
    }

    #[test]
    fn lower_index_commits_weak_floats_before_folding_their_consumer() {
        let neighbor = 1.0 + 2f64.powi(-23);
        let vconst = |values: [f64; 2]| {
            UOp::vconst(
                vec![ConstValue::Float(values[0]), ConstValue::Float(values[1]), ConstValue::Invalid],
                DType::WeakFloat,
            )
        };

        let comparison = production_value(vconst([midpoint(), neighbor]).try_cmpeq(&vconst([1.0, 1.0])).unwrap());
        assert_eq!(comparison.dtype(), DType::Bool.vec(3).unwrap(), "{}", comparison.tree());
        assert!(matches!(comparison.op(), Op::VConst(svod_ir::ops::VConst { values })
            if values == &vec![ConstValue::Bool(true), ConstValue::Bool(false), ConstValue::Invalid]));

        let sum = production_value(vconst([midpoint(), neighbor]).try_add(&vconst([midpoint(); 2])).unwrap());
        assert_eq!(sum.dtype(), DType::Float32.vec(3).unwrap());
        assert!(matches!(sum.op(), Op::VConst(svod_ir::ops::VConst { values })
            if values == &vec![ConstValue::Float(2.0), ConstValue::Float(2.0), ConstValue::Invalid]));

        let scalar_comparison = production_value(weak_float(midpoint()).try_cmpeq(&weak_float(1.0)).unwrap());
        assert_eq!(scalar_comparison.dtype(), DType::Bool);
        assert!(matches!(scalar_comparison.op(), Op::Const(value) if value.0 == ConstValue::Bool(true)));

        let scalar_sum = production_value(weak_float(midpoint()).try_add(&weak_float(midpoint())).unwrap());
        assert_eq!(scalar_sum.dtype(), DType::Float32);
        assert!(matches!(scalar_sum.op(), Op::Const(value) if value.0 == ConstValue::Float(2.0)));
    }

    #[test]
    fn lower_index_commits_constant_stack_lanes_before_their_consumer() {
        let lhs = UOp::stack(vec![weak_float(midpoint()), UOp::invalid_marker()].into());
        let rhs = UOp::stack(vec![weak_float(1.0), UOp::invalid_marker()].into());
        let comparison = lhs.try_cmpeq(&rhs).unwrap();

        let lowered = lower_value(comparison);

        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
        let Op::Binary(BinaryOp::Eq, lhs, rhs) = lowered.op() else {
            panic!("expected comparison, got {}", lowered.tree())
        };
        for stack in [lhs, rhs] {
            assert_eq!(stack.dtype(), DType::Float32);
            let Op::Stack(svod_ir::ops::Stack { sources }) = stack.op() else { panic!("expected STACK") };
            assert!(matches!(sources[0].op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)));
            assert!(UOp::is_invalid_marker(&sources[1]));
        }
    }

    #[test]
    fn production_commits_weak_coefficients_before_term_combining() {
        let x = UOp::variable("x".into(), -10, 10, DType::Float32);
        let expression =
            x.try_mul(&weak_float(midpoint())).unwrap().try_add(&x.try_mul(&weak_float(-1.0)).unwrap()).unwrap();

        let lowered = production_value(expression);

        assert!(
            matches!(lowered.op(), Op::Binary(BinaryOp::Mul, value, zero)
                if Arc::ptr_eq(value, &x)
                    && matches!(zero.op(), Op::Const(value) if value.0 == ConstValue::Float(0.0))),
            "{}",
            lowered.tree()
        );
    }

    #[test]
    fn production_commits_weak_comparison_before_where_bounds() {
        let condition = weak_float(1.0).try_cmplt(&weak_float(midpoint())).unwrap();
        let expression = UOp::try_where(condition, UOp::native_const(7i32), UOp::native_const(9i32)).unwrap();

        let lowered = production_value(expression);

        assert!(matches!(lowered.op(), Op::Const(value) if value.0 == ConstValue::Int(9)), "{}", lowered.tree());
    }

    #[test]
    fn production_commits_weak_base_before_power_decomposition() {
        let exponent = weak_float(1.0).try_add(&weak_float(1.0)).unwrap();
        let expression = weak_float(midpoint()).try_pow(&exponent).unwrap();

        let lowered = production_value(expression);

        assert!(matches!(lowered.op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)), "{}", lowered.tree());
    }

    #[test]
    fn production_scalar_midpoint_neighbors_match_f32_commitment() {
        for (value, expected) in [
            (f64::from_bits(midpoint().to_bits() - 1), true),
            (midpoint(), true),
            (f64::from_bits(midpoint().to_bits() + 1), false),
        ] {
            let comparison = weak_float(value).try_cmpeq(&weak_float(1.0)).unwrap();
            let lowered = production_value(comparison);
            assert!(
                matches!(lowered.op(), Op::Const(result) if result.0 == ConstValue::Bool(expected)),
                "value={value:?} expected={expected}: {}",
                lowered.tree()
            );
        }
    }
}
