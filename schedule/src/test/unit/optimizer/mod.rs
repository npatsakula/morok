pub mod heuristics;
pub mod implicit_barriers;
pub mod mod_internal;
pub mod opts_to_apply;
pub mod opts_validation;
pub mod scheduler;
pub mod tc;

#[cfg(test)]
mod pipeline_composition {
    use crate::linearize::pm_split_ends;
    use crate::optimizer::apply_pre_optimization;
    use crate::rewrite::graph_rewrite;
    use smallvec::smallvec;
    use std::sync::Arc;
    use svod_ir::{AxisId, AxisType, CanonicalGraph, ConstValue, DType, Op, ReduceOp, UOp};

    fn postopt_symbolic(root: Arc<UOp>) -> Arc<UOp> {
        graph_rewrite(&*crate::optimizer::POST_OPT_SYM, root, &mut ())
    }

    #[test]
    fn test_postopt_symbolic_removes_range_unparented_after_zero_fold() {
        let range = UOp::range_axis(UOp::index_const(7), AxisId::Renumbered(0), AxisType::Reduce);
        let src = range.cast(DType::Int32).mul(&UOp::native_const(0i32));
        let reduce = src.reduce(smallvec![range], ReduceOp::Add);

        let result = postopt_symbolic(reduce);

        assert!(!result.toposort().iter().any(|u| matches!(u.op(), Op::Reduce(..))));
        assert!(matches!(result.op(), Op::Const(value) if value.0 == ConstValue::Int(0)));
    }

    #[test]
    fn test_postopt_symbolic_keeps_parented_reduction() {
        let range = UOp::range_axis(UOp::index_const(7), AxisId::Renumbered(0), AxisType::Reduce);
        let reduce = range.cast(DType::Int32).reduce(smallvec![range], ReduceOp::Add);

        let result = postopt_symbolic(reduce);

        assert!(matches!(result.op(), Op::Reduce(svod_ir::ops::Reduce { ranges, .. }) if ranges.len() == 1));
    }

    #[test]
    fn test_postopt_where_load_keeps_index_dependent_condition() {
        let data = UOp::param(0, 8, DType::Float32, None);
        let predicates = UOp::param(1, 8, DType::Bool, None);
        let offset = UOp::index_const(0);
        let data_index = UOp::index().buffer(data).indices(vec![offset.clone()]).call().expect("data index");
        let loaded_condition = UOp::index().buffer(predicates).indices(vec![offset]).call().expect("predicate index");
        let masked = UOp::try_where(loaded_condition, data_index, UOp::native_const(0.0f32)).expect("WHERE");

        let result = postopt_symbolic(masked);

        assert!(
            matches!(result.op(), Op::Ternary(_, _, _, _)),
            "index-dependent WHERE must not move: {}",
            result.tree()
        );
    }

    #[test]
    fn test_pm_split_ends_reattaches_bool_and_void_backedges_outermost() {
        let computation = UOp::native_const(1.0f32);
        let outer_range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);
        let inner_range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(1), AxisType::Loop);
        let bool_backedge = UOp::const_(DType::Bool, ConstValue::Bool(true));
        let void_backedge = UOp::noop();
        let original = computation
            .end(smallvec![bool_backedge.clone(), outer_range.clone(), void_backedge.clone(), inner_range.clone()])
            .with_tag(smallvec![17, 23]);

        let result = graph_rewrite(pm_split_ends(), original, &mut ());

        assert_eq!(result.tag().as_deref(), Some(&[17, 23][..]));
        let Op::End(svod_ir::ops::End { computation: range_ends, ranges: backedges }) = result.op() else {
            panic!("expected outer backedge END, got {}", result.tree());
        };
        assert_eq!(backedges.len(), 2);
        assert!(Arc::ptr_eq(&backedges[0], &bool_backedge));
        assert!(Arc::ptr_eq(&backedges[1], &void_backedge));
        assert_eq!(range_ends.tag(), &None);

        let Op::End(svod_ir::ops::End { computation: inner_end, ranges: outer_ranges }) = range_ends.op() else {
            panic!("expected outer range END, got {}", result.tree());
        };
        assert_eq!(outer_ranges.len(), 1);
        assert!(Arc::ptr_eq(&outer_ranges[0], &outer_range));
        let Op::End(svod_ir::ops::End { computation: leaf, ranges: inner_ranges }) = inner_end.op() else {
            panic!("expected inner range END, got {}", result.tree());
        };
        assert!(Arc::ptr_eq(leaf, &computation));
        assert_eq!(inner_ranges.len(), 1);
        assert!(Arc::ptr_eq(&inner_ranges[0], &inner_range));
    }

    #[test]
    fn test_pm_split_ends_empty_target_ranges_preserves_backedges_and_identity() {
        let computation = UOp::native_const(2.0f32);
        let bool_backedge = UOp::const_(DType::Bool, ConstValue::Bool(false));
        let void_backedge = UOp::noop();
        let original = computation.end(smallvec![bool_backedge, void_backedge]).with_tag(smallvec![99]);

        let result = graph_rewrite(pm_split_ends(), original.clone(), &mut ());

        assert!(Arc::ptr_eq(&result, &original));
        assert_eq!(result.tag().as_deref(), Some(&[99][..]));
    }

    #[test]
    fn test_pm_split_ends_sorts_nested_axis_ids_and_preserves_range_dependencies() {
        let computation = UOp::native_const(3.0f32);
        let dependency = UOp::const_(DType::Bool, ConstValue::Bool(true));
        let parent = AxisId::Renumbered(2);
        let parent_range = UOp::range_axis(UOp::index_const(2), parent.clone(), AxisType::Loop);
        let child_zero = UOp::range_axis(UOp::index_const(3), parent.child(0), AxisType::Loop);
        let child_one = UOp::range_axis(UOp::index_const(5), parent.child(1), AxisType::Loop)
            .with_sources(vec![UOp::index_const(5), dependency.clone()]);
        let original = computation.end(smallvec![parent_range.clone(), child_zero.clone(), child_one.clone()]);

        let result = graph_rewrite(pm_split_ends(), original, &mut ());

        let mut cursor = result.clone();
        for expected in [&parent_range, &child_zero, &child_one] {
            let Op::End(svod_ir::ops::End { computation: next, ranges }) = cursor.op() else {
                panic!("expected nested END chain, got {}", result.tree());
            };
            assert_eq!(ranges.len(), 1);
            assert!(Arc::ptr_eq(&ranges[0], expected));
            cursor = next.clone();
        }
        assert!(Arc::ptr_eq(&cursor, &computation));
        let Op::Range(svod_ir::ops::Range { deps, .. }) = child_one.op() else { unreachable!() };
        assert_eq!(deps.len(), 1);
        assert!(Arc::ptr_eq(&deps[0], &dependency));

        let expected = computation.end(smallvec![child_one]).end(smallvec![child_zero]).end(smallvec![parent_range]);
        assert!(Arc::ptr_eq(&result, &expected), "split form must remain hash-canonical\n{}", result.tree());
        assert_eq!(
            CanonicalGraph::from_root("split_end", &result).unwrap(),
            CanonicalGraph::from_root("split_end", &expected).unwrap()
        );
        assert_eq!(result.tree(), expected.tree());
    }

    #[test]
    fn test_preopt_split_exposes_range_flattening_in_same_rewrite() {
        let range = UOp::range_axis(UOp::index_const(12), AxisId::Renumbered(0), AxisType::Loop);
        let computation = range.mod_(&UOp::index_const(4));
        let sink = UOp::sink(vec![computation.end(smallvec![range])]);

        let result = apply_pre_optimization(sink).unwrap();
        let Op::Sink(svod_ir::ops::Sink { sources, .. }) = result.op() else {
            panic!("expected SINK, got {:?}", result.op())
        };
        let Op::End(svod_ir::ops::End { ranges, .. }) = sources[0].op() else {
            panic!("expected END, got {:?}", sources[0].op())
        };
        assert_eq!(ranges.len(), 2, "split RANGE dependencies must be flattened into the END");
        assert!(ranges.iter().all(|range| matches!(range.op(), Op::Range(..))));
    }

    #[test]
    fn test_preopt_cast_const_fold_enables_range_end_arithmetic() {
        let cast = UOp::const_(DType::Int32, ConstValue::Int(3)).cast(DType::Index);
        let end = cast.add(&UOp::const_(DType::Index, ConstValue::Int(1)));
        let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);
        let sink = UOp::sink(vec![range.clone().end(smallvec![range])]);

        let result = apply_pre_optimization(sink).unwrap();
        let range = result
            .toposort()
            .into_iter()
            .find(|u| matches!(u.op(), Op::Range(..)))
            .expect("range must survive pre-optimization");
        let Op::Range(svod_ir::ops::Range { end, .. }) = range.op() else { unreachable!() };
        assert!(
            matches!(end.op(), Op::Const(value) if value.0 == ConstValue::Int(4)),
            "range end was not folded: {:?}",
            end.op()
        );
    }
}
