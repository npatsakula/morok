use super::*;
use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::types::{AxisId, AxisType, ConstValue, InsArg, RendererDevice, WmmaMetadata};

use crate::linearize::line_rewrite_cleanups;
use test_case::test_case;

#[test]
fn tinygrad_partial_range_comparison_rejects_path_prefixes() {
    let short = ArgKey::Range(vec![0], axis_type_value(AxisType::Weak));
    let long = ArgKey::Range(vec![0, 1], axis_type_value(AxisType::Weak));
    assert_eq!(partial_arg_cmp(&short, &long), None);
    assert_eq!(partial_arg_cmp(&long, &short), None);
}

#[test]
fn tinygrad_partial_param_comparison_stops_at_first_difference() {
    let projected = arg_key(&Op::Param {
        shape: UOp::index_const(1),
        arg: ParamArg::variable("projected".to_string(), DType::WeakInt, 0, 8).into(),
    });
    assert!(matches!(projected, ArgKey::Param(ParamKey { addrspace: Some(4), .. })));

    let mut left = param_key(&ParamArg::variable("a".to_string(), DType::WeakInt, 0, 8));
    let mut right = param_key(&ParamArg::variable("b".to_string(), DType::WeakInt, 0, 8));
    left.addrspace = None;
    right.addrspace = Some(1);
    assert_eq!(partial_param_cmp(&left, &right), Some(Ordering::Less));
    assert_eq!(partial_param_cmp(&right, &left), Some(Ordering::Greater));
}

#[test]
fn tinygrad_float_keys_coalesce_signed_zero_and_nan_payloads() {
    assert_eq!(const_key(ConstValue::Float(-0.0)), const_key(ConstValue::Float(0.0)));
    let left = const_key(ConstValue::Float(f64::from_bits(0x7ff8_0000_0000_0001)));
    let right = const_key(ConstValue::Float(f64::from_bits(0x7ff8_0000_0000_0002)));
    assert_eq!(left, right);
    assert_eq!(partial_const_cmp(&left, &right), None);
}

#[test]
fn tinygrad_vconst_linearizer_key_is_stack_of_constants() {
    let vconst = UOp::vconst(vec![ConstValue::Int(1), ConstValue::Int(2)], DType::WeakInt);
    let keys = compute_tuplize(&vconst.toposort());
    let key = &keys[&vconst.id];
    assert_eq!(key.op, 16);
    assert_eq!(key.arg, ArgKey::None);
    assert_eq!(key.dtype, dtype_key(&DType::WeakInt));
    assert_eq!(key.src.len(), 2);
    assert!(key.src.iter().all(|source| source.op == 61 && source.dtype == dtype_key(&DType::WeakInt)));

    let stack = UOp::stack(smallvec![UOp::range_const(8, 0), UOp::special(UOp::index_const(8), "gidx0".to_string())]);
    assert_eq!(tinygrad_tuplize_cmp(&stack, &vconst), Some(Ordering::Less));
}

#[test]
fn tinygrad_wmma_key_omits_svod_only_metadata() {
    let value = UOp::native_const(0.0f32);
    let mut left = WmmaMetadata {
        name: "z_svod_name".to_string(),
        dims: (16, 16, 16),
        dtype_in: DType::Float16,
        dtype_out: DType::Float32,
        device: RendererDevice::CudaSm80,
        threads: 32,
        upcast_axes: None,
        reduce_axes: vec![AxisId::Renumbered(3)],
        tile_grid: (2, 2),
    };
    let mut right = left.clone();
    right.name = "a_svod_name".to_string();
    right.dtype_out = DType::Int32;
    right.reduce_axes.clear();
    right.tile_grid = (1, 1);

    let make_op =
        |metadata| Op::Wmma { a: value.clone(), b: value.clone(), c: value.clone(), metadata: Box::new(metadata) };
    assert_eq!(arg_key(&make_op(left.clone())), arg_key(&make_op(right.clone())));

    right.device = RendererDevice::CudaSm89;
    right.upcast_axes = Some(svod_ir::WmmaUpcastAxes { a: vec![(AxisId::Unrenumbered(3), 2)], b: vec![], c: vec![] });
    left.upcast_axes =
        Some(svod_ir::WmmaUpcastAxes { a: vec![(AxisId::RenumberedPath(smallvec![3, 1]), 2)], b: vec![], c: vec![] });
    assert_eq!(arg_key(&make_op(left.clone())), arg_key(&make_op(right.clone())));

    let left_uop = UOp::new(make_op(left.clone()), DType::Float32);
    let right_uop = UOp::new(make_op(right.clone()), DType::Float32);
    assert_eq!(tinygrad_tuplize_cmp(&left_uop, &right_uop), Some(Ordering::Equal));

    left.dims = (8, 16, 16);
    assert_ne!(
        arg_key(&make_op(left)),
        arg_key(&make_op(WmmaMetadata {
            name: "ignored".to_string(),
            dims: (16, 16, 16),
            dtype_in: DType::Float16,
            dtype_out: DType::Float32,
            device: RendererDevice::CudaSm80,
            threads: 32,
            upcast_axes: None,
            reduce_axes: vec![],
            tile_grid: (1, 1),
        }))
    );
}

/// Linearization is a topological order over the whole graph: each node is emitted
/// once, after every source it depends on, with the root SINK last.
#[test]
fn linearize_emits_each_node_after_its_sources() {
    // A diamond over three constants (shared source), wrapped in a loop: covers
    // constants, binary ops, RANGE/END and the terminating SINK.
    let shared = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let left = shared.try_add(&UOp::const_(DType::Float32, ConstValue::Float(2.0))).unwrap();
    let right = shared.try_add(&UOp::const_(DType::Float32, ConstValue::Float(3.0))).unwrap();
    let sink = UOp::sink(vec![left.try_add(&right).unwrap().end(smallvec![UOp::range_const(10, 0)])]);

    let order = linearize(sink.clone());
    let at = |node: &Arc<UOp>| order.iter().position(|emitted| Arc::ptr_eq(emitted, node));

    for node in sink.toposort() {
        let position = at(&node).unwrap_or_else(|| panic!("{:?} was never emitted", node.op()));
        for source in node.op().sources() {
            let source_position = at(&source).unwrap_or_else(|| panic!("{:?} was never emitted", source.op()));
            assert!(source_position < position, "{:?} emitted before its source {:?}", node.op(), source.op());
        }
    }
    assert!(Arc::ptr_eq(order.last().expect("a non-empty linearization"), &sink));
}

/// Upstream dropped the CONST arm (52b989c6c "don't place consts early") and the
/// DEFINE_VAR arm (4a4b6956d): a symbolic variable is placed as a PARAM.
#[test_case(UOp::param(3, 1, DType::Float32, Some(DeviceSpec::Cpu)), (-20, Some(3)); "param carries its slot")]
#[test_case(UOp::variable("n".to_string(), 0, 8, DType::Int32), (-20, Some(-1)); "define var is a param")]
#[test_case(UOp::buffer(1, 1, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu)), (-18, None); "global buffer")]
#[test_case(UOp::buffer(2, 1, DType::Float32, AddrSpace::Reg, None), (-18, None); "register buffer")]
#[test_case(UOp::buffer(0, 1, DType::Float32, AddrSpace::Local, None), (-17, None); "local buffer")]
#[test_case(UOp::const_(DType::Int32, ConstValue::Int(7)), (0, None); "const is not placed early")]
#[test_case(UOp::range_const(10, 0), (5, None); "range is placed late")]
fn tinygrad_placement_priorities(node: Arc<UOp>, expected: (i32, Option<i64>)) {
    assert_eq!(priority(&node), expected);
}

#[test]
fn deep_precast_chain_linearizes_in_tuplize_order() {
    let mut low = UOp::const_(DType::Int32, ConstValue::Int(1));
    let mut high = UOp::const_(DType::Int32, ConstValue::Int(2));
    for _ in 0..140 {
        low = UOp::new(Op::Precast { src: low }, DType::Int32);
        high = UOp::new(Op::Precast { src: high }, DType::Int32);
    }
    let sink = UOp::sink(vec![high.clone(), low.clone()]);
    let keys = compute_tuplize(&sink.toposort());
    assert!(keys[&low.id] < keys[&high.id]);

    let order = linearize(sink);
    assert!(order.iter().position(|u| Arc::ptr_eq(u, &low)) < order.iter().position(|u| Arc::ptr_eq(u, &high)));
}

#[test]
fn equal_dependency_side_effects_use_full_arg_order() {
    let dependency = UOp::native_const(0i32);
    let later = UOp::new(Op::CustomI { deps: smallvec![dependency.clone()], code: "z".into() }, DType::Void);
    let earlier = UOp::new(Op::CustomI { deps: smallvec![dependency], code: "a".into() }, DType::Void);
    let order = linearize(UOp::sink(vec![later.clone(), earlier.clone()]));

    assert!(order.iter().position(|u| Arc::ptr_eq(u, &earlier)) < order.iter().position(|u| Arc::ptr_eq(u, &later)));
}

#[test]
fn nested_axis_and_ins_arguments_participate_in_tuplize() {
    let end = UOp::index_const(4);
    let outer = UOp::range_axis(end.clone(), AxisId::RenumberedPath(smallvec![0, 1]), AxisType::Loop);
    let inner = UOp::range_axis(end, AxisId::RenumberedPath(smallvec![0, 2]), AxisType::Loop);
    assert!(arg_key(outer.op()) < arg_key(inner.op()));

    let source = UOp::native_const(1i32);
    let ins_a = UOp::new(
        Op::Ins {
            sources: smallvec![source.clone()],
            arg: InsArg::with_attributes("v_add", vec![("axis".into(), "1".into())]),
        },
        DType::Int32,
    );
    let ins_b = UOp::new(
        Op::Ins {
            sources: smallvec![source],
            arg: InsArg::with_attributes("v_add", vec![("axis".into(), "2".into())]),
        },
        DType::Int32,
    );
    assert!(arg_key(ins_a.op()) < arg_key(ins_b.op()));
    assert_eq!(op_value(ins_a.op()), 64);
}

#[test]
fn linearize_cleanup_expands_a_gated_store_into_if_endif() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let gate = UOp::native_const(true);
    let store = index.store_gated(UOp::native_const(1.0f32), gate.clone());

    let result = line_rewrite_cleanups(vec![store]);
    assert_eq!(result.len(), 3);
    let Op::If { condition, body } = result[0].op() else { panic!("expected IF") };
    assert!(Arc::ptr_eq(condition, &gate));
    assert_eq!(body.len(), 1);
    assert!(matches!(result[1].op(), Op::Store { gate: None, .. }));
    let Op::EndIf { if_op } = result[2].op() else { panic!("expected ENDIF") };
    assert!(Arc::ptr_eq(if_op, &result[0]));
}

#[test]
fn tuplize_comparison_survives_a_forty_thousand_deep_chain() {
    // 2 MiB is a typical non-main thread stack. The recursive comparison
    // overflowed even the 8 MiB main stack somewhere past 20k levels.
    std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024)
        .spawn(|| {
            let mut low = UOp::const_(DType::Int32, ConstValue::Int(1));
            let mut high = UOp::const_(DType::Int32, ConstValue::Int(2));
            for _ in 0..40_000 {
                low = UOp::new(Op::Precast { src: low }, DType::Int32);
                high = UOp::new(Op::Precast { src: high }, DType::Int32);
            }
            let topo = UOp::sink(vec![high.clone(), low.clone()]).toposort();
            let keys = compute_tuplize(&topo);
            let order = compare_tuplize(&keys[&low.id], &keys[&high.id], &mut HashMap::new());
            assert_eq!(order, Ordering::Less);

            // Releasing a 40k-deep Arc chain recurses in drop glue, which is a
            // separate problem from the comparison under test. Hold the whole
            // graph alive so the thread exits without unwinding it.
            std::mem::forget(keys);
            std::mem::forget(topo);
        })
        .expect("spawn comparison thread")
        .join()
        .expect("deep tuplize comparison must not overflow the stack");
}
