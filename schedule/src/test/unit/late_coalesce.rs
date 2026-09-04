use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{BinaryOp, ConstValue, Op, ParamArg, TernaryOp, UOp};
use test_case::test_case;

use crate::devectorize::devectorize;
use crate::graph_rewrite;
use crate::late::{AddImageContext, indexing_simplify, memory_coalescing, pm_simplify_add_image};
use crate::optimizer::Renderer;
use crate::symbolic::patterns::sym;

fn weak(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

fn x() -> Arc<UOp> {
    UOp::define_var("x".into(), 0, 7)
}

fn image_param() -> Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec![1usize.into(), 4usize.into(), 4usize.into()]);
    let arg = ParamArg::buffer(0, DType::Float32, AddrSpace::Global, None);
    UOp::new(Op::Param { shape, arg: arg.into() }, DType::Float32)
}

fn gated_index(index: &Arc<UOp>) -> (&Arc<UOp>, &Arc<UOp>) {
    let Op::Index { indices, .. } = index.op() else { panic!("expected INDEX, got {}", index.tree()) };
    let Op::Ternary(TernaryOp::Where, valid, idx, invalid) = indices[0].op() else {
        panic!("expected gated index, got {}", index.tree())
    };
    assert!(UOp::is_invalid_marker(invalid));
    (valid, idx)
}

fn gated_scalar_index(start: Arc<UOp>, valid: Arc<UOp>) -> Arc<UOp> {
    let buffer = UOp::param(0, 8, DType::Int32, None);
    UOp::index().buffer(buffer).indices(vec![start.valid(valid)]).call().unwrap()
}

#[test_case(x().mod_(&weak(4)), x().lt(&weak(4)), x(); "modulo folds away under an upper bound")]
#[test_case(x().floor_div(&weak(4)), weak(3).lt(&x()), weak(1); "floor-div folds to a constant under a lower bound")]
fn a_gated_index_is_simplified_under_its_validity(start: Arc<UOp>, valid: Arc<UOp>, expected: Arc<UOp>) {
    let matcher = sym().clone() + indexing_simplify().clone();
    let result = graph_rewrite(&matcher, gated_scalar_index(start, valid.clone()), &mut ());

    let (result_valid, result_idx) = gated_index(&result);
    assert!(Arc::ptr_eq(result_valid, &valid), "validity must survive: {}", result_valid.tree());
    assert!(Arc::ptr_eq(result_idx, &expected), "got {}", result_idx.tree());
}

#[test_case(gated_scalar_index(x(), x().eq(&weak(3))); "validity that parse_valid cannot read")]
#[test_case(UOp::index()
    .buffer(UOp::param(0, 64, DType::Float32, None))
    .indices(vec![weak(0).valid(x().lt(&weak(4))), x().valid(x().lt(&weak(4)))])
    .call()
    .unwrap(); "two coordinates on a non-image buffer")]
fn indexing_simplify_declines(index: Arc<UOp>) {
    let result = graph_rewrite(indexing_simplify(), index.clone(), &mut ());
    assert!(Arc::ptr_eq(&result, &index), "rewrote {} into {}", index.tree(), result.tree());
}

#[test]
fn image_clause_is_dropped_only_when_wrong_side_is_out_of_bounds() {
    let valid = x().lt(&weak(4));
    let index = UOp::index()
        .buffer(image_param())
        .indices(vec![weak(0).valid(valid.clone()), x().valid(valid)])
        .dtype(DType::Float32)
        .call()
        .unwrap();

    let result = graph_rewrite(indexing_simplify(), index, &mut ());
    let Op::Index { indices, .. } = result.op() else { panic!("expected INDEX") };
    assert_eq!(indices.len(), 2);
    assert!(indices.iter().all(|idx| !matches!(idx.op(), Op::Ternary(TernaryOp::Where, ..))));
    assert!(matches!(indices[0].op(), Op::Const(value) if value.0 == ConstValue::Int(0)));
    assert!(Arc::ptr_eq(&indices[1], &x()));
}

fn load_at(buffer: &Arc<UOp>, index: Arc<UOp>) -> Arc<UOp> {
    UOp::load().index(UOp::index().buffer(buffer.clone()).indices(vec![index]).call().unwrap()).call()
}

fn loads(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    root.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Load { .. })).collect()
}

fn stores(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    root.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Store { .. })).collect()
}

fn shrink_count(root: &Arc<UOp>) -> usize {
    root.toposort().iter().filter(|uop| matches!(uop.op(), Op::Shrink { .. })).count()
}

fn no_float4() -> Renderer {
    let mut renderer = Renderer::cpu();
    renderer.supports_float4 = false;
    renderer
}

fn target_coalesce(sink: Arc<UOp>, renderer: &Renderer) -> Arc<UOp> {
    let devectorized = devectorize(&sink, renderer);
    let simplified = graph_rewrite(sym(), devectorized, &mut ());
    memory_coalescing(simplified, renderer)
}

/// A single shaped (already-vectorized) LOAD over `offsets`.
fn shaped_load(offsets: &[i64]) -> Arc<UOp> {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let indices = UOp::stack(offsets.iter().copied().map(UOp::index_const).collect());
    let index = UOp::new(Op::Index { buffer, indices: smallvec![indices] }, DType::Float32);
    UOp::sink(vec![UOp::new(Op::Load { index, alt: None, gate: None }, DType::Float32)])
}

/// Devectorization turns a shaped access into lanes; coalescing regroups them
/// into runs no wider than the target's vector width, keeping the *scalar*
/// memory dtype.
#[test_case(&[0, 1, 2, 3], Renderer::cpu(), 1, 4; "width four is one group")]
#[test_case(&[0, 1, 2, 3, 4, 5, 6, 7], Renderer::cpu(), 2, 4; "width eight is two float4 groups")]
#[test_case(&[0, 1, 2, 3, 8, 9, 10, 11], Renderer::cpu(), 2, 4; "a gap is not bridged into one access")]
#[test_case(&(0..16).collect::<Vec<_>>(), Renderer::apple_amx(), 1, 16; "amx folds a whole sixteen lane register")]
fn a_shaped_load_splits_into_target_width_groups(
    offsets: &[i64],
    renderer: Renderer,
    groups: usize,
    group_width: usize,
) {
    let result = target_coalesce(shaped_load(offsets), &renderer);

    let folded = loads(&result);
    assert_eq!(folded.len(), groups, "{}", result.tree());
    assert!(folded.iter().all(|load| load.dtype() == DType::Float32), "memory dtype must stay scalar");
    assert_eq!(folded[0].shape().unwrap().unwrap()[0].as_const(), Some(group_width));
    let Op::Load { index, .. } = folded[0].op() else { unreachable!() };
    assert_eq!(index.dtype(), DType::Float32);
}

fn shaped_store() -> Arc<UOp> {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let indices = UOp::stack((0..4).map(UOp::index_const).collect());
    let index = UOp::new(Op::Index { buffer, indices: smallvec![indices] }, DType::Float32);
    let value = UOp::stack((0..4).map(|v| UOp::const_(DType::Float32, ConstValue::Float(v as f64))).collect());
    UOp::sink(vec![UOp::new(Op::Store { index, value, gate: None }, DType::Void)])
}

fn scalar_stores() -> Arc<UOp> {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    UOp::sink(
        (0..4)
            .map(|offset| {
                UOp::index()
                    .buffer(buffer.clone())
                    .indices(vec![UOp::index_const(offset)])
                    .call()
                    .unwrap()
                    .store(UOp::const_(DType::Float32, ConstValue::Float(offset as f64)))
            })
            .collect(),
    )
}

#[test_case(shaped_store(); "one shaped store")]
#[test_case(scalar_stores(); "four contiguous scalar stores")]
fn contiguous_stores_fold_to_one_shaped_scalar_store(sink: Arc<UOp>) {
    let result = target_coalesce(sink, &Renderer::cpu());

    let folded = stores(&result);
    assert_eq!(folded.len(), 1, "{}", result.tree());
    let Op::Store { index, value, .. } = folded[0].op() else { unreachable!() };
    assert_eq!(index.dtype(), DType::Float32);
    assert_eq!(value.dtype(), DType::Float32);
    assert!(matches!(value.op(), Op::Stack { sources } if sources.len() == 4));
}

fn contiguous_loads(buffer: Arc<UOp>) -> Arc<UOp> {
    UOp::sink((0..4).map(|offset| load_at(&buffer, UOp::index_const(offset))).collect())
}

fn loads_at(buffer: &Arc<UOp>, indices: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::sink(indices.into_iter().map(|index| load_at(buffer, index)).collect())
}

fn mismatched_validity_loads() -> Arc<UOp> {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let indices = vec![UOp::index_const(0).valid(x().lt(&weak(4))), UOp::index_const(1).valid(x().lt(&weak(5)))];
    loads_at(&buffer, indices)
}

fn two_base_loads() -> Arc<UOp> {
    let buffer = UOp::param(0, 64, DType::Float32, None);
    let bases = [x().mul(&weak(2)), UOp::define_var("y".into(), 0, 7).mul(&weak(2))];
    let indices = bases.iter().flat_map(|base| (0..2).map(|offset| base.add(&weak(offset)))).collect();
    loads_at(&buffer, indices)
}

/// Which contiguous runs the pass is willing to merge, and how many SHRINK
/// group headers it leaves behind (one per merged run, none for scalar runs).
#[test_case(contiguous_loads(UOp::param(0, 16, DType::Float32, None)), Renderer::cpu(), 1, 1; "four contiguous loads fold")]
#[test_case(contiguous_loads(UOp::buffer(0, 4, DType::Float32, AddrSpace::Reg, None).after(smallvec![UOp::noop()])), Renderer::cpu(), 4, 0; "reg accesses never coalesce")]
#[test_case(contiguous_loads(UOp::param(0, 16, DType::Float32, None)), no_float4(), 4, 0; "no float4 keeps scalar accesses")]
#[test_case(contiguous_loads(image_param()), no_float4(), 1, 1; "images use fixed width four regardless of float4")]
#[test_case(mismatched_validity_loads(), Renderer::cpu(), 2, 0; "different validity identities stay apart")]
#[test_case(two_base_loads(), Renderer::cpu(), 2, 2; "different base identities form their own runs")]
#[test_case(loads_at(&UOp::param(0, 16, DType::Float32, None), [0, 1, 3, 4].into_iter().map(UOp::index_const).collect()), Renderer::cpu(), 3, 1; "an unaligned run is not realigned")]
fn coalescing_groups_scalar_loads_by_run(sink: Arc<UOp>, renderer: Renderer, groups: usize, shrinks: usize) {
    let result = memory_coalescing(sink, &renderer);
    assert_eq!(loads(&result).len(), groups, "{}", result.tree());
    assert_eq!(shrink_count(&result), shrinks, "{}", result.tree());
}

/// Ported from tinygrad's `test/test_linearizer.py` grouped-store expectations:
/// a symbolic base plus constant offsets groups in fours, and the remainder
/// stays scalar with its base intact.
#[test]
fn grouped_weak_index_offsets_match_tinygrad_for_widths_four_five_and_eight() {
    let buffer = UOp::param(0, 32, DType::Float32, None);
    let base = UOp::define_var("group_width_base".into(), 0, 3).mul(&weak(8));

    for width in [4usize, 5, 8] {
        let accesses = (0..width).map(|offset| load_at(&buffer, base.add(&base.const_like(offset as i64)))).collect();
        let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
        let folded = loads(&result);
        assert_eq!(folded.len(), width.div_ceil(4), "width {width}: {}", result.tree());

        let mut grouped_widths = Vec::new();
        let mut scalar_offsets = Vec::new();
        for load in folded {
            let Op::Load { index, .. } = load.op() else { unreachable!() };
            match index.op() {
                Op::Shrink { offsets, sizes, .. } => {
                    assert_eq!(offsets.dtype(), DType::WeakInt);
                    assert_eq!(sizes.dtype(), DType::WeakInt);
                    assert_eq!(offsets.shape().unwrap().unwrap().as_slice(), &[]);
                    assert_eq!(sizes.shape().unwrap().unwrap().as_slice(), &[]);
                    let Op::Const(value) = sizes.op() else { panic!("width must be CONST") };
                    grouped_widths.push(value.0);
                }
                Op::Index { indices, .. } => {
                    assert_eq!(indices[0].dtype(), DType::WeakInt);
                    let Op::Binary(BinaryOp::Add, _, offset) = indices[0].op() else {
                        panic!("scalar offset must preserve its base")
                    };
                    let Op::Const(value) = offset.op() else { panic!("offset must be CONST") };
                    scalar_offsets.push(value.0);
                }
                _ => panic!("expected SHRINK or INDEX, got {}", index.tree()),
            }
        }

        let expected_widths =
            if width == 8 { vec![ConstValue::Int(4), ConstValue::Int(4)] } else { vec![ConstValue::Int(4)] };
        assert_eq!(grouped_widths, expected_widths, "width {width}");
        let expected_scalars = if width == 5 { vec![ConstValue::Int(4)] } else { vec![] };
        assert_eq!(scalar_offsets, expected_scalars, "width {width}");
    }
}

#[test]
fn a_shaped_load_with_shared_validity_keeps_one_group_gate() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let valid = x().lt(&weak(4));
    let indices = UOp::stack((0..4).map(|offset| UOp::index_const(offset).valid(valid.clone())).collect());
    let index = UOp::new(Op::Index { buffer, indices: smallvec![indices] }, DType::Float32);
    let load = UOp::new(Op::Load { index, alt: None, gate: None }, DType::Float32);

    let result = target_coalesce(UOp::sink(vec![load]), &Renderer::cpu());
    let folded = loads(&result);
    assert_eq!(folded.len(), 1, "shared validity should produce one shaped access: {}", result.tree());
    let Op::Load { index, .. } = folded[0].op() else { unreachable!() };
    let Op::Shrink { offsets, sizes, .. } = index.op() else { panic!("expected SHRINK: {}", index.tree()) };
    assert!(Arc::ptr_eq(&offsets.get_valid(), &valid));
    assert!(matches!(offsets.get_idx().op(), Op::Const(value) if value.0 == ConstValue::Int(0)));
    assert!(matches!(sizes.op(), Op::Const(value) if value.0 == ConstValue::Int(4)));
}

/// The M=5 WMMA output mapping writes C[0..16), C[32..48) and a gated C[64..80);
/// coalescing must not merge them, and only the last keeps the M=5 validity.
#[test]
fn wmma_output_stores_stay_distinct_through_coalescing() {
    let output = UOp::param(0, 80, DType::Float32, None);
    let lidx = UOp::special(weak(32), "lidx0".to_string());
    let valid = lidx.lt(&weak(16));
    let indices = [lidx.clone(), lidx.add(&weak(32)), lidx.add(&weak(64)).valid(valid.clone())];
    let before = UOp::sink(
        indices
            .into_iter()
            .enumerate()
            .map(|(value, index)| {
                UOp::index()
                    .buffer(output.clone())
                    .indices(vec![index])
                    .call()
                    .unwrap()
                    .store(UOp::const_(DType::Float32, ConstValue::Float(value as f64)))
            })
            .collect(),
    );

    let after = memory_coalescing(before, &Renderer::cpu());
    let after_stores = stores(&after);
    assert_eq!(after_stores.len(), 3, "{}", after.tree());
    assert_eq!(
        after_stores
            .iter()
            .filter(|store| matches!(store.op(), Op::Store { index, .. }
                if matches!(index.op(), Op::Index { indices, .. }
                    if Arc::ptr_eq(&indices[0].get_valid(), &valid))))
            .count(),
        1,
        "only the C[64..80) store uses the M=5 validity identity",
    );
}

#[test]
fn multiple_stores_to_one_group_offset_are_left_un_coalesced() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let first = index.store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
    let second = index.store(UOp::const_(DType::Float32, ConstValue::Float(2.0)));

    let result = memory_coalescing(UOp::sink(vec![first, second]), &Renderer::cpu());
    assert_eq!(stores(&result).len(), 2, "both stores survive; coalescing declines the group");
}

#[test]
fn a_gated_load_is_skipped_rather_than_aborting_the_pass() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let gated = UOp::new(
        Op::Load {
            index,
            alt: Some(UOp::const_(DType::Float32, ConstValue::Float(0.0))),
            gate: Some(UOp::const_(DType::Bool, ConstValue::Bool(true))),
        },
        DType::Float32,
    );

    let folded = loads(&memory_coalescing(UOp::sink(vec![gated.clone()]), &Renderer::cpu()));
    assert_eq!(folded.len(), 1);
    assert!(Arc::ptr_eq(&folded[0], &gated));
}

#[test]
fn image_float_half_float_roundtrip_is_removed() {
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let roundtrip = value.cast(DType::Float16).cast(DType::Float32);
    let mut ctx: AddImageContext = (std::collections::HashMap::new(), Renderer::cpu());

    let result = graph_rewrite(&pm_simplify_add_image(), roundtrip, &mut ctx);

    assert!(Arc::ptr_eq(&result, &value));
}
