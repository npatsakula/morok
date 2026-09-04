//! Move `WHERE(gate, index, Invalid)` validity into memory operations.
//!
//! Direct port of Tinygrad's `codegen/late/gater.py`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, ConstValueHash, Op, TernaryOp, UOp};

use crate::TypedPatternMatcher;
use svod_ir::ops;

fn valid_index(index: &Arc<UOp>) -> Option<(&Arc<UOp>, &Arc<UOp>)> {
    let Op::Ternary(TernaryOp::Where, gate, index, invalid) = index.op() else {
        return None;
    };
    matches!(invalid.op(), Op::Const(ConstValueHash(ConstValue::Invalid))).then_some((gate, index))
}

/// The two-coordinate image form, gated by one shared condition.
///
/// Only image buffers take this path: a plain two-index INDEX keeps its own dtype
/// and goes through the generic rules below.
fn image_gate(buffer: &Arc<UOp>, indices: &[Arc<UOp>]) -> Option<(Arc<UOp>, Vec<Arc<UOp>>)> {
    if indices.len() != 2 || !buffer.dtype().is_image() {
        return None;
    }
    let (gate_y, index_y) = valid_index(&indices[0])?;
    let (gate_x, index_x) = valid_index(&indices[1])?;
    Arc::ptr_eq(gate_y, gate_x).then(|| (gate_y.clone(), vec![index_y.clone(), index_x.clone()]))
}

fn move_image_load(load: &Arc<UOp>, index: &Arc<UOp>, buffer: &Arc<UOp>, indices: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let Op::Load(ops::Load { alt: None, gate: None, .. }) = load.op() else {
        return None;
    };
    let (gate, indices) = image_gate(buffer, indices)?;
    let index = UOp::new(Op::Index(ops::Index { buffer: buffer.clone(), indices: indices.into() }), index.dtype());
    let result = UOp::load().index(index.clone()).alt(load.vconst_like(0)).gate(gate).call();
    Some(if result.dtype() == load.dtype() { result } else { result.cast(load.dtype()) })
}

fn move_image_store(
    index: &Arc<UOp>,
    value: &Arc<UOp>,
    gate: Option<&Arc<UOp>>,
    buffer: &Arc<UOp>,
    indices: &[Arc<UOp>],
) -> Option<Arc<UOp>> {
    if gate.is_some() {
        return None;
    }
    let (gate, indices) = image_gate(buffer, indices)?;
    let index = UOp::new(Op::Index(ops::Index { buffer: buffer.clone(), indices: indices.into() }), index.dtype());
    Some(index.store_gated(value.clone(), gate))
}

fn move_shrink_load(
    load: &Arc<UOp>,
    shrink: &Arc<UOp>,
    src: &Arc<UOp>,
    offsets: &Arc<UOp>,
    sizes: &Arc<UOp>,
) -> Option<Arc<UOp>> {
    let Op::Load(ops::Load { alt: None, gate: None, .. }) = load.op() else {
        return None;
    };
    let (gate, offsets) = valid_index(offsets)?;
    let shrink = UOp::new(
        Op::Shrink(ops::Shrink { src: src.clone(), offsets: offsets.clone(), sizes: sizes.clone() }),
        shrink.dtype(),
    );
    Some(UOp::load().index(shrink).alt(load.vconst_like(0)).gate(gate.clone()).call())
}

fn move_shrink_store(
    shrink: &Arc<UOp>,
    src: &Arc<UOp>,
    offsets: &Arc<UOp>,
    sizes: &Arc<UOp>,
    value: &Arc<UOp>,
    gate: Option<&Arc<UOp>>,
) -> Option<Arc<UOp>> {
    if gate.is_some() {
        return None;
    }
    let (gate, offsets) = valid_index(offsets)?;
    let shrink = UOp::new(
        Op::Shrink(ops::Shrink { src: src.clone(), offsets: offsets.clone(), sizes: sizes.clone() }),
        shrink.dtype(),
    );
    Some(shrink.store_gated(value.clone(), gate.clone()))
}

fn move_load(load: &Arc<UOp>, index: &Arc<UOp>, buffer: &Arc<UOp>, indices: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let Op::Load(ops::Load { alt: None, gate: None, .. }) = load.op() else {
        return None;
    };
    let (gate, clean_index) = valid_index(indices.first()?)?;
    let mut indices = indices.to_vec();
    indices[0] = clean_index.clone();
    let index = UOp::new(Op::Index(ops::Index { buffer: buffer.clone(), indices: indices.into() }), index.dtype());
    Some(UOp::load().index(index).alt(load.vconst_like(0)).gate(gate.clone()).call())
}

fn move_store(
    index: &Arc<UOp>,
    value: &Arc<UOp>,
    gate: Option<&Arc<UOp>>,
    buffer: &Arc<UOp>,
    indices: &[Arc<UOp>],
) -> Option<Arc<UOp>> {
    if gate.is_some() {
        return None;
    }
    let (gate, clean_index) = valid_index(indices.first()?)?;
    let mut indices = indices.to_vec();
    indices[0] = clean_index.clone();
    let index = UOp::new(Op::Index(ops::Index { buffer: buffer.clone(), indices: indices.into() }), index.dtype());
    Some(index.store_gated(value.clone(), gate.clone()))
}

fn gated_load<'a>(value: &'a Arc<UOp>, gate: &Arc<UOp>, inverted: bool) -> Option<&'a Arc<UOp>> {
    if gate.dtype() != DType::Bool {
        return None;
    }
    let value = match value.op() {
        Op::Cast(ops::Cast { src, .. }) => src,
        _ => value,
    };
    let Op::Load(ops::Load { alt: Some(_), gate: Some(load_gate), .. }) = value.op() else {
        return None;
    };
    let matches = if inverted {
        matches!(load_gate.op(), Op::Unary(svod_ir::UnaryOp::Not, inner) if Arc::ptr_eq(inner, gate))
    } else {
        Arc::ptr_eq(load_gate, gate)
    };
    matches.then_some(value)
}

fn move_where_load(
    where_: &Arc<UOp>,
    gate: &Arc<UOp>,
    value: &Arc<UOp>,
    alt: &Arc<UOp>,
    inverted: bool,
) -> Option<Arc<UOp>> {
    let load = gated_load(value, gate, inverted)?;
    let Op::Load(ops::Load { index, gate: Some(load_gate), .. }) = load.op() else {
        return None;
    };
    let alt = if matches!(alt.op(), Op::Const(ConstValueHash(ConstValue::Invalid))) {
        load.vconst_like(0)
    } else if let Op::Cast(ops::Cast { src, .. }) = alt.op() {
        if src.dtype() == load.dtype() { src.clone() } else { alt.cast(load.dtype()) }
    } else {
        alt.cast(load.dtype())
    };
    Some(UOp::load().index(index.clone()).alt(alt).gate(load_gate.clone()).call().cast(where_.dtype()))
}

/// Tinygrad's `pm_move_gates_from_index`, in source rule order.
pub fn pm_move_gates_from_index() -> TypedPatternMatcher {
    crate::patterns! {
        // Two-index image-form rules must precede the generic INDEX rules.
        load @ Load { index: idx @ Index { buffer, indices }, .. }
            if image_gate(buffer, indices).is_some()
            => move_image_load(load, idx, buffer, indices),
        Store { index: idx @ Index { buffer, indices }, value, gate }
            if image_gate(buffer, indices).is_some()
            => move_image_store(idx, value, gate.as_ref(), buffer, indices),

        load @ Load { index: idx @ Index { buffer, indices }, .. }
            if indices.first().and_then(valid_index).is_some()
            => move_load(load, idx, buffer, indices),
        load @ Load { index: shrink @ Shrink { src, offsets, sizes }, .. }
            if valid_index(offsets).is_some()
            => move_shrink_load(load, shrink, src, offsets, sizes),
        Store { index: idx @ Index { buffer, indices }, value, gate }
            if indices.first().and_then(valid_index).is_some()
            => move_store(idx, value, gate.as_ref(), buffer, indices),
        Store { index: shrink @ Shrink { src, offsets, sizes }, value, gate }
            if valid_index(offsets).is_some()
            => move_shrink_store(shrink, src, offsets, sizes, value, gate.as_ref()),

        where_ @ Where(gate, value, alt)
            if gated_load(value, gate, false).is_some()
            => move_where_load(where_, gate, value, alt, false),
        where_ @ Where(gate, alt, value)
            if gated_load(value, gate, true).is_some()
            => move_where_load(where_, gate, value, alt, true),
    }
}
