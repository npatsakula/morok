//! Weak dtype lowering, ported from Tinygrad `tinygrad/uop/weak.py`.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::VminVmaxProperty;
use svod_ir::{ConstValue, Op, TernaryOp, UOp};

use crate::TypedPatternMatcher;
use svod_ir::ops;

pub fn select_dtype(u: &Arc<UOp>) -> DType {
    if u.dtype().base() == svod_dtype::ScalarDType::WeakFloat {
        return DType::default_float().vec(u.dtype().vcount()).expect("default dtype is scalar");
    }
    let (vmin, vmax) = VminVmaxProperty::get(u);
    let fits_i32 = match (vmin, vmax) {
        (ConstValue::Int(lo), ConstValue::Int(hi)) => *lo >= i32::MIN as i64 && *hi <= i32::MAX as i64,
        (ConstValue::UInt(lo), ConstValue::UInt(hi)) => *lo <= i32::MAX as u64 && *hi <= i32::MAX as u64,
        (ConstValue::Bool(_), ConstValue::Bool(_)) => true,
        _ => false,
    };
    (if fits_i32 { DType::default_int() } else { DType::Int64 })
        .vec(u.dtype().vcount())
        .expect("selected dtype is scalar")
}

fn is_lower_weak_node(op: &Op) -> bool {
    matches!(
        op,
        Op::Unary(..)
            | Op::Binary(..)
            | Op::Ternary(TernaryOp::Where, ..)
            | Op::Range(..)
            | Op::Stack(..)
            | Op::Special(..)
    )
}

pub fn lower_weak_node(u: &Arc<UOp>) -> Option<Arc<UOp>> {
    let start = usize::from(matches!(u.op(), Op::Ternary(TernaryOp::Where, ..)));
    let old_src = u.op().sources();
    let src: Vec<_> = old_src
        .iter()
        .map(|s| match s.op() {
            Op::Cast(ops::Cast { src, dtype }) if dtype.is_weak() => src.clone(),
            _ => s.clone(),
        })
        .collect();
    let unwrapped = src.iter().zip(&old_src).any(|(a, b)| !Arc::ptr_eq(a, b));
    if (!u.dtype().is_weak() && !unwrapped) || src[start..].iter().any(|s| s.dtype().is_weak()) {
        return None;
    }

    let mut dt = if matches!(u.op(), Op::Binary(..)) {
        let mut dtypes = Vec::with_capacity(src.len() + 1);
        dtypes.push(select_dtype(u).scalar_dtype());
        dtypes.extend(src.iter().map(|s| s.dtype().scalar_dtype()));
        let scalar = DType::least_upper_dtype(&dtypes)?.strong_dtype();
        let vcount = if matches!(u.op(), Op::Binary(op, ..) if op.is_comparison()) {
            src.iter().map(|s| s.dtype().vcount()).max().unwrap_or(1)
        } else {
            u.dtype().vcount()
        };
        scalar.vec(vcount)?
    } else {
        svod_ir::dtype_from_op(u.with_sources(src.clone()).op())?.strong_dtype()
    };
    if matches!(u.op(), Op::Stack(..)) {
        dt = dt.scalar_dtype();
    }
    let lowered = src[..start]
        .iter()
        .cloned()
        .chain(src[start..].iter().map(|s| {
            if UOp::is_invalid_marker(s) {
                s.clone()
            } else if dt.vcount() == 1 && s.dtype().vcount() > 1 {
                UOp::stack((0..s.dtype().vcount()).map(|lane| s.index_axes(vec![lane]).cast(dt.clone())).collect())
            } else {
                s.cast(dt.clone())
            }
        }))
        .collect();
    let lowered = u.with_sources(lowered);
    let lowered = lowered.with_dtype(svod_ir::dtype_from_op(lowered.op())?.strong_dtype());
    // STACK derives its promoted scalar dtype from its lanes. Keeping an outer
    // weak CAST reconstructs the original STACK during cast folding and cycles.
    let lowered = if matches!(u.op(), Op::Stack(..)) || !unwrapped || u.dtype().vcount() > 1 {
        lowered
    } else {
        lowered.cast(u.dtype())
    };
    (!Arc::ptr_eq(&lowered, u)).then_some(lowered)
}

pub fn pm_lower_weak() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        u @const(value) if u.dtype().is_weak() => {
            Some(UOp::const_(select_dtype(u), value).cast(u.dtype()))
        },
        u @ VConst { values } if u.dtype().is_weak() => {
            Some(UOp::try_vconst(values.clone(), select_dtype(u).scalar_dtype()).ok()?.cast(u.dtype()))
        },
        u @ Cast { src: inner, dtype } if dtype.is_weak() => {
            let Op::Cast(ops::Cast { src: x, dtype: inner_dtype }) = inner.op() else { return None };
            if !inner_dtype.is_weak() || x.dtype().is_weak() { return None; }
            Some(x.cast(select_dtype(u)).cast(u.dtype()))
        },
        u @ Index { buffer, indices } if u.dtype().is_weak() => {
            let buffer = buffer.cast(select_dtype(u));
            let indices = indices.iter().map(|index| {
                if index.dtype().is_weak() { commit_weak(index, select_dtype(index)) } else { index.clone() }
            });
            Some(u.with_sources(std::iter::once(buffer).chain(indices).collect()))
        },
        u if is_lower_weak_node(u.op()) => lower_weak_node(u),
        u @ Param { shape, arg } if u.dtype() == DType::WeakInt => {
            if arg.addrspace.is_some() { return None; }
            let mut arg = arg.clone();
            arg.dtype = select_dtype(u);
            Some(UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg }), select_dtype(u)).cast(DType::WeakInt))
        },
    }
}

/// The `ctx:dict[UOp, UOp]` upstream threads through `pm_lower_index_dtype`
/// (`tinygrad/uop/weak.py:29-40`, `:70`), created once per `to_program` by the
/// single `ctx={}` at `tinygrad/codegen/__init__.py:349`. Keyed by source
/// identity: hash-consing makes `s.id` the same key `UOp` is upstream.
pub type WeakMemo = rustc_hash::FxHashMap<u64, Arc<UOp>>;

pub fn lower_weak_srcs(memo: &mut WeakMemo, u: &Arc<UOp>) -> Option<Arc<UOp>> {
    fn lower(memo: &mut WeakMemo, s: &Arc<UOp>) -> Arc<UOp> {
        if let Some(cached) = memo.get(&s.id) {
            return cached.clone();
        }
        let r = crate::rewrite::graph_rewrite(pm_lower_weak(), s.clone(), &mut ());
        // the consumer absorbs the cast on its own edge
        let r = match r.op() {
            Op::Cast(ops::Cast { src, dtype }) if dtype.is_weak() => src.clone(),
            _ => r,
        };
        memo.insert(s.id, r.clone());
        r
    }

    if matches!(u.op(), Op::Binary(op, ..) if op.is_comparison()) {
        let ret = lower(memo, u);
        return (!Arc::ptr_eq(&ret, u)).then_some(ret);
    }
    let old_src = u.op().sources();
    let src: Vec<_> = old_src.iter().map(|s| if s.dtype().is_weak() { lower(memo, s) } else { s.clone() }).collect();
    if src.iter().zip(&old_src).all(|(a, b)| Arc::ptr_eq(a, b)) {
        return None;
    }
    let lowered = u.with_sources(src);
    (!Arc::ptr_eq(&lowered, u)).then_some(lowered)
}

pub fn commit_weak(s: &Arc<UOp>, dt: DType) -> Arc<UOp> {
    match s.op() {
        Op::Const(value) => UOp::const_(dt, value.0),
        _ => s.cast(dt),
    }
}

pub fn commit_weak_srcs(u: &Arc<UOp>) -> Option<Arc<UOp>> {
    let src = u.op().sources();
    if !src.iter().any(|s| s.dtype().is_weak()) {
        return None;
    }
    let dt = DType::least_upper_dtype(&src.iter().map(|s| s.dtype()).collect::<Vec<_>>())?;
    if dt.is_weak() {
        return None;
    }
    Some(u.with_sources(
        src.iter().map(|s| if s.dtype().is_weak() { commit_weak(s, dt.clone()) } else { s.clone() }).collect(),
    ))
}

pub fn pm_commit_weak() -> TypedPatternMatcher {
    crate::patterns! {
        u if matches!(u.op(), Op::Binary(..) | Op::Ternary(..)) => commit_weak_srcs(u),
        u @ Store { index, value, gate } if value.dtype().is_weak() => {
            let mut src = vec![index.clone(), commit_weak(value, index.dtype())];
            src.extend(gate.iter().cloned());
            Some(u.with_sources(src))
        },
    }
}

pub fn cast_weak_srcs(c: &Arc<UOp>, u: &Arc<UOp>) -> Option<Arc<UOp>> {
    if c.dtype().is_weak() || c.dtype().weak_dtype() != u.dtype() {
        return None;
    }
    let dt = DType::least_upper_dtype(&[c.dtype(), select_dtype(u)])?;
    let src = u.op().sources();
    let lowered = u
        .with_sources(
            src.iter().map(|s| if s.dtype().is_weak() { commit_weak(s, dt.clone()) } else { s.clone() }).collect(),
        )
        .cast(c.dtype());
    (!Arc::ptr_eq(&lowered, c)).then_some(lowered)
}

pub fn pm_cast_weak() -> TypedPatternMatcher {
    crate::patterns! {
        c @ Cast { src: u, dtype: _ } if matches!(u.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..)) && u.dtype().is_weak() => {
            cast_weak_srcs(c, u)
        },
    }
}

fn max_numel_fits_i32(buf: &Arc<UOp>) -> bool {
    buf.shape()
        .ok()
        .flatten()
        .and_then(|shape| shape.iter().try_fold(1usize, |n, dim| n.checked_mul(dim.vmax()?)))
        .is_some_and(|n| n.saturating_sub(1) <= i32::MAX as usize)
}

pub fn pm_lower_index_dtype() -> TypedPatternMatcher<WeakMemo> {
    pm_commit_weak().with_context::<WeakMemo>()
        + pm_cast_weak().with_context()
        + crate::patterns! {
            @context WeakMemo;
            u @ Shrink { src, offsets, sizes }
                if offsets.dtype().is_weak() || sizes.dtype().is_weak() => {
                let offsets = if offsets.dtype().is_weak() {
                    commit_weak(offsets, select_dtype(offsets))
                } else {
                    offsets.clone()
                };
                let sizes = if sizes.dtype().is_weak() {
                    commit_weak(sizes, select_dtype(sizes))
                } else {
                    sizes.clone()
                };
                Some(u.with_sources(vec![src.clone(), offsets, sizes]))
            },
            u if !u.dtype().is_weak() && u.op().sources().iter().any(|s| s.dtype().is_weak()) => lower_weak_srcs(ctx, u),
            u @ Index { buffer, indices } => {
                let first = indices.first()?;
                let Op::Ternary(TernaryOp::Where, gate, idx, invalid) = first.op() else { return None };
                if idx.dtype() != DType::Int64 || !UOp::is_invalid_marker(invalid) || !max_numel_fits_i32(buffer) { return None; }
                let mut new_indices = indices.clone();
                new_indices[0] = idx.cast(DType::Int32).valid(gate.clone());
                Some(u.with_sources(std::iter::once(buffer.clone()).chain(new_indices).collect()))
            },
            u @ Shrink { src, offsets, sizes } => {
                let Op::Ternary(TernaryOp::Where, gate, idx, invalid) = offsets.op() else { return None };
                if idx.dtype() != DType::Int64 || !UOp::is_invalid_marker(invalid) || !max_numel_fits_i32(src) { return None; }
                Some(u.with_sources(vec![src.clone(), idx.cast(DType::Int32).valid(gate.clone()), sizes.clone()]))
            },
        }
}
