//! Flat addressing for tile buffers.
//!
//! Tiles carry a *logical* (multi-dimensional) shape but back onto a flat 1-D
//! pointer. [`flat_offset`] collapses multi-dim indices into a single
//! `Index`-typed offset (folding the all-constant part), and [`flat_index`] /
//! [`load_at`] turn that into the INDEX / LOAD the renderer expects.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, Op, UOp};

/// An index component: a compile-time constant or a runtime `Index`-typed UOp
/// (a loop range, `Special`, or derived lane arithmetic).
#[derive(Clone)]
pub enum Idx {
    Const(i64),
    Uop(Arc<UOp>),
}

impl Idx {
    /// Materialize this index component as an `Index`-typed UOp (a constant
    /// folds to a `cidx`, a dynamic component passes through).
    pub fn to_uop(&self) -> Arc<UOp> {
        match self {
            Idx::Const(c) => cidx(*c),
            Idx::Uop(u) => u.clone(),
        }
    }
}

impl From<i64> for Idx {
    fn from(v: i64) -> Self {
        Idx::Const(v)
    }
}
impl From<usize> for Idx {
    fn from(v: usize) -> Self {
        Idx::Const(v as i64)
    }
}
impl From<i32> for Idx {
    fn from(v: i32) -> Self {
        Idx::Const(v as i64)
    }
}
impl From<Arc<UOp>> for Idx {
    fn from(u: Arc<UOp>) -> Self {
        Idx::Uop(u)
    }
}
impl From<&Arc<UOp>> for Idx {
    fn from(u: &Arc<UOp>) -> Self {
        Idx::Uop(u.clone())
    }
}

mod private {
    pub trait Sealed {}
}

/// A fixed-arity tuple (or slice/array) of `Into<Idx>` elements, collected
/// into a stack-allocated `SmallVec<[Idx; 4]>`. Implemented for arities 1..=4
/// (covers every `MoveIdx` call site in the crate) plus `&[Idx]`, `[Idx; N]`,
/// and `Vec<Idx>` for back-compat. Sealed — extend by adding more tuple impls.
pub trait IntoIdxs: private::Sealed {
    fn into_idxs(self) -> smallvec::SmallVec<[Idx; 4]>;
}

impl private::Sealed for Idx {}
impl IntoIdxs for Idx {
    fn into_idxs(self) -> smallvec::SmallVec<[Idx; 4]> {
        smallvec::smallvec![self]
    }
}

macro_rules! impl_tuple_intoids {
    ($($a:ident:$i:tt),+) => {
        impl<$($a),+> private::Sealed for ($($a,)+) {}
        impl<$($a),+> IntoIdxs for ($($a,)+)
        where
            $($a: Into<Idx>,)+
        {
            fn into_idxs(self) -> smallvec::SmallVec<[Idx; 4]> {
                smallvec::smallvec!($(self.$i.into(),)+)
            }
        }
    };
}

impl_tuple_intoids!(A:0);
impl_tuple_intoids!(A:0, B:1);
impl_tuple_intoids!(A:0, B:1, C:2);
impl_tuple_intoids!(A:0, B:1, C:2, D:3);

impl<const N: usize> private::Sealed for [Idx; N] {}
impl<const N: usize> IntoIdxs for [Idx; N] {
    fn into_idxs(self) -> smallvec::SmallVec<[Idx; 4]> {
        self.into_iter().collect()
    }
}

impl private::Sealed for &[Idx] {}
impl IntoIdxs for &[Idx] {
    fn into_idxs(self) -> smallvec::SmallVec<[Idx; 4]> {
        self.iter().cloned().collect()
    }
}

impl private::Sealed for Vec<Idx> {}
impl IntoIdxs for Vec<Idx> {
    fn into_idxs(self) -> smallvec::SmallVec<[Idx; 4]> {
        smallvec::SmallVec::from_vec(self)
    }
}

/// A constant `Index`-typed UOp.
pub(crate) fn cidx(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(v))
}

/// Row-major strides for `shape`.
pub fn strides(shape: &[usize]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1] as i64;
    }
    s
}

/// Collapse multi-dim `idxs` into a single `Index`-typed offset UOp, folding the
/// all-constant contribution into one constant and chaining only the dynamic
/// terms with `try_add`/`try_mul`.
pub fn flat_offset(shape: &[usize], idxs: &[Idx]) -> Arc<UOp> {
    assert_eq!(shape.len(), idxs.len(), "flat_offset: rank mismatch (shape {} vs idx {})", shape.len(), idxs.len());
    let st = strides(shape);
    let mut konst: i64 = 0;
    let mut dynamic: Option<Arc<UOp>> = None;
    for (i, idx) in idxs.iter().enumerate() {
        match idx {
            Idx::Const(c) => konst += c * st[i],
            Idx::Uop(u) => {
                let term =
                    if st[i] == 1 { u.clone() } else { u.try_mul(&cidx(st[i])).expect("flat_offset: stride mul") };
                dynamic = Some(match dynamic {
                    Some(a) => a.try_add(&term).expect("flat_offset: term add"),
                    None => term,
                });
            }
        }
    }
    match dynamic {
        None => cidx(konst),
        Some(a) if konst == 0 => a,
        Some(a) => a.try_add(&cidx(konst)).expect("flat_offset: const add"),
    }
}

/// Unwrap a `custom_kernel` placeholder (`PARAM` or `RESHAPE(PARAM)`) to its flat
/// 1-D pointer buffer plus element dtype. Hand-built kernels index the flat
/// PARAM directly rather than the multi-dim reshape view.
pub fn flat_ptr(placeholder: &Arc<UOp>) -> (Arc<UOp>, DType) {
    let buf = match placeholder.op() {
        Op::Reshape { src, .. } => src.clone(),
        _ => placeholder.clone(),
    };
    let elem = match buf.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        dt => dt,
    };
    (buf, elem)
}

/// INDEX (ptr=true) into `buf` at the flattened offset — usable as both a STORE
/// target and a LOAD source.
pub fn flat_index(buf: &Arc<UOp>, shape: &[usize], idxs: &[Idx]) -> Arc<UOp> {
    let off = flat_offset(shape, idxs);
    UOp::index().buffer(buf.clone()).indices(vec![off]).ptr(true).call().expect("flat_index: INDEX construction")
}

/// LOAD from `buf` at the flattened offset (element dtype inferred from the
/// buffer's pointer base).
pub fn load_at(buf: &Arc<UOp>, shape: &[usize], idxs: &[Idx]) -> Arc<UOp> {
    let idx = flat_index(buf, shape, idxs);
    UOp::load().buffer(buf.clone()).index(idx).call()
}

/// INDEX (ptr=true) into `buf` at an already-flattened element `offset` — the
/// 1-D form used for flat GLOBAL buffer access (`srcf[src_i]` in tinygrad).
pub fn index_off(buf: &Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buf.clone()).indices(vec![offset]).ptr(true).call().expect("index_off: INDEX construction")
}

/// LOAD from `buf` at an already-flattened element `offset`.
pub fn load_off(buf: &Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
    let idx = index_off(buf, offset);
    UOp::load().buffer(buf.clone()).index(idx).call()
}

/// Vector LOAD of `lanes` contiguous elements from `buf` starting at the flattened
/// `idxs` position (the fragment's innermost `ept` run, trailing index `0`) — ONE
/// aligned `load <lanes x T>` instead of `lanes` scalar loads + an `insertelement`
/// chain. Requires the trailing tile axis (the `ept` element) to be unit-stride
/// (svod's register-tile layout), so `[.., 0..lanes)` is contiguous.
pub fn load_vec_at(buf: &Arc<UOp>, shape: &[usize], idxs: &[Idx], lanes: usize) -> Arc<UOp> {
    let idx = flat_index(buf, shape, idxs);
    let elem = match buf.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        dt => dt,
    };
    UOp::load().buffer(buf.clone()).index(idx).dtype(elem.vec(lanes).expect("load_vec_at: scalar element")).call()
}

/// Gated INDEX (ptr=true) at a flat `offset`: a STORE through it writes only when
/// `gate` is true (out-of-bounds writes are dropped) — the masked-store form.
pub fn index_off_gated(buf: &Arc<UOp>, offset: Arc<UOp>, gate: Arc<UOp>) -> Arc<UOp> {
    UOp::index()
        .buffer(buf.clone())
        .indices(vec![offset])
        .gate(gate)
        .ptr(true)
        .call()
        .expect("index_off_gated: INDEX construction")
}

/// Gated LOAD at a flat `offset`: returns the loaded value when `gate` is true,
/// else `alt` (out-of-bounds reads fold to the fill) — the masked-load form.
pub fn load_off_gated(buf: &Arc<UOp>, offset: Arc<UOp>, gate: Arc<UOp>, alt: Arc<UOp>) -> Arc<UOp> {
    let idx = index_off_gated(buf, offset, gate);
    UOp::load().buffer(buf.clone()).index(idx).alt(alt).call()
}

/// Wide LOAD of `lanes` contiguous elements from `buf` starting at element
/// `offset` — a single `<lanes × elem>` vector load (the 128-bit coalesced
/// GLOBAL fill: `bf16` × 8 = `global_load_dwordx4`). The INDEX points at the
/// first element; the vector LOAD reads the `lanes`-wide contiguous run. The
/// caller must guarantee `offset` is `lanes`-aligned (16-byte for `vec8` bf16).
pub fn load_vec(buf: &Arc<UOp>, offset: Arc<UOp>, lanes: usize) -> Arc<UOp> {
    let idx = index_off(buf, offset);
    let elem = match buf.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        dt => dt,
    };
    UOp::load().buffer(buf.clone()).index(idx).dtype(elem.vec(lanes).expect("load_vec element must be a scalar")).call()
}
