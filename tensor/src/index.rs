//! ndarray-style indexing: the [`s!`] macro plus [`Tensor::getitem`] / [`Tensor::set`].
//!
//! This is an ergonomic, numpy-faithful front-end over the existing movement /
//! gather / `where_` backend. Basic slicing lowers to `try_shrink` (graph-identical
//! to a hand-written shrink) or, when steps / reverse are involved, to
//! [`Tensor::slice_with`]. Advanced (tensor) indices lower to `index_select` /
//! gather. [`Tensor::set`] is a *functional* setitem — it returns a new tensor
//! (the lib is immutable; `assign` covers realized-buffer in-place writes).

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use svod_ir::SInt;

use super::*;

/// One axis selector in an indexing expression (built by [`s!`]).
#[derive(Clone)]
pub enum Idx {
    /// Integer index — selects one element and collapses the dimension.
    Index(i64),
    /// `a..b` / `a..` / `..b` / `a..b;step` — half-open slice (concrete bounds).
    Range { start: Option<i64>, end: Option<i64>, step: Option<i64> },
    /// Symbolic-bound range — for JIT batch dims (`Idx::sint(0, b)`).
    SymRange(SInt, SInt),
    /// `..` — keep the whole dimension.
    Full,
    /// Insert a new dimension of size 1 (consumes no source dim).
    NewAxis,
    /// `...` — expands to as many [`Idx::Full`] as needed (at most one per spec).
    Ellipsis,
    /// Advanced index: an integer tensor (gather / `index_select`).
    Fancy(Tensor),
}

impl Idx {
    /// A symbolic-bound range, e.g. `Idx::sint(SInt::Const(0), batch.as_sint())`.
    pub fn sint(begin: impl Into<SInt>, end: impl Into<SInt>) -> Self {
        Idx::SymRange(begin.into(), end.into())
    }

    /// Attach a step to a range produced by [`s!`]'s `a..b;step` syntax. Internal.
    #[doc(hidden)]
    pub fn __with_step(range: impl Into<Idx>, step: i64) -> Self {
        match range.into() {
            Idx::Range { start, end, .. } => Idx::Range { start, end, step: Some(step) },
            Idx::Full => Idx::Range { start: None, end: None, step: Some(step) },
            other => other,
        }
    }
}

macro_rules! impl_idx_from {
    ($($ty:ty => |$v:pat_param| $body:expr),+ $(,)?) => {
        $(impl From<$ty> for Idx { fn from($v: $ty) -> Self { $body } })+
    };
}
impl_idx_from! {
    i64 => |i| Idx::Index(i),
    RangeFull => |_| Idx::Full,
    Range<i64> => |r| Idx::Range { start: Some(r.start), end: Some(r.end), step: None },
    RangeFrom<i64> => |r| Idx::Range { start: Some(r.start), end: None, step: None },
    RangeTo<i64> => |r| Idx::Range { start: None, end: Some(r.end), step: None },
    Tensor => |t| Idx::Fancy(t),
    &Tensor => |t| Idx::Fancy(t.clone()),
    Vec<i64> => |v| Idx::Fancy(Tensor::from_slice(v)),
    Vec<usize> => |v| Idx::Fancy(Tensor::from_slice(v.into_iter().map(|x| x as i64).collect::<Vec<_>>())),
}

/// What [`s!`] builds and [`Tensor::getitem`] / [`Tensor::set`] accept.
#[derive(Clone, Default)]
pub struct IndexSpec(pub Vec<Idx>);

impl From<Vec<Idx>> for IndexSpec {
    fn from(v: Vec<Idx>) -> Self {
        IndexSpec(v)
    }
}

/// ndarray-style indexing macro. Builds an [`IndexSpec`] for `getitem`/`set`.
///
/// | spelling | meaning |
/// |---|---|
/// | `..` | full axis |
/// | `a..b` / `a..` / `..b` | half-open range (concrete `i64` bounds) |
/// | `a..b ; step` | stepped range (negative step reverses) |
/// | `i` | integer index (collapses the axis) |
/// | `Idx::sint(b, e)` | symbolic-bound range (JIT batch dims) |
/// | `tensor` / `vec` | advanced (fancy) index |
/// | `NewAxis` | insert a size-1 dimension |
/// | `Ellipsis` | fill the remaining axes with `..` |
///
/// ```
/// use ndarray::array;
/// use svod_tensor::{Tensor, s};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let t = Tensor::from_ndarray(&array![[1.0f32, 2., 3.], [4., 5., 6.]]); // [2, 3]
/// let _row  = t.getitem(s![0, ..])?;        // integer index collapses axis 0 → [3]
/// let _cols = t.getitem(s![.., 0..2])?;     // ranges → [2, 2]
/// let heads = vec![0usize, 2];
/// let _sel  = t.getitem(s![.., heads])?;    // single-axis fancy (gather) → [2, 2]
/// let block = Tensor::from_slice([7.0f32, 8., 9.]);
/// let _t2   = t.set(s![1, ..], &block)?;    // functional setitem → new [2, 3]
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! s {
    ($($t:tt)*) => {
        $crate::IndexSpec($crate::__s_collect!([] $($t)*))
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __s_collect {
    // terminal: emit the accumulated items as a vec (no init-then-push).
    ([$($acc:expr),*]) => { ::std::vec![$($acc),*] };
    ([$($acc:expr),*] ,) => { ::std::vec![$($acc),*] };
    // markers (matched before the generic expr arm).
    ([$($acc:expr),*] Ellipsis $(, $($rest:tt)*)?) => {
        $crate::__s_collect!([$($acc,)* $crate::Idx::Ellipsis] $($($rest)*)?)
    };
    ([$($acc:expr),*] NewAxis $(, $($rest:tt)*)?) => {
        $crate::__s_collect!([$($acc,)* $crate::Idx::NewAxis] $($($rest)*)?)
    };
    // stepped range: `a..b ; step`.
    ([$($acc:expr),*] $r:expr ; $step:expr $(, $($rest:tt)*)?) => {
        $crate::__s_collect!([$($acc,)* $crate::Idx::__with_step($r, $step)] $($($rest)*)?)
    };
    // everything else via `From`: `..`, `a..b`, ints, tensors, `Idx::sint(..)`.
    ([$($acc:expr),*] $e:expr $(, $($rest:tt)*)?) => {
        $crate::__s_collect!([$($acc,)* $crate::Idx::from($e)] $($($rest)*)?)
    };
}

// =========================================================================
// Helpers
// =========================================================================

fn idx_err(details: impl Into<String>) -> Error {
    ErrorKind::IrConstruction { details: details.into() }.into()
}

fn concrete(shape: &svod_ir::shape::Shape, d: usize, what: &str) -> Result<usize> {
    shape[d]
        .as_const()
        .ok_or_else(|| ErrorKind::SymbolicShapeUnsupported { operation: what.to_string() })
        .map_err(Into::into)
}

/// Resolve a (possibly negative) integer index against a concrete dim.
fn resolve_index(i: i64, dim: usize) -> Result<usize> {
    let d = dim as i64;
    let r = if i < 0 { i + d } else { i };
    if r < 0 || r >= d {
        return Err(idx_err(format!("index {i} out of bounds for axis of size {dim}")));
    }
    Ok(r as usize)
}

/// Resolve a half-open `[start, end)` slice against a concrete dim (clamped; `b ≤ e`).
fn resolve_range(start: Option<i64>, end: Option<i64>, dim: usize) -> (usize, usize) {
    let b = start.unwrap_or(0).clamp(0, dim as i64) as usize;
    let e = (end.unwrap_or(dim as i64).clamp(0, dim as i64) as usize).max(b);
    (b, e)
}

/// Normalize a spec: validate ≤1 ellipsis, expand ellipsis / implicit-trailing
/// into `Full` so the number of source-consuming items equals `ndim`.
fn normalize_spec(spec: Vec<Idx>, ndim: usize) -> Result<Vec<Idx>> {
    let ellipsis = spec.iter().filter(|i| matches!(i, Idx::Ellipsis)).count();
    if ellipsis > 1 {
        return Err(idx_err("at most one `Ellipsis` allowed per index"));
    }
    let consumed = spec.iter().filter(|i| !matches!(i, Idx::NewAxis | Idx::Ellipsis)).count();
    if consumed > ndim {
        return Err(idx_err(format!("too many indices: {consumed} for {ndim}-D tensor")));
    }
    let fill = ndim - consumed;

    let mut out = Vec::with_capacity(spec.len() + fill);
    let mut filled = false;
    for item in spec {
        match item {
            Idx::Ellipsis => {
                out.extend(std::iter::repeat_n(Idx::Full, fill));
                filled = true;
            }
            other => out.push(other),
        }
    }
    if !filled {
        out.extend(std::iter::repeat_n(Idx::Full, fill));
    }
    Ok(out)
}

/// Does this range need the `slice_with` path (steps / reverse / negative bounds)?
fn range_needs_slice_with(start: Option<i64>, end: Option<i64>, step: Option<i64>) -> bool {
    step.is_some_and(|s| s != 1) || start.is_some_and(|s| s < 0) || end.is_some_and(|e| e < 0)
}

/// Per-source-axis classification of a normalized (no-Ellipsis, no-stepped) spec,
/// shared by the basic-shrink, advanced, and set paths. Each path then validates
/// the fields it forbids (e.g. advanced rejects `collapse`/`newaxis`).
struct Parsed {
    /// Per source axis: shrink bound (`None` = full, or a fancy axis kept full).
    bounds: Vec<Option<(SInt, SInt)>>,
    /// Source axes selected by an integer index (collapsed in the output).
    collapse: Vec<usize>,
    /// `(source axis, index tensor)` for advanced indices.
    fancy: Vec<(usize, Tensor)>,
    /// Whether any `NewAxis` appears (inserts a size-1 output dim).
    newaxis: bool,
}

/// One pass over the spec. Errors on stepped/reverse ranges (those route through
/// `slice_with` before this is reached).
fn parse_spec(spec: &[Idx], shape: &svod_ir::shape::Shape, ctx: &str) -> Result<Parsed> {
    let (mut bounds, mut collapse, mut fancy, mut newaxis) = (Vec::new(), Vec::new(), Vec::new(), false);
    let mut d = 0;
    for item in spec {
        match item {
            Idx::NewAxis => {
                newaxis = true;
                continue; // consumes no source axis
            }
            Idx::Full => bounds.push(None),
            Idx::SymRange(b, e) => bounds.push(Some((b.clone(), e.clone()))),
            Idx::Range { start, end, step } => {
                if range_needs_slice_with(*start, *end, *step) {
                    return Err(idx_err("stepped / reverse range is not allowed in this position"));
                }
                let (b, e) = resolve_range(*start, *end, concrete(shape, d, ctx)?);
                bounds.push(Some((SInt::Const(b), SInt::Const(e))));
            }
            Idx::Index(i) => {
                let ii = resolve_index(*i, concrete(shape, d, ctx)?)?;
                bounds.push(Some((SInt::Const(ii), SInt::Const(ii + 1))));
                collapse.push(d);
            }
            Idx::Fancy(t) => {
                bounds.push(None);
                fancy.push((d, t.clone()));
            }
            Idx::Ellipsis => unreachable!("ellipsis expanded by normalize_spec"),
        }
        d += 1;
    }
    Ok(Parsed { bounds, collapse, fancy, newaxis })
}

impl Tensor {
    /// numpy-style read indexing. Build `spec` with [`s!`].
    #[track_caller]
    pub fn getitem(&self, spec: impl Into<IndexSpec>) -> Result<Tensor> {
        origin_call!("getitem");
        self.index_impl(spec.into().0, None)
    }

    /// Functional numpy-style write indexing: returns a new tensor equal to `self`
    /// with the `spec` region overwritten by `value` (broadcast). Nothing mutates.
    #[track_caller]
    pub fn set(&self, spec: impl Into<IndexSpec>, value: &Tensor) -> Result<Tensor> {
        origin_call!("set");
        self.index_impl(spec.into().0, Some(value))
    }

    fn index_impl(&self, spec: Vec<Idx>, value: Option<&Tensor>) -> Result<Tensor> {
        let ndim = self.ndim()?;
        let spec = normalize_spec(spec, ndim)?;
        let has_fancy = spec.iter().any(|i| matches!(i, Idx::Fancy(_)));

        if has_fancy {
            return self.index_advanced(&spec, value);
        }
        match value {
            None => self.getitem_basic(&spec),
            Some(v) => self.set_basic(&spec, v),
        }
    }

    /// Basic read: route plain/symbolic shrinks straight to `try_shrink` (graph-
    /// identical to a hand-written shrink); only steps / reverse use `slice_with`.
    fn getitem_basic(&self, spec: &[Idx]) -> Result<Tensor> {
        let shape = self.shape()?;
        let use_slice_with = spec
            .iter()
            .any(|i| matches!(i, Idx::Range { start, end, step } if range_needs_slice_with(*start, *end, *step)));

        let x = if use_slice_with {
            self.basic_via_slice_with(spec, &shape)?
        } else {
            self.try_shrink(parse_spec(spec, &shape, "basic index on a symbolic dim")?.bounds)?
        };

        // Final reshape only if a dim collapses (int) or is inserted (NewAxis).
        let need_reshape = spec.iter().any(|i| matches!(i, Idx::Index(_) | Idx::NewAxis));
        if !need_reshape {
            return Ok(x);
        }
        let xs = x.shape()?;
        let mut view: Vec<SInt> = Vec::new();
        let mut d = 0;
        for item in spec {
            match item {
                Idx::NewAxis => view.push(SInt::Const(1)),
                Idx::Index(_) => d += 1, // collapsed: drop
                _ => {
                    view.push(xs[d].clone());
                    d += 1;
                }
            }
        }
        x.try_reshape(view)
    }

    /// Step / reverse / negative-bound slicing via the `slice_with` backend (concrete-only).
    fn basic_via_slice_with(&self, spec: &[Idx], shape: &svod_ir::shape::Shape) -> Result<Tensor> {
        let (mut starts, mut ends, mut steps, mut axes) = (vec![], vec![], vec![], vec![]);
        let mut d = 0;
        for item in spec {
            match item {
                Idx::NewAxis => {}
                Idx::Full => d += 1, // untouched axes stay full
                Idx::SymRange(..) => return Err(idx_err("cannot combine a symbolic range with stepped slicing")),
                Idx::Range { start, end, step } => {
                    let dim = concrete(shape, d, "stepped slice on a symbolic dim")? as i64;
                    let st = (*step).unwrap_or(1);
                    let (ds, de) = if st > 0 { (0, dim) } else { (dim - 1, -dim - 1) };
                    axes.push(d as i64);
                    starts.push((*start).unwrap_or(ds));
                    ends.push((*end).unwrap_or(de));
                    steps.push(st);
                    d += 1;
                }
                Idx::Index(i) => {
                    let dim = concrete(shape, d, "integer index on a symbolic dim")?;
                    let ii = resolve_index(*i, dim)? as i64;
                    axes.push(d as i64);
                    starts.push(ii);
                    ends.push(ii + 1);
                    steps.push(1);
                    d += 1;
                }
                Idx::Ellipsis | Idx::Fancy(_) => unreachable!("normalized / non-fancy"),
            }
        }
        self.slice_with().starts(&starts).ends(&ends).axes(&axes).steps(&steps).call()
    }

    /// Advanced (fancy) indexing. Non-fancy axes may be `Full` / step-1 `Range` /
    /// `SymRange`; `NewAxis`, integer collapse, and steps cannot mix with fancy
    /// indices (decompose into separate calls).
    fn index_advanced(&self, spec: &[Idx], value: Option<&Tensor>) -> Result<Tensor> {
        let shape = self.shape()?;
        let p = parse_spec(spec, &shape, "advanced index on a symbolic dim")?;
        if !p.collapse.is_empty() || p.newaxis {
            return Err(idx_err("NewAxis / integer-collapse cannot mix with advanced indices"));
        }
        // Pre-apply ranges on non-fancy axes (identity when all are Full → graph-identical).
        let has_range = p.bounds.iter().any(|b| b.is_some());
        let x = self.try_shrink(p.bounds)?;

        // Fast path: a single 1-D fancy axis is exactly `index_select` (the head-pruning case).
        if value.is_none() && p.fancy.len() == 1 && p.fancy[0].1.ndim()? == 1 {
            let (axis, idx) = &p.fancy[0];
            return x.index_select(*axis as isize, idx);
        }
        match value {
            None => x.gather_advanced(&p.fancy),
            // A range/slice on another axis would shrink `x`, so writing back via
            // `set_advanced` would drop the region outside the range. Fail loud
            // instead of silently corrupting; the caller can split it explicitly.
            Some(_) if has_range => Err(idx_err(
                "advanced `set` combined with a range/slice on another axis is unsupported; \
                 split it: `let s = t.getitem(range)?; let s = s.set(fancy, v)?; t.set(range, &s)`",
            )),
            Some(v) => x.set_advanced(&p.fancy, v),
        }
    }

    /// General N-axis advanced gather (one-hot / `where_` / sum, numpy axis order).
    fn gather_advanced(&self, fancy: &[(usize, Tensor)]) -> Result<Tensor> {
        use svod_ir::shape::{Shape, align_shapes_left, broadcast_shapes};

        let xs = self.shape()?;
        let n = xs.len();
        let dims: Vec<usize> = fancy.iter().map(|(d, _)| *d).collect();
        let d0 = dims[0];

        // Fold negative indices and compute the broadcast index shape.
        let folded: Vec<Tensor> = fancy
            .iter()
            .map(|(d, t)| {
                let dim = concrete(&xs, *d, "advanced index on a symbolic gather axis")?;
                t.normalize_negative_indices(dim as i64)
            })
            .collect::<Result<_>>()?;
        let idx_shapes: Vec<Shape> = folded.iter().map(|t| t.shape()).collect::<Result<_>>()?;
        let big = broadcast_shapes(&align_shapes_left(&idx_shapes)).context(UOpSnafu)?;
        let g = big.len();

        // pre_reduce_shape = xs[..d0] ++ big ++ xs[d0..]
        let mut pre: Vec<SInt> = xs[..d0].to_vec();
        pre.extend(big.iter().cloned());
        pre.extend_from_slice(&xs[d0..]);
        let pre: Shape = pre.into_iter().collect();

        // Reshape x to inject g size-1 axes for `big` at d0, then build the AND of one-hot masks.
        let mut xr: Vec<SInt> = xs[..d0].to_vec();
        xr.extend(std::iter::repeat_n(SInt::Const(1), g));
        xr.extend_from_slice(&xs[d0..]);
        let x_re = self.try_reshape(xr)?;

        let mut mask: Option<Tensor> = None;
        for ((d, _), idx) in fancy.iter().zip(&folded) {
            let num_classes = concrete(&xs, *d, "advanced index on a symbolic gather axis")?;
            // Broadcast the index to `big`, then place that block at positions
            // [d0 .. d0+g] with size-1 axes before and after, so the expand to `pre`
            // works for any first-fancy-axis position d0 and for indices of differing
            // rank (our `try_expand` requires equal rank — no implicit left-pad).
            let mut ir: Vec<SInt> = vec![SInt::Const(1); d0];
            ir.extend(big.iter().cloned());
            ir.extend(std::iter::repeat_n(SInt::Const(1), n - d0));
            let i = idx.broadcast_to(&big)?.try_reshape(ir)?.try_expand(pre.clone())?;
            let oh = i.one_hot_along_dim(num_classes, *d as isize - n as isize)?;
            mask = Some(match mask {
                None => oh,
                Some(m) => m.try_bitand(&oh)?,
            });
        }
        let mask = mask.expect("advanced path has ≥1 fancy axis");

        let zero = Tensor::new(x_re.uop().const_like(0));
        let sum_axes: Vec<isize> = dims.iter().map(|&d| (d + g) as isize).collect();
        let out = x_re.where_(&mask, &zero)?.sum_with().axes(sum_axes).dtype(self.uop().dtype()).call()?;

        // numpy: non-contiguous advanced axes move the broadcast block to the front.
        let contiguous = dims.windows(2).all(|w| w[1] == w[0] + 1);
        if contiguous {
            return Ok(out);
        }
        let rank = n - dims.len() + g;
        let mut perm: Vec<isize> = (d0..d0 + g).map(|x| x as isize).collect();
        perm.extend((0..d0).map(|x| x as isize));
        perm.extend((d0 + g..rank).map(|x| x as isize));
        out.try_permute(&perm)
    }

    /// Single-axis fancy set via `scatter` (last-writer-wins). Multi-axis → error.
    fn set_advanced(&self, fancy: &[(usize, Tensor)], value: &Tensor) -> Result<Tensor> {
        if fancy.len() != 1 {
            return Err(idx_err("only single-axis fancy assignment is supported; use `scatter` directly"));
        }
        let (axis, idx) = &fancy[0];
        let shape = self.shape()?;
        let ndim = shape.len();
        let k = concrete(&idx.shape()?, 0, "fancy assignment index length")?;

        // Broadcast the 1-D index and the value to self.shape with K along `axis`.
        let mut nd = vec![1isize; ndim];
        nd[*axis] = k as isize;
        let idx_nd = idx.try_reshape(&nd)?;
        let mut tgt: Vec<SInt> = shape.iter().cloned().collect();
        tgt[*axis] = SInt::Const(k);
        let tgt: svod_ir::shape::Shape = tgt.into_iter().collect();
        let idx_nd = idx_nd.try_expand(tgt.clone())?;
        let src = value.broadcast_to(&tgt)?;
        self.scatter(*axis as isize, &idx_nd, &src)
    }

    /// Basic functional setitem: overwrite the `spec` region with `value`.
    /// Constrained axes must be concrete; `Full` axes may be symbolic. No steps / fancy.
    fn set_basic(&self, spec: &[Idx], value: &Tensor) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();

        // One pass → per-axis bounds + collapsed axes. `set` pads the region, so it
        // needs concrete bounds: reject symbolic ranges (and NewAxis / fancy) here.
        let p = parse_spec(spec, &shape, "set on a symbolic dim")?;
        if p.newaxis || !p.fancy.is_empty() {
            return Err(idx_err("set supports only Full / range / integer indices (no NewAxis or fancy)"));
        }
        let sym = || ErrorKind::SymbolicShapeUnsupported { operation: "set on a symbolic dim".to_string() };
        let bounds: Vec<Option<(usize, usize)>> = p
            .bounds
            .iter()
            .map(|b| match b {
                None => Ok(None),
                Some((s, e)) => Ok(Some((s.as_const().ok_or_else(sym)?, e.as_const().ok_or_else(sym)?))),
            })
            .collect::<Result<_>>()?;
        let collapse = p.collapse;

        // Lift `value` to the slice shape. Broadcast it to the *kept* (non-collapsed)
        // extents first — so a scalar / lower-rank value lines up by numpy rules —
        // then re-insert size-1 axes at the collapsed positions. Unsqueezing the raw
        // value at the collapsed axes directly would panic when its rank is below a
        // non-leading collapsed axis (e.g. a 0-D value with `s![.., 0]`).
        let slice_dims: Vec<SInt> = bounds
            .iter()
            .enumerate()
            .map(|(d, b)| match b {
                Some((s, e)) => SInt::Const(e - s),
                None => shape[d].clone(),
            })
            .collect();
        let kept: svod_ir::shape::Shape =
            slice_dims.iter().enumerate().filter(|(d, _)| !collapse.contains(d)).map(|(_, s)| s.clone()).collect();
        let mut vb = value.broadcast_to(&kept)?;
        for &ax in &collapse {
            vb = vb.try_unsqueeze(ax as isize)?;
        }
        // Broadcast into the full slice shape, then pad back to self's shape at the offset.
        let slice_shape: svod_ir::shape::Shape = slice_dims.into_iter().collect();
        vb = vb.broadcast_to(&slice_shape)?;
        let pads: Vec<(isize, isize)> = bounds
            .iter()
            .enumerate()
            .map(|(d, b)| match b {
                Some((s, e)) => {
                    let dim = shape[d].as_const().expect("constrained axis is concrete");
                    Ok::<_, Error>((*s as isize, (dim - e) as isize))
                }
                None => Ok((0, 0)),
            })
            .collect::<Result<_>>()?;
        vb = vb.try_pad(&pads)?;

        // Region mask: AND over constrained axes of (arange >= begin) & (arange < end).
        let mut mask: Option<Tensor> = None;
        for (d, b) in bounds.iter().enumerate() {
            let Some((s, e)) = b else { continue };
            let dim = shape[d].as_const().expect("constrained axis is concrete");
            let ar = Tensor::arange(0, Some(dim as i64), None)?;
            let mut ar_shape = vec![1isize; ndim];
            ar_shape[d] = dim as isize;
            let ar = ar.try_reshape(&ar_shape)?;
            let dt = ar.uop().dtype();
            let lo = ar.try_ge(&Tensor::const_(ConstValue::Int(*s as i64), dt.clone()))?;
            let hi = ar.try_lt(&Tensor::const_(ConstValue::Int(*e as i64), dt))?;
            let cond = lo.try_bitand(&hi)?;
            mask = Some(match mask {
                None => cond,
                Some(m) => m.try_bitand(&cond)?,
            });
        }
        match mask {
            // No constrained axes (all Full) → overwrite everything.
            None => value.broadcast_to(&shape),
            Some(mask) => vb.where_(&mask, self),
        }
    }
}
