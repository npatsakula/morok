//! Devectorize pass — single-pass combined matcher matching Tinygrad's `pm_devectorize`.
//!
//! Composition: `sym + devectorize + load_store_folding + correct_load_store + load_store_indexing`
//!
//! All patterns run in one `graph_rewrite` call with fixed-point convergence.
//! PtrCat is an intermediate created by `fold_expanded_index` and eliminated by
//! `distribute_ptrcat_load/store` within the same pass. It must never reach codegen.
//!
//! # pm_render (called AFTER devectorize)
//!
//! - CAT → VECTORIZE (CAT can't be rendered directly)
//! - Multi-index GEP → VECTORIZE of single-index GEPs
//! - Single-element VECTORIZE/PTRCAT → unwrap

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::LazyLock;

use itertools::Itertools;
use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::{BinaryOp, ConstValue, Op, ReduceOp, TernaryOp, UOp, UOpKey, UnaryOp, WmmaMetadata};

use crate::TypedPatternMatcher;
use smallvec::SmallVec;

/// Context for REDUCE transformation (Tinygrad devectorizer.py:280-281).
///
/// Tracks END nodes created per reduce-range set so that multiple ENDs sharing
/// the same ranges can be merged into a single END with a GROUP body.
#[derive(Debug, Default)]
pub struct ReduceContext {
    range_to_ends: HashMap<SmallVec<[u64; 4]>, Vec<Arc<UOp>>>,
}

impl ReduceContext {
    /// Register an END node under its reduce-range key.
    pub fn register_end(&mut self, end: &Arc<UOp>) {
        if let Op::End { ranges, .. } = end.op() {
            let mut key: SmallVec<[u64; 4]> = ranges.iter().map(|r| r.id).collect();
            key.sort_unstable();
            self.range_to_ends.entry(key).or_default().push(end.clone());
        }
    }

    /// Merge END nodes that share the same reduce ranges.
    ///
    /// For each group of >1 ENDs with identical range sets, creates a single
    /// `GROUP(computation1, computation2, ...).end(ranges)` and substitutes it
    /// throughout the SINK subgraph. Clears tracking state after merge.
    ///
    /// Matches Tinygrad's `merge_reduce_ends` (devectorizer.py:333-336).
    pub fn merge_reduce_ends(&mut self, sources: &SmallVec<[Arc<UOp>; 4]>) -> Option<Arc<UOp>> {
        #[allow(clippy::mutable_key_type)]
        let subs = build_end_merge_subs(&self.range_to_ends);
        self.range_to_ends.clear();
        if subs.is_empty() {
            return None;
        }
        Some(UOp::sink(sources.to_vec()).substitute(&subs))
    }
}

/// Core merge logic: given a map of range-key → END nodes, build substitutions.
#[allow(clippy::mutable_key_type)]
fn build_end_merge_subs(range_to_ends: &HashMap<SmallVec<[u64; 4]>, Vec<Arc<UOp>>>) -> HashMap<UOpKey, Arc<UOp>> {
    let mut subs = HashMap::new();
    for ends in range_to_ends.values() {
        if ends.len() <= 1 {
            continue;
        }
        let computations: Vec<Arc<UOp>> = ends
            .iter()
            .map(|e| match e.op() {
                Op::End { computation, .. } => computation.clone(),
                _ => unreachable!(),
            })
            .collect();
        let ranges = match ends[0].op() {
            Op::End { ranges, .. } => ranges.clone(),
            _ => unreachable!(),
        };
        let merged = UOp::group(computations).end(ranges);
        for end in ends {
            subs.insert(UOpKey(end.clone()), merged.clone());
        }
    }
    subs
}

/// Merge sibling END nodes that share the same reduce ranges (standalone pass).
///
/// Walks the SINK subgraph, discovers all END nodes, groups by range key,
/// and merges groups of >1 into `GROUP(computations...).end(ranges)`.
///
/// This is the same merge as `ReduceContext::merge_reduce_ends` but doesn't
/// require tracking during pm_reduce — it discovers ENDs from the graph directly.
/// Needed because later passes (e.g. pm_decomp+pm_render) can create new sibling
/// ENDs that weren't present when pm_reduce ran.
pub fn merge_sibling_ends(sink: &Arc<UOp>) -> Arc<UOp> {
    let Op::Sink { sources, .. } = sink.op() else { return sink.clone() };

    let mut range_to_ends: HashMap<SmallVec<[u64; 4]>, Vec<Arc<UOp>>> = HashMap::new();
    for node in sink.toposort() {
        if let Op::End { ranges, .. } = node.op() {
            let mut key: SmallVec<[u64; 4]> = ranges.iter().map(|r| r.id).collect();
            key.sort_unstable();
            range_to_ends.entry(key).or_default().push(node.clone());
        }
    }

    #[allow(clippy::mutable_key_type)]
    let subs = build_end_merge_subs(&range_to_ends);
    if subs.is_empty() {
        return sink.clone();
    }
    UOp::sink(sources.to_vec()).substitute(&subs)
}

use crate::passes::linearize_index::pm_linearize_multi_index;
use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::sym;

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run devectorize pass. Call AFTER `pre_expand`, BEFORE codegen.
///
/// Single-pass combined matcher (Tinygrad `pm_devectorize`):
/// sym + linearize_multi_index + devectorize + load_store_folding + correct_load_store + load_store_indexing
///
/// Note: `bool_storage_patterns()` called separately (backend-specific).
/// Note: `pm_render()` should be applied AFTER this pass.
pub fn devectorize(ast: &Arc<UOp>) -> Arc<UOp> {
    static COMBINED: LazyLock<TypedPatternMatcher> = LazyLock::new(|| {
        sym()
            + pm_linearize_multi_index()
            + devectorize_patterns()
            + load_store_folding_patterns()
            + correct_load_store_patterns()
            + load_store_indexing_patterns()
    });
    graph_rewrite(&*COMBINED, ast.clone(), &mut ())
}

/// Bool LOAD/STORE via uint8. LLVM i1 can have garbage in upper bits.
/// Also rewrites BitCast involving Bool to Cast (bitcast requires same bit-width).
pub fn bool_storage_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // STORE bool: cast to uint8 before storing
        Store { index, value, ranges } if value.dtype().base().is_bool() => {
            let uint8_dtype = value.dtype().with_base(ScalarDType::UInt8);
            Some(index.store_with_ranges(value.cast(uint8_dtype), ranges.clone()))
        },

        // LOAD bool: load as uint8, then cast to bool
        load @ Load { buffer, index, alt } if load.dtype().base().is_bool() => {
            let uint8_dtype = load.dtype().with_base(ScalarDType::UInt8);
            let uint8_alt = alt.clone().map(|a| a.cast(uint8_dtype.clone()));
            let uint8_load = UOp::load()
                .buffer(buffer.clone())
                .index(index.clone())
                .maybe_alt(uint8_alt)
                .dtype(uint8_dtype)
                .call();
            Some(uint8_load.cast(load.dtype()))
        },

        // BitCast with Bool: i1 has different bit-width than i8+, use Cast instead
        BitCast { src, dtype } if src.dtype().base().is_bool() || dtype.base().is_bool() => {
            Some(src.cast(dtype.clone()))
        },
    }
}

// ============================================================================
// FP8 Float Decomposition (Tinygrad: pm_float_decomp, decompositions.py:504-522)
// ============================================================================

/// Context for FP8 float decomposition.
/// `from` is the FP8 dtype being decomposed, `to` is the target float dtype.
#[derive(Debug, Clone)]
pub struct Fp8DecompCtx {
    pub from: ScalarDType,
    pub to: ScalarDType,
}

/// Round-to-nearest-even for integer bitwise rounding.
/// Port of Tinygrad's `rne(v, s)` (decompositions.py:383).
fn rne(v: &Arc<UOp>, s: u32) -> Arc<UOp> {
    let one = v.const_like(1);
    let shifted = v.shr(&v.const_like(s));
    let half_bit = v.shr(&v.const_like(s - 1)).and_(&one);
    let remainder_mask = v.const_like((1i64 << (s - 1)) - 1);
    let has_remainder = v.and_(&remainder_mask).ne(&v.const_like(0)).cast(v.dtype());
    let lsb = shifted.and_(&one);
    let round_up = half_bit.and_(&has_remainder.or_(&lsb));
    shifted.try_add(&round_up).expect("rne: add failed")
}

/// Bitwise float-to-float format conversion.
/// Port of Tinygrad's `f2f(v, fr, to)` (decompositions.py:385-404).
///
/// `v` is a UInt value holding the raw bits of the source float.
/// Returns a UOp holding raw bits of the target float, which must be bitcast to get the float value.
fn f2f(v: &Arc<UOp>, fr: ScalarDType, to: ScalarDType) -> Arc<UOp> {
    let (fe, fm) = fr.finfo();
    let (te, tm) = to.finfo();
    let fs = fr.bitsize();
    let ts = to.bitsize();
    let fb = fr.exponent_bias() as i64;
    let tb = to.exponent_bias() as i64;
    let fr_uint = DType::Scalar(fr.float_to_uint());
    let to_uint = DType::Scalar(to.float_to_uint());

    if fe <= te && fm < tm {
        // Upcast path: e.g. FP8 → Float16
        let sign_mask = v.const_like(1i64 << (fs - 1));
        let sign = v.and_(&sign_mask).cast(to_uint.clone()).shl(&v.const_like(ts - fs).cast(to_uint.clone()));
        let nosign_mask = v.const_like((1i64 << (fs - 1)) - 1);
        let nosign = v.and_(&nosign_mask).cast(to_uint.clone());
        let exp = nosign.shr(&nosign.const_like(fm));
        let norm = nosign
            .shl(&nosign.const_like(tm - fm))
            .try_add(&nosign.const_like((tb - fb) << tm))
            .expect("f2f: add failed");
        let nan_val = nosign.shl(&nosign.const_like(tm - fm)).or_(&nosign.const_like(((1i64 << te) - 1) << tm));

        // FP8E4M3 has a single NaN value (all exponent+mantissa bits set)
        let is_nan = if fr == ScalarDType::FP8E4M3 {
            nosign.eq(&nosign.const_like((1i64 << (fm + fe)) - 1))
        } else {
            exp.eq(&exp.const_like((1i64 << fe) - 1))
        };

        let zero = nosign.const_like(0);
        let exp_is_zero = exp.eq(&zero);
        let inner = UOp::try_where(is_nan, nan_val, norm).expect("f2f: where failed");
        let result = UOp::try_where(exp_is_zero, zero, inner).expect("f2f: where failed");
        sign.or_(&result).bitcast(DType::Scalar(to))
    } else if fe >= te && fm > tm {
        // Downcast path: e.g. Float16 → FP8
        let clamped = f2f_clamp(&v.bitcast(DType::Scalar(fr)), to);
        let v = clamped.bitcast(fr_uint);
        let sign = v.shr(&v.const_like(fs - ts)).and_(&v.const_like(1i64 << (ts - 1)));
        let nosign_mask = v.const_like((1i64 << (fs - 1)) - 1);
        let nosign = v.and_(&nosign_mask);
        let norm = rne(&nosign, fm - tm)
            .try_sub(&nosign.const_like((fb - tb) << tm))
            .expect("f2f: sub failed")
            .cast(to_uint.clone());

        let exp_field = nosign.shr(&nosign.const_like(fm)).and_(&nosign.const_like((1i64 << fe) - 1));
        let underflow = exp_field.lt(&exp_field.const_like(1 + fb - tb));

        let nan_mantissa = if to == ScalarDType::FP8E4M3 {
            sign.const_like((1i64 << tm) - 1).cast(to_uint.clone())
        } else {
            nosign.shr(&nosign.const_like(fm - tm)).and_(&nosign.const_like((1i64 << tm) - 1)).cast(to_uint.clone())
        };
        let nan_exp = sign.const_like(((1i64 << te) - 1) << tm).cast(to_uint.clone());
        let nan = sign.cast(to_uint.clone()).or_(&nan_mantissa).or_(&nan_exp);

        let is_nan = exp_field.eq(&exp_field.const_like((1i64 << fe) - 1));
        let zero = sign.const_like(0).cast(to_uint.clone());
        let normal = sign.cast(to_uint.clone()).or_(&UOp::try_where(underflow, zero, norm).expect("f2f: where failed"));
        UOp::try_where(is_nan, nan, normal).expect("f2f: where failed")
    } else {
        panic!("f2f: unsupported conversion {fr:?} -> {to:?}")
    }
}

/// Clamp a float value to the representable range of a target FP8 dtype.
/// Port of Tinygrad's `f2f_clamp` (decompositions.py:406-412).
fn f2f_clamp(val: &Arc<UOp>, dt: ScalarDType) -> Arc<UOp> {
    let (e, m) = dt.finfo();
    let (max_exp, max_man): (i64, i64) =
        if dt == ScalarDType::FP8E4M3 { ((1 << e) - 1, (1 << m) - 2) } else { ((1 << e) - 2, (1 << m) - 1) };
    let mx_f64 =
        f64::powi(2.0, (max_exp - dt.exponent_bias() as i64) as i32) * (1.0 + max_man as f64 / (1i64 << m) as f64);
    let mx = val.const_like(mx_f64);
    let neg_mx = val.const_like(-mx_f64);

    // For FP8 types, clamp to ±max; for others, clamp to ±inf
    let sat = if dt.is_fp8() { mx.clone() } else { val.const_like(f64::INFINITY) };
    let neg_sat = if dt.is_fp8() { neg_mx.clone() } else { val.const_like(f64::NEG_INFINITY) };

    // nan → nan, < -mx → -sat, > mx → sat, otherwise → val
    let is_nan = val.ne(val);
    let below = val.lt(&neg_mx);
    let above = mx.lt(val);
    let clamped_above = UOp::try_where(above, sat, val.clone()).expect("f2f_clamp: where failed");
    let clamped = UOp::try_where(below, neg_sat, clamped_above).expect("f2f_clamp: where failed");
    UOp::try_where(is_nan, val.clone(), clamped).expect("f2f_clamp: where failed")
}

/// FP8 STORE decomposition patterns (bpm — sees ORIGINAL children).
///
/// The STORE pattern must run in the bpm slot so it sees the ORIGINAL index dtype
/// (still FP8) before Pattern 1 changes it to UInt8. This is the Morok equivalent
/// of Tinygrad's `tag` mechanism in `pm_float_decomp`.
pub fn pm_float_decomp_store() -> crate::TypedPatternMatcher<Fp8DecompCtx> {
    crate::patterns! {
        @context Fp8DecompCtx;

        // STORE to FP8 buffer → f2f convert value→UInt8, store
        // In bpm, index still has FP8 ptr (ORIGINAL children, before Pattern 1 runs).
        Store { index, value, ranges }
            if index.dtype().base() == ctx.from
        => {
            let target_float = DType::Scalar(ctx.to);
            let target_uint = DType::Scalar(ctx.to.float_to_uint());
            // Cast value to target float (handles FP8, Float32, etc. → Float16)
            let float_val = value.cast(target_float);
            // Bitwise float→FP8 conversion (includes clamping internally)
            let result = f2f(&float_val.bitcast(target_uint), ctx.to, ctx.from);
            // Change index ptr to UInt8
            let uint8_ptr = index.dtype().with_ptr_base(DType::Scalar(ctx.from.float_to_uint()))?;
            let new_index = index.with_dtype(uint8_ptr);
            Some(new_index.store_with_ranges(result, ranges.clone()))
        },
    }
}

/// FP8 float decomposition patterns (pm — sees OPTIMIZED children).
///
/// Port of Tinygrad's `pm_float_decomp` (decompositions.py:504-522).
/// Run via `graph_rewrite_with_bpm` together with `pm_float_decomp_store()`.
pub fn pm_float_decomp() -> crate::TypedPatternMatcher<Fp8DecompCtx> {
    crate::patterns! {
        @context Fp8DecompCtx;

        // Pattern 1: INDEX/DEFINE with FP8 ptr base → change ptr to UInt8
        x if matches!(x.op(), Op::Param { device: None, .. } | Op::DefineLocal(_) | Op::Index { .. })
            && x.dtype().base() == ctx.from
        => {
            let uint8_ptr = x.dtype().with_ptr_base(DType::Scalar(ctx.from.float_to_uint()))?;
            Some(x.with_dtype(uint8_ptr))
        },

        // Pattern 2: LOAD with FP8 dtype → load as UInt8, f2f upcast to target float
        load @ Load { buffer, index, alt } if load.dtype().base() == ctx.from => {
            let uint_dtype = DType::Scalar(ctx.from.float_to_uint()).vec(load.dtype().vcount());
            let uint_alt = alt.clone().map(|a| {
                if a.dtype().base() == ctx.from {
                    a.bitcast(uint_dtype.clone())
                } else {
                    let target_float = DType::Scalar(ctx.to).vec(load.dtype().vcount());
                    let target_uint = DType::Scalar(ctx.to.float_to_uint()).vec(load.dtype().vcount());
                    let float_alt = a.cast(target_float);
                    f2f(&float_alt.bitcast(target_uint), ctx.to, ctx.from)
                }
            });
            let uint_load = UOp::load()
                .buffer(buffer.clone())
                .index(index.clone())
                .maybe_alt(uint_alt)
                .dtype(uint_dtype)
                .call();
            Some(f2f(&uint_load, ctx.from, ctx.to))
        },

        // Pattern 5: CAST to FP8 → full round-trip (Float16→FP8 bytes→Float16).
        // Must do the complete conversion (not just clamp) because the kernel may fuse
        // Cast(Float16→FP8) and Cast(FP8→Float32) without materializing the FP8 buffer.
        x @ Cast { src: val, .. } if x.dtype().base() == ctx.from => {
            let target = DType::Scalar(ctx.to);
            let target_uint = DType::Scalar(ctx.to.float_to_uint());
            let float_val = val.cast(target);
            // Downcast: Float16 bits → FP8 bytes (includes clamping)
            let fp8_bytes = f2f(&float_val.bitcast(target_uint.clone()), ctx.to, ctx.from);
            // Upcast: FP8 bytes → Float16 (proper FP8-quantized value)
            Some(f2f(&fp8_bytes, ctx.from, ctx.to))
        },

        // Pattern 6: Any op with FP8 output dtype → promote to target float, cast FP8 sources
        x if !matches!(x.op(), Op::BitCast { .. })
            && x.dtype().is_float()
            && x.dtype().base() == ctx.from
        => {
            let target_dtype = DType::Scalar(ctx.to);
            let new_dtype = if x.dtype().vcount() > 1 {
                target_dtype.vec(x.dtype().vcount())
            } else {
                target_dtype.clone()
            };
            let new_sources: Vec<Arc<UOp>> = x.op().sources().iter().map(|s| {
                if s.dtype().base() == ctx.from {
                    s.cast(target_dtype.clone())
                } else {
                    s.clone()
                }
            }).collect();
            Some(x.with_sources(new_sources).with_dtype(new_dtype))
        },
    }
}

/// Post-devectorize rendering patterns (devectorizer.py:258-275).
/// Called during codegen, NOT part of pm_devectorize.
pub fn pm_render() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Vector CONST → VECTORIZE of scalar CONST (devectorizer.py:260-261)
        c @ Const(_) if c.dtype().vcount() > 1 => |c| {
            let vcount = c.dtype().vcount();
            let Op::Const(cv) = c.op() else { return None };
            let scalar_const = UOp::const_(c.dtype().scalar_dtype(), cv.0);
            let elements: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|_| scalar_const.clone()).collect();
            Some(UOp::vectorize(elements))
        },

        // VCONST → VECTORIZE of scalar CONSTs (devectorizer.py:262)
        vc @ VConst { values } => |vc, values| {
            let scalar_dtype = vc.dtype().scalar_dtype();
            let elements: SmallVec<[Arc<UOp>; 4]> = values.iter()
                .map(|v| UOp::const_(scalar_dtype.clone(), *v))
                .collect();
            Some(UOp::vectorize(elements))
        },

        // CAT → VECTORIZE (CAT can't be rendered)
        Cat { sources } if sources.len() == 1 => Some(sources[0].clone()),
        Cat { sources } => {
            let elements: SmallVec<[Arc<UOp>; 4]> = sources.iter()
                .flat_map(|src| {
                    let n = src.dtype().vcount();
                    (0..n).map(move |i| if n == 1 { src.clone() } else { src.gep(vec![i]) })
                })
                .collect();
            Some(UOp::vectorize(elements))
        },

        // GEP on scalar → identity
        Gep { vector, indices } if vector.dtype().vcount() == 1 && indices.len() == 1 && indices[0] == 0
            ~> |vector| Arc::clone(vector),

        // GEP identity: [0,1,...,n-1] → unwrap (must be before multi-index GEP)
        Gep { vector, indices } if is_identity_gep(vector, indices) => Some(vector.clone()),

        // GEP(VECTORIZE) → extract (must be before multi-index GEP)
        Gep { vector: Vectorize { elements }, indices } => {
            let extracted: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .filter_map(|&i| elements.get(i).cloned())
                .collect();
            if extracted.len() != indices.len() { return None; }
            Some(if extracted.len() == 1 { extracted[0].clone() } else { UOp::vectorize(extracted) })
        },

        // Multi-index GEP → VECTORIZE (fallback, must be last GEP pattern)
        Gep { vector, indices } if indices.len() > 1 => |vector, indices| {
            let geps: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .map(|&i| vector.gep(vec![i]))
                .collect();
            Some(UOp::vectorize(geps))
        },

        // Single-element unwrap
        Vectorize { elements } if elements.len() == 1 => Some(elements[0].clone()),
        PtrCat { sources } if sources.len() == 1 => Some(sources[0].clone()),

        // =========================================================================
        // Gated Load Alt Patterns (devectorizer.py:266-274)
        // =========================================================================

        // Give any gated LOADs without alt a const 0 alt value (devectorizer.py:267-269)
        // LOAD(INDEX(buf, idx, gate)) where alt is None → LOAD with alt=0
        load @ Load { index, alt: None, .. } if has_gate(index) => |load, index| {
            let alt_value = load.const_like(ConstValue::Int(0));
            Some(UOp::load().buffer(load.load_buffer()?).index(index.clone()).alt(alt_value).dtype(load.dtype()).call())
        },

        // WHERE(c, LOAD(INDEX(buf, idx, c)), alt) → LOAD with alt value (devectorizer.py:289-291)
        // The load's gate matches the WHERE condition.
        // Matches Tinygrad's allow_any_len=True (no alt: None guard).
        // NOTE: if alt is CAST and alt.src.dtype == load.dtype, use alt.src to avoid
        // roundtrip cast (e.g. uint->float->uint).
        Where(cond, load @ Load { index, .. }, alt)
            if index_has_gate_matching(index, cond)
            => |cond, load, index, alt| {
                let casted_alt = cast_alt_avoiding_roundtrip(alt, &load.dtype());
                let new_load = UOp::load()
                    .buffer(load.load_buffer()?)
                    .index(index.clone())
                    .alt(casted_alt)
                    .dtype(load.dtype())
                    .call();
                Some(new_load.cast(alt.dtype()))
            },

        // WHERE(c, alt, LOAD(INDEX(buf, idx, !c))) → LOAD with alt value (devectorizer.py:292-294)
        // Same pattern but with inverted condition in WHERE.
        // is_negation_of handles NOT(cond) and pm_comparison_negations simplified forms.
        Where(cond, alt, load @ Load { index, .. })
            if index_has_inverted_gate_matching(index, cond)
            => |cond, alt, load, index| {
                let casted_alt = cast_alt_avoiding_roundtrip(alt, &load.dtype());
                let new_load = UOp::load()
                    .buffer(load.load_buffer()?)
                    .index(index.clone())
                    .alt(casted_alt)
                    .dtype(load.dtype())
                    .call();
                Some(new_load.cast(alt.dtype()))
            },
    }
}

/// Cast alt value to load dtype, avoiding roundtrip casts (devectorizer.py:290).
/// If alt is CAST(inner) and inner.dtype == load_dtype, use inner directly
/// to avoid e.g. uint→float→uint.
fn cast_alt_avoiding_roundtrip(alt: &Arc<UOp>, load_dtype: &DType) -> Arc<UOp> {
    if let Op::Cast { src: inner, .. } = alt.op()
        && inner.dtype() == *load_dtype
    {
        return inner.clone();
    }
    alt.cast(load_dtype.clone())
}

/// Check if GEP is identity: GEP(x, [0,1,...,n-1]) where n == x.vcount
fn is_identity_gep(vector: &Arc<UOp>, indices: &[usize]) -> bool {
    let vcount = vector.dtype().vcount();
    indices.len() == vcount && indices.iter().enumerate().all(|(i, &j)| i == j)
}

/// Check if index (or casted index) has a gate.
fn has_gate(index: &Arc<UOp>) -> bool {
    match index.op() {
        Op::Index { gate: Some(_), .. } => true,
        Op::Cast { src, .. } => has_gate(src),
        _ => false,
    }
}

/// Check if index has a gate that matches the given condition (pointer equality).
fn index_has_gate_matching(index: &Arc<UOp>, cond: &Arc<UOp>) -> bool {
    match index.op() {
        Op::Index { gate: Some(g), .. } => Arc::ptr_eq(g, cond),
        Op::Cast { src, .. } => index_has_gate_matching(src, cond),
        _ => false,
    }
}

/// Check if index has an inverted gate that matches the given condition.
///
/// Matches INDEX gate that is semantically NOT(cond). Handles three forms:
/// 1. `NOT(cond)` — structural NOT, pointer-equal inner
/// 2. `Lt(c-1, x)` when cond = `Lt(x, c)` — result of pm_comparison_negations on NOT(Lt(x,c))
/// 3. `Lt(x, c+1)` when cond = `Lt(c, x)` — result of pm_comparison_negations on NOT(Lt(c,x))
///
/// Form 2/3 arise because dce_dsl_patterns swaps WHERE(NOT(Lt), t, f) → WHERE(Lt, f, t),
/// then pm_comparison_negations converts the NOT(Lt) on the INDEX gate to a reversed Lt.
fn index_has_inverted_gate_matching(index: &Arc<UOp>, cond: &Arc<UOp>) -> bool {
    match index.op() {
        Op::Index { gate: Some(g), .. } => is_negation_of(g, cond),
        Op::Cast { src, .. } => index_has_inverted_gate_matching(src, cond),
        _ => false,
    }
}

/// Check if `gate` is semantically NOT(cond).
fn is_negation_of(gate: &Arc<UOp>, cond: &Arc<UOp>) -> bool {
    // Form 1: NOT(cond) — structural
    if let Op::Unary(UnaryOp::Not, inner) = gate.op()
        && Arc::ptr_eq(inner, cond)
    {
        return true;
    }

    // Form 2: gate = Lt(c-1, x), cond = Lt(x, c) — from pm_comparison_negations on NOT(Lt(x,c))
    // Form 3: gate = Lt(x, c+1), cond = Lt(c, x) — from pm_comparison_negations on NOT(Lt(c,x))
    if let Op::Binary(BinaryOp::Lt, gate_lhs, gate_rhs) = gate.op()
        && let Op::Binary(BinaryOp::Lt, cond_lhs, cond_rhs) = cond.op()
    {
        // Form 2: gate_rhs == cond_lhs (same x), gate_lhs == c-1 where cond_rhs == c
        if Arc::ptr_eq(gate_rhs, cond_lhs)
            && let (Op::Const(gate_cv), Op::Const(cond_cv)) = (gate_lhs.op(), cond_rhs.op())
            && is_const_minus_one(&cond_cv.0, &gate_cv.0)
        {
            return true;
        }
        // Form 3: gate_lhs == cond_rhs (same x), gate_rhs == c+1 where cond_lhs == c
        if Arc::ptr_eq(gate_lhs, cond_rhs)
            && let (Op::Const(cond_cv), Op::Const(gate_cv)) = (cond_lhs.op(), gate_rhs.op())
            && is_const_minus_one(&cond_cv.0, &gate_cv.0)
        {
            return true;
        }
    }

    false
}

/// Check if `a - 1 == b` (i.e., b = a - 1).
fn is_const_minus_one(a: &ConstValue, b: &ConstValue) -> bool {
    match (a, b) {
        (ConstValue::Int(av), ConstValue::Int(bv)) => av.checked_sub(1) == Some(*bv),
        (ConstValue::UInt(av), ConstValue::UInt(bv)) => av.checked_sub(1) == Some(*bv),
        _ => false,
    }
}

// ============================================================================
// ALU Devectorization
// ============================================================================

/// Generic ALU devectorization: Vector ALU → VECTORIZE of scalar ALU.
///
/// Mirrors Tinygrad's `no_vectorized_alu` (devectorizer.py:219-223):
/// ```python
/// alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg)
///              for i in range(alu.dtype.vcount))
/// return UOp(Ops.VECTORIZE, alu.dtype, alus)
/// ```
fn devectorize_alu(alu: &Arc<UOp>) -> Option<Arc<UOp>> {
    let vcount = alu.dtype().vcount();
    if vcount <= 1 {
        return None;
    }

    // Skip WHERE(cond, t, Invalid) - used for image indexing (devectorizer.py:232)
    // Handles both scalar Invalid and vectorized VECTORIZE(Invalid,...) from expansion.
    if let Op::Ternary(TernaryOp::Where, _, _, f) = alu.op()
        && UOp::is_invalid_marker(f)
    {
        return None;
    }

    let scalar_dtype = alu.dtype().scalar_dtype();
    let sources = alu.op().sources();

    let elements: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
        .map(|i| {
            // Apply GEP to each source, broadcasting scalars
            let new_sources: Vec<Arc<UOp>> =
                sources.iter().map(|s| if s.dtype().vcount() > 1 { s.gep(vec![i]) } else { s.clone() }).collect();

            // CAST and BITCAST need special handling: Op::Cast/BitCast has its own dtype field
            // that must be updated to scalar, not just the UOp's result dtype.
            // The generic replace chain doesn't update Op::Cast::dtype.
            match alu.op() {
                Op::Cast { .. } => new_sources[0].cast(scalar_dtype.clone()),
                Op::BitCast { .. } => new_sources[0].bitcast(scalar_dtype.clone()),
                _ => alu.replace().dtype(scalar_dtype.clone()).src(new_sources).call(),
            }
        })
        .collect();

    Some(UOp::vectorize(elements))
}

/// Vector ALU → VECTORIZE of scalar ALU (devectorizer.py:219-223).
/// LLVM SLP can re-vectorize when beneficial.
#[allow(unused_variables)]
pub fn no_vectorized_alu() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // All binary ops
        for op in binary [*] {
            alu @ op(_, _) if alu.dtype().vcount() > 1 => devectorize_alu(alu),
        },
        // All unary ops
        for op in unary [*] {
            alu @ op(_) if alu.dtype().vcount() > 1 => devectorize_alu(alu),
        },
        // All ternary ops (Where, MulAcc)
        for op in ternary [*] {
            alu @ op(_, _, _) if alu.dtype().vcount() > 1 => devectorize_alu(alu),
        },
        // Cast and BitCast
        alu @ Cast { src: _, .. } if alu.dtype().vcount() > 1 => devectorize_alu(alu),
        alu @ BitCast { src: _, .. } if alu.dtype().vcount() > 1 => devectorize_alu(alu),
    }
}

// ============================================================================
// Devectorize Patterns (devectorizer.py:250-256)
// ============================================================================

/// Combined devectorize patterns: cast_after, ALU, WMMA, buffer/index devectorization.
pub fn devectorize_patterns() -> &'static TypedPatternMatcher {
    use std::sync::LazyLock;
    static CACHED: LazyLock<TypedPatternMatcher> = LazyLock::new(|| {
        cast_after_pattern() + no_vectorized_alu() + no_vectorized_wmma() + devectorize_buf_and_index_patterns()
    });
    &CACHED
}

/// WMMA devectorization (devectorizer.py:208-217).
fn no_vectorized_wmma() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        wmma @ Wmma { a, b, c, metadata } if wmma.dtype().vcount() > wmma_expected_size(metadata)
            => devectorize_wmma(wmma, a, b, c, metadata),
    }
}

fn wmma_expected_size(metadata: &WmmaMetadata) -> usize {
    metadata.upcast_axes.c.iter().map(|(_, size)| size).product::<usize>().max(1)
}

fn devectorize_wmma(
    wmma: &Arc<UOp>,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &WmmaMetadata,
) -> Option<Arc<UOp>> {
    let out_sz = wmma_expected_size(metadata);
    if wmma.dtype().vcount() == out_sz {
        return None;
    }

    // Split each source by its OWN axis sizes (A, B, C may differ).
    // For CUDA 8-16-16 with elements_per_thread=(8,4,4):
    //   A split by 8, B split by 4, C split by 4.
    let sources: [&Arc<UOp>; 3] = [a, b, c];
    let tsrcs: Vec<Vec<Arc<UOp>>> = sources
        .iter()
        .enumerate()
        .map(|(i, src)| {
            let ssz = metadata.upcast_axes.source_size(i);
            let n = src.dtype().vcount();
            (0..n).step_by(ssz).map(|g| src.gep((g..g + ssz.min(n - g)).collect())).collect()
        })
        .collect();

    // Verify all sources have same number of groups
    let num_groups = tsrcs[0].len();
    if tsrcs.iter().any(|t| t.len() != num_groups) {
        tracing::warn!("WMMA devectorization: mismatched source group counts");
        return None;
    }

    // Create new WMMA for each group, flatten with GEP
    let wmma_ex: SmallVec<[Arc<UOp>; 4]> = (0..num_groups)
        .flat_map(|g| {
            let w = UOp::wmma(tsrcs[0][g].clone(), tsrcs[1][g].clone(), tsrcs[2][g].clone(), metadata.clone());
            (0..out_sz).map(move |i| w.gep(vec![i]))
        })
        .collect();

    Some(UOp::vectorize(wmma_ex))
}

/// AFTER(CAST(x), deps) → CAST(AFTER(x, deps)) - allows cast to be optimized independently.
fn cast_after_pattern() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        After { passthrough: Cast { src, dtype }, deps }
            => |src, dtype, deps| {
                let new_after = src.after(deps.clone());
                Some(new_after.cast(dtype.clone()))
            },
    }
}

/// LOCAL/REG buffer devectorization (devectorizer.py:241-248).
///
/// Extended beyond Tinygrad: handles vector indices (not just scalar) by expanding
/// each index lane separately. Tinygrad asserts `idx.dtype.count == 1` which would
/// crash on local buffers with vector indices from UPCAST — Morok's optimizer can
/// produce such kernels (e.g., u3u3 upcast on matmul with local buffers).
fn devectorize_buf_and_index_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // DEFINE_LOCAL/REG with vector pointer → scalar pointer + CAST
        def if matches!(def.op(), Op::DefineLocal(_) | Op::DefineReg { .. })
            && def.ptrdtype().is_some_and(|(base, _, _)| base.vcount() > 1)
            => no_vectorized_buf(def),

        // INDEX(CAST(DEFINE_LOCAL/REG), idx) → scaled vector index
        // Handles both scalar and vector idx (Tinygrad only handles scalar).
        Index { buffer: Cast { src: buf, dtype: cast_dtype }, indices, gate }
            if is_vectorized_local_reg_cast(buf, cast_dtype)
            => no_vectorized_index(buf, indices, gate, cast_dtype),

        // INDEX(BROADCAST(CAST(...)), idx)
        Index { buffer: Vectorize { elements }, indices, gate }
            if is_vectorized_broadcast_cast(elements)
            => {
                let first = elements.first()?;
                let Op::Cast { src: buf, dtype: DType::Ptr { base, .. } } = first.op() else { return None };
                let idx = indices.first()?;
                no_vectorized_index_precnt(buf, idx, gate, base.vcount(), &vec![0; elements.len()])
            },

        // INDEX(GEP(CAST(...)), idx)
        Index { buffer: Gep { vector: Cast { src: buf, dtype: cast_dtype }, indices: gep_indices }, indices, gate }
            if is_vectorized_local_reg_cast(buf, cast_dtype)
            => {
                let DType::Ptr { base, .. } = cast_dtype else { return None };
                let idx = indices.first()?;
                no_vectorized_index_precnt(buf, idx, gate, base.vcount(), gep_indices)
            },
    }
}

fn is_vectorized_local_reg_cast(buf: &Arc<UOp>, cast_dtype: &DType) -> bool {
    matches!(cast_dtype, DType::Ptr { base, .. } if base.vcount() > 1) && is_define_local_or_reg_or_after(buf)
}

fn is_vectorized_broadcast_cast(elements: &SmallVec<[Arc<UOp>; 4]>) -> bool {
    elements.first().is_some_and(|f| {
        matches!(f.op(), Op::Cast { dtype: DType::Ptr { base, .. }, src }
        if base.vcount() > 1 && is_define_local_or_reg_or_after(src))
    })
}

/// Uses `unwrap_after()` to handle `.or_after()` pattern.
fn is_define_local_or_reg_or_after(uop: &Arc<UOp>) -> bool {
    matches!(uop.unwrap_after().op(), Op::DefineLocal(_) | Op::DefineReg { .. })
}

/// Vector pointer → scalar pointer + CAST (devectorizer.py:225-226).
fn no_vectorized_buf(buf: &Arc<UOp>) -> Option<Arc<UOp>> {
    let (base, addrspace, size) = buf.ptrdtype()?;
    let vcount = base.vcount();
    if vcount <= 1 {
        return None;
    }

    let scalar_base = base.base();
    let new_size = size.map(|s| s * vcount);
    let scalar_ptr_dtype =
        DType::Ptr { base: Box::new(DType::Scalar(scalar_base)), addrspace, size: new_size, vcount: 1 };

    let scalar_def = buf.with_dtype(scalar_ptr_dtype);
    Some(scalar_def.cast(buf.dtype()))
}

/// INDEX(CAST(buf), idx) → INDEX(VECTORIZE([buf,...]), scaled_vec_idx) (devectorizer.py:228-231)
///
/// Handles both scalar idx (original Tinygrad path) and vector idx (Morok extension).
/// For vector idx with vcount=V and pointer vcount=cnt, produces total = V*cnt lanes:
///   for each lane i in idx: idx[i]*cnt + [0, 1, ..., cnt-1]
fn no_vectorized_index(
    buf: &Arc<UOp>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    gate: &Option<Arc<UOp>>,
    cast_dtype: &DType,
) -> Option<Arc<UOp>> {
    let idx = indices.first()?;
    let DType::Ptr { base, .. } = cast_dtype else { return None };
    let cnt = base.vcount();
    if cnt <= 1 {
        return None;
    }

    let idx_vcount = idx.dtype().vcount();
    let total = cnt * idx_vcount;
    let buf_broadcast = buf.broadcast(total);

    let final_idx = if idx_vcount == 1 {
        // Scalar path (original Tinygrad logic)
        let idx_broadcast = idx.broadcast(cnt);
        let cnt_broadcast = idx.const_like(cnt as i64).broadcast(cnt);
        idx_broadcast.mul(&cnt_broadcast).add(&create_index_vector(0..cnt as i64))
    } else {
        // Vector path: expand each lane by cnt elements
        // idx = [a, b, c], cnt = 3 → [a*3+0, a*3+1, a*3+2, b*3+0, b*3+1, b*3+2, c*3+0, c*3+1, c*3+2]
        let elements: SmallVec<[Arc<UOp>; 4]> = (0..idx_vcount)
            .flat_map(|i| {
                let lane = idx.gep(vec![i]);
                let cnt_const = UOp::const_(lane.dtype(), ConstValue::Int(cnt as i64));
                let scaled = lane.mul(&cnt_const);
                (0..cnt).map(move |j| scaled.add(&UOp::const_(scaled.dtype(), ConstValue::Int(j as i64))))
            })
            .collect();
        UOp::vectorize(elements)
    };

    // Expand gate to match total lanes if it's vectorized
    let expanded_gate = if idx_vcount > 1 {
        gate.as_ref().map(|g| {
            if g.dtype().vcount() > 1 {
                let elements: SmallVec<[Arc<UOp>; 4]> = (0..idx_vcount)
                    .flat_map(|i| {
                        let lane = g.gep(vec![i]);
                        std::iter::repeat_n(lane, cnt)
                    })
                    .collect();
                UOp::vectorize(elements)
            } else {
                g.broadcast(total)
            }
        })
    } else {
        gate.clone()
    };

    Some(
        UOp::index()
            .buffer(buf_broadcast)
            .indices(vec![final_idx])
            .maybe_gate(expanded_gate)
            .ptr(true)
            .call()
            .expect("ICE unable to create index"),
    )
}

fn create_index_vector(values: impl IntoIterator<Item = i64>) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> = values.into_iter().map(UOp::index_const).collect();
    UOp::vectorize(elements)
}

/// INDEX with precnt multiplier (broadcast or gep case) (devectorizer.py:233-239)
///
/// Handles both scalar and vector idx. For vector idx, each lane is expanded
/// independently with the same precnt/cnt scaling.
fn no_vectorized_index_precnt(
    buf: &Arc<UOp>,
    idx: &Arc<UOp>,
    gate: &Option<Arc<UOp>>,
    cnt: usize,
    input_gep: &[usize],
) -> Option<Arc<UOp>> {
    let precnt = input_gep.len();
    let idx_vcount = idx.dtype().vcount();

    if idx_vcount == 1 {
        // Scalar path (original Tinygrad logic)
        let total = cnt * precnt;
        let gep_arg: Vec<usize> = (0..cnt).flat_map(|_| 0..precnt).collect();
        let sum_arg = (0..cnt).flat_map(|i| input_gep.iter().map(move |&y| (i + y) as i64));

        let buf_broadcast = buf.broadcast(total);
        let final_idx =
            idx.gep(gep_arg).mul(&idx.const_like(cnt as i64).broadcast(total)).add(&create_index_vector(sum_arg));

        Some(
            UOp::index()
                .buffer(buf_broadcast)
                .indices(vec![final_idx])
                .maybe_gate(gate.clone())
                .ptr(true)
                .call()
                .expect("ICE: unable to create index"),
        )
    } else {
        // Vector path: expand each lane with the same precnt/cnt scaling
        let per_lane = cnt * precnt;
        let total = per_lane * idx_vcount;

        let buf_broadcast = buf.broadcast(total);
        let elements: SmallVec<[Arc<UOp>; 4]> = (0..idx_vcount)
            .flat_map(|i| {
                let lane = idx.gep(vec![i]);
                let cnt_const = UOp::const_(lane.dtype(), ConstValue::Int(cnt as i64));
                let scaled = lane.mul(&cnt_const);
                (0..cnt).flat_map(move |c| {
                    let s = scaled.clone();
                    input_gep.iter().map(move |&y| s.add(&UOp::const_(s.dtype(), ConstValue::Int((c + y) as i64))))
                })
            })
            .collect();
        let final_idx = UOp::vectorize(elements);

        let expanded_gate = gate.as_ref().map(|g| {
            if g.dtype().vcount() > 1 {
                let elements: SmallVec<[Arc<UOp>; 4]> = (0..idx_vcount)
                    .flat_map(|i| {
                        let lane = g.gep(vec![i]);
                        std::iter::repeat_n(lane, per_lane)
                    })
                    .collect();
                UOp::vectorize(elements)
            } else {
                g.broadcast(total)
            }
        });

        Some(
            UOp::index()
                .buffer(buf_broadcast)
                .indices(vec![final_idx])
                .maybe_gate(expanded_gate)
                .ptr(true)
                .call()
                .expect("ICE: unable to create index"),
        )
    }
}

// ============================================================================
// Load Store Indexing Patterns (devectorizer.py:48-55)
// ============================================================================

/// INDEX(buf, x, true) → INDEX(buf, x, None)
///
/// Tinygrad (devectorizer.py:48-55) has 2 additional IMAGE-specific patterns:
///
/// 1. `simplify_valid_load(buf, x, cond)` for `INDEX(buf, WHERE(cond, x, Invalid))`
/// 2. `simplify_valid_load(buf, x, c)` for `INDEX(buf, x:long, c:bool)` (post-lowering)
///
/// These use `uop_given_valid`/`parse_valid` and are tied to ImageDType.
/// TODO: Add when implementing IMAGE backend support.
pub fn load_store_indexing_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // INDEX(buf, idx, true) → INDEX(buf, idx, None) — remove trivially-true gate.
        // Uses UOp::new directly to preserve the original dtype without builder inference,
        // since the builder's dtype logic (ptr flag, element extraction) may not match
        // the already-determined dtype on the matched INDEX node.
        index @ Index { buffer, indices, gate: Some(g) }
            if matches!(g.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Bool(true)))
            ~> UOp::new(Op::Index { buffer: buffer.clone(), indices: indices.clone(), gate: None }, index.dtype())
    }
}

// ============================================================================
// Add Loads Patterns (devectorizer.py:320-326)
// ============================================================================

/// Add LOAD to non-pointer INDEX, remove LOAD wrapper from STORE.
pub fn pm_add_loads() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Add LOAD to non-ptr INDEX: INDEX(buf, idx) → LOAD(INDEX(buf, idx))
        // Skip if dtype is already Ptr (devectorizer.py:322-323)
        idx @ Index { buffer, .. } if !is_ptr_or_image_dtype(&idx.dtype()) => {
            let new_idx = idx.with_dtype(buffer.dtype());
            Some(UOp::load().buffer(buffer.clone()).index(new_idx).dtype(idx.dtype().scalar_dtype()).call())
        },

        // Remove LOAD wrapper from STORE: STORE(LOAD(x), ...) → STORE(x, ...)
        // (devectorizer.py:325)
        Store { index: Load { index: inner_idx, .. }, value, ranges }
            => Some(inner_idx.store_with_ranges(value.clone(), ranges.clone())),
    }
}

fn is_ptr_or_image_dtype(dtype: &DType) -> bool {
    matches!(dtype, DType::Ptr { .. } | DType::Image { .. })
}

// ============================================================================
// WMMA Accumulation Patterns (devectorizer.py:314-315)
// ============================================================================

/// Fuse Add into WMMA's accumulator: WMMA(a,b,c) + add → WMMA(a,b,c+add)
/// Tensor cores have built-in accumulation, so this is more efficient.
pub fn pm_wmma_accumulate() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // WMMA + add → WMMA with fused accumulator (devectorizer.py:314-315)
        // Pattern: Add(WMMA(a, b, c), add) → WMMA(a, b, Add(c, add))
        Add(wmma @ Wmma { a, b, c, metadata }, add) => |wmma, a, b, c, metadata, add| {
            // Only fuse if types match
            if wmma.dtype() != add.dtype() {
                return None;
            }
            let new_c = c.add(add);
            Some(UOp::wmma(a.clone(), b.clone(), new_c, metadata.clone()))
        },

        // Commutative: add + WMMA → WMMA with fused accumulator
        Add(add, wmma @ Wmma { a, b, c, metadata }) => |wmma, add, a, b, c, metadata| {
            if wmma.dtype() != add.dtype() {
                return None;
            }
            let new_c = c.add(add);
            Some(UOp::wmma(a.clone(), b.clone(), new_c, metadata.clone()))
        },
    }
}

// ============================================================================
// Load Store Folding Patterns (devectorizer.py:114-126)
// ============================================================================
/// Tinygrad load_store_folding (devectorizer.py:119-132).
/// 6 patterns in one matcher, exactly matching Tinygrad's order.
pub fn load_store_folding_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // 1. expand_index: INDEX(VECTORIZE(buf), vec) → VECTORIZE(INDEX(buf, gep(0)), ...)
        index if is_vector_index(index) => expand_index_to_vectorize(index),

        // 2. fold_expanded_index: VECTORIZE(INDEX, INDEX, ...) → GEP(PTRCAT(...), indices)
        midx @ Vectorize { elements } if elements.iter().all(|e| matches!(e.op(), Op::Index { .. }))
            => fold_expanded_index(midx),

        // 3. GEP after LOAD: LOAD(buf, GEP(x)) → GEP(LOAD(buf, x))
        load @ Load { buffer, index: Gep { vector, indices } }
            => move_gep_after_load(load, buffer, vector, indices),

        // 4. GEP on STORE: STORE(GEP(x), data) → STORE(x, GEP⁻¹(data))
        Store { index: Gep { vector, indices }, value, ranges }
            => move_gep_on_store(vector, indices, value, ranges),

        // 5. PTRCAT after LOAD: LOAD(buf, PTRCAT) → CAT(LOAD(buf_i, ptr_i), ...)
        load @ Load { buffer, index: ptrcat @ PtrCat { sources } }
            => distribute_ptrcat_load(load, buffer, ptrcat, sources),

        // 6. PTRCAT after STORE: STORE(PTRCAT, data) → GROUP(STORE(ptr_i, gep(data, i)), ...)
        Store { index: PtrCat { sources }, value, ranges }
            => distribute_ptrcat_store(sources, value, ranges),
    }
}

// ============================================================================
// Correct Load Store Patterns (devectorizer.py:198-203)
// ============================================================================

/// LOAD/STORE(CAST(INDEX)) → split by device fold lengths + image fixup.
pub fn correct_load_store_patterns() -> &'static TypedPatternMatcher {
    crate::cached_patterns! {
        // Split LOAD/STORE by device fold lengths
        ls @ Load { index: Cast { src: idx @ Index { buffer: _, .. }, .. }, .. }
            => split_load_store(ls, idx),

        ls @ Store { index: Cast { src: idx @ Index { buffer: _, .. }, .. }, .. }
            => split_load_store(ls, idx),

        // Image fixup patterns (devectorizer.py:176-196)
        ls @ Load { buffer: _, index: _, alt: _ } => image_fixup(ls),
        ls @ Store { index: _, value: _, ranges: _ } => image_fixup(ls),
    }
}

// ============================================================================
// Pattern Predicates
// ============================================================================

fn is_define_or_after(uop: &Arc<UOp>) -> bool {
    matches!(uop.unwrap_after().op(), Op::DefineLocal(_) | Op::DefineReg { .. } | Op::Param { device: None, .. })
}

/// Matches INDEX(VECTORIZE(Defines.or_after()), vec_idx) only.
/// Tinygrad devectorizer.py:115 - expand_index only matches VECTORIZE of defines.
fn is_vector_index(uop: &Arc<UOp>) -> bool {
    let Op::Index { buffer, indices, gate } = uop.op() else { return false };
    if indices.len() != 1 || gate.is_some() {
        return false;
    }
    let Some(idx) = indices.first() else { return false };
    if idx.dtype().vcount() <= 1 {
        return false;
    }
    let Op::Vectorize { elements } = buffer.op() else { return false };
    !elements.is_empty() && elements.iter().all(is_define_or_after)
}

// ============================================================================
// GEP Movement Patterns (devectorizer.py:106-120)
// ============================================================================

/// LOAD(GEP(ptr)) → GEP(LOAD(ptr)).
///
/// Tinygrad (devectorizer.py:117-118):
/// ```python
/// lambda gep, ld: ld.replace(dtype=ld.dtype.scalar().vec(gep.dtype.count),
///                            src=(gep.src[0],)+ld.src[1:]).gep(gep.arg)
/// ```
fn move_gep_after_load(
    load: &Arc<UOp>,
    buffer: &Arc<UOp>,
    gep_inner: &Arc<UOp>,
    gep_indices: &[usize],
) -> Option<Arc<UOp>> {
    let new_dtype = load.dtype().scalar_dtype().vec(gep_indices.len());
    let inner_load = load.replace().dtype(new_dtype).src(vec![buffer.clone(), gep_inner.clone()]).call();
    Some(inner_load.gep(gep_indices.to_vec()))
}

/// STORE(GEP(ptr), data) → STORE(ptr, GEP⁻¹(data)). Inverts GEP indices.
fn move_gep_on_store(
    gep_inner: &Arc<UOp>,
    gep_indices: &[usize],
    value: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
) -> Option<Arc<UOp>> {
    // Invert GEP: [2,0,1] → sorted by key → [1,2,0]
    let mut inverse_map: Vec<(usize, usize)> = gep_indices.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    inverse_map.sort_by_key(|&(x, _)| x);
    let inverse_indices: Vec<usize> = inverse_map.iter().map(|&(_, i)| i).collect();

    let reordered_value = value.gep(inverse_indices);
    Some(gep_inner.store_with_ranges(reordered_value, ranges.clone()))
}

// ============================================================================
// expand_index (devectorizer.py:59-95)
// ============================================================================

/// Vector INDEX → grouped PTRCAT. Generates scalar indices, simplifies, groups by root+offset.
/// Phase 1a: Expand vector INDEX into VECTORIZE of scalar INDEXes.
/// Matches Tinygrad's `expand_index` (devectorizer.py:59-62).
/// NO inner rewrite — the outer fixed-point (sym) simplifies GEP expressions.
fn expand_index_to_vectorize(index: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate } = index.op() else { return None };
    if indices.len() != 1 {
        return None;
    }
    let vec = indices.first()?;
    let count = vec.dtype().vcount();

    let buf = if let Op::Vectorize { elements } = buffer.op() { elements.first()?.clone() } else { buffer.clone() };

    let scalar_indices: Vec<_> = (0..count)
        .map(|i| {
            let lane_gate = gate.as_ref().map(|g| if g.dtype().vcount() > 1 { g.gep(vec![i]) } else { g.clone() });
            UOp::index()
                .buffer(buf.clone())
                .indices(vec![vec.gep(vec![i])])
                .maybe_gate(lane_gate)
                .ptr(true)
                .call()
                .expect("ICE: unable to create index")
        })
        .collect();

    Some(UOp::vectorize(scalar_indices.into()))
}

/// Phase 1b: Fold VECTORIZE of scalar INDEXes into PTRCAT groupings.
/// Matches Tinygrad's `fold_expanded_index` (devectorizer.py:64-100).
/// By this point, the outer sym fixed-point has simplified GEP expressions
/// into concrete root+offset form so grouping can identify consecutive accesses.
fn fold_expanded_index(midx: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Vectorize { elements: sources } = midx.op() else { return None };
    let count = sources.len();
    if count == 0 {
        return None;
    }

    // Verify all elements are INDEX and share the same buffer
    let first_buf = match sources[0].op() {
        Op::Index { buffer, .. } => buffer,
        _ => return None,
    };
    if !sources.iter().all(|s| matches!(s.op(), Op::Index { buffer, .. } if Arc::ptr_eq(buffer, first_buf))) {
        return None;
    }
    let buf = first_buf;

    // Extract (valid, root, offset, gate) for each lane.
    struct LaneData {
        valid: Arc<UOp>,
        root: Arc<UOp>,
        offset: i64,
        gate_id: u64,
    }
    let mut lane_data: Vec<(usize, LaneData)> = Vec::with_capacity(count);

    for (lane, idx_op) in sources.iter().enumerate() {
        let Op::Index { indices: simp_indices, gate: lane_gate, .. } = idx_op.op() else { continue };
        let idx = simp_indices.first()?.get_idx();
        let valid = simp_indices.first()?.get_valid();
        let gate_id = lane_gate.as_ref().map_or(u64::MAX, |g| g.id);

        let (root, offset) = match idx.op() {
            Op::Invalid => (UOp::invalid_marker(), 0),
            Op::Binary(BinaryOp::Add, l, r) if matches!(r.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Int(_))) => {
                let Op::Const(cv) = r.op() else { unreachable!() };
                let ConstValue::Int(off) = cv.0 else { unreachable!() };
                (l.clone(), off)
            }
            Op::Binary(BinaryOp::Add, l, r) if matches!(l.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Int(_))) => {
                let Op::Const(cv) = l.op() else { unreachable!() };
                let ConstValue::Int(off) = cv.0 else { unreachable!() };
                (r.clone(), off)
            }
            Op::Const(cv) if matches!(cv.0, ConstValue::Int(_)) => {
                let ConstValue::Int(off) = cv.0 else { unreachable!() };
                (UOp::index_const(0), off)
            }
            _ => (idx.clone(), 0),
        };

        lane_data.push((lane, LaneData { valid, root, offset, gate_id }));
    }

    // Build grouping map
    let mut offsets_by_root: HashMap<(u64, u64, u64), HashMap<i64, Vec<usize>>> = HashMap::new();
    for (lane, data) in &lane_data {
        let key = (data.valid.id, data.root.id, data.gate_id);
        offsets_by_root.entry(key).or_default().entry(data.offset).or_default().push(*lane);
    }

    // Group consecutive offsets and build PTRCAT
    let mut ret = Vec::new();
    let mut idxs: Vec<Option<usize>> = vec![None; count];
    let mut global_offset = 0;

    for offsets in offsets_by_root.values() {
        let groups = group_consecutive_offsets_from_map(offsets);
        for grp in groups {
            let lidx = sources[offsets[&grp[0]][0]].clone();
            let ptr = if grp.len() > 1 { lidx.cast(make_vec_ptr_dtype(buf, grp.len())) } else { lidx };
            for (i, &offset) in grp.iter().enumerate() {
                for &lane in &offsets[&offset] {
                    idxs[lane] = Some(global_offset + i);
                }
            }
            ret.push(ptr);
            global_offset += grp.len();
        }
    }

    if idxs.iter().any(|x| x.is_none()) {
        return None;
    }

    let DType::Ptr { base, addrspace, size, .. } = buf.dtype().clone() else { return None };
    let scalar_ptr = DType::Ptr { base: Box::new(DType::Scalar(base.scalar()?)), addrspace, size, vcount: 1 };
    let ptrcat_dtype = scalar_ptr.vec(global_offset);
    let ptrcat = UOp::ptrcat().sources(ret).dtype(ptrcat_dtype).call();
    let gep_indices: Vec<usize> = idxs.into_iter().map(|x| x.unwrap()).collect();

    Some(ptrcat.gep(gep_indices))
}

/// Groups offsets where `offset - index` is constant.
///
/// Returns groups of consecutive offset keys. Multi-lane offsets (broadcasts)
/// are handled by the caller — all lanes sharing an offset get the same PTRCAT slot.
/// Matches Tinygrad's `itertools.groupby(enumerate(sorted(offsets.keys())), lambda x: x[1]-x[0])`.
fn group_consecutive_offsets_from_map(offsets_map: &HashMap<i64, Vec<usize>>) -> Vec<Vec<i64>> {
    if offsets_map.is_empty() {
        return vec![];
    }

    let sorted: Vec<_> = offsets_map.keys().copied().sorted().collect();
    sorted
        .iter()
        .copied()
        .enumerate()
        .chunk_by(|(idx, offset)| offset - (*idx as i64))
        .into_iter()
        .map(|(_, group)| group.map(|(_, offset)| offset).collect())
        .collect()
}

fn make_vec_ptr_dtype(buffer: &Arc<UOp>, vec_len: usize) -> DType {
    let (base_dtype, addrspace) = buffer
        .ptrdtype()
        .map(|(base, addrspace, _)| (base.base(), addrspace))
        .unwrap_or_else(|| (buffer.dtype().base(), AddrSpace::Global));
    let vec_dtype = DType::Vector { scalar: base_dtype, count: vec_len };
    DType::Ptr { base: Box::new(vec_dtype), addrspace, size: Some(vec_len), vcount: 1 }
}

// ============================================================================
// PTRCAT Distribution (devectorizer.py:97-104, 122-123)
// ============================================================================

/// LOAD(PTRCAT) → CAT(LOADs). CAT dtype = ptrcat.dtype.base.vec(ptrcat.dtype.vcount)
/// LOAD(buf, PTRCAT(idx0,idx1,...)) → CAT(LOAD(buf_i, idx_i), ...)
///
/// Matches Tinygrad devectorizer.py:128-129:
///   ld.replace(dtype=x.dtype.base, src=(x,)+ld.src[1:]) for x in cat.src
///
/// Each PtrCat source is a scalar INDEX(buf, offset). The distributed scalar
/// LOAD uses that INDEX directly, with the scalar buffer from GEP(buffer, i).
fn distribute_ptrcat_load(
    load: &Arc<UOp>,
    buffer: &Arc<UOp>,
    ptrcat: &Arc<UOp>,
    sources: &[Arc<UOp>],
) -> Option<Arc<UOp>> {
    let loads: Vec<Arc<UOp>> = sources
        .iter()
        .enumerate()
        .map(|(i, ptr)| {
            let load_dtype = match ptr.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other.clone(),
            };
            // Extract scalar buffer for this lane.
            // Tinygrad: ld.replace(src=(x,)+ld.src[1:]) — PtrCat source IS the full
            // INDEX(buf, idx), so the scalar load doesn't need the outer buffer at all.
            // Each PtrCat source already contains its own buffer reference.
            // For VECTORIZE(buf, buf, ...) just use the scalar element.
            let scalar_buf = match buffer.op() {
                Op::Vectorize { elements, .. } => elements.get(i).cloned().unwrap_or_else(|| buffer.clone()),
                _ => buffer.clone(),
            };
            let alt = match load.op() {
                Op::Load { alt, .. } => alt.clone(),
                _ => None,
            };
            UOp::load().buffer(scalar_buf).index(ptr.clone()).maybe_alt(alt).dtype(load_dtype).call()
        })
        .collect();

    let cat_dtype = DType::Scalar(ptrcat.dtype().base()).vec(ptrcat.dtype().vcount());
    Some(UOp::cat().sources(loads).dtype(cat_dtype).call())
}

/// STORE(PTRCAT, data) → GROUP(STOREs with GEP-sliced data)
fn distribute_ptrcat_store(
    sources: &[Arc<UOp>],
    value: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
) -> Option<Arc<UOp>> {
    let value_vcount = value.dtype().vcount();
    let mut stores = Vec::new();
    let mut offset = 0usize;

    for ptr in sources.iter() {
        let ptr_count = ptr_element_count(ptr);
        debug_assert!(offset + ptr_count <= value_vcount, "PTRCAT size mismatch");
        let gep_indices: Vec<usize> = (offset..offset + ptr_count).collect();
        let store_value = value.gep(gep_indices);
        stores.push(ptr.store_with_ranges(store_value, ranges.clone()));
        offset += ptr_count;
    }

    Some(UOp::group(stores.into_iter().collect()))
}

/// Get the element count for a PTRCAT source pointer.
///
/// This should return the vcount of the base type, NOT the buffer size.
/// For `Ptr { base: Scalar(Float32), size: Some(4), .. }` → 1 (scalar access)
/// For `Ptr { base: Vector { count: 2, .. }, size: Some(2), .. }` → 2 (vec2 access)
///
/// Tinygrad uses `dtype.count` which returns the base type's element count.
fn ptr_element_count(ptr: &Arc<UOp>) -> usize {
    match ptr.dtype() {
        DType::Ptr { base, .. } => base.vcount(),
        _ => 1,
    }
}

// ============================================================================
// split_load_store (devectorizer.py:130-174)
// ============================================================================

/// Split LOAD/STORE into multiple chunks by device fold lengths (devectorizer.py:130-174).
fn split_load_store(ls: &Arc<UOp>, idx: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer: buf, indices, .. } = idx.op() else { return None };

    // sz = ls.src[0].dtype.count (Tinygrad: size from index dtype)
    // For Ptr types, we need base.vcount() — the pointee's vector count.
    // index.dtype().vcount() returns the pointer's vector count (always 1 for CAST'd pointers).
    let sz = match ls.op() {
        Op::Load { index, .. } | Op::Store { index, .. } => ptr_element_count(index),
        _ => return None,
    };
    if sz == 1 {
        return None;
    }

    // Fold lengths (devectorizer.py:138-152)
    let buf_dtype = buf.dtype();
    static IS_AMX: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("MOROK_AMX").is_ok_and(|v| v == "1"));
    let is_amx = *IS_AMX;

    // AMX TC accumulators are stored in DEFINE_REG (AddrSpace::Reg) but need vector stores.
    // For STORE: check if VALUE comes from an AMX TC accumulator (DEFINE_REG with AddrSpace::Reg).
    // For LOAD: check if BUFFER is an AMX TC accumulator.
    fn is_amx_tc_reg_ptr(dtype: &DType, sz: usize) -> bool {
        sz >= 16
            && dtype.base().is_float()
            && matches!(dtype, DType::Ptr { addrspace: AddrSpace::Reg, .. } | DType::Vector { .. })
    }

    // Helper to find underlying LOAD through GEP chains
    fn find_underlying_load(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
        match uop.op() {
            Op::Gep { vector, .. } => find_underlying_load(vector),
            Op::Load { .. } => Some(uop.clone()),
            _ => None,
        }
    }

    let is_amx_tc_acc = match ls.op() {
        Op::Store { value, .. } => {
            // Check if value comes from LOAD of DEFINE_REG (AMX accumulator)
            // Value may be GEP(LOAD(...)), so trace through GEP chains
            if let Some(load) = find_underlying_load(value) {
                if let Op::Load { index, .. } = load.op() {
                    if let Op::Index { buffer, .. } = index.op() {
                        let buf_dtype = buffer.dtype();
                        is_amx && is_amx_tc_reg_ptr(&buf_dtype, sz)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        }
        Op::Load { .. } => is_amx && is_amx_tc_reg_ptr(&buf_dtype, sz),
        _ => false,
    };

    // Don't fold for non-float types or Image, but allow AMX TC accumulators.
    // Tinygrad: no_fold is False for AMX operations since they use in-memory accumulation.
    let no_fold = (!buf_dtype.base().is_float() && !matches!(buf_dtype, DType::Image { .. }))
        || (matches!(buf_dtype, DType::Ptr { addrspace: AddrSpace::Reg, .. }) && !is_amx_tc_acc);

    let mut lengths = if no_fold {
        vec![1]
    } else if matches!(buf_dtype, DType::Image { .. }) {
        vec![4, 1]
    } else if is_amx {
        vec![16, 8, 4, 2, 1] // AMX: wider folds matching 64-byte row stride
    } else {
        // Tinygrad uses ctx.supports_float4 from Renderer context (devectorizer.py:155-157).
        // Hardcoded [4,2,1] matches the default supports_float4 path.
        // TODO: Pass Renderer context through when adding backends with different fold lengths.
        vec![4, 2, 1]
    };

    // Filter by divisibility (devectorizer.py:155-156)
    // NOTE: Tinygrad has `must_divide=False` for DSP devices which skips this check.
    // DSP uses larger fold lengths [128,64,32,16,8,4] without divisibility requirement.
    // AMX TC accumulators also skip divisibility check to allow vector stores.
    if !is_amx_tc_acc && let Some(offset) = indices.first() {
        lengths.retain(|&len| offset_divides_evenly(offset, len));
    }

    // Split loop (devectorizer.py:159-170)
    let scalar_dtype = buf_dtype.scalar_dtype();
    let mut ret = Vec::new();
    let mut pos = 0usize;

    while pos < sz {
        for &fold_len in &lengths {
            if pos + fold_len > sz {
                continue;
            }
            let lidx = if pos == 0 { idx.clone() } else { offset_index(idx, pos as i64) };
            let lidx = if fold_len > 1 { lidx.cast(make_vec_ptr_dtype(buf, fold_len)) } else { lidx };

            match ls.op() {
                Op::Store { value, ranges, .. } => {
                    ret.push(lidx.store_with_ranges(value.gep((pos..pos + fold_len).collect()), ranges.clone()));
                }
                Op::Load { buffer, alt, .. } => {
                    let load = if let Some(alt) = alt {
                        UOp::load()
                            .buffer(buffer.clone())
                            .index(lidx)
                            .dtype(scalar_dtype.vec(fold_len))
                            .alt(slice_vector_lane(alt, pos, fold_len))
                            .call()
                    } else {
                        UOp::load().buffer(buffer.clone()).index(lidx).dtype(scalar_dtype.vec(fold_len)).call()
                    };
                    ret.push(load);
                }
                _ => return None,
            }
            pos += fold_len;
            break;
        }
    }

    if ret.len() <= 1 {
        return None;
    }

    match ls.op() {
        Op::Load { .. } => Some(UOp::cat().sources(ret).dtype(scalar_dtype.vec(sz)).call()),
        Op::Store { .. } => Some(UOp::group(ret.into_iter().collect())),
        _ => None,
    }
}

/// Check if offset expression divides evenly by len (devectorizer.py:703-711).
/// Conservative: false for unknown expressions.
fn offset_divides_evenly(offset: &Arc<UOp>, len: usize) -> bool {
    // len==0 is invalid (can't divide by zero), return false defensively
    if len == 0 {
        return false;
    }
    // len==1 means no vectorization, trivially true
    if len == 1 {
        return true;
    }
    let v = len as i64;

    match offset.op() {
        // CONST: check modulo
        Op::Const(cv) => matches!(cv.0, ConstValue::Int(n) if n % v == 0),

        // VCONST: all elements must divide evenly
        Op::VConst { values } => values.iter().all(|val| matches!(val, ConstValue::Int(n) if n % v == 0)),

        // ADD: both operands must divide
        Op::Binary(BinaryOp::Add, left, right) => offset_divides_evenly(left, len) && offset_divides_evenly(right, len),

        // MUL: either operand divides (matching Tinygrad - no n >= len check!)
        Op::Binary(BinaryOp::Mul, left, right) => {
            let check_const =
                |c: &Arc<UOp>| matches!(c.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Int(n) if n % v == 0));
            check_const(left)
                || check_const(right)
                || offset_divides_evenly(left, len)
                || offset_divides_evenly(right, len)
        }

        _ => false,
    }
}

fn offset_index(idx: &Arc<UOp>, offset: i64) -> Arc<UOp> {
    let Op::Index { buffer, indices, gate } = idx.op() else {
        return idx.clone();
    };
    let new_indices: SmallVec<[Arc<UOp>; 4]> = indices
        .iter()
        .enumerate()
        .map(|(i, index_expr)| if i == 0 { index_expr.add(&index_expr.const_like(offset)) } else { index_expr.clone() })
        .collect();

    UOp::index()
        .buffer(buffer.clone())
        .indices(new_indices)
        .maybe_gate(gate.clone())
        .ptr(true)
        .call()
        .expect("ICE: unable to create index")
}

fn slice_vector_lane(value: &Arc<UOp>, pos: usize, fold_len: usize) -> Arc<UOp> {
    if value.dtype().vcount() == 1 {
        // Scalar inputs are broadcast across the lane (mirrors tinygrad's
        // split_load_store, where a shared scalar alt is reused across every
        // per-lane LOAD; the codegen-side invariant that alt vcount matches
        // the LOAD lane width is restored here by splatting at split time).
        return value.broadcast(fold_len);
    }
    if fold_len == 1 {
        return value.gep(vec![pos]);
    }
    value.gep((pos..pos + fold_len).collect())
}

// ============================================================================
// image_fixup (devectorizer.py:176-196)
// ============================================================================

/// Convert linear image index to 2D (x, y) coordinates.
///
/// For images with shape [height, width]:
///   x_coord = (linear_idx // 4) % width
///   y_coord = linear_idx // (4 * width)
///
/// Handles two cases:
/// 1. Normal image load/store with CAST from expand_index (dtype.count == 4)
/// 2. Unfoldable image load (no CAST, direct INDEX with ImageDType)
fn image_fixup(ls: &Arc<UOp>) -> Option<Arc<UOp>> {
    // Case 1: LOAD/STORE(CAST(INDEX)) where INDEX.buffer is ImageDType
    // The CAST should be to a vec4 pointer
    let (index, is_load) = match ls.op() {
        Op::Load { index, .. } => (index, true),
        Op::Store { index, .. } => (index, false),
        _ => return None,
    };

    // Check for CAST(INDEX) pattern
    if let Op::Cast { src: inner_idx, dtype: cast_dtype } = index.op()
        && let Op::Index { buffer: img_buf, indices, gate } = inner_idx.op()
    {
        // Check if buffer is ImageDType
        let DType::Image { shape, .. } = img_buf.dtype() else { return None };

        // Image must be casted to vec4 (RGBA)
        if cast_dtype.vcount() != 4 {
            return None;
        }

        // Get the first index (linear index)
        let lin_idx = indices.first()?;
        let x = lin_idx.get_idx();
        let valid = lin_idx.get_valid();

        // Get image width (shape[1])
        let width = shape.get(1).copied().unwrap_or(1) as i64;

        // Create 2D index: x_coord = (x // 4) % width, y_coord = x // (4 * width)
        let four = UOp::index_const(4);
        let width_const = UOp::index_const(width);
        let stride = UOp::index_const(4 * width);

        let x_coord = x.idiv(&four).mod_(&width_const);
        let y_coord = x.idiv(&stride);

        // Create vec2 index
        let oidx = UOp::vectorize(smallvec::smallvec![x_coord, y_coord]);

        // Apply validity if not always true
        let new_idx_expr = if matches!(valid.op(), Op::Const(cv) if cv.0 == ConstValue::Bool(true)) {
            oidx
        } else {
            oidx.valid(valid)
        };

        // Create new INDEX with 2D coordinates
        // Use ptr(true) when inner_idx has Ptr dtype, otherwise preserve element dtype
        let new_idx = if matches!(inner_idx.dtype(), DType::Ptr { .. }) {
            UOp::index()
                .buffer(img_buf.clone())
                .indices(vec![new_idx_expr])
                .maybe_gate(gate.clone())
                .ptr(true)
                .call()
                .ok()?
        } else {
            UOp::index()
                .buffer(img_buf.clone())
                .indices(vec![new_idx_expr])
                .maybe_gate(gate.clone())
                .dtype(inner_idx.dtype())
                .call()
                .ok()?
        };

        // Replace the index in LOAD/STORE while preserving all other sources (including LOAD.alt).
        let mut src = ls.op().sources().to_vec();
        let index_pos = if is_load { 1 } else { 0 };
        src[index_pos] = new_idx;
        return Some(ls.replace().src(src).call());
    }

    // Case 2: Direct INDEX with ImageDType (unfoldable image, no CAST)
    if let Op::Index { buffer: img_buf, indices, gate } = index.op() {
        let DType::Image { shape, .. } = img_buf.dtype() else { return None };

        // Get the first index
        let lin_idx = indices.first()?;
        let x = lin_idx.get_idx();

        // Check if it's already a 2D index (vec2)
        if x.dtype().vcount() == 2 {
            return None; // Already converted
        }

        // Only LOAD is supported for unfoldable images
        if !is_load {
            tracing::warn!("image_fixup: STORE with unfoldable image not supported");
            return None;
        }

        let valid = lin_idx.get_valid();

        // Get image width
        let width = shape.get(1).copied().unwrap_or(1) as i64;

        // Create 2D index
        let four = UOp::index_const(4);
        let width_const = UOp::index_const(width);
        let stride = UOp::index_const(4 * width);

        let x_coord = x.idiv(&four).mod_(&width_const);
        let y_coord = x.idiv(&stride);

        let oidx = UOp::vectorize(smallvec::smallvec![x_coord, y_coord]);

        let new_idx_expr = if matches!(valid.op(), Op::Const(cv) if cv.0 == ConstValue::Bool(true)) {
            oidx
        } else {
            oidx.valid(valid)
        };

        // Use ptr(true) when index has Ptr dtype, otherwise preserve element dtype
        let new_idx = if matches!(index.dtype(), DType::Ptr { .. }) {
            UOp::index()
                .buffer(img_buf.clone())
                .indices(vec![new_idx_expr])
                .maybe_gate(gate.clone())
                .ptr(true)
                .call()
                .ok()?
        } else {
            UOp::index()
                .buffer(img_buf.clone())
                .indices(vec![new_idx_expr])
                .maybe_gate(gate.clone())
                .dtype(index.dtype())
                .call()
                .ok()?
        };

        // For unfoldable images: load vec4, then select correct element
        // result = reduce(lambda ret, i: (x % 4).ne(i).where(ret, vec_load.gep(i)), range(4), nan)
        let vec4_dtype = ls.dtype().vec(4);
        let vec_load = UOp::load().buffer(ls.load_buffer()?).index(new_idx).dtype(vec4_dtype).call();

        // Build: WHERE(x%4 != 0, WHERE(x%4 != 1, WHERE(x%4 != 2, WHERE(x%4 != 3, nan, gep3), gep2), gep1), gep0)
        let x_mod_4 = x.mod_(&four);
        let nan = ls.const_like(ConstValue::Float(f64::NAN));

        let result = (0..4).rev().fold(nan, |ret, i| {
            let i_const = UOp::index_const(i);
            let not_eq = x_mod_4.ne(&i_const);
            let gep_i = vec_load.gep(vec![i as usize]);
            UOp::try_where(not_eq, ret, gep_i).expect("WHERE")
        });

        return Some(result);
    }

    None
}

// ============================================================================
// pm_reduce: Convert REDUCE to explicit accumulator pattern (Tinygrad devectorizer.py:310-316)
// ============================================================================

use crate::symbolic::dce::reduce_identity;

/// Convert REDUCE to explicit DEFINE_REG + LOAD/STORE accumulation pattern.
///
/// Transforms:
/// ```text
/// REDUCE(src, ranges, Add) with dtype Float32
/// ```
///
/// To:
/// ```text
/// acc = DEFINE_REG_TYPED(1, Float32)
/// idx = INDEX(acc, [0])
/// store_init = STORE(acc, idx, identity)  // Initialize with 0 for Add
/// // Loop body (ranges provide iteration):
/// acc_after = AFTER(acc, [store_init, ranges...])
/// idx_loop = INDEX(acc_after, [0])
/// val = LOAD(acc, idx_loop)
/// new_val = val + src
/// store_loop = STORE(acc, idx_loop, new_val)
/// // After loop:
/// end = END(store_loop, ranges)
/// acc_final = AFTER(acc, [end])
/// idx_final = INDEX(acc_final, [0])
/// result = LOAD(acc, idx_final)
/// ```
///
/// This runs EARLY (before pm_add_loads, before main devectorize) to eliminate
/// REDUCE before other patterns see it. Matches Tinygrad's pm_reduce.
pub fn pm_reduce() -> TypedPatternMatcher<ReduceContext> {
    crate::patterns! {
        @context ReduceContext;

        // Match ALL REDUCEs - empty ranges handled by returning reduced value directly
        red @ Reduce(_, ..) => {
            reduce_to_acc(red, ctx)
        },

        // Merge END nodes sharing the same reduce ranges (Tinygrad merge_reduce_ends)
        Sink { sources: _sources } => {
            ctx.merge_reduce_ends(_sources)
        },
    }
}

/// Horizontal reduce for accumulator pattern (devectorizer.py:283-289).
fn horizontal_reduce(inp: &Arc<UOp>, out_dtype: &DType) -> Vec<Arc<UOp>> {
    if inp.dtype() == *out_dtype {
        return vec![inp.clone()];
    }
    let inp_vcount = inp.dtype().vcount();
    let out_vcount = out_dtype.vcount();
    assert!(
        inp_vcount.is_multiple_of(out_vcount),
        "horizontal mismatch: inp.dtype={:?} (vcount={}), out_dtype={:?} (vcount={})",
        inp.dtype(),
        inp_vcount,
        out_dtype,
        out_vcount
    );
    let horizontal_amount = inp_vcount / out_vcount;
    (0..horizontal_amount).map(|i| inp.gep((i..inp_vcount).step_by(horizontal_amount).collect())).collect()
}

/// Convert REDUCE to explicit accumulator pattern (devectorizer.py:291-308).
fn reduce_to_acc(red: &Arc<UOp>, ctx: &mut ReduceContext) -> Option<Arc<UOp>> {
    let Op::Reduce { src: inp, ranges: reduce_range, reduce_op } = red.op() else { return None };

    let lst = horizontal_reduce(inp, &red.dtype());
    debug_assert!(lst.iter().all(|x| x.dtype() == red.dtype()), "horizontal reduction mismatch");

    // No ranges → just horizontal reduction
    if reduce_range.is_empty() {
        return lst.into_iter().reduce(|a, b| apply_reduce_binary(*reduce_op, a, b, &red.dtype()));
    }

    // Find input_ranges: ranges in topo that are not reduce_range and not ended
    let topo = inp.toposort();
    let ended: HashSet<u64> = topo
        .iter()
        .filter_map(|n| if let Op::End { ranges, .. } = n.op() { Some(ranges.iter().map(|r| r.id)) } else { None })
        .flatten()
        .collect();
    let reduce_ids: HashSet<u64> = reduce_range.iter().map(|r| r.id).collect();
    let input_ranges: SmallVec<[Arc<UOp>; 4]> = topo
        .iter()
        .filter(|n| matches!(n.op(), Op::Range { .. }) && !reduce_ids.contains(&n.id) && !ended.contains(&n.id))
        .cloned()
        .collect();

    // Set up accumulator
    let identity = reduce_identity(*reduce_op, red.dtype());
    let acc = UOp::define_reg_typed(1, red.dtype());
    let zero = UOp::index_const(0);
    let make_idx = |buf: Arc<UOp>| UOp::index().buffer(buf).indices(vec![zero.clone()]).call().expect("index");

    // acc_init = acc.after(*input_ranges).index(0).store(identity)
    let acc_init = make_idx(acc.after(input_ranges)).store_value(identity);

    // lst = [acc.after(acc_init, *reduce_range).index(0)] + lst
    let mut loop_deps: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![acc_init];
    loop_deps.extend(reduce_range.iter().cloned());
    let acc_loop = make_idx(acc.after(loop_deps));
    let lst_with_acc = std::iter::once(acc_loop).chain(lst);

    // ret = functools.reduce(lambda x,y: x.alu(red.arg, y), lst)
    let ret = lst_with_acc.reduce(|a, b| apply_reduce_binary(*reduce_op, a, b, &red.dtype()))?;

    // return acc.after(acc.index(0).store(ret).end(*reduce_range)).index(0)
    let store_end = make_idx(acc.clone()).store_value(ret).end(reduce_range.clone());
    ctx.register_end(&store_end);
    Some(make_idx(acc.after(smallvec::smallvec![store_end])))
}

/// Apply binary reduce operation between two values.
fn apply_reduce_binary(reduce_op: ReduceOp, a: Arc<UOp>, b: Arc<UOp>, dtype: &DType) -> Arc<UOp> {
    debug_assert!(a.dtype() == b.dtype(), "reduce operand dtype mismatch");
    match reduce_op {
        ReduceOp::Add => UOp::new(Op::Binary(BinaryOp::Add, a, b), dtype.clone()),
        ReduceOp::Mul => UOp::new(Op::Binary(BinaryOp::Mul, a, b), dtype.clone()),
        ReduceOp::Max => UOp::new(Op::Binary(BinaryOp::Max, a, b), dtype.clone()),
        ReduceOp::Min => {
            let cond = UOp::new(Op::Binary(BinaryOp::Lt, a.clone(), b.clone()), DType::Bool.vec(dtype.vcount()));
            UOp::try_where(cond, a, b).expect("WHERE")
        }
    }
}
