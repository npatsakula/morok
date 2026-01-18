//! Devectorize pass for contiguous memory access optimization.
//!
//! Transforms vectorized memory operations AFTER expansion to enable
//! contiguous vector loads/stores instead of scalar gather/scatter.
//!
//! # Problem
//!
//! After K-vec + output upcasting with `do_expand`, LOAD operations have
//! vector indices like `LOAD(buffer, [i*4+0, i*4+1, i*4+2, i*4+3])`.
//! Without this pass, codegen emits scalar gather (4x load + insertelement).
//!
//! # Solution (Tinygrad-aligned)
//!
//! Run AFTER `pre_expand` to detect and group consecutive indices:
//!
//! 1. **expand_index**: Explode vector index into scalar GEPs, simplify each,
//!    extract (root, offset) pairs, group consecutive offsets, create PTRCAT.
//!    Pattern: INDEX(buf, vec_idx) → PTRCAT of grouped CAST(INDEX) pointers
//!
//! 2. **load_store_folding**: Distribute PTRCAT through LOAD/STORE.
//!    LOAD(PTRCAT(a,b,c)) → CAT(LOAD(a), LOAD(b), LOAD(c))
//!
//! 3. **split_load_store**: For each LOAD(CAST(INDEX)), emit contiguous
//!    vector load based on fold length divisibility.
//!    The CAST signals vector pointer type → codegen emits `load <N x T>`.
//!
//! # References
//!
//! Based on Tinygrad's `codegen/late/devectorizer.py`:
//! - `expand_index` (lines 59-95)
//! - `split_load_store` (lines 130-174)
//! - Pattern (line 200): `LOAD/STORE(CAST(INDEX))` triggers contiguous access

use std::collections::HashMap;
use std::sync::Arc;

use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp};

use crate::TypedPatternMatcher;
use smallvec::SmallVec;

use crate::rewrite::graph_rewrite_bottom_up;
use crate::symbolic::patterns::{gep_pushing_patterns, symbolic};

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run devectorize pass on kernel AST.
///
/// Call this AFTER `pre_expand` but BEFORE codegen.
/// Transforms vector indices into grouped contiguous accesses.
///
/// Matches Tinygrad's devectorizer pipeline (devectorizer.py):
/// - Phase 1: expand_index → PTRCAT grouping
/// - Phase 2: load_store_folding (GEP movement + PTRCAT distribution + split)
///
/// Note: Bool storage conversion (`bool_storage_patterns()`) is called separately
/// from `optimizer/mod.rs` as it's backend-specific (LLVM/PTX).
pub fn devectorize(ast: &Arc<UOp>) -> Arc<UOp> {
    // Phase 1: Expand vector indices into grouped PTRCAT
    let phase1 = expand_index_patterns();
    let ast = graph_rewrite_bottom_up(&phase1, ast.clone(), &mut ());

    // Phase 2: GEP movement + PTRCAT distribution + LOAD/STORE splitting + CAT→VECTORIZE
    // All patterns run together so:
    // - LOAD(GEP(PTRCAT)) → GEP(LOAD(PTRCAT)) → GEP(CAT(LOADs))
    // - split_load creates CAT([LOAD<4>×N]), CAT→VECTORIZE converts it
    // This matches Tinygrad where gep_pushing (with CAT→VECTORIZE) is part of symbolic.
    let phase2 = gep_ptrcat_patterns() + load_store_patterns();
    graph_rewrite_bottom_up(&phase2, ast, &mut ())
}
/// Phase 3 patterns: Convert bool LOAD/STORE to use uint8 storage.
///
/// LLVM's i1 type when stored to memory can have garbage in upper 7 bits.
/// We cast bool→uint8 before storing and uint8→bool after loading.
/// This matches Tinygrad's approach in PTX and NIR renderers.
pub fn bool_storage_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // STORE bool value: cast to uint8 before storing
        store if is_bool_store(store) => |store| rewrite_bool_store(store),

        // LOAD bool value: load as uint8, then cast to bool
        load if is_bool_load(load) => |load| rewrite_bool_load(load),
    }
}

/// Check if STORE has a bool-typed value.
fn is_bool_store(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Store { value, .. } => value.dtype().base().is_bool(),
        Op::StoreGated { value, .. } => value.dtype().base().is_bool(),
        _ => false,
    }
}

/// Check if LOAD has bool result dtype.
fn is_bool_load(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::Load { .. } | Op::LoadGated { .. } => uop.dtype().base().is_bool(),
        _ => false,
    }
}

/// Rewrite STORE of bool value to cast to uint8 first.
/// STORE(buf, idx, bool_val) → STORE(buf, idx, cast(bool_val, uint8))
fn rewrite_bool_store(store: &Arc<UOp>) -> Option<Arc<UOp>> {
    match store.op() {
        Op::Store { buffer, index, value, ranges } => {
            tracing::debug!(value_dtype = ?value.dtype(), "rewrite_bool_store: casting bool to uint8");
            // Cast bool value to uint8
            let uint8_dtype = value.dtype().with_base(ScalarDType::UInt8);
            let value_as_uint8 = UOp::cast(value.clone(), uint8_dtype);
            Some(UOp::store_with_ranges(buffer.clone(), index.clone(), value_as_uint8, ranges.clone()))
        }
        Op::StoreGated { buffer, index, value, gate, ranges } => {
            // Cast bool value to uint8
            let uint8_dtype = value.dtype().with_base(ScalarDType::UInt8);
            let value_as_uint8 = UOp::cast(value.clone(), uint8_dtype);
            Some(UOp::store_gated_with_ranges(
                buffer.clone(),
                index.clone(),
                value_as_uint8,
                gate.clone(),
                ranges.clone(),
            ))
        }
        _ => None,
    }
}

/// Rewrite LOAD of bool to load as uint8 and cast back to bool.
/// LOAD(buf, idx) : bool → cast(LOAD(buf, idx) : uint8, bool)
fn rewrite_bool_load(load: &Arc<UOp>) -> Option<Arc<UOp>> {
    match load.op() {
        Op::Load { buffer, index } => {
            // Create a modified buffer with uint8 element type for loading
            let bool_dtype = load.dtype();
            let uint8_dtype = bool_dtype.with_base(ScalarDType::UInt8);

            // Create load with uint8 dtype
            let uint8_load = UOp::new(Op::Load { buffer: buffer.clone(), index: index.clone() }, uint8_dtype);

            // Cast back to bool
            Some(UOp::cast(uint8_load, bool_dtype))
        }
        Op::LoadGated { buffer, index, gate } => {
            let bool_dtype = load.dtype();
            let uint8_dtype = bool_dtype.with_base(ScalarDType::UInt8);

            let uint8_load = UOp::new(
                Op::LoadGated { buffer: buffer.clone(), index: index.clone(), gate: gate.clone() },
                uint8_dtype,
            );

            Some(UOp::cast(uint8_load, bool_dtype))
        }
        _ => None,
    }
}

/// GEP/CAT/VECTORIZE patterns for memory access devectorization.
///
/// Matches Tinygrad's approach: CAT→VECTORIZE runs in the same PatternMatcher
/// as load_store_patterns, so when split_load creates CAT, it's immediately
/// converted to VECTORIZE. This ensures GEP(CAT) only sees scalar sources.
///
/// Pattern order matters:
/// 1. CAT → VECTORIZE (converts multi-element CAT sources to scalar GEPs)
/// 2. GEP(VECTORIZE) → element extraction (simplifies GEPs on VECTORIZE)
/// 3. GEP(CAT) → reorder (only scalar sources after step 1)
/// 4. GEP(PTRCAT) → reorder pointers
pub(crate) fn gep_ptrcat_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // === CAT → VECTORIZE (Tinygrad symbolic.py:169-171) ===
        // CAT can't be rendered. Expand to VECTORIZE with element-wise GEPs.
        // CAT([a<4>, b<4>]) → VECTORIZE(a.gep(0), ..., a.gep(3), b.gep(0), ..., b.gep(3))
        // Must run BEFORE GEP(CAT) so multi-element sources are eliminated first.
        cat if matches!(cat.op(), Op::Cat { .. }) => |cat| {
            let Op::Cat { sources } = cat.op() else { return None };
            // Skip single-source CAT (handled by identity pattern below)
            if sources.len() <= 1 { return None; }
            // Skip if all sources are scalar (let simple GEP(CAT) handle it)
            if sources.iter().all(|s| s.dtype().vcount() == 1) { return None; }
            // Flatten: each source contributes vcount elements via GEP
            let elements: SmallVec<[Arc<UOp>; 4]> = sources.iter()
                .flat_map(|src| {
                    let n = src.dtype().vcount();
                    (0..n).map(move |i| if n == 1 { src.clone() } else { UOp::gep(src.clone(), vec![i]) })
                })
                .collect();
            Some(UOp::vectorize(elements))
        },

        // === GEP(VECTORIZE) → element extraction ===
        // GEP(VECTORIZE([e0, ..., en-1]), [i]) → ei
        // GEP(VECTORIZE([e0, ..., en-1]), [i,j,k]) → VECTORIZE([ei, ej, ek])
        Gep { vector, indices } if matches!(vector.op(), Op::Vectorize { .. }) => |vector, indices| {
            let Op::Vectorize { elements } = vector.op() else { return None };
            let extracted: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .filter_map(|&i| elements.get(i).cloned())
                .collect();
            if extracted.len() != indices.len() { return None; }
            Some(if extracted.len() == 1 { extracted.into_iter().next().unwrap() }
                else { UOp::vectorize(extracted) })
        },

        // === GEP(CAT) → reorder (scalar sources only after CAT→VECTORIZE) ===
        // GEP(CAT([a, b, c]), [1, 2]) → CAT([b, c])
        Gep { vector, indices } if matches!(vector.op(), Op::Cat { .. }) => |vector, indices| {
            let Op::Cat { sources } = vector.op() else { return None };
            let reordered: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .filter_map(|&i| sources.get(i).cloned())
                .collect();
            if reordered.len() != indices.len() { return None; }
            Some(if reordered.len() == 1 { reordered.into_iter().next().unwrap() }
                else { UOp::cat(reordered.to_vec()) })
        },

        // === GEP(PTRCAT) → reorder pointers ===
        Gep { vector, indices } if matches!(vector.op(), Op::PtrCat { .. }) => |vector, indices| {
            let Op::PtrCat { sources } = vector.op() else { return None };
            let reordered: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .filter_map(|&i| sources.get(i).cloned())
                .collect();
            (reordered.len() == indices.len()).then(|| UOp::ptrcat(reordered.to_vec()))
        },

        // === Identity patterns ===
        // Single-source CAT/PTRCAT → unwrap
        cat if matches!(cat.op(), Op::Cat { sources } if sources.len() == 1) => |cat| {
            let Op::Cat { sources } = cat.op() else { return None };
            Some(Arc::clone(&sources[0]))
        },
        ptrcat if matches!(ptrcat.op(), Op::PtrCat { sources } if sources.len() == 1) => |ptrcat| {
            let Op::PtrCat { sources } = ptrcat.op() else { return None };
            Some(Arc::clone(&sources[0]))
        },

        // CAT(GEP(x,[0]), GEP(x,[1]), ..., GEP(x,[n-1])) → x (identity reconstruction)
        cat if matches!(cat.op(), Op::Cat { .. }) => |cat| {
            let Op::Cat { sources } = cat.op() else { return None };
            if sources.is_empty() { return None; }
            let Op::Gep { vector: first, indices: idx0 } = sources[0].op() else { return None };
            if idx0.len() != 1 || idx0[0] != 0 { return None; }
            for (i, src) in sources.iter().enumerate() {
                let Op::Gep { vector, indices } = src.op() else { return None };
                if !Arc::ptr_eq(vector, first) || indices != &[i] { return None; }
            }
            (sources.len() == first.dtype().vcount()).then(|| Arc::clone(first))
        },

        // === Devectorize WHERE ===
        // WHERE(<N x i1>, ...) → VECTORIZE(WHERE(i1, ...), ...)
        ternary if is_vectorized_where(ternary) => |ternary| devectorize_where(ternary),
    }
}

/// Check if UOp is a WHERE with vector condition (vcount > 1).
fn is_vectorized_where(uop: &Arc<UOp>) -> bool {
    if let Op::Ternary(TernaryOp::Where, cond, _, _) = uop.op() { cond.dtype().vcount() > 1 } else { false }
}

/// Devectorize WHERE operation by extracting elements with GEP and rebuilding with VECTORIZE.
///
/// Based on Tinygrad's no_vectorized_alu (devectorizer.py:219-223):
/// ```python
/// def no_vectorized_alu(alu:UOp):
///   if alu.dtype.vcount == 1: return None
///   alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg)
///                for i in range(alu.dtype.vcount))
///   return UOp(Ops.VECTORIZE, alu.dtype, alus)
/// ```
///
/// Transforms: WHERE(<N x i1>, <N x T>, <N x T>) → VECTORIZE(WHERE(i1, T, T), ...)
fn devectorize_where(ternary: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Ternary(TernaryOp::Where, cond, t, f) = ternary.op() else {
        return None;
    };

    let vcount = cond.dtype().vcount();
    if vcount <= 1 {
        return None;
    }

    // Create scalar WHERE for each vector element
    let scalar_wheres: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
        .map(|i| {
            let cond_elem = UOp::gep(cond.clone(), vec![i]);
            let t_elem = UOp::gep(t.clone(), vec![i]);
            let f_elem = UOp::gep(f.clone(), vec![i]);
            UOp::try_where(cond_elem, t_elem, f_elem).expect("WHERE construction should succeed")
        })
        .collect();

    Some(UOp::vectorize(scalar_wheres))
}

// ============================================================================
// ALU Devectorization
// ============================================================================

/// Convert vectorized ALU ops to VECTORIZE of scalar ALU ops.
///
/// Based on Tinygrad's no_vectorized_alu (devectorizer.py:219-223):
/// ```python
/// def no_vectorized_alu(alu:UOp):
///   if alu.dtype.vcount == 1: return None  # skip scalars
///   alus = tuple(UOp(alu.op, alu.dtype.scalar(), tuple(s.gep(i) for s in alu.src), alu.arg)
///                for i in range(alu.dtype.vcount))
///   return UOp(Ops.VECTORIZE, alu.dtype, alus)
/// ```
///
/// Transforms:
/// - Binary<vec32>(a, b) → VECTORIZE(Binary(GEP(a,0), GEP(b,0)), ..., Binary(GEP(a,31), GEP(b,31)))
/// - Unary<vec32>(a) → VECTORIZE(Unary(GEP(a,0)), ..., Unary(GEP(a,31)))
/// - Cast<vec32>(a) → VECTORIZE(Cast(GEP(a,0)), ..., Cast(GEP(a,31)))
///
/// This runs BEFORE pm_vectorize_normalize so that:
/// - Parent VECTORIZE sees VECTORIZE children, not Binary/Unary children
/// - No need to flatten Binary/Unary elements (already scalar)
/// - Prevents exponential growth when gep_pushing interacts with flatten patterns
///
/// # Performance Note
///
/// This follows Tinygrad's default (DEVECTORIZE=1). LLVM's SLP vectorizer can
/// re-vectorize the scalar ops when beneficial. Future work could add a flag
/// to preserve vectors for performance-critical paths.
pub fn no_vectorized_alu() -> TypedPatternMatcher {
    crate::patterns! {
        // Binary ops with vector dtype → VECTORIZE of scalar binaries
        // Covers: Add, Mul, Sub, etc. on vector types
        for op in binary [*] {
            result @ op(lhs, rhs) if result.dtype().vcount() > 1 => |result, lhs, rhs| {
                let vcount = result.dtype().vcount();
                let scalar_dtype = result.dtype().scalar().map(DType::Scalar)?;
                let alus: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|i| {
                    let gep_lhs = UOp::gep(lhs.clone(), vec![i]);
                    let gep_rhs = UOp::gep(rhs.clone(), vec![i]);
                    UOp::new(Op::Binary(op, gep_lhs, gep_rhs), scalar_dtype.clone())
                }).collect();
                Some(UOp::vectorize(alus))
            },
        },

        // Unary ops with vector dtype → VECTORIZE of scalar unaries
        // Covers: Neg, Sqrt, Exp, etc. on vector types
        for op in unary [*] {
            result @ op(src) if result.dtype().vcount() > 1 => |result, src| {
                let vcount = result.dtype().vcount();
                let scalar_dtype = result.dtype().scalar().map(DType::Scalar)?;
                let alus: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|i| {
                    let gep_src = UOp::gep(src.clone(), vec![i]);
                    UOp::new(Op::Unary(op, gep_src), scalar_dtype.clone())
                }).collect();
                Some(UOp::vectorize(alus))
            },
        },

        // Cast with vector dtype → VECTORIZE of scalar casts
        Cast { src, .. } if src.dtype().vcount() > 1 => |src| {
            let vcount = src.dtype().vcount();
            let src_scalar_dtype = src.dtype().scalar().map(DType::Scalar)?;
            // Cast output dtype scalar version - infer from cast target
            // NOTE: The cast target dtype comes from the Cast op's dtype field
            let casts: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|i| {
                let gep_src = UOp::gep(src.clone(), vec![i]);
                // Cast to scalar version of the destination dtype
                UOp::cast(gep_src, src_scalar_dtype.clone())
            }).collect();
            Some(UOp::vectorize(casts))
        },

        // WHERE with vector condition → VECTORIZE of scalar WHEREs
        // (Already handled in pm_local_optimizations, but included for completeness)
        Where(cond, t, f) if cond.dtype().vcount() > 1 => |cond, t, f| {
            let vcount = cond.dtype().vcount();
            let t_vcount = t.dtype().vcount();
            let f_vcount = f.dtype().vcount();

            let scalar_wheres: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
                .map(|i| {
                    let cond_elem = UOp::gep(cond.clone(), vec![i]);
                    let t_elem = if t_vcount > 1 { UOp::gep(t.clone(), vec![i]) } else { t.clone() };
                    let f_elem = if f_vcount > 1 { UOp::gep(f.clone(), vec![i]) } else { f.clone() };
                    UOp::try_where(cond_elem, t_elem, f_elem).expect("WHERE construction should succeed")
                })
                .collect();
            Some(UOp::vectorize(scalar_wheres))
        },

        // MulAcc (FMA) with vector dtype → VECTORIZE of scalar MulAccs
        // Required for 8×8 matmul tiling to work with split_load (which creates 4-element chunks)
        MulAcc(a, b, c) if a.dtype().vcount() > 1 => |a, b, c| {
            let vcount = a.dtype().vcount();
            let a_vcount = a.dtype().vcount();
            let b_vcount = b.dtype().vcount();
            let c_vcount = c.dtype().vcount();

            let scalar_mulaccs: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
                .map(|i| {
                    let a_elem = if a_vcount > 1 { UOp::gep(a.clone(), vec![i]) } else { a.clone() };
                    let b_elem = if b_vcount > 1 { UOp::gep(b.clone(), vec![i]) } else { b.clone() };
                    let c_elem = if c_vcount > 1 { UOp::gep(c.clone(), vec![i]) } else { c.clone() };
                    UOp::try_mulacc(a_elem, b_elem, c_elem).expect("MulAcc construction should succeed")
                })
                .collect();
            Some(UOp::vectorize(scalar_mulaccs))
        },
    }
}

// ============================================================================
// VECTORIZE Normalization
// ============================================================================

/// Normalize VECTORIZE and GEP for rendering.
///
/// Based on Tinygrad's pm_render (devectorizer.py:258-275):
/// ```python
/// pm_render = PatternMatcher([
///   (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.VECTORIZE, gep.dtype,
///        tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
///   (UPat(Ops.GEP, name='gep'), lambda gep: gep.src[0] if gep.src[0].dtype.vcount == 1 and gep.arg == (0,) else None),
///   (UPat(Ops.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
/// ])
/// ```
///
/// NOTE: With `no_vectorized_alu()` running first, all vector ALU ops are already
/// converted to VECTORIZE(scalar_alu, ...). This pattern only needs to handle:
/// - Multi-index GEP → VECTORIZE with single-index GEPs
/// - GEP on scalar → identity
/// - Single-source VECTORIZE → unwrap
pub fn pm_vectorize_normalize() -> TypedPatternMatcher {
    crate::patterns! {
        // Multi-index GEP → VECTORIZE with single-index GEPs
        // GEP(x, [0, 1, 2, 3]) → VECTORIZE(GEP(x, [0]), GEP(x, [1]), GEP(x, [2]), GEP(x, [3]))
        Gep { vector, indices } if indices.len() > 1 => |vector, indices| {
            let geps: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .map(|&i| UOp::gep(vector.clone(), vec![i]))
                .collect();
            Some(UOp::vectorize(geps))
        },

        // GEP on scalar → identity (no elements to extract)
        // GEP(scalar, [0]) → scalar
        Gep { vector, indices } if vector.dtype().vcount() == 1 && indices.len() == 1 && indices[0] == 0
            ~> |vector| Arc::clone(vector),

        // Single-source VECTORIZE → unwrap
        // VECTORIZE([x]) → x
        Vectorize { elements } if elements.len() == 1 => |elements| Some(elements[0].clone()),
    }
}

/// Phase 1 patterns: expand vector INDEX into grouped PTRCAT.
pub(crate) fn expand_index_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // INDEX(buffer, vector_index) with vector index → expand and group
        index if is_vector_index(index) => |index| expand_vector_index(index),
    }
}

/// Phase 2 patterns: GEP movement, PTRCAT distribution, and LOAD/STORE splitting.
///
/// Based on Tinygrad's load_store_folding (devectorizer.py:114-126).
/// Pattern order matters:
/// 1. Move GEP after LOAD/STORE (enables PTRCAT distribution)
/// 2. Distribute PTRCAT through LOAD/STORE
/// 3. Split LOAD/STORE by fold length
pub(crate) fn load_store_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // === GEP Movement (Tinygrad devectorizer.py:117-120) ===
        // These MUST come first to transform LOAD(GEP(PTRCAT)) → GEP(LOAD(PTRCAT))
        // so that PTRCAT distribution can fire.

        // LOAD(GEP(x)) → GEP(LOAD(x))
        // Tinygrad: ld.replace(dtype=ld.dtype.scalar().vec(gep.dtype.count), src=(gep.src[0],)+ld.src[1:]).gep(gep.arg)
        load @ Load { buffer, index: Gep { vector, indices } }
            => |load, buffer, vector, indices| move_gep_after_load(load, buffer, vector, indices),

        // STORE(GEP(x), data) → STORE(x, GEP⁻¹(data))
        // Tinygrad: gep_on_store (devectorizer.py:106-113) - preserves sto.src[2:]
        Store { buffer, index: Gep { vector, indices }, value, ranges }
            => |buffer, vector, indices, value, ranges| move_gep_on_store(buffer, vector, indices, value, ranges),

        // === PTRCAT Distribution (Tinygrad devectorizer.py:122-125) ===

        // LOAD(PTRCAT(a,b)) → CAT(LOAD(a), LOAD(b))
        Load { buffer, index: PtrCat { sources } }
            => |buffer, sources| distribute_ptrcat_load(buffer, sources),

        // STORE(PTRCAT(a,b), data) → GROUP(STORE(a, gep(data,0..n)), ...)
        // Tinygrad: cat_after_store preserves sto.src[2:] (ranges)
        Store { buffer, index: PtrCat { sources }, value, ranges }
            => |buffer, sources, value, ranges| distribute_ptrcat_store(buffer, sources, value, ranges),

        // === Split by Fold Length (Tinygrad correct_load_store) ===

        // LOAD(CAST(INDEX)) → split by fold length
        Load { buffer, index: Cast { src: idx @ Index { buffer: _b, indices: _i, gate: _g }, dtype: cast_dtype } }
            => |buffer, idx, cast_dtype| split_load(buffer, idx, cast_dtype),

        // STORE(CAST(INDEX), data) → split by fold length
        Store { buffer, index: Cast { src: idx @ Index { buffer: _b, indices: _i, gate: _g }, dtype: cast_dtype }, value, ranges }
            => |buffer, idx, cast_dtype, value, ranges| split_store(buffer, idx, cast_dtype, value, ranges),
    }
}

// ============================================================================
// Pattern Predicates
// ============================================================================

/// Check if INDEX has a vector index (dtype.vcount() > 1).
fn is_vector_index(uop: &Arc<UOp>) -> bool {
    if let Op::Index { indices, .. } = uop.op()
        && let Some(idx) = indices.first()
    {
        return idx.dtype().vcount() > 1;
    }
    false
}

// ============================================================================
// GEP Movement Patterns (Tinygrad devectorizer.py:117-120, 106-113)
// ============================================================================

/// Move GEP after LOAD: LOAD(GEP(ptr, indices)) → GEP(LOAD(ptr), indices)
///
/// Tinygrad (devectorizer.py:117-118):
/// ```python
/// lambda gep, ld: ld.replace(dtype=ld.dtype.scalar().vec(gep.dtype.count),
///                            src=(gep.src[0],)+ld.src[1:]).gep(gep.arg)
/// ```
///
/// Key insight: new LOAD dtype = `ld.dtype.scalar().vec(gep.dtype.count)`
/// - Uses the LOAD's dtype scalar base, NOT the buffer's dtype
/// - Vector count = number of elements GEP selects (gep_indices.len())
///
/// This enables PTRCAT distribution: LOAD(GEP(PTRCAT)) → GEP(LOAD(PTRCAT)) → GEP(CAT(LOADs))
fn move_gep_after_load(
    load: &Arc<UOp>,
    buffer: &Arc<UOp>,
    gep_inner: &Arc<UOp>,
    gep_indices: &[usize],
) -> Option<Arc<UOp>> {
    // Tinygrad: ld.dtype.scalar().vec(gep.dtype.count)
    // Use the LOAD's dtype to get the scalar base, not the buffer's dtype
    let gep_count = gep_indices.len();
    let scalar_base = load.dtype().scalar()?;
    let inner_load_dtype = if gep_count > 1 {
        DType::Vector { scalar: scalar_base, count: gep_count }
    } else {
        DType::Scalar(scalar_base)
    };

    // Create the inner LOAD with dtype matching GEP's output size
    // Tinygrad: src=(gep.src[0],)+ld.src[1:] - we use gep_inner as the new index
    let inner_load = UOp::new(Op::Load { buffer: buffer.clone(), index: gep_inner.clone() }, inner_load_dtype);

    // Apply GEP to the loaded result (this may become identity if indices are [0,1,2,...])
    Some(UOp::gep(inner_load, gep_indices.to_vec()))
}

/// Move GEP on STORE: STORE(GEP(ptr, indices), data) → STORE(ptr, GEP⁻¹(data))
///
/// Tinygrad (devectorizer.py:106-113):
/// ```python
/// def gep_on_store(gep:UOp, st:UOp, sto:UOp):
///   # NOTE: we need to invert the gep here, but it may be an expanding gep
///   # fake argsort. TODO: handle duplicates
///   a = {}
///   for i,x in enumerate(gep.arg): a[x] = i
///   new_arg = tuple(x[1] for x in sorted(a.items()))
///   return gep.src[0].store(st.gep(new_arg), *sto.src[2:])  # preserves ranges
/// ```
fn move_gep_on_store(
    buffer: &Arc<UOp>,
    gep_inner: &Arc<UOp>,
    gep_indices: &[usize],
    value: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
) -> Option<Arc<UOp>> {
    // Invert the GEP indices: build a map from output position to input position
    // GEP([2,0,1]) means: result[0]=src[2], result[1]=src[0], result[2]=src[1]
    // Inverse: to write result, we need src[gep_idx[i]] = input[i]
    // So inverse_gep[gep_idx[i]] = i, i.e., inverse[2]=0, inverse[0]=1, inverse[1]=2
    // Sorted by key: [(0,1), (1,2), (2,0)] → [1, 2, 0]

    let mut inverse_map: Vec<(usize, usize)> = gep_indices.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    inverse_map.sort_by_key(|&(x, _)| x);
    let inverse_indices: Vec<usize> = inverse_map.iter().map(|&(_, i)| i).collect();

    // Apply inverse GEP to the value
    let reordered_value = UOp::gep(value.clone(), inverse_indices);

    // Create STORE to the inner pointer with reordered value, preserving ranges
    // Tinygrad: *sto.src[2:] preserves additional store arguments
    Some(UOp::store_with_ranges(buffer.clone(), gep_inner.clone(), reordered_value, ranges.clone()))
}

// ============================================================================
// Root Key for Grouping (Tinygrad: offsets_rootsrc)
// ============================================================================

/// Key for grouping indices by root expression and validity.
///
/// Tinygrad groups by (validity, root) tuple. We use content_hash() for structural
/// equality - UOp IDs don't work because extract_root_and_offset rebuilds expressions
/// and identical structures can have different IDs if they came from different code paths.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
enum RootKey {
    /// Expression-based root with optional gate (validity) hash
    Expr { valid_id: Option<u64>, root_id: u64 },
    /// Constant index
    Const,
}

impl RootKey {
    fn expr(root: &Arc<UOp>, gate: Option<&Arc<UOp>>) -> Self {
        // Use content_hash() for structural equality, not .id (pointer identity)
        RootKey::Expr { valid_id: gate.map(|g| g.content_hash()), root_id: root.content_hash() }
    }
}

// ============================================================================
// expand_index: Explode-Simplify-Extract-Group
// ============================================================================

/// Expand a vector INDEX into grouped consecutive accesses.
///
/// Based on Tinygrad's expand_index (devectorizer.py:59-95).
///
/// Steps:
/// 1. Explode: Create scalar INDEX for each lane via GEP on vector index
/// 2. Simplify: Apply gep_pushing + symbolic patterns to simplify each scalar
/// 3. Extract: Get (root, offset) from each simplified scalar index
/// 4. Group: Collect consecutive offsets with same root
/// 5. Create: PTRCAT of CAST(INDEX) pointers with GEP reorder
fn expand_vector_index(index: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate } = index.op() else {
        return None;
    };

    // Only handle single-index case (most common)
    if indices.len() != 1 {
        return None;
    }

    let vec_idx = &indices[0];
    let vec_count = vec_idx.dtype().vcount();

    if vec_count <= 1 {
        return None;
    }

    // CRITICAL: Don't vectorize loads for i1/bool types.
    // LLVM's `load <N x i1>` reads N BITS from memory, not N bytes.
    // Since bools are stored as bytes (0x00/0x01), vector loads give wrong results.
    // Example: `load <2 x i1>` from byte[0]=0x01 reads bits 0,1 giving <1,0>
    // but we wanted byte[0]=1, byte[1]=1 giving <1,1>.
    let base_dtype = buffer.dtype().base();
    if matches!(base_dtype, morok_dtype::ScalarDType::Bool) {
        return None;
    }

    // ASSERT: Index must have integer or index dtype
    let is_integer_or_index_dtype = vec_idx.dtype().is_int()
        || vec_idx.dtype() == DType::Index
        || matches!(vec_idx.dtype(), DType::Vector { scalar, .. }
            if scalar.is_signed() || scalar.is_unsigned() || matches!(scalar, morok_dtype::ScalarDType::Index));
    if !is_integer_or_index_dtype {
        return None;
    }

    // Step 1: Generate scalar INDEX ops via GEP for each lane
    // Tinygrad: midx = graph_rewrite(UOp.sink(*[buf.index(vec.gep(i), ptr=True) for i in range(vec.dtype.count)]))
    // Use GEP to extract each lane's index from the vectorized index
    let scalar_idx_dtype = index.dtype().clone();
    let scalar_indices: Vec<Arc<UOp>> = (0..vec_count)
        .map(|i| {
            // Extract lane i's index from vectorized index via GEP
            let scalar_idx = UOp::gep(vec_idx.clone(), vec![i]);
            UOp::new(
                Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![scalar_idx], gate: gate.clone() },
                scalar_idx_dtype.clone(),
            )
        })
        .collect();

    // Step 2: Apply GEP pushing to FIXED POINT before symbolic simplification
    // This is critical: Tinygrad runs gep_pushing until no more changes occur.
    // If we mix with symbolic too early, identity folding (Add(x,0)→x) creates
    // asymmetry between lanes before GEP is fully pushed through.
    let sink = UOp::sink(scalar_indices);
    let gep_patterns = gep_pushing_patterns();

    // Phase 2a: Run GEP pushing to fixpoint
    let mut current = sink;
    let mut iterations = 0;
    loop {
        let next = graph_rewrite_bottom_up(&gep_patterns, current.clone(), &mut ());
        if next.id == current.id {
            break;
        }
        current = next;
        iterations += 1;
        if iterations > 100 {
            tracing::warn!("GEP pushing exceeded 100 iterations");
            break;
        }
    }

    // Phase 2b: Now apply full symbolic simplification
    // At this point, all GEPs should be pushed to leaves, and hash-consing
    // ensures identical subexpressions share the same UOp instance.
    let full_simplifier = gep_pushing_patterns() + symbolic();
    let simplified = graph_rewrite_bottom_up(&full_simplifier, current, &mut ());
    tracing::trace!(simplified = %simplified.tree(), "expand_vector_index: after symbolic simplification");

    // Step 3: Extract (root, offset) from each simplified scalar INDEX
    let Op::Sink { sources } = simplified.op() else {
        return None;
    };

    // offsets_by_root: RootKey -> { offset -> [lane_indices] }
    let mut offsets_by_root: HashMap<RootKey, HashMap<i64, Vec<usize>>> = HashMap::new();

    for (lane, idx_op) in sources.iter().enumerate() {
        let Op::Index { indices: simp_indices, .. } = idx_op.op() else {
            continue;
        };

        if simp_indices.is_empty() {
            continue;
        }

        let (root_key, offset) = extract_root_and_offset(&simp_indices[0], gate.as_ref());
        offsets_by_root.entry(root_key).or_default().entry(offset).or_default().push(lane);
    }

    // Step 4: Group consecutive offsets
    let mut result_ptrs = Vec::new();
    let mut idx_mapping: Vec<Option<usize>> = vec![None; vec_count];
    let mut global_offset = 0usize;

    for (root_key, offsets_map) in &offsets_by_root {
        let groups = group_consecutive_offsets_from_map(offsets_map);

        tracing::debug!(
            root = ?root_key,
            num_groups = groups.len(),
            "expand_vector_index: grouping for root"
        );

        for (_first_offset, lanes) in groups {
            let group_len = lanes.len();
            let first_lane = lanes[0];

            // Get the simplified INDEX for the first lane of this group
            let first_idx = &sources[first_lane];

            // CAST to vector pointer if group > 1
            let ptr = if group_len > 1 {
                let vec_ptr_dtype = make_vec_ptr_dtype(buffer, group_len);
                UOp::cast(first_idx.clone(), vec_ptr_dtype)
            } else {
                first_idx.clone()
            };

            result_ptrs.push(ptr);

            // Track lane → output position mapping
            for (i, &lane) in lanes.iter().enumerate() {
                idx_mapping[lane] = Some(global_offset + i);
            }
            global_offset += group_len;
        }
    }

    // Verify all lanes are mapped
    if idx_mapping.iter().any(|x| x.is_none()) {
        return None;
    }

    // Step 5: Create PTRCAT (always, even for single group per Issue 2)
    let ptrcat = UOp::ptrcat(result_ptrs);

    // Check if GEP reorder is needed
    let gep_indices: Vec<usize> = idx_mapping.iter().map(|x| x.unwrap()).collect();
    let needs_reorder = !gep_indices.iter().enumerate().all(|(i, &idx)| i == idx);

    let result = if needs_reorder { UOp::gep(ptrcat, gep_indices) } else { ptrcat };

    Some(result)
}

/// Extract (root_key, offset) from a simplified scalar index.
///
/// Matches Tinygrad's approach (devectorizer.py:68-72): only strip the OUTERMOST
/// Add with a constant. This is simpler and works correctly with hash consing
/// since we don't rebuild expressions.
///
/// Patterns:
/// - Add(root, CONST(offset)) → (Expr(root), offset)
/// - Add(CONST(offset), root) → (Expr(root), offset)
/// - CONST(offset) → (Const, offset)
/// - other → (Expr(other), 0)
fn extract_root_and_offset(idx: &Arc<UOp>, gate: Option<&Arc<UOp>>) -> (RootKey, i64) {
    match idx.op() {
        // Add(root, CONST(offset)) or Add(CONST(offset), root)
        Op::Binary(BinaryOp::Add, left, right) => {
            // Check if right is a constant offset
            if let Op::Const(cv) = right.op()
                && let ConstValue::Int(offset) = cv.0
            {
                return (RootKey::expr(left, gate), offset);
            }
            // Check if left is a constant offset
            if let Op::Const(cv) = left.op()
                && let ConstValue::Int(offset) = cv.0
            {
                return (RootKey::expr(right, gate), offset);
            }
            // Neither side is a constant - no offset extracted
            (RootKey::expr(idx, gate), 0)
        }
        // Pure CONST(offset)
        Op::Const(cv) => {
            if let ConstValue::Int(offset) = cv.0 {
                (RootKey::Const, offset)
            } else {
                (RootKey::expr(idx, gate), 0)
            }
        }
        // Anything else: no offset extracted
        _ => (RootKey::expr(idx, gate), 0),
    }
}

/// Group consecutive offsets from an offset->lanes map.
///
/// Returns [(first_offset, [lanes]), ...] sorted by first_offset.
fn group_consecutive_offsets_from_map(offsets_map: &HashMap<i64, Vec<usize>>) -> Vec<(i64, Vec<usize>)> {
    if offsets_map.is_empty() {
        return vec![];
    }

    // Collect and sort by offset
    let mut sorted_offsets: Vec<_> = offsets_map.keys().copied().collect();
    sorted_offsets.sort();

    let mut groups: Vec<(i64, Vec<usize>)> = Vec::new();
    let mut current_start = sorted_offsets[0];
    let mut current_lanes: Vec<usize> = offsets_map[&current_start].clone();
    let mut expected_next = current_start + 1;

    for &offset in &sorted_offsets[1..] {
        if offset == expected_next && offsets_map[&offset].len() == 1 {
            // Consecutive and single lane - extend current group
            current_lanes.extend(offsets_map[&offset].iter().copied());
            expected_next = offset + 1;
        } else {
            // Not consecutive or multiple lanes at offset - start new group
            groups.push((current_start, std::mem::take(&mut current_lanes)));
            current_start = offset;
            current_lanes = offsets_map[&offset].clone();
            expected_next = offset + 1;
        }
    }

    // Don't forget the last group
    groups.push((current_start, current_lanes));

    groups
}

/// Create vector pointer dtype for contiguous access.
///
/// Issue 1 fix: size encodes element count (not always 1).
fn make_vec_ptr_dtype(buffer: &Arc<UOp>, vec_len: usize) -> DType {
    let base_dtype = buffer.dtype().base();
    let addrspace = match buffer.dtype() {
        DType::Ptr { addrspace, .. } => addrspace,
        _ => AddrSpace::Global,
    };
    let vec_dtype = DType::Vector { scalar: base_dtype, count: vec_len };
    DType::Ptr { base: Box::new(vec_dtype), addrspace, size: Some(vec_len) }
}

// ============================================================================
// load_store_folding: Distribute PTRCAT through LOAD/STORE
// ============================================================================

/// Distribute PTRCAT through LOAD.
///
/// Based on Tinygrad's load_store_folding pattern (devectorizer.py:122-123).
/// LOAD(PTRCAT(a, b, c)) → CAT(LOAD(a), LOAD(b), LOAD(c))
fn distribute_ptrcat_load(buffer: &Arc<UOp>, sources: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    tracing::debug!(num_sources = sources.len(), "distribute_ptrcat_load: distributing PTRCAT through LOAD");

    // Create individual LOADs for each pointer
    let loads: Vec<Arc<UOp>> = sources
        .iter()
        .map(|ptr| {
            // Determine dtype for this load based on pointer
            let load_dtype = ptr_to_load_dtype(ptr);
            UOp::new(Op::Load { buffer: buffer.clone(), index: ptr.clone() }, load_dtype)
        })
        .collect();

    // CAT them together
    Some(UOp::cat(loads))
}

/// Distribute PTRCAT through STORE.
///
/// Based on Tinygrad's cat_after_store (devectorizer.py:97-104):
/// ```python
/// def cat_after_store(cat:UOp, data:UOp, sto:UOp):
///   offset = 0
///   ret: list[UOp] = []
///   for s in cat.src:
///     ret.append(s.store(data.gep(tuple(range(offset, offset+s.dtype.count))), *sto.src[2:]))
///     offset += s.dtype.count
///   return UOp.group(*ret)
/// ```
///
/// STORE(PTRCAT(a, b), data) → GROUP(STORE(a, gep(data, 0..n)), STORE(b, gep(data, n..)))
fn distribute_ptrcat_store(
    buffer: &Arc<UOp>,
    sources: &[Arc<UOp>],
    value: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
) -> Option<Arc<UOp>> {
    tracing::debug!(num_sources = sources.len(), "distribute_ptrcat_store: distributing PTRCAT through STORE");

    // Create individual STOREs for each pointer
    let mut stores = Vec::new();
    let mut offset = 0usize;

    for ptr in sources.iter() {
        let ptr_count = ptr_element_count(ptr);

        // GEP to extract data elements for this store
        // Tinygrad: data.gep(tuple(range(offset, offset+s.dtype.count)))
        let gep_indices: Vec<usize> = (offset..offset + ptr_count).collect();
        let store_value = UOp::gep(value.clone(), gep_indices);

        // Create STORE with preserved ranges
        // Tinygrad: *sto.src[2:] preserves additional store arguments
        let store_op = UOp::store_with_ranges(buffer.clone(), ptr.clone(), store_value, ranges.clone());
        stores.push(store_op);

        offset += ptr_count;
    }

    // GROUP them together
    Some(UOp::group(stores.into_iter().collect()))
}

/// Get load dtype from pointer.
fn ptr_to_load_dtype(ptr: &Arc<UOp>) -> DType {
    match ptr.dtype() {
        DType::Ptr { base, size, .. } => {
            if let Some(sz) = size
                && sz > 1
            {
                // Pointer has size > 1: load produces vector
                return DType::Vector { scalar: base.base(), count: sz };
            }
            // Size is 1 or None: load produces whatever base is
            base.as_ref().clone()
        }
        _ => ptr.dtype().clone(),
    }
}

/// Get element count from pointer.
fn ptr_element_count(ptr: &Arc<UOp>) -> usize {
    match ptr.dtype() {
        DType::Ptr { size, .. } => size.unwrap_or(1),
        _ => 1,
    }
}

// ============================================================================
// split_load_store: Split by fold length divisibility
// ============================================================================

/// Get device-specific fold lengths for vectorized memory operations.
///
/// Based on Tinygrad's devectorizer.py:138-155 which uses different fold lengths
/// based on device capabilities:
/// - DSP: [128, 64, 32, 16, 8, 4] for high-throughput DSP operations
/// - AMX: [16, 8, 4, 2] for Apple AMX matrix coprocessor
/// - Image dtypes: [4] for image memory operations
/// - Default: [4, 2] for float4 support on most GPUs
///
/// TODO: Add device parameter when device context is threaded through patterns.
/// For now, uses conservative default that works across devices.
fn get_device_fold_lengths(load_dtype: &DType) -> Vec<usize> {
    // Check for image dtype (always use [4] for image operations)
    if let DType::Ptr { base, .. } = load_dtype
        && matches!(base.as_ref(), DType::Image { .. })
    {
        return vec![4, 1];
    }

    // Default fold lengths for float4 support
    // TODO: When device info is available, use device-specific lengths:
    // - DSP: [128, 64, 32, 16, 8, 4]
    // - AMX: [16, 8, 4, 2]
    // - Half precision with ALLOW_HALF8: [8, 4, 2]
    vec![4, 2, 1]
}

/// Split LOAD based on fold length divisibility.
///
/// Based on Tinygrad's split_load_store (devectorizer.py:130-174).
/// For LOAD(CAST(INDEX)), determine maximum fold length based on offset divisibility.
fn split_load(buffer: &Arc<UOp>, idx: &Arc<UOp>, cast_dtype: &DType) -> Option<Arc<UOp>> {
    let Op::Index { indices, .. } = idx.op() else {
        return None;
    };

    // Get vector size from the cast pointer dtype
    let sz = match cast_dtype {
        DType::Ptr { size: Some(sz), .. } => *sz,
        DType::Ptr { base, .. } => base.vcount(),
        _ => return None,
    };

    if sz <= 1 {
        return None;
    }

    tracing::debug!(sz = sz, "split_load: processing LOAD(CAST(INDEX))");

    // Get scalar base type from buffer
    let scalar_base = buffer.dtype().base();

    // Determine fold lengths (based on device capability and dtype)
    let load_dtype = DType::Vector { scalar: scalar_base, count: sz };
    let mut lengths = get_device_fold_lengths(&load_dtype);

    // Filter by offset divisibility (Issue 4: conservative default)
    if let Some(offset) = indices.first() {
        lengths.retain(|&len| offset_divides_evenly(offset, len));
    }

    // Ensure 1 is always available as fallback
    if !lengths.contains(&1) {
        lengths.push(1);
    }

    // Greedy split
    let mut chunks = Vec::new();
    let mut pos = 0usize;

    while pos < sz {
        for &fold_len in &lengths {
            if pos + fold_len > sz {
                continue;
            }

            // Create INDEX for this chunk
            let chunk_idx = if pos == 0 { idx.clone() } else { offset_index(idx, pos as i64) };

            // CAST to vector pointer if fold_len > 1
            let chunk_ptr = if fold_len > 1 {
                let vec_ptr_dtype = make_vec_ptr_dtype(buffer, fold_len);
                UOp::cast(chunk_idx, vec_ptr_dtype)
            } else {
                chunk_idx
            };

            // Create LOAD with appropriate dtype
            let chunk_dtype = if fold_len > 1 {
                DType::Vector { scalar: scalar_base, count: fold_len }
            } else {
                DType::Scalar(scalar_base)
            };

            let chunk_load = UOp::new(Op::Load { buffer: buffer.clone(), index: chunk_ptr }, chunk_dtype);
            chunks.push(chunk_load);

            pos += fold_len;
            break;
        }
    }

    // If not split (single chunk), no change needed
    if chunks.len() <= 1 {
        return None;
    }

    tracing::debug!(num_chunks = chunks.len(), "split_load: split into chunks");

    Some(UOp::cat(chunks))
}

/// Split STORE based on fold length divisibility.
fn split_store(
    buffer: &Arc<UOp>,
    idx: &Arc<UOp>,
    cast_dtype: &DType,
    value: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
) -> Option<Arc<UOp>> {
    let Op::Index { indices, .. } = idx.op() else {
        return None;
    };

    // Get vector size from the value dtype or cast pointer dtype
    let sz = value.dtype().vcount().max(match cast_dtype {
        DType::Ptr { size: Some(sz), .. } => *sz,
        DType::Ptr { base, .. } => base.vcount(),
        _ => 1,
    });

    if sz <= 1 {
        return None;
    }

    tracing::debug!(sz = sz, "split_store: processing STORE(CAST(INDEX), ...)");

    // Determine fold lengths (based on device capability and dtype)
    let mut lengths = get_device_fold_lengths(&value.dtype());

    // Filter by offset divisibility (Issue 4: conservative default)
    if let Some(offset) = indices.first() {
        lengths.retain(|&len| offset_divides_evenly(offset, len));
    }

    if !lengths.contains(&1) {
        lengths.push(1);
    }

    // Greedy split
    let mut stores = Vec::new();
    let mut pos = 0usize;

    while pos < sz {
        for &fold_len in &lengths {
            if pos + fold_len > sz {
                continue;
            }

            // Create INDEX for this chunk
            let chunk_idx = if pos == 0 { idx.clone() } else { offset_index(idx, pos as i64) };

            // CAST to vector pointer if fold_len > 1
            let chunk_ptr = if fold_len > 1 {
                let vec_ptr_dtype = make_vec_ptr_dtype(buffer, fold_len);
                UOp::cast(chunk_idx, vec_ptr_dtype)
            } else {
                chunk_idx
            };

            // GEP to extract value elements for this chunk
            let gep_indices: Vec<usize> = (pos..pos + fold_len).collect();
            let chunk_value = UOp::gep(value.clone(), gep_indices);

            // Create STORE with preserved ranges
            let chunk_store = UOp::store_with_ranges(buffer.clone(), chunk_ptr, chunk_value, ranges.clone());
            stores.push(chunk_store);

            pos += fold_len;
            break;
        }
    }

    if stores.len() <= 1 {
        return None;
    }

    tracing::debug!(num_stores = stores.len(), "split_store: split into stores");

    Some(UOp::group(stores.into_iter().collect()))
}

/// Check if offset expression divides evenly by len.
///
/// Based on Tinygrad's offset.divides(x) (devectorizer.py:156).
/// Issue 4 fix: Conservative - returns false for unknown expressions.
fn offset_divides_evenly(offset: &Arc<UOp>, len: usize) -> bool {
    if len <= 1 {
        return true;
    }

    match offset.op() {
        Op::Const(cv) => {
            if let ConstValue::Int(n) = cv.0 {
                return n % (len as i64) == 0;
            }
            false
        }
        Op::Binary(BinaryOp::Mul, left, right) => {
            // X * const → divides if const >= len and const % len == 0
            let check_const = |c: &Arc<UOp>| {
                if let Op::Const(cv) = c.op()
                    && let ConstValue::Int(n) = cv.0
                {
                    return n >= len as i64 && n % (len as i64) == 0;
                }
                false
            };
            check_const(left) || check_const(right)
        }
        Op::Binary(BinaryOp::Add, left, right) => {
            // Add: both sides must divide evenly
            offset_divides_evenly(left, len) && offset_divides_evenly(right, len)
        }
        _ => false, // Issue 4: Conservative default - unknown expressions don't divide
    }
}

/// Create INDEX with offset added.
fn offset_index(idx: &Arc<UOp>, offset: i64) -> Arc<UOp> {
    let Op::Index { buffer, indices, gate } = idx.op() else {
        return idx.clone();
    };

    // Add offset to first index
    let new_indices: SmallVec<[Arc<UOp>; 4]> = indices
        .iter()
        .enumerate()
        .map(|(i, idx)| {
            if i == 0 {
                UOp::new(
                    Op::Binary(BinaryOp::Add, idx.clone(), UOp::const_(DType::Index, ConstValue::Int(offset))),
                    DType::Index,
                )
            } else {
                idx.clone()
            }
        })
        .collect();

    UOp::new(Op::Index { buffer: buffer.clone(), indices: new_indices, gate: gate.clone() }, idx.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_consecutive_offsets_from_map_contiguous() {
        let mut offsets_map = HashMap::new();
        offsets_map.insert(0, vec![0]);
        offsets_map.insert(1, vec![1]);
        offsets_map.insert(2, vec![2]);
        offsets_map.insert(3, vec![3]);

        let groups = group_consecutive_offsets_from_map(&offsets_map);

        // Should be one contiguous group
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].0, 0); // first_offset = 0
        assert_eq!(groups[0].1, vec![0, 1, 2, 3]); // lanes
    }

    #[test]
    fn test_group_consecutive_offsets_from_map_non_contiguous() {
        let mut offsets_map = HashMap::new();
        offsets_map.insert(0, vec![0]);
        offsets_map.insert(2, vec![1]);
        offsets_map.insert(4, vec![2]);
        offsets_map.insert(6, vec![3]);

        let groups = group_consecutive_offsets_from_map(&offsets_map);

        // Should be four separate groups (no consecutive offsets)
        assert_eq!(groups.len(), 4);
        assert_eq!(groups[0].0, 0);
        assert_eq!(groups[0].1, vec![0]);
        assert_eq!(groups[1].0, 2);
        assert_eq!(groups[1].1, vec![1]);
    }

    #[test]
    fn test_group_consecutive_offsets_from_map_mixed() {
        let mut offsets_map = HashMap::new();
        offsets_map.insert(0, vec![0]);
        offsets_map.insert(1, vec![1]);
        offsets_map.insert(2, vec![2]);
        offsets_map.insert(5, vec![3]);
        offsets_map.insert(6, vec![4]);

        let groups = group_consecutive_offsets_from_map(&offsets_map);

        // [0,1,2] consecutive, [5,6] consecutive
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, 0);
        assert_eq!(groups[0].1, vec![0, 1, 2]);
        assert_eq!(groups[1].0, 5);
        assert_eq!(groups[1].1, vec![3, 4]);
    }

    #[test]
    fn test_offset_divides_evenly() {
        let offset_4 = UOp::const_(DType::Index, ConstValue::Int(4));
        assert!(offset_divides_evenly(&offset_4, 4));
        assert!(offset_divides_evenly(&offset_4, 2));
        assert!(offset_divides_evenly(&offset_4, 1));
        assert!(!offset_divides_evenly(&offset_4, 3));

        let offset_0 = UOp::const_(DType::Index, ConstValue::Int(0));
        assert!(offset_divides_evenly(&offset_0, 4));

        // Unknown expression should return false (conservative)
        let range_var = UOp::new(
            Op::Range {
                end: UOp::const_(DType::Index, ConstValue::Int(10)),
                axis_id: morok_ir::AxisId::Renumbered(0),
                axis_type: morok_ir::AxisType::Loop,
            },
            DType::Index,
        );
        assert!(!offset_divides_evenly(&range_var, 4));
    }

    #[test]
    fn test_extract_root_and_offset() {
        // Test Add(root, const) where root is a non-constant expression
        let root = UOp::new(
            Op::Range {
                end: UOp::const_(DType::Index, ConstValue::Int(10)),
                axis_id: morok_ir::AxisId::Renumbered(0),
                axis_type: morok_ir::AxisType::Loop,
            },
            DType::Index,
        );
        let offset_const = UOp::const_(DType::Index, ConstValue::Int(3));
        let add = UOp::new(Op::Binary(BinaryOp::Add, root.clone(), offset_const), DType::Index);

        let (key, offset) = extract_root_and_offset(&add, None);
        assert_eq!(offset, 3);
        assert!(matches!(key, RootKey::Expr { .. }));

        // Test pure const - when both operands are constants, sum them
        let pure_const = UOp::const_(DType::Index, ConstValue::Int(42));
        let (key, offset) = extract_root_and_offset(&pure_const, None);
        assert_eq!(offset, 42);
        assert!(matches!(key, RootKey::Const));

        // Test Add of two constants - we only strip the outermost constant
        // In practice, symbolic simplification would fold Add(100, 3) -> 103 before extraction.
        // Here we test the un-simplified case: Add(100, 3) extracts (Expr(100), 3).
        let const_a = UOp::const_(DType::Index, ConstValue::Int(100));
        let const_b = UOp::const_(DType::Index, ConstValue::Int(3));
        let add_consts = UOp::new(Op::Binary(BinaryOp::Add, const_a, const_b), DType::Index);
        let (key, offset) = extract_root_and_offset(&add_consts, None);
        // With simplified extraction (only strip outermost const), we get the 3 as offset
        // and the left side (100) as the "root" (which happens to be a constant but we don't fold)
        assert_eq!(offset, 3);
        // The root is the CONST(100) node, but RootKey::expr() will mark it as Expr
        // This is fine because symbolic simplification should have already folded this case
        assert!(matches!(key, RootKey::Expr { .. }));
    }
}
