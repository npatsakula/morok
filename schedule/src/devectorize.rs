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

use itertools::Itertools;
use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::{BinaryOp, ConstValue, Op, UOp};

use crate::TypedPatternMatcher;
use smallvec::SmallVec;

use crate::rewrite::graph_rewrite_bottom_up;
use crate::symbolic::patterns::symbolic;

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run devectorize pass on kernel AST.
///
/// Call this AFTER `pre_expand` but BEFORE codegen.
/// Transforms vector indices into grouped contiguous accesses.
///
/// Matches Tinygrad's devectorizer pipeline (codegen/__init__.py:79):
/// ```python
/// pm_devectorize = sym + devectorize + load_store_folding + correct_load_store + load_store_indexing
/// ```
///
/// Morok's two-phase approach:
/// - Phase 1: devectorize_patterns + expand_index → PTRCAT grouping
/// - Phase 2: load_store_folding (GEP movement + PTRCAT distribution + split) + load_store_indexing
///
/// Note: Bool storage conversion (`bool_storage_patterns()`) is called separately
/// from `optimizer/mod.rs` as it's backend-specific (LLVM/PTX).
pub fn devectorize(ast: &Arc<UOp>) -> Arc<UOp> {
    // Phase 1: Devectorize patterns (cast_after, buf_and_index) + expand vector indices
    // This matches Tinygrad's `devectorize` + `load_store_folding.expand_index`
    let phase1 = devectorize_patterns() + expand_index_patterns();
    let ast = graph_rewrite_bottom_up(&phase1, ast.clone(), &mut ());

    // Phase 2: GEP movement + PTRCAT distribution + LOAD/STORE splitting + gate dropping
    // All patterns run together so:
    // - LOAD(GEP(PTRCAT)) → GEP(LOAD(PTRCAT)) → GEP(CAT(LOADs))
    // - split_load creates CAT([LOAD<4>×N]), CAT→VECTORIZE converts it
    // - INDEX(buf, x, true) → INDEX(buf, x, None) (gate dropping)
    let phase2 = gep_ptrcat_patterns() + load_store_patterns() + load_store_indexing_patterns();
    graph_rewrite_bottom_up(&phase2, ast, &mut ())
}
/// Phase 3 patterns: Convert bool LOAD/STORE to use uint8 storage.
///
/// LLVM's i1 type when stored to memory can have garbage in upper 7 bits.
/// We cast bool→uint8 before storing and uint8→bool after loading.
/// This matches Tinygrad's approach in PTX and NIR renderers.
pub fn bool_storage_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // STORE bool: cast to uint8 before storing
        Store { buffer, index, value, ranges } if value.dtype().base().is_bool() => {
            let uint8_dtype = value.dtype().with_base(ScalarDType::UInt8);
            Some(UOp::store_with_ranges(buffer.clone(), index.clone(), UOp::cast(value.clone(), uint8_dtype), ranges.clone()))
        },

        // LOAD bool: load as uint8, then cast to bool
        load @ Load { buffer, index } if load.dtype().base().is_bool() => {
            let uint8_dtype = load.dtype().with_base(ScalarDType::UInt8);
            let uint8_load = UOp::new(Op::Load { buffer: buffer.clone(), index: index.clone() }, uint8_dtype);
            Some(UOp::cast(uint8_load, load.dtype()))
        },
    }
}

/// GEP/CAT/VECTORIZE patterns for memory access devectorization.
///
/// Simplified to match Tinygrad's approach (symbolic.py + devectorizer.py):
/// - CAT → VECTORIZE: CAT can't be rendered, expand to VECTORIZE with GEPs
/// - GEP(VECTORIZE) → element extraction
/// - GEP identity removal
/// - Single-element VECTORIZE/CAT/PTRCAT → unwrap
///
/// Note: GEP(CAT) and GEP(PTRCAT) patterns removed because:
/// - CAT→VECTORIZE runs first, so GEP(CAT) becomes GEP(VECTORIZE)
/// - PTRCAT is created with GEP already applied, then immediately consumed by LOAD/STORE
pub(crate) fn gep_ptrcat_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // CAT → VECTORIZE: CAT([a<4>, b<4>]) → VECTORIZE(a.gep(0), ..., b.gep(3))
        Cat { sources } if sources.len() == 1 => Some(sources[0].clone()),
        Cat { sources } => {
            let elements: SmallVec<[Arc<UOp>; 4]> = sources.iter()
                .flat_map(|src| {
                    let n = src.dtype().vcount();
                    (0..n).map(move |i| if n == 1 { src.clone() } else { UOp::gep(src.clone(), vec![i]) })
                })
                .collect();
            Some(UOp::vectorize(elements))
        },

        // GEP(VECTORIZE) → element extraction
        Gep { vector: Vectorize { elements }, indices } => {
            let extracted: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .filter_map(|&i| elements.get(i).cloned())
                .collect();
            if extracted.len() != indices.len() { return None; }
            Some(if extracted.len() == 1 { extracted[0].clone() } else { UOp::vectorize(extracted) })
        },

        // GEP(PTRCAT) → reorder pointers
        Gep { vector: PtrCat { sources }, indices } => {
            let reordered: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .filter_map(|&i| sources.get(i).cloned())
                .collect();
            if reordered.len() != indices.len() { return None; }
            Some(if reordered.len() == 1 { reordered[0].clone() } else { UOp::ptrcat(reordered.to_vec()) })
        },

        // GEP identity: GEP(x, [0,1,...,n-1]) → x
        Gep { vector, indices } if is_identity_gep(vector, indices) => Some(vector.clone()),

        // Single-element unwrap
        Vectorize { elements } if elements.len() == 1 => Some(elements[0].clone()),
        PtrCat { sources } if sources.len() == 1 => Some(sources[0].clone()),

        // WHERE devectorize: WHERE(<N x i1>, ...) → VECTORIZE(WHERE(i1, ...), ...)
        Where(cond, t, f) if cond.dtype().vcount() > 1 => {
            let vcount = cond.dtype().vcount();
            let scalar_wheres: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
                .map(|i| {
                    let c = UOp::gep(cond.clone(), vec![i]);
                    let tv = UOp::gep(t.clone(), vec![i]);
                    let fv = UOp::gep(f.clone(), vec![i]);
                    UOp::try_where(c, tv, fv).expect("WHERE construction should succeed")
                })
                .collect();
            Some(UOp::vectorize(scalar_wheres))
        },
    }
}

/// Check if GEP is identity: GEP(x, [0,1,...,n-1]) where n == x.vcount
fn is_identity_gep(vector: &Arc<UOp>, indices: &[usize]) -> bool {
    let vcount = vector.dtype().vcount();
    indices.len() == vcount && indices.iter().enumerate().all(|(i, &j)| i == j)
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

// ============================================================================
// Devectorize Patterns (Tinygrad devectorizer.py:250-256)
// ============================================================================

/// Combined devectorize patterns matching Tinygrad's `devectorize` PatternMatcher.
///
/// Includes:
/// - cast_after_pattern: AFTER(CAST(x), deps) → CAST(AFTER(x, deps))
/// - no_vectorized_alu: Vector ALU ops → VECTORIZE(scalar ALU ops)
/// - no_vectorized_wmma: Vector WMMA ops → VECTORIZE(scalar WMMA ops)
/// - devectorize_buf_and_index: LOCAL/REG buffer vectorization
///
/// Tinygrad (devectorizer.py:250-256):
/// ```python
/// devectorize = PatternMatcher([
///   (UPat(Ops.CAST, name="c").f(Ops.AFTER, allow_any_len=True, name="a"), ...),
///   (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name="alu"), no_vectorized_alu),
///   (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
/// ])+devectorize_buf_and_index
/// ```
pub fn devectorize_patterns() -> TypedPatternMatcher {
    cast_after_pattern() + no_vectorized_alu() + no_vectorized_wmma() + devectorize_buf_and_index_patterns()
}

/// Pattern for WMMA devectorization.
///
/// Based on Tinygrad's no_vectorized_wmma (devectorizer.py:208-217).
/// Splits vectorized WMMA into multiple scalar WMMA operations.
fn no_vectorized_wmma() -> TypedPatternMatcher {
    crate::patterns! {
        wmma if is_vectorized_wmma(wmma) => |wmma| devectorize_wmma(wmma),
    }
}

/// Check if WMMA needs devectorization.
///
/// Returns true if WMMA's dtype.vcount > expected output size from metadata.
fn is_vectorized_wmma(uop: &Arc<UOp>) -> bool {
    let Op::Wmma { metadata, .. } = uop.op() else {
        return false;
    };

    // Calculate expected output size from upcast_axes
    let out_sz: usize = metadata.upcast_axes.iter().map(|(_, size)| size).product();

    // If out_sz is 0 or 1, use 1 as minimum
    let expected_sz = out_sz.max(1);

    uop.dtype().vcount() > expected_sz
}

/// Devectorize WMMA operation.
///
/// Based on Tinygrad's no_vectorized_wmma (devectorizer.py:208-217):
/// ```python
/// def no_vectorized_wmma(wmma:UOp):
///   out_sz = prod(x[1] for x in wmma.arg[6][-1])
///   if wmma.dtype.count == out_sz: return None
///   tsrcs = []
///   for s,sz in zip(wmma.src, wmma.arg[6]):
///     ssz = prod(x[1] for x in sz)
///     tsrcs.append([s.gep(tuple(range(grp, grp+ssz))) for grp in range(0, s.dtype.count, ssz)])
///   wmmas = [UOp(Ops.WMMA, wmma.dtype.scalar().vec(out_sz), tsrc, wmma.arg) for tsrc in zip(*tsrcs)]
///   wmma_ex = flatten([[e.gep(i) for i in range(out_sz)] for e in wmmas])
///   return UOp(Ops.VECTORIZE, wmma.dtype, tuple(wmma_ex))
/// ```
fn devectorize_wmma(wmma: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Wmma { a, b, c, metadata } = wmma.op() else {
        return None;
    };

    // Calculate output size from upcast_axes
    let out_sz: usize = metadata.upcast_axes.iter().map(|(_, size)| size).product::<usize>().max(1);

    if wmma.dtype().vcount() == out_sz {
        return None;
    }

    // For each source (a, b, c), calculate its expected size and split into groups
    // In Tinygrad, arg[6] contains size info for each source. In Morok, we use upcast_axes.
    // Simplified approach: split by out_sz for all sources

    let sources = [a, b, c];
    let mut tsrcs: Vec<Vec<Arc<UOp>>> = vec![Vec::new(), Vec::new(), Vec::new()];

    for (i, src) in sources.iter().enumerate() {
        let src_count = src.dtype().vcount();
        let ssz = out_sz; // Simplified: use out_sz for all sources

        for grp in (0..src_count).step_by(ssz) {
            let gep_indices: Vec<usize> = (grp..grp + ssz.min(src_count - grp)).collect();
            tsrcs[i].push(UOp::gep((*src).clone(), gep_indices));
        }
    }

    // Verify all sources have same number of groups
    let num_groups = tsrcs[0].len();
    if tsrcs.iter().any(|t| t.len() != num_groups) {
        tracing::warn!("WMMA devectorization: mismatched source group counts");
        return None;
    }

    // Create new WMMA for each group
    let scalar_dtype = wmma.dtype().scalar()?;
    let wmma_dtype = if out_sz > 1 { DType::Scalar(scalar_dtype).vec(out_sz) } else { DType::Scalar(scalar_dtype) };

    let wmmas: Vec<Arc<UOp>> = (0..num_groups)
        .map(|g| UOp::wmma(tsrcs[0][g].clone(), tsrcs[1][g].clone(), tsrcs[2][g].clone(), metadata.clone()))
        .collect();

    // Flatten: for each WMMA, extract each element with GEP
    let wmma_ex: SmallVec<[Arc<UOp>; 4]> =
        wmmas.iter().flat_map(|w| (0..out_sz).map(move |i| UOp::gep(w.clone(), vec![i]))).collect();

    Some(UOp::vectorize(wmma_ex))
}

/// Pattern: AFTER(CAST(x), deps) → CAST(AFTER(x, deps))
///
/// Tinygrad (devectorizer.py:252):
/// ```python
/// (UPat(Ops.CAST, name="c").f(Ops.AFTER, allow_any_len=True, name="a"),
///  lambda c,a: c.src[0].after(*a.src[1:]).cast(c.dtype)),
/// ```
///
/// This reorders AFTER and CAST so that dependencies are tracked on the inner
/// value, allowing the cast to be optimized independently.
fn cast_after_pattern() -> TypedPatternMatcher {
    crate::patterns! {
        // AFTER(CAST(x), deps) → CAST(AFTER(x, deps))
        // DSL nested pattern directly captures the structure
        After { passthrough: Cast { src, dtype }, deps }
            => |src, dtype, deps| {
                let new_after = UOp::after(src.clone(), deps.clone());
                Some(UOp::cast(new_after, dtype.clone()))
            },
    }
}

/// Patterns for LOCAL/REG buffer devectorization.
///
/// Tinygrad (devectorizer.py:241-248):
/// ```python
/// devectorize_buf_and_index = PatternMatcher([
///   (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG), name="buf"), no_vectorized_buf),
///   (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").index(UPat.var("idx")),
///    no_vectorized_index),
///   (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").broadcast(name="bcast").index(UPat.var("idx")),
///    no_vectorized_index_broadcast),
///   (UPat((Ops.DEFINE_LOCAL, Ops.DEFINE_REG)).or_after(name="buf").cast(name="cast").gep(name="bcast").index(UPat.var("idx")),
///    no_vectorized_index_broadcast),
/// ])
/// ```
fn devectorize_buf_and_index_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // Pattern 1: DEFINE_LOCAL/DEFINE_REG with vector pointer → scalar pointer + CAST
        def if is_vectorized_define_local_or_reg(def) => |def| no_vectorized_buf(def),

        // Pattern 2: INDEX(CAST(DEFINE_LOCAL/REG), scalar_idx) → INDEX(VECTORIZE([buf...]), scaled_vec_idx)
        // Transforms local/register memory indexing for tensor cores / shared memory
        index if is_no_vectorized_index_pattern(index) => |index| no_vectorized_index(index),

        // Pattern 3: INDEX(BROADCAST(CAST(DEFINE_LOCAL/REG)), scalar_idx) → similar transform
        index if is_no_vectorized_index_broadcast_pattern(index) => |index| no_vectorized_index_broadcast(index),

        // Pattern 4: INDEX(GEP(CAST(DEFINE_LOCAL/REG)), scalar_idx) → similar transform with GEP
        index if is_no_vectorized_index_gep_pattern(index) => |index| no_vectorized_index_gep(index),
    }
}

/// Check if DEFINE_LOCAL or DEFINE_REG has vectorized pointer dtype.
fn is_vectorized_define_local_or_reg(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::DefineLocal(_) | Op::DefineReg { .. } => uop.ptrdtype().is_some_and(|(base, _, _)| base.vcount() > 1),
        _ => false,
    }
}

/// Transform DEFINE_LOCAL/DEFINE_REG with vector pointer to scalar pointer + CAST.
///
/// Tinygrad (devectorizer.py:225-226):
/// ```python
/// def no_vectorized_buf(buf:UOp):
///   return buf.replace(dtype=buf.ptrdtype.base.scalar().ptr(buf.ptrdtype.size*buf.ptrdtype.count,
///                      buf.ptrdtype.addrspace)).cast(buf.dtype)
/// ```
fn no_vectorized_buf(buf: &Arc<UOp>) -> Option<Arc<UOp>> {
    let (base, addrspace, size) = buf.ptrdtype()?;

    // Get the vector count and create scalar base
    let vcount = base.vcount();
    if vcount <= 1 {
        return None;
    }

    let scalar_base = base.scalar()?;
    let new_size = size.map(|s| s * vcount);
    let scalar_ptr_dtype = DType::Ptr { base: Box::new(DType::Scalar(scalar_base)), addrspace, size: new_size, vcount: 1 };

    // Create new define with scalar pointer dtype, then cast back
    let scalar_def = buf.with_dtype(scalar_ptr_dtype);
    Some(UOp::cast(scalar_def, buf.dtype()))
}

/// Check if INDEX matches the no_vectorized_index pattern:
/// INDEX(CAST(DEFINE_LOCAL/REG.or_after()), scalar_idx)
fn is_no_vectorized_index_pattern(uop: &Arc<UOp>) -> bool {
    let Op::Index { buffer: cast_node, indices, .. } = uop.op() else {
        return false;
    };

    // Check if index is scalar
    if indices.first().is_none_or(|idx| idx.dtype().vcount() != 1) {
        return false;
    }

    // Check if buffer is CAST with vectorized pointer dtype
    let Op::Cast { src: inner, dtype: cast_dtype } = cast_node.op() else {
        return false;
    };
    let DType::Ptr { base, .. } = cast_dtype else {
        return false;
    };
    if base.vcount() <= 1 {
        return false;
    }

    // Check if inner is DEFINE_LOCAL/REG or AFTER(DEFINE_LOCAL/REG)
    is_define_local_or_reg_or_after(inner)
}

/// Check if a UOp is DEFINE_LOCAL, DEFINE_REG, or AFTER wrapping one of those.
///
/// Uses `unwrap_after()` to walk through AFTER nodes (Tinygrad's `.or_after()` pattern).
fn is_define_local_or_reg_or_after(uop: &Arc<UOp>) -> bool {
    matches!(uop.unwrap_after().op(), Op::DefineLocal(_) | Op::DefineReg { .. })
}

/// Transform INDEX(CAST(DEFINE_LOCAL/REG), scalar_idx) → INDEX(VECTORIZE([buf,...]), scaled_vec_idx)
///
/// Tinygrad (devectorizer.py:228-231):
/// ```python
/// def no_vectorized_index(buf:UOp, cast:UOp, idx:UOp):
///   cnt = cast.dtype.count
///   assert idx.dtype.count == 1, f"idx dtype must be 1 {idx.dtype}"
///   return buf.broadcast(cnt).index(idx.broadcast(cnt)*cnt+UOp.const(dtypes.index.vec(cnt), tuple(range(cnt))), ptr=True)
/// ```
fn no_vectorized_index(index_uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer: cast_node, indices, gate } = index_uop.op() else {
        return None;
    };

    let Op::Cast { src: buf, dtype: cast_dtype } = cast_node.op() else {
        return None;
    };

    let idx = indices.first()?;

    // Get cnt from cast_dtype
    let DType::Ptr { base, .. } = cast_dtype else {
        return None;
    };
    let cnt = base.vcount();
    if cnt <= 1 {
        return None;
    }

    // buf.broadcast(cnt) → VECTORIZE([buf, buf, ..., buf])
    let buf_broadcast = UOp::broadcast(buf.clone(), cnt);

    // idx.broadcast(cnt) → VECTORIZE([idx, idx, ..., idx])
    let idx_broadcast = UOp::broadcast(idx.clone(), cnt);

    // Create the constant vector [0, 1, 2, ..., cnt-1]
    let offset_vec = create_index_vector(cnt);

    // Create cnt constant and broadcast
    let cnt_broadcast = UOp::broadcast(idx.const_like(cnt as i64), cnt);

    // idx.broadcast(cnt) * cnt + offset_vec (using panicking wrappers for validated types)
    let final_idx = idx_broadcast.mul(&cnt_broadcast).add(&offset_vec);

    // Create INDEX with Ptr dtype (ptr=True equivalent)
    let buf_dtype = buf_broadcast.dtype();
    let indices_smallvec: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![final_idx];
    Some(UOp::new(
        Op::Index { buffer: buf_broadcast, indices: indices_smallvec, gate: gate.clone() },
        buf_dtype, // Use buffer dtype for ptr=True
    ))
}

/// Create a constant vector [0, 1, 2, ..., cnt-1] as a VECTORIZE of consts.
fn create_index_vector(cnt: usize) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> = (0..cnt).map(|i| UOp::index_const(i as i64)).collect();
    UOp::vectorize(elements)
}

/// Check if INDEX matches the no_vectorized_index_broadcast pattern:
/// INDEX(BROADCAST(CAST(DEFINE_LOCAL/REG.or_after())), scalar_idx)
fn is_no_vectorized_index_broadcast_pattern(uop: &Arc<UOp>) -> bool {
    let Op::Index { buffer: outer, indices, .. } = uop.op() else {
        return false;
    };

    // Index must be scalar
    if indices.first().is_none_or(|idx| idx.dtype().vcount() != 1) {
        return false;
    }

    // Buffer must be VECTORIZE containing CAST(DEFINE_LOCAL/REG) with vectorized pointer
    let Op::Vectorize { elements } = outer.op() else {
        return false;
    };
    let Some(first) = elements.first() else {
        return false;
    };
    let Op::Cast { src: inner, dtype: cast_dtype } = first.op() else {
        return false;
    };
    let DType::Ptr { base, .. } = cast_dtype else {
        return false;
    };

    base.vcount() > 1 && is_define_local_or_reg_or_after(inner)
}

/// Transform INDEX(BROADCAST(CAST(buf)), scalar_idx) similar to no_vectorized_index.
///
/// Tinygrad (devectorizer.py:233-239):
/// ```python
/// def no_vectorized_index_broadcast(buf:UOp, cast:UOp, bcast:UOp, idx:UOp):
///   cnt = cast.dtype.count
///   precnt = bcast.dtype.vcount
///   input_gep = bcast.arg if bcast.op is Ops.GEP else ([0]*precnt)
///   gep_arg = tuple(flatten([range(precnt) for _ in range(cnt)]))
///   sum_arg = tuple(flatten([[i+y for y in input_gep] for i in range(cnt)]))
///   return buf.broadcast(cnt*precnt).index(idx.gep(gep_arg)*cnt+UOp.const(dtypes.index.vec(cnt*precnt), sum_arg), ptr=True)
/// ```
fn no_vectorized_index_broadcast(index_uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer: bcast, indices, gate } = index_uop.op() else {
        return None;
    };

    let Op::Vectorize { elements } = bcast.op() else {
        return None;
    };

    let first = elements.first()?;
    let Op::Cast { src: buf, dtype: cast_dtype } = first.op() else {
        return None;
    };

    let idx = indices.first()?;

    let DType::Ptr { base, .. } = cast_dtype else {
        return None;
    };
    let cnt = base.vcount();
    let precnt = elements.len();

    if cnt <= 1 {
        return None;
    }

    // input_gep = [0] * precnt (since this is VECTORIZE, not GEP)
    let input_gep: Vec<usize> = vec![0; precnt];

    // gep_arg = flatten([range(precnt) for _ in range(cnt)])
    let gep_arg: Vec<usize> = (0..cnt).flat_map(|_| 0..precnt).collect();

    // sum_arg = flatten([[i+y for y in input_gep] for i in range(cnt)])
    let sum_arg: Vec<i64> = (0..cnt).flat_map(|i| input_gep.iter().map(move |&y| (i + y) as i64)).collect();

    // buf.broadcast(cnt*precnt)
    let total_cnt = cnt * precnt;
    let buf_broadcast = UOp::broadcast(buf.clone(), total_cnt);

    // idx.gep(gep_arg) - extract elements according to gep_arg
    let idx_gep = UOp::gep(idx.clone(), gep_arg);

    // cnt constant broadcast
    let cnt_broadcast = UOp::broadcast(idx.const_like(cnt as i64), total_cnt);

    // Create sum_arg vector
    let sum_vec = create_const_index_vector(&sum_arg);

    // idx_gep * cnt + sum_vec (using panicking wrappers for validated types)
    let final_idx = idx_gep.mul(&cnt_broadcast).add(&sum_vec);

    let buf_dtype = buf_broadcast.dtype();
    let indices_smallvec: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![final_idx];
    Some(UOp::new(Op::Index { buffer: buf_broadcast, indices: indices_smallvec, gate: gate.clone() }, buf_dtype))
}

/// Create a constant vector from i64 values.
fn create_const_index_vector(values: &[i64]) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> = values.iter().map(|&v| UOp::index_const(v)).collect();
    UOp::vectorize(elements)
}

/// Check if INDEX matches the no_vectorized_index_gep pattern:
/// INDEX(GEP(CAST(DEFINE_LOCAL/REG.or_after())), scalar_idx)
fn is_no_vectorized_index_gep_pattern(uop: &Arc<UOp>) -> bool {
    let Op::Index { buffer: outer, indices, .. } = uop.op() else {
        return false;
    };

    // Index must be scalar
    if indices.first().is_none_or(|idx| idx.dtype().vcount() != 1) {
        return false;
    }

    // Buffer must be GEP(CAST(...)) with vectorized pointer dtype
    let Op::Gep { vector: cast_node, .. } = outer.op() else {
        return false;
    };
    let Op::Cast { src: inner, dtype: cast_dtype } = cast_node.op() else {
        return false;
    };
    let DType::Ptr { base, .. } = cast_dtype else {
        return false;
    };

    base.vcount() > 1 && is_define_local_or_reg_or_after(inner)
}

/// Transform INDEX(GEP(CAST(buf)), scalar_idx) similar to no_vectorized_index_broadcast.
fn no_vectorized_index_gep(index_uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer: gep_node, indices, gate } = index_uop.op() else {
        return None;
    };

    let Op::Gep { vector: cast_node, indices: gep_indices } = gep_node.op() else {
        return None;
    };

    let Op::Cast { src: buf, dtype: cast_dtype } = cast_node.op() else {
        return None;
    };

    let idx = indices.first()?;

    let DType::Ptr { base, .. } = cast_dtype else {
        return None;
    };
    let cnt = base.vcount();
    let precnt = gep_indices.len();

    if cnt <= 1 {
        return None;
    }

    // input_gep = gep_indices (the actual GEP indices)
    let input_gep: Vec<usize> = gep_indices.clone();

    // gep_arg = flatten([range(precnt) for _ in range(cnt)])
    let gep_arg: Vec<usize> = (0..cnt).flat_map(|_| 0..precnt).collect();

    // sum_arg = flatten([[i+y for y in input_gep] for i in range(cnt)])
    let sum_arg: Vec<i64> = (0..cnt).flat_map(|i| input_gep.iter().map(move |&y| (i + y) as i64)).collect();

    let total_cnt = cnt * precnt;
    let buf_broadcast = UOp::broadcast(buf.clone(), total_cnt);

    let idx_gep = UOp::gep(idx.clone(), gep_arg);

    // cnt constant broadcast
    let cnt_broadcast = UOp::broadcast(idx.const_like(cnt as i64), total_cnt);

    // Create sum_arg vector
    let sum_vec = create_const_index_vector(&sum_arg);

    // idx_gep * cnt + sum_vec (using panicking wrappers for validated types)
    let final_idx = idx_gep.mul(&cnt_broadcast).add(&sum_vec);

    let buf_dtype = buf_broadcast.dtype();
    let indices_smallvec: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![final_idx];
    Some(UOp::new(Op::Index { buffer: buf_broadcast, indices: indices_smallvec, gate: gate.clone() }, buf_dtype))
}

// ============================================================================
// Load Store Indexing Patterns (Tinygrad devectorizer.py:48-55)
// ============================================================================

/// Drop constant true gate: INDEX(buf, x, true) → INDEX(buf, x, None)
pub fn load_store_indexing_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        index @ Index { buffer, indices, gate: Some(g) }
            if matches!(g.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Bool(true)))
            => Some(UOp::new(Op::Index { buffer: buffer.clone(), indices: indices.clone(), gate: None }, index.dtype())),
    }
}

// ============================================================================
// Expand Index Patterns
// ============================================================================

/// Phase 1 patterns: expand vector INDEX into grouped PTRCAT.
///
/// Matches Tinygrad's load_store_folding (devectorizer.py:115): expand_index
/// Handles both INDEX(VECTORIZE([defines...]), vec) and INDEX(buffer, vec_idx).
pub(crate) fn expand_index_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // INDEX with vector index → expand into grouped PTRCAT
        // Tinygrad: (UPat(Ops.INDEX, ...), expand_index)
        index if is_vector_index(index) => expand_vector_index(index),
    }
}

/// Phase 2 patterns: GEP movement, PTRCAT distribution, and LOAD/STORE splitting.
///
/// Based on Tinygrad's load_store_folding (devectorizer.py:114-126).
pub(crate) fn load_store_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // GEP Movement: LOAD(GEP(x)) → GEP(LOAD(x))
        load @ Load { buffer, index: Gep { vector, indices } }
            => move_gep_after_load(load, buffer, vector, indices),

        // GEP Movement: STORE(GEP(x), data) → STORE(x, GEP⁻¹(data))
        Store { buffer, index: Gep { vector, indices }, value, ranges }
            => move_gep_on_store(buffer, vector, indices, value, ranges),

        // PTRCAT Distribution: LOAD(PTRCAT(a,b)) → CAT(LOAD(a), LOAD(b))
        Load { buffer, index: PtrCat { sources } }
            => distribute_ptrcat_load(buffer, sources),

        // PTRCAT Distribution: STORE(PTRCAT(a,b), data) → GROUP(STORE(a, gep(data,0..n)), ...)
        Store { buffer, index: PtrCat { sources }, value, ranges }
            => distribute_ptrcat_store(buffer, sources, value, ranges),

        // Split by Fold Length: LOAD(CAST(INDEX)) → split
        Load { buffer, index: Cast { src: idx @ Index { buffer: _, .. }, dtype: cast_dtype } }
            => split_load(buffer, idx, cast_dtype),

        // Split by Fold Length: STORE(CAST(INDEX), data) → split
        Store { buffer, index: Cast { src: idx @ Index { buffer: _, .. }, dtype: cast_dtype }, value, ranges }
            => split_store(buffer, idx, cast_dtype, value, ranges),
    }
}

// ============================================================================
// Pattern Predicates
// ============================================================================

/// Check if UOp is a define (LOCAL, REG, GLOBAL) or AFTER wrapping one.
///
/// Uses `unwrap_after()` to walk through AFTER nodes (Tinygrad's `.or_after()` pattern).
fn is_define_or_after(uop: &Arc<UOp>) -> bool {
    matches!(
        uop.unwrap_after().op(),
        Op::DefineLocal(_) | Op::DefineReg { .. } | Op::DefineGlobal(_)
    )
}

/// Check if INDEX has a vector index (dtype.vcount() > 1) and is eligible for expansion.
fn is_vector_index(uop: &Arc<UOp>) -> bool {
    let Op::Index { buffer, indices, .. } = uop.op() else { return false };
    let Some(idx) = indices.first() else { return false };
    if idx.dtype().vcount() <= 1 { return false; }

    // Extract actual buffer from VECTORIZE if present
    let actual_buf = if let Op::Vectorize { elements } = buffer.op() {
        if elements.is_empty() || !elements.iter().all(is_define_or_after) { return false; }
        &elements[0]
    } else {
        buffer
    };

    // Don't vectorize bool loads (LLVM i1 vector load is broken)
    !actual_buf.dtype().base().is_bool()
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
// expand_index: Tinygrad devectorizer.py:59-95
// ============================================================================

/// Expand vector INDEX into grouped consecutive accesses (PTRCAT).
///
/// Tinygrad's expand_index: generate scalar indices, simplify, group by root+offset.
fn expand_vector_index(index: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate } = index.op() else { return None };
    let vec = indices.first()?;
    let count = vec.dtype().vcount();

    // Extract actual buffer from VECTORIZE if present
    let buf = if let Op::Vectorize { elements } = buffer.op() {
        elements.first()?.clone()
    } else {
        buffer.clone()
    };

    // Step 1: Generate scalar INDEX ops and simplify
    // Tinygrad: midx = graph_rewrite(UOp.sink(*[buf.index(vec.gep(i), ptr=True) for i in range(count)]), symbolic+load_store_indexing)
    let scalar_indices: Vec<_> = (0..count)
        .map(|i| UOp::new(
            Op::Index { buffer: buf.clone(), indices: smallvec::smallvec![UOp::gep(vec.clone(), vec![i])], gate: gate.clone() },
            buf.dtype().clone(),
        ))
        .collect();

    let midx = graph_rewrite_bottom_up(&(symbolic() + load_store_indexing_patterns()), UOp::sink(scalar_indices), &mut ());
    let Op::Sink { sources } = midx.op() else { return None };

    // Step 2: Extract (valid, root, offset) for each lane
    // Tinygrad: offsets_rootsrc[root_src].setdefault(arg, []).append(i)
    let mut offsets_by_root: HashMap<(u64, u64), HashMap<i64, Vec<usize>>> = HashMap::new();

    for (lane, idx_op) in sources.iter().enumerate() {
        let Op::Index { indices: simp_indices, .. } = idx_op.op() else { continue };
        let idx = simp_indices.first()?.get_idx();
        let valid = simp_indices.first()?.get_valid();

        // Extract root and offset: Add(root, CONST) or CONST or other
        let (root, offset) = match idx.op() {
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
                (UOp::index_const(0), off) // Use sentinel for pure constants
            }
            _ => (idx.clone(), 0),
        };

        let key = (valid.content_hash(), root.content_hash());
        offsets_by_root.entry(key).or_default().entry(offset).or_default().push(lane);
    }

    // Step 3: Group consecutive offsets and build PTRCAT
    let mut ret = Vec::new();
    let mut idxs: Vec<Option<usize>> = vec![None; count];
    let mut global_offset = 0;

    for offsets in offsets_by_root.values() {
        // Group consecutive offsets: [(offset, lanes), ...]
        let groups = group_consecutive_offsets_from_map(offsets);

        for (_, lanes) in groups {
            let lidx = sources[lanes[0]].clone();
            let ptr = if lanes.len() > 1 {
                UOp::cast(lidx, make_vec_ptr_dtype(&buf, lanes.len()))
            } else {
                lidx
            };
            for (i, &lane) in lanes.iter().enumerate() {
                idxs[lane] = Some(global_offset + i);
            }
            ret.push(ptr);
            global_offset += lanes.len();
        }
    }

    if idxs.iter().any(|x| x.is_none()) { return None; }

    // Step 4: Create PTRCAT with correct dtype and apply GEP reorder
    // Tinygrad: buf.ptrdtype.base.ptr(size, addrspace).vec(global_offset)
    // PTRCAT dtype is vec(global_offset) of scalar pointers
    let DType::Ptr { base, addrspace, size, .. } = buf.dtype() else { return None };
    let scalar_ptr = DType::Ptr { base: Box::new(DType::Scalar(base.scalar()?)), addrspace: addrspace.clone(), size: size.clone(), vcount: 1 };
    let ptrcat_dtype = scalar_ptr.vec(global_offset);
    let ptrcat = UOp::new(Op::PtrCat { sources: SmallVec::from_vec(ret) }, ptrcat_dtype);
    let gep_indices: Vec<usize> = idxs.into_iter().map(|x| x.unwrap()).collect();
    Some(UOp::gep(ptrcat, gep_indices))
}

/// Group consecutive offsets for contiguous memory access.
///
/// Returns [(first_offset, [lanes]), ...] sorted by first_offset.
/// Uses itertools to group offsets where `offset - index` is constant.
/// Only extends groups when each offset maps to exactly one lane.
fn group_consecutive_offsets_from_map(offsets_map: &HashMap<i64, Vec<usize>>) -> Vec<(i64, Vec<usize>)> {
    if offsets_map.is_empty() {
        return vec![];
    }

    // Sort offsets and pair with consecutive index
    let sorted: Vec<_> = offsets_map.keys().copied().sorted().collect();

    // Group by (offset - index) which is constant for consecutive runs
    // Also break groups when an offset has multiple lanes
    sorted
        .iter()
        .copied()
        .enumerate()
        .chunk_by(|(idx, offset)| {
            // Break group if multiple lanes at this offset
            let single_lane = offsets_map[offset].len() == 1;
            let group_key = offset - (*idx as i64);
            (single_lane, group_key)
        })
        .into_iter()
        .map(|(_, group)| {
            let offsets_in_group: Vec<_> = group.collect();
            let first_offset = offsets_in_group[0].1;
            let lanes: Vec<usize> = offsets_in_group
                .iter()
                .flat_map(|(_, offset)| offsets_map[offset].iter().copied())
                .collect();
            (first_offset, lanes)
        })
        .collect()
}

/// Create vector pointer dtype for contiguous access.
///
/// Handles both global and LOCAL/REG memory by preferring ptrdtype() when available.
fn make_vec_ptr_dtype(buffer: &Arc<UOp>, vec_len: usize) -> DType {
    // Prefer ptrdtype() for LOCAL/REG memory (gives correct base and addrspace)
    // Fall back to dtype().base() for global memory
    let (base_dtype, addrspace) = buffer
        .ptrdtype()
        .map(|(base, addrspace, _)| (base.base(), addrspace))
        .unwrap_or_else(|| (buffer.dtype().base(), AddrSpace::Global));
    let vec_dtype = DType::Vector { scalar: base_dtype, count: vec_len };
    DType::Ptr { base: Box::new(vec_dtype), addrspace, size: Some(vec_len), vcount: 1 }
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

    // Add offset to first index using new helpers
    let new_indices: SmallVec<[Arc<UOp>; 4]> = indices
        .iter()
        .enumerate()
        .map(|(i, index_expr)| {
            if i == 0 {
                index_expr.add(&index_expr.const_like(offset))
            } else {
                index_expr.clone()
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
        let offset_4 = UOp::index_const(4);
        assert!(offset_divides_evenly(&offset_4, 4));
        assert!(offset_divides_evenly(&offset_4, 2));
        assert!(offset_divides_evenly(&offset_4, 1));
        assert!(!offset_divides_evenly(&offset_4, 3));

        let offset_0 = UOp::index_const(0);
        assert!(offset_divides_evenly(&offset_0, 4));

        // Unknown expression should return false (conservative)
        let range_var = UOp::new(
            Op::Range {
                end: UOp::index_const(10),
                axis_id: morok_ir::AxisId::Renumbered(0),
                axis_type: morok_ir::AxisType::Loop,
            },
            DType::Index,
        );
        assert!(!offset_divides_evenly(&range_var, 4));
    }

}
