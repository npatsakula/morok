//! Devectorize pass for contiguous memory access optimization.
//!
//! Transforms vectorized INDEX into grouped consecutive accesses (PTRCAT) after expansion.
//!
//! # Pipeline (Tinygrad codegen/__init__.py:79)
//!
//! Single pass: `sym + devectorize + load_store_folding + correct_load_store + load_store_indexing`
//!
//! - `devectorize`: cast_after, no_vectorized_alu, no_vectorized_wmma, devectorize_buf_and_index
//! - `load_store_folding`: expand_index, GEP/PTRCAT distribution
//! - `correct_load_store`: split_load_store by device fold lengths
//! - `load_store_indexing`: drop true gates
//!
//! # pm_render (called AFTER devectorize)
//!
//! - CAT → VECTORIZE (CAT can't be rendered directly)
//! - Multi-index GEP → VECTORIZE of single-index GEPs
//! - Single-element VECTORIZE/PTRCAT → unwrap

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use itertools::Itertools;
use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::{AxisType, BinaryOp, ConstValue, Op, ReduceOp, UOp, WmmaMetadata};
use tracing::debug;

use crate::TypedPatternMatcher;
use smallvec::SmallVec;

use crate::rewrite::graph_rewrite_bottom_up;
use crate::symbolic::patterns::symbolic;

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run devectorize pass. Call AFTER `pre_expand`, BEFORE codegen.
///
/// Note: `bool_storage_patterns()` called separately (backend-specific).
/// Note: `pm_render()` should be applied AFTER this pass.
pub fn devectorize(ast: &Arc<UOp>) -> Arc<UOp> {
    let pm_devectorize = symbolic()
        + devectorize_patterns()
        + load_store_folding_patterns()
        + correct_load_store_patterns()
        + load_store_indexing_patterns();

    graph_rewrite_bottom_up(&pm_devectorize, ast.clone(), &mut ())
}

/// Bool LOAD/STORE via uint8. LLVM i1 can have garbage in upper bits.
pub fn bool_storage_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // STORE bool: cast to uint8 before storing
        Store { index, value, ranges } if value.dtype().base().is_bool() => {
            let uint8_dtype = value.dtype().with_base(ScalarDType::UInt8);
            Some(UOp::store_with_ranges(index.clone(), UOp::cast(value.clone(), uint8_dtype), ranges.clone()))
        },

        // LOAD bool: load as uint8, then cast to bool
        load @ Load { buffer, index } if load.dtype().base().is_bool() => {
            let uint8_dtype = load.dtype().with_base(ScalarDType::UInt8);
            let uint8_load = UOp::load().buffer(buffer.clone()).index(index.clone()).dtype(uint8_dtype).call();
            Some(UOp::cast(uint8_load, load.dtype()))
        },
    }
}

/// Post-devectorize rendering patterns (devectorizer.py:258-275).
/// Called during codegen, NOT part of pm_devectorize.
pub fn pm_render() -> TypedPatternMatcher {
    crate::patterns! {
        // Vector CONST → VECTORIZE of scalar CONST (devectorizer.py:260-261)
        c @ Const(_) if c.dtype().vcount() > 1 => |c| {
            let vcount = c.dtype().vcount();
            let Op::Const(cv) = c.op() else { return None };
            let result_dtype = DType::Scalar(c.dtype().base());
            let scalar_const = UOp::const_(result_dtype, cv.0);
            let elements: SmallVec<[Arc<UOp>; 4]> = (0..vcount)
                .map(|_| scalar_const.clone())
                .collect();
            Some(UOp::vectorize(elements))
        },

        // VCONST → VECTORIZE of scalar CONSTs (devectorizer.py:262)
        vc @ VConst { values } => |vc, values| {
            // VConst stores different values per lane - convert each to scalar CONST
            let result_dtype = DType::Scalar(vc.dtype().base());
            let elements: SmallVec<[Arc<UOp>; 4]> = values.iter()
                .map(|v| UOp::const_(result_dtype.clone(), *v))
                .collect();
            Some(UOp::vectorize(elements))
        },

        // CAT → VECTORIZE (CAT can't be rendered)
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

        // GEP(PTRCAT) → reorder (must be before multi-index GEP)
        Gep { vector: PtrCat { sources }, indices } => {
            let reordered: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .filter_map(|&i| sources.get(i).cloned())
                .collect();
            if reordered.len() != indices.len() { return None; }
            Some(if reordered.len() == 1 { reordered[0].clone() } else { UOp::ptrcat().sources(reordered.to_vec()).call() })
        },

        // Multi-index GEP → VECTORIZE (fallback, must be last GEP pattern)
        Gep { vector, indices } if indices.len() > 1 => |vector, indices| {
            let geps: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .map(|&i| UOp::gep(vector.clone(), vec![i]))
                .collect();
            Some(UOp::vectorize(geps))
        },

        // Single-element unwrap
        Vectorize { elements } if elements.len() == 1 => Some(elements[0].clone()),
        PtrCat { sources } if sources.len() == 1 => Some(sources[0].clone()),
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

/// Vector ALU → VECTORIZE of scalar ALU (devectorizer.py:219-223).
/// LLVM SLP can re-vectorize when beneficial.
pub fn no_vectorized_alu() -> TypedPatternMatcher {
    crate::patterns! {
        for op in binary [*] {
            result @ op(lhs, rhs) if result.dtype().vcount() > 1 => |result, lhs, rhs| {
                let vcount = result.dtype().vcount();
                let result_dtype = result.dtype().scalar().map(DType::Scalar)?;
                let alus: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|i| {
                    let gep_lhs = UOp::gep(lhs.clone(), vec![i]);
                    let gep_rhs = UOp::gep(rhs.clone(), vec![i]);
                    UOp::new(Op::Binary(op, gep_lhs, gep_rhs), result_dtype.clone())
                }).collect();
                Some(UOp::vectorize(alus))
            },
        },

        for op in unary [*] {
            result @ op(src) if result.dtype().vcount() > 1 => |result, src| {
                let vcount = result.dtype().vcount();
                let result_dtype = result.dtype().scalar().map(DType::Scalar)?;
                let alus: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|i| {
                    let gep_src = UOp::gep(src.clone(), vec![i]);
                    UOp::new(Op::Unary(op, gep_src), result_dtype.clone())
                }).collect();
                Some(UOp::vectorize(alus))
            },
        },

        Cast { src, .. } if src.dtype().vcount() > 1 => |src| {
            let vcount = src.dtype().vcount();
            let src_result_dtype = src.dtype().scalar().map(DType::Scalar)?;
            let casts: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|i| {
                let gep_src = UOp::gep(src.clone(), vec![i]);
                UOp::cast(gep_src, src_result_dtype.clone())
            }).collect();
            Some(UOp::vectorize(casts))
        },

        // BITCAST devectorization (Tinygrad devectorizer.py:254 includes BITCAST)
        BitCast { src, dtype } if src.dtype().vcount() > 1 => |src, dtype| {
            let vcount = src.dtype().vcount();
            let result_dtype = dtype.scalar().map(DType::Scalar)?;
            let bitcasts: SmallVec<[Arc<UOp>; 4]> = (0..vcount).map(|i| {
                let gep_src = UOp::gep(src.clone(), vec![i]);
                UOp::bitcast(gep_src, result_dtype.clone())
            }).collect();
            Some(UOp::vectorize(bitcasts))
        },

        // Skip WHERE(cond, t, Invalid) - used for image indexing (devectorizer.py:221)
        Where(cond, t, f) if cond.dtype().vcount() > 1 && !matches!(f.op(), Op::Invalid) => |cond, t, f| {
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

        // MulAcc (FMA) - required for matmul tiling with split_load
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

/// Normalize VECTORIZE and GEP for rendering (subset of pm_render).
pub fn pm_vectorize_normalize() -> TypedPatternMatcher {
    crate::patterns! {
        Gep { vector, indices } if indices.len() > 1 => |vector, indices| {
            let geps: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .map(|&i| UOp::gep(vector.clone(), vec![i]))
                .collect();
            Some(UOp::vectorize(geps))
        },

        Gep { vector, indices } if vector.dtype().vcount() == 1 && indices.len() == 1 && indices[0] == 0
            ~> |vector| Arc::clone(vector),

        Vectorize { elements } if elements.len() == 1 => |elements| Some(elements[0].clone()),
    }
}

// ============================================================================
// Devectorize Patterns (devectorizer.py:250-256)
// ============================================================================

/// Combined devectorize patterns: cast_after, ALU, WMMA, buffer/index devectorization.
pub fn devectorize_patterns() -> TypedPatternMatcher {
    cast_after_pattern() + no_vectorized_alu() + no_vectorized_wmma() + devectorize_buf_and_index_patterns()
}

/// WMMA devectorization (devectorizer.py:208-217).
fn no_vectorized_wmma() -> TypedPatternMatcher {
    crate::patterns! {
        wmma @ Wmma { a, b, c, metadata } if wmma.dtype().vcount() > wmma_expected_size(metadata)
            => devectorize_wmma(wmma, a, b, c, metadata),
    }
}

fn wmma_expected_size(metadata: &WmmaMetadata) -> usize {
    metadata.upcast_axes.iter().map(|(_, size)| size).product::<usize>().max(1)
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

    // Split each source by out_sz
    let sources = [a, b, c];
    let mut tsrcs: Vec<Vec<Arc<UOp>>> = vec![Vec::new(), Vec::new(), Vec::new()];

    for (i, src) in sources.iter().enumerate() {
        let src_count = src.dtype().vcount();
        for grp in (0..src_count).step_by(out_sz) {
            let gep_indices: Vec<usize> = (grp..grp + out_sz.min(src_count - grp)).collect();
            tsrcs[i].push(UOp::gep((*src).clone(), gep_indices));
        }
    }

    // Verify all sources have same number of groups
    let num_groups = tsrcs[0].len();
    if tsrcs.iter().any(|t| t.len() != num_groups) {
        tracing::warn!("WMMA devectorization: mismatched source group counts");
        return None;
    }

    // Create new WMMA for each group, flatten with GEP
    let wmmas: Vec<Arc<UOp>> = (0..num_groups)
        .map(|g| UOp::wmma(tsrcs[0][g].clone(), tsrcs[1][g].clone(), tsrcs[2][g].clone(), metadata.clone()))
        .collect();

    let wmma_ex: SmallVec<[Arc<UOp>; 4]> =
        wmmas.iter().flat_map(|w| (0..out_sz).map(move |i| UOp::gep(w.clone(), vec![i]))).collect();

    Some(UOp::vectorize(wmma_ex))
}

/// AFTER(CAST(x), deps) → CAST(AFTER(x, deps)) - allows cast to be optimized independently.
fn cast_after_pattern() -> TypedPatternMatcher {
    crate::patterns! {
        After { passthrough: Cast { src, dtype }, deps }
            => |src, dtype, deps| {
                let new_after = UOp::after(src.clone(), deps.clone());
                Some(UOp::cast(new_after, dtype.clone()))
            },
    }
}

/// LOCAL/REG buffer devectorization (devectorizer.py:241-248).
fn devectorize_buf_and_index_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // DEFINE_LOCAL/REG with vector pointer → scalar pointer + CAST
        def if is_vectorized_define_local_or_reg(def) => no_vectorized_buf(def),

        // INDEX(CAST(DEFINE_LOCAL/REG), scalar_idx) → scaled vector index
        Index { buffer: Cast { src: buf, dtype: cast_dtype }, indices, gate }
            if is_scalar_index(indices) && is_vectorized_local_reg_cast(buf, cast_dtype)
            => no_vectorized_index(buf, indices, gate, cast_dtype),

        // INDEX(BROADCAST(CAST(...)), scalar_idx)
        Index { buffer: Vectorize { elements }, indices, gate }
            if is_scalar_index(indices) && is_vectorized_broadcast_cast(elements)
            => no_vectorized_index_broadcast(elements, indices, gate),

        // INDEX(GEP(CAST(...)), scalar_idx)
        Index { buffer: Gep { vector: Cast { src: buf, dtype: cast_dtype }, indices: gep_indices }, indices, gate }
            if is_scalar_index(indices) && is_vectorized_local_reg_cast(buf, cast_dtype)
            => no_vectorized_index_gep(buf, indices, gate, cast_dtype, gep_indices),
    }
}

fn is_vectorized_define_local_or_reg(uop: &Arc<UOp>) -> bool {
    match uop.op() {
        Op::DefineLocal(_) | Op::DefineReg { .. } => has_vectorized_ptr(uop),
        _ => false,
    }
}

fn has_vectorized_ptr(uop: &Arc<UOp>) -> bool {
    uop.ptrdtype().is_some_and(|(base, _, _)| base.vcount() > 1)
}

fn is_scalar_index(indices: &SmallVec<[Arc<UOp>; 4]>) -> bool {
    indices.first().is_some_and(|idx| idx.dtype().vcount() == 1)
}

fn is_vectorized_local_reg_cast(buf: &Arc<UOp>, cast_dtype: &DType) -> bool {
    let DType::Ptr { base, .. } = cast_dtype else { return false };
    base.vcount() > 1 && is_define_local_or_reg_or_after(buf)
}

fn is_vectorized_broadcast_cast(elements: &SmallVec<[Arc<UOp>; 4]>) -> bool {
    let Some(first) = elements.first() else { return false };
    let Op::Cast { src: inner, dtype: cast_dtype } = first.op() else { return false };
    let DType::Ptr { base, .. } = cast_dtype else { return false };
    base.vcount() > 1 && is_define_local_or_reg_or_after(inner)
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

    let scalar_base = base.scalar()?;
    let new_size = size.map(|s| s * vcount);
    let scalar_ptr_dtype =
        DType::Ptr { base: Box::new(DType::Scalar(scalar_base)), addrspace, size: new_size, vcount: 1 };

    let scalar_def = buf.with_dtype(scalar_ptr_dtype);
    Some(UOp::cast(scalar_def, buf.dtype()))
}

/// INDEX(CAST(buf), scalar_idx) → INDEX(VECTORIZE([buf,...]), scaled_vec_idx) (devectorizer.py:228-231)
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

    let buf_broadcast = UOp::broadcast(buf.clone(), cnt);
    let idx_broadcast = UOp::broadcast(idx.clone(), cnt);
    let offset_vec = create_index_vector(cnt);
    let cnt_broadcast = UOp::broadcast(idx.const_like(cnt as i64), cnt);
    let final_idx = idx_broadcast.mul(&cnt_broadcast).add(&offset_vec);

    let buf_dtype = buf_broadcast.dtype();
    Some(UOp::new(
        Op::Index { buffer: buf_broadcast, indices: smallvec::smallvec![final_idx], gate: gate.clone() },
        buf_dtype,
    ))
}

fn create_index_vector(cnt: usize) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> = (0..cnt).map(|i| UOp::index_const(i as i64)).collect();
    UOp::vectorize(elements)
}

/// INDEX(BROADCAST(CAST(buf)), scalar_idx) → scaled vec idx (devectorizer.py:233-239)
fn no_vectorized_index_broadcast(
    elements: &SmallVec<[Arc<UOp>; 4]>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    gate: &Option<Arc<UOp>>,
) -> Option<Arc<UOp>> {
    let first = elements.first()?;
    let Op::Cast { src: buf, dtype: cast_dtype } = first.op() else { return None };
    let idx = indices.first()?;

    let DType::Ptr { base, .. } = cast_dtype else { return None };
    let cnt = base.vcount();
    let precnt = elements.len();
    if cnt <= 1 {
        return None;
    }

    let input_gep: Vec<usize> = vec![0; precnt];
    let gep_arg: Vec<usize> = (0..cnt).flat_map(|_| 0..precnt).collect();
    let sum_arg: Vec<i64> = (0..cnt).flat_map(|i| input_gep.iter().map(move |&y| (i + y) as i64)).collect();

    let total_cnt = cnt * precnt;
    let buf_broadcast = UOp::broadcast(buf.clone(), total_cnt);
    let idx_gep = UOp::gep(idx.clone(), gep_arg);
    let cnt_broadcast = UOp::broadcast(idx.const_like(cnt as i64), total_cnt);
    let sum_vec = create_const_index_vector(&sum_arg);
    let final_idx = idx_gep.mul(&cnt_broadcast).add(&sum_vec);

    let buf_dtype = buf_broadcast.dtype();
    Some(UOp::new(
        Op::Index { buffer: buf_broadcast, indices: smallvec::smallvec![final_idx], gate: gate.clone() },
        buf_dtype,
    ))
}

fn create_const_index_vector(values: &[i64]) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> = values.iter().map(|&v| UOp::index_const(v)).collect();
    UOp::vectorize(elements)
}

/// INDEX(GEP(CAST(buf)), scalar_idx) → scaled vec idx
fn no_vectorized_index_gep(
    buf: &Arc<UOp>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    gate: &Option<Arc<UOp>>,
    cast_dtype: &DType,
    gep_indices: &[usize],
) -> Option<Arc<UOp>> {
    let idx = indices.first()?;
    let DType::Ptr { base, .. } = cast_dtype else { return None };
    let cnt = base.vcount();
    let precnt = gep_indices.len();
    if cnt <= 1 {
        return None;
    }

    let input_gep: Vec<usize> = gep_indices.to_vec();
    let gep_arg: Vec<usize> = (0..cnt).flat_map(|_| 0..precnt).collect();
    let sum_arg: Vec<i64> = (0..cnt).flat_map(|i| input_gep.iter().map(move |&y| (i + y) as i64)).collect();

    let total_cnt = cnt * precnt;
    let buf_broadcast = UOp::broadcast(buf.clone(), total_cnt);
    let idx_gep = UOp::gep(idx.clone(), gep_arg);
    let cnt_broadcast = UOp::broadcast(idx.const_like(cnt as i64), total_cnt);
    let sum_vec = create_const_index_vector(&sum_arg);
    let final_idx = idx_gep.mul(&cnt_broadcast).add(&sum_vec);

    let buf_dtype = buf_broadcast.dtype();
    Some(UOp::new(
        Op::Index { buffer: buf_broadcast, indices: smallvec::smallvec![final_idx], gate: gate.clone() },
        buf_dtype,
    ))
}

// ============================================================================
// Load Store Indexing Patterns (devectorizer.py:48-55)
// ============================================================================

/// INDEX(buf, x, true) → INDEX(buf, x, None)
pub fn load_store_indexing_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        index @ Index { buffer, indices, gate: Some(g) }
            if matches!(g.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Bool(true)))
            => Some(UOp::new(Op::Index { buffer: buffer.clone(), indices: indices.clone(), gate: None }, index.dtype())),
    }
}

// ============================================================================
// Add Loads Patterns (devectorizer.py:320-326)
// ============================================================================

/// Add LOAD to non-pointer INDEX, remove LOAD wrapper from STORE.
pub fn pm_add_loads() -> TypedPatternMatcher {
    crate::patterns! {
        // Add LOAD to non-ptr INDEX: INDEX(buf, idx) → LOAD(INDEX(buf, idx))
        // Skip if dtype is already Ptr (devectorizer.py:322-323)
        idx @ Index { buffer, .. } if !is_ptr_or_image_dtype(&idx.dtype()) => {
            let new_idx = idx.with_dtype(buffer.dtype());
            let load_dtype = DType::Scalar(idx.dtype().base());
            Some(UOp::load().buffer(buffer.clone()).index(new_idx).dtype(load_dtype).call())
        },

        // Remove LOAD wrapper from STORE: STORE(LOAD(x), ...) → STORE(x, ...)
        // (devectorizer.py:325)
        Store { index: Load { index: inner_idx, .. }, value, ranges }
            => Some(UOp::store_with_ranges(inner_idx.clone(), value.clone(), ranges.clone())),
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
pub fn pm_wmma_accumulate() -> TypedPatternMatcher {
    crate::patterns! {
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

/// expand_index, GEP movement, PTRCAT distribution.
pub fn load_store_folding_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // expand_index: vector INDEX → PTRCAT grouping
        index if is_vector_index(index) => expand_vector_index(index),

        // GEP after LOAD: LOAD(GEP(x)) → GEP(LOAD(x))
        load @ Load { buffer, index: Gep { vector, indices } }
            => move_gep_after_load(load, buffer, vector, indices),

        // GEP on STORE: STORE(GEP(x), data) → STORE(x, GEP⁻¹(data))
        Store { index: Gep { vector, indices }, value, ranges }
            => move_gep_on_store(vector, indices, value, ranges),

        // PTRCAT after LOAD: LOAD(PTRCAT(a,b)) → CAT(LOAD(a), LOAD(b))
        load @ Load { buffer, index: ptrcat @ PtrCat { sources } }
            => distribute_ptrcat_load(load, buffer, ptrcat, sources),

        // PTRCAT after STORE
        Store { index: PtrCat { sources }, value, ranges }
            => distribute_ptrcat_store(sources, value, ranges),
    }
}

// ============================================================================
// Correct Load Store Patterns (devectorizer.py:198-203)
// ============================================================================

/// LOAD/STORE(CAST(INDEX)) → split by device fold lengths.
pub fn correct_load_store_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        Load { buffer, index: Cast { src: idx @ Index { buffer: _, .. }, dtype: cast_dtype } }
            => split_load(buffer, idx, cast_dtype),

        Store { index: Cast { src: idx @ Index { buffer: idx_buffer, .. }, dtype: cast_dtype }, value, ranges }
            => split_store(idx_buffer, idx, cast_dtype, value, ranges),
    }
}

// ============================================================================
// Pattern Predicates
// ============================================================================

fn is_define_or_after(uop: &Arc<UOp>) -> bool {
    matches!(uop.unwrap_after().op(), Op::DefineLocal(_) | Op::DefineReg { .. } | Op::DefineGlobal(_))
}

/// Matches INDEX(VECTORIZE(Defines.or_after()), vec_idx) only.
/// Tinygrad devectorizer.py:115 - expand_index only matches VECTORIZE of defines.
fn is_vector_index(uop: &Arc<UOp>) -> bool {
    let Op::Index { buffer, indices, .. } = uop.op() else { return false };

    // Index must be vector
    let Some(idx) = indices.first() else { return false };
    if idx.dtype().vcount() <= 1 {
        return false;
    }

    // Buffer MUST be VECTORIZE (not bare buffer) - Tinygrad line 115
    let Op::Vectorize { elements } = buffer.op() else { return false };

    // Elements must be Defines.or_after()
    if elements.is_empty() || !elements.iter().all(is_define_or_after) {
        return false;
    }

    // Don't vectorize bool loads (LLVM i1 vector load is broken)
    !elements[0].dtype().base().is_bool()
}

// ============================================================================
// GEP Movement Patterns (devectorizer.py:106-120)
// ============================================================================

/// LOAD(GEP(ptr)) → GEP(LOAD(ptr)). LOAD dtype = ld.dtype.scalar().vec(gep.dtype.count)
fn move_gep_after_load(
    load: &Arc<UOp>,
    buffer: &Arc<UOp>,
    gep_inner: &Arc<UOp>,
    gep_indices: &[usize],
) -> Option<Arc<UOp>> {
    let gep_count = gep_indices.len();
    let scalar_base = load.dtype().scalar()?;
    let inner_load_dtype = if gep_count > 1 {
        DType::Vector { scalar: scalar_base, count: gep_count }
    } else {
        DType::Scalar(scalar_base)
    };

    let inner_load = UOp::load().buffer(buffer.clone()).index(gep_inner.clone()).dtype(inner_load_dtype).call();
    Some(UOp::gep(inner_load, gep_indices.to_vec()))
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

    let reordered_value = UOp::gep(value.clone(), inverse_indices);
    Some(UOp::store_with_ranges(gep_inner.clone(), reordered_value, ranges.clone()))
}

// ============================================================================
// expand_index (devectorizer.py:59-95)
// ============================================================================

/// Vector INDEX → grouped PTRCAT. Generates scalar indices, simplifies, groups by root+offset.
fn expand_vector_index(index: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate } = index.op() else { return None };
    let vec = indices.first()?;
    let count = vec.dtype().vcount();

    let buf = if let Op::Vectorize { elements } = buffer.op() { elements.first()?.clone() } else { buffer.clone() };

    // Generate scalar INDEX ops and simplify
    let scalar_indices: Vec<_> = (0..count)
        .map(|i| {
            UOp::new(
                Op::Index {
                    buffer: buf.clone(),
                    indices: smallvec::smallvec![UOp::gep(vec.clone(), vec![i])],
                    gate: gate.clone(),
                },
                buf.dtype().clone(),
            )
        })
        .collect();

    let midx =
        graph_rewrite_bottom_up(&(symbolic() + load_store_indexing_patterns()), UOp::sink(scalar_indices), &mut ());
    let Op::Sink { sources } = midx.op() else { return None };

    // Extract (valid, root, offset) for each lane
    let mut offsets_by_root: HashMap<(u64, u64), HashMap<i64, Vec<usize>>> = HashMap::new();

    for (lane, idx_op) in sources.iter().enumerate() {
        let Op::Index { indices: simp_indices, .. } = idx_op.op() else { continue };
        let idx = simp_indices.first()?.get_idx();
        let valid = simp_indices.first()?.get_valid();

        let (root, offset) = match idx.op() {
            // Invalid grouped separately (devectorizer.py:72-77)
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

        let key = (valid.content_hash(), root.content_hash());
        offsets_by_root.entry(key).or_default().entry(offset).or_default().push(lane);
    }

    // Group consecutive offsets and build PTRCAT
    let mut ret = Vec::new();
    let mut idxs: Vec<Option<usize>> = vec![None; count];
    let mut global_offset = 0;

    for offsets in offsets_by_root.values() {
        let groups = group_consecutive_offsets_from_map(offsets);
        for (_, lanes) in groups {
            let lidx = sources[lanes[0]].clone();
            let ptr = if lanes.len() > 1 { UOp::cast(lidx, make_vec_ptr_dtype(&buf, lanes.len())) } else { lidx };
            for (i, &lane) in lanes.iter().enumerate() {
                idxs[lane] = Some(global_offset + i);
            }
            ret.push(ptr);
            global_offset += lanes.len();
        }
    }

    if idxs.iter().any(|x| x.is_none()) {
        return None;
    }

    // Create PTRCAT with vec(global_offset) of scalar pointers
    let DType::Ptr { base, addrspace, size, .. } = buf.dtype().clone() else { return None };
    let scalar_ptr = DType::Ptr { base: Box::new(DType::Scalar(base.scalar()?)), addrspace, size, vcount: 1 };
    let ptrcat_dtype = scalar_ptr.vec(global_offset);
    let ptrcat = UOp::ptrcat().sources(ret).dtype(ptrcat_dtype).call();
    let gep_indices: Vec<usize> = idxs.into_iter().map(|x| x.unwrap()).collect();

    Some(UOp::gep(ptrcat, gep_indices))
}

/// Groups offsets where `offset - index` is constant. Breaks on multi-lane offsets.
fn group_consecutive_offsets_from_map(offsets_map: &HashMap<i64, Vec<usize>>) -> Vec<(i64, Vec<usize>)> {
    if offsets_map.is_empty() {
        return vec![];
    }

    let sorted: Vec<_> = offsets_map.keys().copied().sorted().collect();
    sorted
        .iter()
        .copied()
        .enumerate()
        .chunk_by(|(idx, offset)| (offsets_map[offset].len() == 1, offset - (*idx as i64)))
        .into_iter()
        .map(|(_, group)| {
            let offsets_in_group: Vec<_> = group.collect();
            let first_offset = offsets_in_group[0].1;
            let lanes: Vec<usize> =
                offsets_in_group.iter().flat_map(|(_, offset)| offsets_map[offset].iter().copied()).collect();
            (first_offset, lanes)
        })
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
fn distribute_ptrcat_load(
    _load: &Arc<UOp>,
    buffer: &Arc<UOp>,
    ptrcat: &Arc<UOp>,
    sources: &[Arc<UOp>],
) -> Option<Arc<UOp>> {
    let loads: Vec<Arc<UOp>> = sources
        .iter()
        .map(|ptr| {
            let load_dtype = match ptr.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other.clone(),
            };
            UOp::load().buffer(buffer.clone()).index(ptr.clone()).dtype(load_dtype).call()
        })
        .collect();

    let ptrcat_vcount = ptrcat.dtype().vcount();
    let base_scalar = ptrcat.dtype().base();
    let cat_dtype = DType::Scalar(base_scalar).vec(ptrcat_vcount);

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

        // Safety check: GEP indices must not exceed value's element count
        // If this triggers, there's a bug in earlier passes that created mismatched shapes
        if offset + ptr_count > value_vcount {
            panic!(
                "ICE: incorrect Morok IR produced; PTRCAT size mismatch in distribute_ptrcat_store: \
                 offset={}, ptr_count={}, value_vcount={} (expected value_vcount >= {})",
                offset,
                ptr_count,
                value_vcount,
                offset + ptr_count
            );
        }

        let gep_indices: Vec<usize> = (offset..offset + ptr_count).collect();
        let store_value = UOp::gep(value.clone(), gep_indices);
        stores.push(UOp::store_with_ranges(ptr.clone(), store_value, ranges.clone()));
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

/// Device fold lengths: Image=[4], Default=[4,2,1]. TODO: device-specific (DSP, AMX).
fn get_device_fold_lengths(load_dtype: &DType) -> Vec<usize> {
    if let DType::Ptr { base, .. } = load_dtype
        && matches!(base.as_ref(), DType::Image { .. })
    {
        return vec![4, 1];
    }
    vec![4, 2, 1]
}

/// Split LOAD by fold length divisibility.
fn split_load(buffer: &Arc<UOp>, idx: &Arc<UOp>, cast_dtype: &DType) -> Option<Arc<UOp>> {
    let Op::Index { indices, .. } = idx.op() else {
        return None;
    };

    let sz = match cast_dtype {
        DType::Ptr { size: Some(sz), .. } => *sz,
        DType::Ptr { base, .. } => base.vcount(),
        _ => return None,
    };

    if sz <= 1 {
        return None;
    }

    tracing::debug!(sz = sz, "split_load");

    let scalar_base = buffer.dtype().base();
    let load_dtype = DType::Vector { scalar: scalar_base, count: sz };
    let mut lengths = get_device_fold_lengths(&load_dtype);

    if let Some(offset) = indices.first() {
        lengths.retain(|&len| offset_divides_evenly(offset, len));
    }
    if !lengths.contains(&1) {
        lengths.push(1);
    }

    let mut chunks = Vec::new();
    let mut pos = 0usize;

    while pos < sz {
        for &fold_len in &lengths {
            if pos + fold_len > sz {
                continue;
            }

            let chunk_idx = if pos == 0 { idx.clone() } else { offset_index(idx, pos as i64) };
            let chunk_ptr =
                if fold_len > 1 { UOp::cast(chunk_idx, make_vec_ptr_dtype(buffer, fold_len)) } else { chunk_idx };
            let chunk_dtype = if fold_len > 1 {
                DType::Vector { scalar: scalar_base, count: fold_len }
            } else {
                DType::Scalar(scalar_base)
            };

            chunks.push(UOp::load().buffer(buffer.clone()).index(chunk_ptr).dtype(chunk_dtype).call());
            pos += fold_len;
            break;
        }
    }

    if chunks.len() <= 1 {
        return None;
    }

    let cat_dtype = DType::Vector { scalar: scalar_base, count: sz };
    Some(UOp::cat().sources(chunks).dtype(cat_dtype).call())
}

fn split_store(
    idx_buffer: &Arc<UOp>,
    idx: &Arc<UOp>,
    cast_dtype: &DType,
    value: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
) -> Option<Arc<UOp>> {
    let Op::Index { indices, .. } = idx.op() else {
        return None;
    };

    // Use the VALUE's vcount - we can only extract elements that exist in the value.
    // Tinygrad uses the pointer type's count (devectorizer.py:132) but ensures earlier
    // passes make value and pointer types match.
    let sz = value.dtype().vcount();

    // ICE check: pointer type should not expect more elements than value provides
    let ptr_sz = match cast_dtype {
        DType::Ptr { size: Some(sz), .. } => *sz,
        DType::Ptr { base, .. } => base.vcount(),
        _ => 1,
    };
    if ptr_sz > sz {
        panic!(
            "ICE: incorrect Morok IR produced; split_store pointer/value size mismatch: \
             ptr expects {} elements but value only has {}",
            ptr_sz, sz
        );
    }

    if sz <= 1 {
        return None;
    }

    tracing::debug!(sz = sz, "split_store");

    let mut lengths = get_device_fold_lengths(&value.dtype());
    if let Some(offset) = indices.first() {
        lengths.retain(|&len| offset_divides_evenly(offset, len));
    }
    if !lengths.contains(&1) {
        lengths.push(1);
    }

    let mut stores = Vec::new();
    let mut pos = 0usize;

    while pos < sz {
        for &fold_len in &lengths {
            if pos + fold_len > sz {
                continue;
            }

            let chunk_idx = if pos == 0 { idx.clone() } else { offset_index(idx, pos as i64) };
            let chunk_ptr =
                if fold_len > 1 { UOp::cast(chunk_idx, make_vec_ptr_dtype(idx_buffer, fold_len)) } else { chunk_idx };
            let gep_indices: Vec<usize> = (pos..pos + fold_len).collect();
            let chunk_value = UOp::gep(value.clone(), gep_indices);
            stores.push(UOp::store_with_ranges(chunk_ptr, chunk_value, ranges.clone()));

            pos += fold_len;
            break;
        }
    }

    if stores.len() <= 1 {
        return None;
    }

    tracing::debug!(num_stores = stores.len(), "split_store");
    Some(UOp::group(stores.into_iter().collect()))
}

/// Conservative: false for unknown expressions (devectorizer.py:156).
fn offset_divides_evenly(offset: &Arc<UOp>, len: usize) -> bool {
    if len <= 1 {
        return true;
    }
    match offset.op() {
        Op::Const(cv) => matches!(cv.0, ConstValue::Int(n) if n % (len as i64) == 0),
        Op::Binary(BinaryOp::Mul, left, right) => {
            let check = |c: &Arc<UOp>| matches!(c.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Int(n) if n >= len as i64 && n % (len as i64) == 0));
            check(left) || check(right)
        }
        Op::Binary(BinaryOp::Add, left, right) => offset_divides_evenly(left, len) && offset_divides_evenly(right, len),
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

    UOp::new(Op::Index { buffer: buffer.clone(), indices: new_indices, gate: gate.clone() }, idx.dtype())
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
pub fn pm_reduce() -> TypedPatternMatcher {
    crate::patterns! {
        // Convert REDUCE to accumulator pattern
        // Skip if ranges are empty (handled by codegen or identity)
        red @ Reduce { src, ranges, reduce_op } if !ranges.is_empty() => |red, src, ranges, reduce_op| {
            reduce_to_acc(red, src, ranges, reduce_op)
        },
    }
}

/// Convert ThreadScheduled ranges to Thread ranges.
///
/// Runs AFTER pm_reduce, matching Tinygrad's pm_add_gpudims ordering.
/// This ensures reduce_to_acc never sees Thread ranges in its input_ranges,
/// fixing incorrect accumulator placement for threaded reduce kernels.
///
/// # Implementation
///
/// Follows Tinygrad's `pm_add_gpudims` approach:
/// 1. Match on SINK (root node) to process the entire graph at once
/// 2. Collect all ThreadScheduled ranges from toposort
/// 3. Build a substitution map: ThreadScheduled → Thread
/// 4. Call `sink.substitute(&map)` to atomically replace all references
///
/// This is necessary because per-node pattern matching doesn't propagate
/// substitutions correctly through the DAG (same node referenced multiple times).
///
/// # Pipeline Position
///
/// ```text
/// 1. pm_reduce          → REDUCE → accumulator pattern (ThreadScheduled excluded from input_ranges)
/// 2. pm_add_thread_dims → ThreadScheduled → Thread
/// 3. pm_add_loads       → Add LOAD wrappers
/// ```
pub fn pm_add_thread_dims() -> TypedPatternMatcher {
    use morok_ir::UOpKey;

    crate::patterns! {
        // Match on SINK to process the entire graph at once (like Tinygrad's pm_add_gpudims)
        sink @ Sink { sources } => |sink, sources| {
            let _ = sources; // Suppress unused warning
            let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

            // Collect all ThreadScheduled ranges from toposort
            for node in sink.toposort() {
                if let Op::Range { axis_type, end, axis_id } = node.op()
                    && *axis_type == AxisType::ThreadScheduled {
                        let new_range = UOp::range_axis(end.clone(), *axis_id, AxisType::Thread);
                        debug!(old_id = node.id, new_id = new_range.id, "pm_add_thread_dims: converting ThreadScheduled -> Thread");
                        subs.insert(UOpKey(node.clone()), new_range);
                    }
            }

            if subs.is_empty() {
                debug!("pm_add_thread_dims: no ThreadScheduled ranges found");
                None // No transformation needed
            } else {
                debug!(num_substitutions = subs.len(), "pm_add_thread_dims: applying substitutions");
                let result = sink.substitute(&subs);
                debug!(old_sink_id = sink.id, new_sink_id = result.id, "pm_add_thread_dims: substitution complete");
                Some(result)
            }
        },
    }
}

/// Filter non-RANGE ops from END/REDUCE/STORE ranges.
///
/// Based on Tinygrad's `pm_flatten_range` (simplify.py:7-16).
///
/// # Problem
///
/// The `dead_loop_patterns()` in symbolic/patterns.rs converts trivial RANGE(end=1)
/// nodes to CONST(0) when vmin == vmax. When these RANGEs are referenced in END.ranges,
/// the graph rewrite substitutes the CONST, causing END to have `ranges: [CONST, CONST]`
/// instead of `ranges: [RANGE, RANGE]`. This breaks linearization.
///
/// # Solution
///
/// This pattern filters END/REDUCE/STORE ranges to keep only actual RANGE ops,
/// removing any CONST substitutions that occurred during symbolic simplification.
///
/// ```python
/// # Tinygrad simplify.py:7-16
/// def flatten_range(r:UOp):
///   off = range_start[r.op]
///   rngs = r.src[off:]
///   if not len(rngs): return None
///   new_rngs = [x for x in UOp.sink(*rngs).toposort() if x.op is Ops.RANGE]
///   return r.replace(src=r.src[:off]+tuple(new_rngs))
/// ```
pub fn pm_flatten_range() -> TypedPatternMatcher {
    /// Extract actual RANGE ops from a list of range sources.
    /// Creates a sink, toposorts, and filters to only RANGE ops.
    fn extract_ranges(ranges: &SmallVec<[Arc<UOp>; 4]>) -> SmallVec<[Arc<UOp>; 4]> {
        if ranges.is_empty() {
            return SmallVec::new();
        }
        let sink = UOp::sink(ranges.iter().cloned().collect());
        sink.toposort()
            .into_iter()
            .filter(|node| matches!(node.op(), Op::Range { .. }))
            .collect()
    }

    /// Check if ranges changed (optimization to avoid unnecessary rewrites).
    fn ranges_unchanged(original: &SmallVec<[Arc<UOp>; 4]>, filtered: &SmallVec<[Arc<UOp>; 4]>) -> bool {
        original.len() == filtered.len()
            && original.iter().zip(filtered.iter()).all(|(a, b)| a.id == b.id)
    }

    crate::patterns! {
        // END: Filter ranges to only RANGE ops
        end @ End { computation, ranges } => |end, computation, ranges| {
            let actual_ranges = extract_ranges(ranges);

            if ranges_unchanged(ranges, &actual_ranges) {
                return None; // No change needed
            }

            tracing::debug!(
                end_id = end.id,
                original_len = ranges.len(),
                filtered_len = actual_ranges.len(),
                "pm_flatten_range: filtering END ranges"
            );

            if actual_ranges.is_empty() {
                // All ranges were non-RANGE (e.g., all CONST) - unwrap END
                Some(computation.clone())
            } else {
                Some(UOp::end(computation.clone(), actual_ranges))
            }
        },

        // REDUCE: Filter ranges to only RANGE ops
        reduce @ Reduce { src, ranges, reduce_op } => |reduce, src, ranges, reduce_op| {
            let actual_ranges = extract_ranges(ranges);

            if ranges_unchanged(ranges, &actual_ranges) {
                return None;
            }

            tracing::debug!(
                reduce_id = reduce.id,
                original_len = ranges.len(),
                filtered_len = actual_ranges.len(),
                "pm_flatten_range: filtering REDUCE ranges"
            );

            Some(UOp::new(
                Op::Reduce { src: src.clone(), ranges: actual_ranges, reduce_op: *reduce_op },
                reduce.dtype(),
            ))
        },

        // STORE: Filter ranges to only RANGE ops
        Store { index, value, ranges } => |index, value, ranges| {
            let actual_ranges = extract_ranges(ranges);

            if ranges_unchanged(ranges, &actual_ranges) {
                return None;
            }

            tracing::debug!(
                original_len = ranges.len(),
                filtered_len = actual_ranges.len(),
                "pm_flatten_range: filtering STORE ranges"
            );

            Some(UOp::store_with_ranges(index.clone(), value.clone(), actual_ranges))
        },
    }
}

/// Convert a REDUCE operation to explicit accumulator pattern.
///
/// Follows Tinygrad's approach (devectorizer.py:reduce_to_acc):
/// 1. Find input_ranges (outer ranges that are NOT the reduce ranges)
/// 2. Initialize accumulator BEFORE reduce loop (depends only on input_ranges)
/// 3. Access accumulator INSIDE reduce loop (depends on reduce ranges)
/// 4. Read final value AFTER reduce loop completes
///
/// Note: This creates INDEX ops with element dtype. pm_add_loads will:
/// 1. Wrap INDEX with LOAD for values used in arithmetic
/// 2. STORE cleanup will remove LOAD from STORE's index position
fn reduce_to_acc(
    red: &Arc<UOp>,
    src: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    reduce_op: &ReduceOp,
) -> Option<Arc<UOp>> {
    // Use the REDUCE's actual output dtype (could be vec2 if upcasted)
    // Tinygrad: acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, ...), ...)
    let result_dtype = red.dtype();

    // 1. Find input_ranges: outer ranges that are NOT the reduce ranges
    // These are ranges from the input that we need to preserve (nested loops)
    let reduce_range_set: HashSet<u64> = ranges.iter().map(|r| r.id).collect();

    // Get all ranges from input's toposort that aren't reduce ranges or ended
    let topo = src.toposort();
    let ended_ranges: HashSet<u64> =
        topo.iter()
            .filter_map(|node| {
                if let Op::End { ranges: ended, .. } = node.op() { Some(ended.iter().map(|r| r.id)) } else { None }
            })
            .flatten()
            .collect();

    let input_ranges: SmallVec<[Arc<UOp>; 4]> = topo
        .iter()
        .filter(|node| {
            if let Op::Range { axis_type, .. } = node.op() {
                // Exclude parallel dispatch axes (Thread, ThreadScheduled, Global, Local, Warp)
                // These don't represent sequential iterations for accumulator placement.
                // This is critical for Tinygrad alignment: pm_reduce runs BEFORE pm_add_gpudims,
                // so Thread axes shouldn't exist yet. ThreadScheduled exists but is excluded.
                !axis_type.is_parallel() && !reduce_range_set.contains(&node.id) && !ended_ranges.contains(&node.id)
            } else {
                false
            }
        })
        .cloned()
        .collect();

    debug!(
        reduce_id = red.id,
        reduce_ranges = ?ranges.iter().map(|r| r.id).collect::<Vec<_>>(),
        input_ranges = ?input_ranges.iter().map(|r| r.id).collect::<Vec<_>>(),
        ended_ranges = ?ended_ranges.iter().collect::<Vec<_>>(),
        src_id = src.id,
        "reduce_to_acc: processing REDUCE"
    );

    // 2. Create identity element for the reduction
    let identity = reduce_identity(*reduce_op, result_dtype.clone());

    // 3. Create DEFINE_REG accumulator with explicit element type
    let acc = UOp::define_reg_typed(1, result_dtype.clone());

    // 4. Create index into accumulator (always index 0 for single-element acc)
    let zero_idx = UOp::index_const(0);

    // 5. Initialize accumulator BEFORE the reduce loop
    // If there are outer ranges, initialization happens after those start
    // If no outer ranges, initialization happens directly (no AFTER needed)
    let store_init = if input_ranges.is_empty() {
        // No outer ranges: acc.index(0).store(identity)
        let idx_init = UOp::new(
            Op::Index { buffer: acc.clone(), indices: smallvec::smallvec![zero_idx.clone()], gate: None },
            result_dtype.clone(),
        );
        idx_init.store_value(identity)
    } else {
        // Has outer ranges: acc.after(*input_ranges).index(0).store(identity)
        let acc_after_outer = UOp::after(acc.clone(), input_ranges);
        let idx_init = UOp::new(
            Op::Index { buffer: acc_after_outer, indices: smallvec::smallvec![zero_idx.clone()], gate: None },
            result_dtype.clone(),
        );
        idx_init.store_value(identity)
    };

    // 6. Create accumulator access INSIDE the reduce loop
    // acc.after(store_init, *reduce_range).index(0)
    let mut loop_deps: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![store_init.clone()];
    loop_deps.extend(ranges.iter().cloned());
    let acc_inside_loop = UOp::after(acc.clone(), loop_deps);

    let idx_loop = UOp::new(
        Op::Index { buffer: acc_inside_loop.clone(), indices: smallvec::smallvec![zero_idx.clone()], gate: None },
        result_dtype.clone(),
    );

    // 7. Apply reduce operation: new_val = op(idx_loop, src)
    let new_val = apply_reduce_binary(*reduce_op, idx_loop.clone(), src.clone(), &result_dtype);

    // 8. Store new value back to accumulator (idx_loop contains the INDEX, not acc directly)
    let store_loop = idx_loop.store_value(new_val);

    // 9. End the reduce ranges after the store
    let end = UOp::end(store_loop.clone(), ranges.clone());

    // 10. Final value AFTER loop completes
    // acc.after(end).index(0)
    let acc_after_end = UOp::after(acc.clone(), smallvec::smallvec![end.clone()]);
    let idx_final = UOp::new(
        Op::Index { buffer: acc_after_end.clone(), indices: smallvec::smallvec![zero_idx], gate: None },
        result_dtype,
    );

    debug!(
        acc_id = acc.id,
        store_init_id = store_init.id,
        acc_inside_loop_id = acc_inside_loop.id,
        idx_loop_id = idx_loop.id,
        store_loop_id = store_loop.id,
        end_id = end.id,
        acc_after_end_id = acc_after_end.id,
        idx_final_id = idx_final.id,
        "reduce_to_acc: created accumulator pattern"
    );

    Some(idx_final)
}

/// Apply binary reduce operation between two values.
fn apply_reduce_binary(reduce_op: ReduceOp, a: Arc<UOp>, b: Arc<UOp>, dtype: &DType) -> Arc<UOp> {
    match reduce_op {
        ReduceOp::Add => UOp::new(Op::Binary(BinaryOp::Add, a, b), dtype.clone()),
        ReduceOp::Mul => UOp::new(Op::Binary(BinaryOp::Mul, a, b), dtype.clone()),
        ReduceOp::Max => UOp::new(Op::Binary(BinaryOp::Max, a, b), dtype.clone()),
        ReduceOp::Min => {
            // Min(a, b) = Where(a < b, a, b)
            let cond_dtype = DType::Bool.vec(dtype.vcount());
            let cond = UOp::new(Op::Binary(BinaryOp::Lt, a.clone(), b.clone()), cond_dtype);
            UOp::try_where(cond, a, b).expect("WHERE construction should succeed")
        }
    }
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
