//! Devectorize pass for contiguous memory access optimization.
//!
//! Transforms vectorized INDEX into grouped consecutive accesses (PTRCAT) after expansion.
//!
//! # Multi-Pass Architecture
//!
//! The devectorize pass uses a multi-pass architecture to ensure pattern dependencies
//! are respected by the bottom-up rewrite engine:
//!
//! - **Phase 1**: Devectorize ALU/WMMA/buffers + expand vector INDEX → GEP(PTRCAT)
//! - **Phase 2**: Move GEP through LOAD/STORE → creates LOAD(PTRCAT) from LOAD(GEP(PTRCAT))
//! - **Phase 3**: Distribute PTRCAT through LOAD/STORE → LOAD(PTRCAT) → CAT(LOADs)
//! - **Phase 4**: Split loads/stores by device fold lengths + drop true gates
//!
//! This separation is necessary because the bottom-up rewrite engine processes children
//! first, then does fixed-point matching on the result. When `move_gep_after_load`
//! transforms `LOAD(GEP(PTRCAT))` to `GEP(LOAD(PTRCAT))`, the inner `LOAD(PTRCAT)` needs
//! to be matched by `distribute_ptrcat_load` in a subsequent pass.
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
use morok_ir::{BinaryOp, ConstValue, Op, ReduceOp, TernaryOp, UOp, WmmaMetadata};
use tracing::debug;

use crate::TypedPatternMatcher;
use smallvec::SmallVec;

/// Context for REDUCE transformation (Tinygrad devectorizer.py:280-281)
#[derive(Debug, Default)]
pub struct ReduceContext {
    pub acc_num: u32,
}

use crate::rewrite::graph_rewrite_bottom_up;
use crate::symbolic::patterns::symbolic;

// ============================================================================
// Main Entry Point
// ============================================================================

/// Run devectorize pass. Call AFTER `pre_expand`, BEFORE codegen.
///
/// Uses multi-pass architecture to ensure pattern dependencies are respected:
/// - Phase 1: Devectorize ALU/WMMA/buffers + expand vector INDEX → GEP(PTRCAT)
/// - Phase 2: Move GEP through LOAD/STORE (creates LOAD(PTRCAT) from LOAD(GEP(PTRCAT)))
/// - Phase 3: Distribute PTRCAT through LOAD/STORE (LOAD(PTRCAT) → CAT(LOADs))
/// - Phase 4: Split loads/stores by device fold lengths + drop true gates
///
/// Note: `bool_storage_patterns()` called separately (backend-specific).
/// Note: `pm_render()` should be applied AFTER this pass.
pub fn devectorize(ast: &Arc<UOp>) -> Arc<UOp> {
    // Phase 1: Devectorize ALU, WMMA, buffers, and expand vector indices
    let pm_phase1 = symbolic() + devectorize_patterns() + expand_index_patterns();
    let ast = graph_rewrite_bottom_up(&pm_phase1, ast.clone(), &mut ());

    // Phase 2: Move GEP through LOAD/STORE AND distribute PTRCAT
    // These must be in the same pass because:
    // - move_gep_after_load creates LOAD(PTRCAT) from LOAD(GEP(PTRCAT))
    // - distribute_ptrcat_load needs to match the newly created LOAD(PTRCAT)
    // - The rewrite engine's fixed-point matching allows this in a single pass
    let pm_phase2 = symbolic() + gep_movement_patterns() + ptrcat_distribution_patterns();
    let ast = graph_rewrite_bottom_up(&pm_phase2, ast, &mut ());

    // Phase 3: Split loads/stores by device fold lengths, drop true gates
    let pm_phase3 = symbolic() + correct_load_store_patterns() + load_store_indexing_patterns();
    graph_rewrite_bottom_up(&pm_phase3, ast, &mut ())
}

/// Bool LOAD/STORE via uint8. LLVM i1 can have garbage in upper bits.
pub fn bool_storage_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // STORE bool: cast to uint8 before storing
        Store { index, value, ranges } if value.dtype().base().is_bool() => {
            let uint8_dtype = value.dtype().with_base(ScalarDType::UInt8);
            Some(index.store_with_ranges(value.cast(uint8_dtype), ranges.clone()))
        },

        // LOAD bool: load as uint8, then cast to bool
        load @ Load { buffer, index } if load.dtype().base().is_bool() => {
            let uint8_dtype = load.dtype().with_base(ScalarDType::UInt8);
            let uint8_load = UOp::load().buffer(buffer.clone()).index(index.clone()).dtype(uint8_dtype).call();
            Some(uint8_load.cast(load.dtype()))
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

        // GEP(PTRCAT) → distribute element indices across sources
        // PTRCAT vcount = total elements, NOT source count. GEP indices are element indices.
        // Example: PTRCAT with 8 sources (each ptr<vec4<float>>) has vcount=32
        // GEP([0..31]) should extract elements across all sources
        Gep { vector: PtrCat { sources }, indices } => {
            // Build element-to-source mapping: element_idx -> (source_idx, offset_in_source)
            let mut element_map: Vec<(usize, usize)> = Vec::new();
            for (src_idx, src) in sources.iter().enumerate() {
                let src_count = ptr_element_count(src);
                for offset in 0..src_count {
                    element_map.push((src_idx, offset));
                }
            }

            // Fast path: if indices match sources 1:1, just reorder sources
            if element_map.len() == sources.len() && element_map.iter().all(|(_, off)| *off == 0) {
                let reordered: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                    .filter_map(|&i| sources.get(i).cloned())
                    .collect();
                if reordered.len() == indices.len() {
                    return Some(if reordered.len() == 1 {
                        reordered[0].clone()
                    } else {
                        UOp::ptrcat().sources(reordered.to_vec()).call()
                    });
                }
            }

            // General case: distribute element indices across sources
            let mut result_elements: Vec<Arc<UOp>> = Vec::with_capacity(indices.len());
            for &elem_idx in indices.iter() {
                if elem_idx >= element_map.len() {
                    return None; // Index out of bounds
                }
                let (src_idx, offset) = element_map[elem_idx];
                let src = &sources[src_idx];
                let src_count = ptr_element_count(src);
                let elem = if src_count == 1 {
                    src.clone()
                } else {
                    src.gep(vec![offset])
                };
                result_elements.push(elem);
            }

            Some(if result_elements.len() == 1 {
                result_elements[0].clone()
            } else {
                // Result is VECTORIZE of extracted pointer elements
                UOp::vectorize(result_elements.into_iter().collect())
            })
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

    // Skip WHERE(cond, t, Invalid) - used for image indexing (devectorizer.py:221)
    if let Op::Ternary(TernaryOp::Where, _, _, f) = alu.op()
        && matches!(f.op(), Op::Invalid)
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
            alu.replace().dtype(scalar_dtype.clone()).src(new_sources).call()
        })
        .collect();

    Some(UOp::vectorize(elements))
}

/// Vector ALU → VECTORIZE of scalar ALU (devectorizer.py:219-223).
/// LLVM SLP can re-vectorize when beneficial.
#[allow(unused_variables)]
pub fn no_vectorized_alu() -> TypedPatternMatcher {
    crate::patterns! {
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
// VECTORIZE Normalization
// ============================================================================

/// Normalize VECTORIZE and GEP for rendering (subset of pm_render).
pub fn pm_vectorize_normalize() -> TypedPatternMatcher {
    crate::patterns! {
        Gep { vector, indices } if indices.len() > 1 => |vector, indices| {
            let geps: SmallVec<[Arc<UOp>; 4]> = indices.iter()
                .map(|&i| vector.gep(vec![i]))
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
            tsrcs[i].push((*src).gep(gep_indices));
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
        wmmas.iter().flat_map(|w| (0..out_sz).map(move |i| w.gep(vec![i]))).collect();

    Some(UOp::vectorize(wmma_ex))
}

/// AFTER(CAST(x), deps) → CAST(AFTER(x, deps)) - allows cast to be optimized independently.
fn cast_after_pattern() -> TypedPatternMatcher {
    crate::patterns! {
        After { passthrough: Cast { src, dtype }, deps }
            => |src, dtype, deps| {
                let new_after = src.after(deps.clone());
                Some(new_after.cast(dtype.clone()))
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
    Some(scalar_def.cast(buf.dtype()))
}

/// INDEX(CAST(buf), scalar_idx) → INDEX(VECTORIZE([buf,...]), scaled_vec_idx) (devectorizer.py:228-231)
fn no_vectorized_index(
    buf: &Arc<UOp>,
    indices: &SmallVec<[Arc<UOp>; 4]>,
    gate: &Option<Arc<UOp>>,
    cast_dtype: &DType,
) -> Option<Arc<UOp>> {
    let idx = indices.first()?;

    // Tinygrad alignment: devectorizer.py expects scalar index
    debug_assert!(
        idx.dtype().vcount() == 1,
        "no_vectorized_index: expected scalar index, got vcount={}",
        idx.dtype().vcount()
    );

    let DType::Ptr { base, .. } = cast_dtype else { return None };
    let cnt = base.vcount();
    if cnt <= 1 {
        return None;
    }

    let buf_broadcast = buf.broadcast(cnt);
    let idx_broadcast = idx.broadcast(cnt);
    let offset_vec = create_index_vector(cnt);
    let cnt_broadcast = idx.const_like(cnt as i64).broadcast(cnt);
    let final_idx = idx_broadcast.mul(&cnt_broadcast).add(&offset_vec);

    let buf_dtype = buf_broadcast.dtype();
    Some(
        UOp::index()
            .buffer(buf_broadcast)
            .indices(vec![final_idx])
            .maybe_gate(gate.clone())
            .call()
            .expect("ICE unable to create index")
            .with_dtype(buf_dtype),
    )
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

    // Tinygrad alignment: devectorizer.py expects scalar index
    debug_assert!(
        idx.dtype().vcount() == 1,
        "no_vectorized_index_broadcast: expected scalar index, got vcount={}",
        idx.dtype().vcount()
    );

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
    let buf_broadcast = buf.broadcast(total_cnt);
    let idx_gep = idx.gep(gep_arg);
    let cnt_broadcast = idx.const_like(cnt as i64).broadcast(total_cnt);
    let sum_vec = create_const_index_vector(&sum_arg);
    let final_idx = idx_gep.mul(&cnt_broadcast).add(&sum_vec);

    let buf_dtype = buf_broadcast.dtype();
    Some(
        UOp::index()
            .buffer(buf_broadcast)
            .indices(vec![final_idx])
            .maybe_gate(gate.clone())
            .call()
            .expect("ICE: unabel to create index")
            .with_dtype(buf_dtype),
    )
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

    // Tinygrad alignment: devectorizer.py expects scalar index
    debug_assert!(
        idx.dtype().vcount() == 1,
        "no_vectorized_index_gep: expected scalar index, got vcount={}",
        idx.dtype().vcount()
    );

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
    let buf_broadcast = buf.broadcast(total_cnt);
    let idx_gep = idx.gep(gep_arg);
    let cnt_broadcast = idx.const_like(cnt as i64).broadcast(total_cnt);
    let sum_vec = create_const_index_vector(&sum_arg);
    let final_idx = idx_gep.mul(&cnt_broadcast).add(&sum_vec);

    let buf_dtype = buf_broadcast.dtype();
    Some(
        UOp::index()
            .buffer(buf_broadcast)
            .indices(vec![final_idx])
            .maybe_gate(gate.clone())
            .call()
            .expect("ICE: unabel to create index")
            .with_dtype(buf_dtype),
    )
}

// ============================================================================
// Load Store Indexing Patterns (devectorizer.py:48-55)
// ============================================================================

/// INDEX(buf, x, true) → INDEX(buf, x, None)
pub fn load_store_indexing_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        index @ Index { buffer, indices, gate: Some(g) }
            if matches!(g.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Bool(true)))
            ~> UOp::index().buffer(buffer.clone()).indices(indices.clone()).call().expect("ICE: unable to crate index").with_dtype(index.dtype())
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
//
// Multi-pass architecture: The patterns are split into phases because the
// bottom-up rewrite engine processes children before parents and does fixed-point
// matching on the result. When patterns create nested structures (e.g., GEP(LOAD(PTRCAT))),
// inner patterns may not match if they're in the same pass.
//
// Phase order:
// 1. expand_index: vector INDEX → GEP(PTRCAT)
// 2. gep_movement: LOAD(GEP(PTRCAT)) → GEP(LOAD(PTRCAT))
// 3. ptrcat_distribution: LOAD(PTRCAT) → CAT(LOADs)

/// Phase 1: Expand vector INDEX into GEP(PTRCAT) groupings.
fn expand_index_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        index if is_vector_index(index) => expand_vector_index(index),
    }
}

/// Phase 2: Move GEP through LOAD/STORE.
fn gep_movement_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // GEP after LOAD: LOAD(GEP(x)) → GEP(LOAD(x))
        load @ Load { buffer, index: Gep { vector, indices } }
            => move_gep_after_load(load, buffer, vector, indices),

        // GEP on STORE: STORE(GEP(x), data) → STORE(x, GEP⁻¹(data))
        Store { index: Gep { vector, indices }, value, ranges }
            => move_gep_on_store(vector, indices, value, ranges),
    }
}

/// Phase 3: Distribute PTRCAT through LOAD/STORE.
fn ptrcat_distribution_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // PTRCAT after LOAD: LOAD(PTRCAT(a,b)) → CAT(LOAD(a), LOAD(b))
        load @ Load { buffer, index: ptrcat @ PtrCat { sources } }
            => distribute_ptrcat_load(load, buffer, ptrcat, sources),

        // PTRCAT after STORE
        Store { index: PtrCat { sources }, value, ranges }
            => distribute_ptrcat_store(sources, value, ranges),
    }
}

/// Combined load/store folding patterns (for backward compatibility).
/// Prefer using `devectorize()` which applies these in proper phase order.
pub fn load_store_folding_patterns() -> TypedPatternMatcher {
    expand_index_patterns() + gep_movement_patterns() + ptrcat_distribution_patterns()
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
    let idx_vcount = idx.dtype().vcount();
    if idx_vcount <= 1 {
        tracing::trace!(idx_vcount = idx_vcount, "is_vector_index: idx not vector");
        return false;
    }

    // Buffer MUST be VECTORIZE (not bare buffer) - Tinygrad line 115
    let Op::Vectorize { elements } = buffer.op() else {
        tracing::trace!("is_vector_index: buffer not VECTORIZE");
        return false;
    };

    // Elements must be Defines.or_after()
    if elements.is_empty() || !elements.iter().all(is_define_or_after) {
        tracing::trace!("is_vector_index: elements not defines");
        return false;
    }

    // Don't vectorize bool loads (LLVM i1 vector load is broken)
    let is_bool = elements[0].dtype().base().is_bool();
    if is_bool {
        tracing::trace!("is_vector_index: bool type, skipping");
        return false;
    }

    true
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
fn expand_vector_index(index: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Index { buffer, indices, gate } = index.op() else { return None };
    let vec = indices.first()?;
    let count = vec.dtype().vcount();

    let buf = if let Op::Vectorize { elements } = buffer.op() { elements.first()?.clone() } else { buffer.clone() };

    // Generate scalar INDEX ops and simplify
    let scalar_indices: Vec<_> = (0..count)
        .map(|i| {
            UOp::index()
                .buffer(buf.clone())
                .indices(vec![vec.gep(vec![i])])
                .maybe_gate(gate.clone())
                .call()
                .expect("ICE: unable to create index")
                .with_dtype(buf.dtype().clone())
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
            let ptr = if lanes.len() > 1 { lidx.cast(make_vec_ptr_dtype(&buf, lanes.len())) } else { lidx };
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

    Some(ptrcat.gep(gep_indices))
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
            // Tinygrad: ld.replace(dtype=x.dtype.base, ...) - preserves vec type
            // If ptr is ptr<vec4<float>>, load dtype should be vec4<float>, NOT scalar float
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

    let scalar_dtype = buffer.dtype().scalar_dtype();
    let mut lengths = get_device_fold_lengths(&scalar_dtype.vec(sz));

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
            let chunk_ptr = if fold_len > 1 { chunk_idx.cast(make_vec_ptr_dtype(buffer, fold_len)) } else { chunk_idx };
            let chunk_dtype = scalar_dtype.vec(fold_len);

            chunks.push(UOp::load().buffer(buffer.clone()).index(chunk_ptr).dtype(chunk_dtype).call());
            pos += fold_len;
            break;
        }
    }

    if chunks.len() <= 1 {
        return None;
    }

    Some(UOp::cat().sources(chunks).dtype(scalar_dtype.vec(sz)).call())
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
                if fold_len > 1 { chunk_idx.cast(make_vec_ptr_dtype(idx_buffer, fold_len)) } else { chunk_idx };
            let gep_indices: Vec<usize> = (pos..pos + fold_len).collect();
            let chunk_value = value.gep(gep_indices);
            stores.push(chunk_ptr.store_with_ranges(chunk_value, ranges.clone()));

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

    UOp::index()
        .buffer(buffer.clone())
        .indices(new_indices)
        .maybe_gate(gate.clone())
        .call()
        .expect("ICE: unabel to create index")
        .with_dtype(idx.dtype())
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
            tracing::debug!(
                red_id = red.id,
                red_dtype = ?red.dtype(),
                red_dtype_vcount = red.dtype().vcount(),
                src_id = src.id,
                src_op = src.op().as_ref(),
                src_dtype = ?src.dtype(),
                src_dtype_vcount = src.dtype().vcount(),
                "pm_reduce: REDUCE pattern matched"
            );
            reduce_to_acc(red, src, ranges, reduce_op)
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
#[tracing::instrument]
pub fn pm_flatten_range() -> TypedPatternMatcher {
    /// Extract actual RANGE ops from a list of range sources.
    /// Creates a sink, toposorts, and filters to only RANGE ops.
    fn extract_ranges(ranges: &SmallVec<[Arc<UOp>; 4]>) -> SmallVec<[Arc<UOp>; 4]> {
        if ranges.is_empty() {
            return SmallVec::new();
        }
        let sink = UOp::sink(ranges.iter().cloned().collect());
        sink.toposort().into_iter().filter(|node| matches!(node.op(), Op::Range { .. })).collect()
    }

    /// Check if ranges changed (optimization to avoid unnecessary rewrites).
    fn ranges_unchanged(original: &SmallVec<[Arc<UOp>; 4]>, filtered: &SmallVec<[Arc<UOp>; 4]>) -> bool {
        original.len() == filtered.len() && original.iter().zip(filtered.iter()).all(|(a, b)| a.id == b.id)
    }

    crate::patterns! {
        // END: Filter ranges to only RANGE ops
        end @ End { computation, ranges } => {
            let actual_ranges = extract_ranges(ranges);

            if ranges_unchanged(ranges, &actual_ranges) {
                return None; // No change needed
            }

            tracing::debug!(
                end_id = end.id, original_len = ranges.len(), filtered_len = actual_ranges.len(),
                "filtering END ranges"
            );

            if actual_ranges.is_empty() {
                // All ranges were non-RANGE (e.g., all CONST) - unwrap END
                Some(computation.clone())
            } else {
                Some(computation.end(actual_ranges))
            }
        },

        // REDUCE: Filter ranges to only RANGE ops
        reduce @ Reduce { src, ranges, reduce_op } => |reduce, src, ranges, reduce_op| {
            let actual_ranges = extract_ranges(ranges);
            if ranges_unchanged(ranges, &actual_ranges) {
                return None;
            }

            tracing::debug!(
                reduce_id = reduce.id, original_len = ranges.len(), filtered_len = actual_ranges.len(),
                "pm_flatten_range: filtering REDUCE ranges"
            );

            Some(src.reduce(actual_ranges, *reduce_op).with_dtype(reduce.dtype()))
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

            Some(index.store_with_ranges(value.clone(), actual_ranges))
        },
    }
}

/// Horizontal reduce for accumulator pattern.
///
/// If src dtype matches out_dtype, returns `vec![src]`.
/// Otherwise, creates stride-pattern GEPs to reduce src to out_dtype elements.
///
/// Based on Tinygrad's horizontal_reduce (devectorizer.py:284-289):
/// ```python
/// def horizontal_reduce(inp, out_dtype):
///   if inp.dtype != out_dtype:
///     horizontal_amount = inp.dtype.count // out_dtype.count
///     return [inp.gep(tuple(range(i, inp.dtype.count, horizontal_amount))) for i in range(horizontal_amount)]
///   return [inp]
/// ```
fn horizontal_reduce_for_acc(src: &Arc<UOp>, out_dtype: &DType, reduce_op: ReduceOp) -> Vec<Arc<UOp>> {
    let src_count = src.dtype().vcount();
    let out_count = out_dtype.vcount();

    tracing::debug!(
        src_id = src.id,
        src_op = src.op().as_ref(),
        src_dtype = ?src.dtype(),
        src_count = src_count,
        out_dtype = ?out_dtype,
        out_count = out_count,
        "horizontal_reduce_for_acc: checking dtype match"
    );

    // Types already match - return single-element list
    if src_count == out_count {
        tracing::debug!("horizontal_reduce_for_acc: dtypes match, returning src as-is");
        return vec![src.clone()];
    }

    // Need horizontal reduction
    let horizontal_amount = src_count / out_count;

    // Edge case: uneven division - fall back to full scalar reduction
    if !src_count.is_multiple_of(out_count) || horizontal_amount == 0 {
        let scalar_dtype = src.dtype().scalar_dtype();
        let elements: Vec<Arc<UOp>> = (0..src_count).map(|i| src.gep(vec![i])).collect();
        return vec![
            elements
                .into_iter()
                .reduce(|acc, elem| apply_reduce_binary(reduce_op, acc, elem, &scalar_dtype))
                .expect("src_count >= 1"),
        ];
    }

    // Create stride pattern GEPs
    // e.g., for src=vec8 -> out=vec2, horizontal_amount=4:
    //   GEP([0,4]), GEP([1,5]), GEP([2,6]), GEP([3,7])
    (0..horizontal_amount)
        .map(|i| {
            let indices: Vec<usize> = (i..src_count).step_by(horizontal_amount).collect();
            src.gep(indices)
        })
        .collect()
}

/// Convert a REDUCE operation to explicit accumulator pattern.
///
/// Follows Tinygrad's approach (devectorizer.py:reduce_to_acc):
/// 1. First call horizontal_reduce to ensure src matches result dtype
/// 2. Find input_ranges (outer ranges that are NOT the reduce ranges)
/// 3. Initialize accumulator BEFORE reduce loop (depends only on input_ranges)
/// 4. Access accumulator INSIDE reduce loop (depends on reduce ranges)
/// 5. Read final value AFTER reduce loop completes
///
/// Note: This creates INDEX ops with element dtype. pm_add_loads will:
/// 1. Wrap INDEX with LOAD for values used in arithmetic
/// 2. STORE cleanup will remove LOAD from STORE's index position
#[tracing::instrument(skip_all)]
fn reduce_to_acc(
    red: &Arc<UOp>,
    src: &Arc<UOp>,
    ranges: &SmallVec<[Arc<UOp>; 4]>,
    reduce_op: &ReduceOp,
) -> Option<Arc<UOp>> {
    // Use the REDUCE's actual output dtype (could be vec2 if upcasted)
    // Tinygrad: acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, ...), ...)
    let result_dtype = red.dtype();

    // Step 0: Apply horizontal_reduce to ensure src dtype matches result dtype
    // This handles vectorized sources (e.g., src=vec20, result=scalar).
    // Tinygrad (devectorizer.py:293): lst = horizontal_reduce(inp, red.dtype)
    let src_list = horizontal_reduce_for_acc(src, &result_dtype, *reduce_op);

    // Tinygrad alignment: devectorizer.py:294
    // assert all(x.dtype == red.dtype for x in lst)
    debug_assert!(
        src_list.iter().all(|x| x.dtype() == result_dtype),
        "horizontal reduction dtype mismatch: expected {:?}, got {:?}",
        result_dtype,
        src_list.first().map(|x| x.dtype())
    );

    debug!(
        src_id = src.id,
        src_dtype = ?src.dtype(),
        result_dtype = ?result_dtype,
        src_list_len = src_list.len(),
        "horizontal_reduce applied"
    );

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
        "processing REDUCE"
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
        let idx_init = UOp::index()
            .buffer(acc.clone())
            .indices(vec![zero_idx.clone()])
            .call()
            .expect("ICE: unable to create index")
            .with_dtype(result_dtype.clone());
        idx_init.store_value(identity)
    } else {
        // Has outer ranges: acc.after(*input_ranges).index(0).store(identity)
        let idx_init = UOp::index()
            .buffer(acc.after(input_ranges))
            .indices(vec![zero_idx.clone()])
            .call()
            .expect("ICE: unable to create index")
            .with_dtype(result_dtype.clone());
        idx_init.store_value(identity)
    };

    // 6. Create accumulator access INSIDE the reduce loop
    // acc.after(store_init, *reduce_range).index(0)
    let mut loop_deps: SmallVec<[Arc<UOp>; 4]> = smallvec::smallvec![store_init.clone()];
    loop_deps.extend(ranges.iter().cloned());
    let acc_inside_loop = acc.after(loop_deps);
    let idx_loop = UOp::index()
        .buffer(acc_inside_loop.clone())
        .indices(vec![zero_idx.clone()])
        .call()
        .expect("ICE: unable to build index")
        .with_dtype(result_dtype.clone());

    // 7. Chain accumulator with all horizontal-reduced elements
    // Tinygrad: lst = [acc...] + horizontal_reduce_list
    //           ret = functools.reduce(lambda x,y: x.alu(red.arg, y), lst)
    let mut chain_elements: Vec<Arc<UOp>> = vec![idx_loop.clone()];
    chain_elements.extend(src_list);

    let new_val = chain_elements
        .into_iter()
        .reduce(|acc_val, elem| apply_reduce_binary(*reduce_op, acc_val, elem, &result_dtype))
        .expect("chain_elements should have at least one element (acc)");

    // 8. Store new value back to accumulator (idx_loop contains the INDEX, not acc directly)
    let store_loop = idx_loop.store_value(new_val);
    // 9. End the reduce ranges after the store
    let end = store_loop.end(ranges.clone());
    // 10. Final value AFTER loop completes
    // acc.after(end).index(0)
    let acc_after_end = acc.after(smallvec::smallvec![end.clone()]);
    let idx_final =
        UOp::index().buffer(acc_after_end.clone()).indices(vec![zero_idx]).call().expect("ICE: unable to build index");

    debug!(
        acc_id = acc.id,
        store_init_id = store_init.id,
        acc_inside_loop_id = acc_inside_loop.id,
        idx_loop_id = idx_loop.id,
        store_loop_id = store_loop.id,
        end_id = end.id,
        acc_after_end_id = acc_after_end.id,
        idx_final_id = idx_final.id,
        "created accumulator pattern"
    );

    Some(idx_final)
}

/// Apply binary reduce operation between two values.
fn apply_reduce_binary(reduce_op: ReduceOp, a: Arc<UOp>, b: Arc<UOp>, dtype: &DType) -> Arc<UOp> {
    // Tinygrad alignment: verify operand dtypes match for reduction chaining
    debug_assert!(
        a.dtype() == b.dtype(),
        "apply_reduce_binary: dtype mismatch between operands: a={:?}, b={:?}",
        a.dtype(),
        b.dtype()
    );

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
