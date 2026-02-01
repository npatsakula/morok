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
use morok_ir::{BinaryOp, ConstValue, Op, ReduceOp, TernaryOp, UOp, UnaryOp, WmmaMetadata};

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

        // WHERE(c, LOAD(INDEX(buf, idx, c)), alt) → LOAD with alt value (devectorizer.py:271-272)
        // The load's gate matches the WHERE condition
        Where(cond, load @ Load { index, alt: None, .. }, alt)
            if index_has_gate_matching(index, cond)
            => |cond, load, index, alt| {
                let casted_alt = alt.cast(load.dtype());
                let new_load = UOp::load()
                    .buffer(load.load_buffer()?)
                    .index(index.clone())
                    .alt(casted_alt)
                    .dtype(load.dtype())
                    .call();
                Some(new_load.cast(alt.dtype()))
            },

        // WHERE(c, alt, LOAD(INDEX(buf, idx, !c))) → LOAD with alt value (devectorizer.py:273-274)
        // Same pattern but with inverted condition in WHERE
        Where(cond, alt, load @ Load { index, alt: None, .. })
            if index_has_inverted_gate_matching(index, cond)
            => |cond, alt, load, index| {
                let casted_alt = alt.cast(load.dtype());
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
/// Matches INDEX(..., NOT(cond)) where cond is the WHERE condition.
fn index_has_inverted_gate_matching(index: &Arc<UOp>, cond: &Arc<UOp>) -> bool {
    match index.op() {
        Op::Index { gate: Some(g), .. } => {
            // Check if gate is NOT(cond)
            if let Op::Unary(UnaryOp::Not, inner) = g.op() { Arc::ptr_eq(inner, cond) } else { false }
        }
        Op::Cast { src, .. } => index_has_inverted_gate_matching(src, cond),
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
    let tsrcs: Vec<Vec<Arc<UOp>>> = [a, b, c]
        .iter()
        .map(|src| {
            let n = src.dtype().vcount();
            (0..n).step_by(out_sz).map(|g| src.gep((g..g + out_sz.min(n - g)).collect())).collect()
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
        def if matches!(def.op(), Op::DefineLocal(_) | Op::DefineReg { .. })
            && def.ptrdtype().is_some_and(|(base, _, _)| base.vcount() > 1)
            => no_vectorized_buf(def),

        // INDEX(CAST(DEFINE_LOCAL/REG), scalar_idx) → scaled vector index
        Index { buffer: Cast { src: buf, dtype: cast_dtype }, indices, gate }
            if is_scalar_index(indices) && is_vectorized_local_reg_cast(buf, cast_dtype)
            => no_vectorized_index(buf, indices, gate, cast_dtype),

        // INDEX(BROADCAST(CAST(...)), scalar_idx)
        Index { buffer: Vectorize { elements }, indices, gate }
            if is_scalar_index(indices) && is_vectorized_broadcast_cast(elements)
            => {
                let first = elements.first()?;
                let Op::Cast { src: buf, dtype: DType::Ptr { base, .. } } = first.op() else { return None };
                let idx = indices.first()?;
                no_vectorized_index_precnt(buf, idx, gate, base.vcount(), &vec![0; elements.len()])
            },

        // INDEX(GEP(CAST(...)), scalar_idx)
        Index { buffer: Gep { vector: Cast { src: buf, dtype: cast_dtype }, indices: gep_indices }, indices, gate }
            if is_scalar_index(indices) && is_vectorized_local_reg_cast(buf, cast_dtype)
            => {
                let DType::Ptr { base, .. } = cast_dtype else { return None };
                let idx = indices.first()?;
                no_vectorized_index_precnt(buf, idx, gate, base.vcount(), gep_indices)
            },
    }
}

fn is_scalar_index(indices: &SmallVec<[Arc<UOp>; 4]>) -> bool {
    indices.first().is_some_and(|idx| idx.dtype().vcount() == 1)
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
    let DType::Ptr { base, .. } = cast_dtype else { return None };
    let cnt = base.vcount();
    if cnt <= 1 {
        return None;
    }

    let buf_broadcast = buf.broadcast(cnt);
    let idx_broadcast = idx.broadcast(cnt);
    let offset_vec = create_index_vector(0..cnt as i64);
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

fn create_index_vector(values: impl IntoIterator<Item = i64>) -> Arc<UOp> {
    let elements: SmallVec<[Arc<UOp>; 4]> = values.into_iter().map(UOp::index_const).collect();
    UOp::vectorize(elements)
}

/// INDEX with precnt multiplier (broadcast or gep case) (devectorizer.py:233-239)
fn no_vectorized_index_precnt(
    buf: &Arc<UOp>,
    idx: &Arc<UOp>,
    gate: &Option<Arc<UOp>>,
    cnt: usize,
    input_gep: &[usize],
) -> Option<Arc<UOp>> {
    let precnt = input_gep.len();
    let total = cnt * precnt;
    let gep_arg: Vec<usize> = (0..cnt).flat_map(|_| 0..precnt).collect();
    let sum_arg = (0..cnt).flat_map(|i| input_gep.iter().map(move |&y| (i + y) as i64));

    let buf_broadcast = buf.broadcast(total);
    let final_idx =
        idx.gep(gep_arg).mul(&idx.const_like(cnt as i64).broadcast(total)).add(&create_index_vector(sum_arg));

    Some(
        UOp::index()
            .buffer(buf_broadcast.clone())
            .indices(vec![final_idx])
            .maybe_gate(gate.clone())
            .call()
            .expect("ICE: unable to create index")
            .with_dtype(buf_broadcast.dtype()),
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

/// LOAD/STORE(CAST(INDEX)) → split by device fold lengths + image fixup.
pub fn correct_load_store_patterns() -> TypedPatternMatcher {
    crate::patterns! {
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
    matches!(uop.unwrap_after().op(), Op::DefineLocal(_) | Op::DefineReg { .. } | Op::DefineGlobal(_))
}

/// Matches INDEX(VECTORIZE(Defines.or_after()), vec_idx) only.
/// Tinygrad devectorizer.py:115 - expand_index only matches VECTORIZE of defines.
fn is_vector_index(uop: &Arc<UOp>) -> bool {
    let Op::Index { buffer, indices, .. } = uop.op() else { return false };
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
    let sz = match ls.op() {
        Op::Load { index, .. } | Op::Store { index, .. } => index.dtype().vcount(),
        _ => return None,
    };
    if sz == 1 {
        return None;
    }

    // Fold lengths (devectorizer.py:138-152)
    let buf_dtype = buf.dtype();
    let no_fold = (!buf_dtype.base().is_float() && !matches!(buf_dtype, DType::Image { .. }))
        || matches!(buf_dtype, DType::Ptr { addrspace: AddrSpace::Reg, .. });
    let mut lengths = if no_fold {
        vec![1]
    } else if matches!(buf_dtype, DType::Image { .. }) {
        vec![4, 1]
    } else {
        vec![4, 2, 1] // ctx.supports_float4
    };

    // Filter by divisibility (devectorizer.py:155-156)
    if let Some(offset) = indices.first() {
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
                Op::Load { buffer, .. } => {
                    ret.push(UOp::load().buffer(buffer.clone()).index(lidx).dtype(scalar_dtype.vec(fold_len)).call());
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
        let new_idx = UOp::index()
            .buffer(img_buf.clone())
            .indices(vec![new_idx_expr])
            .maybe_gate(gate.clone())
            .call()
            .ok()?
            .with_dtype(inner_idx.dtype());

        // Replace the index in LOAD/STORE
        return Some(ls.replace().src(vec![new_idx]).call());
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

        let new_idx = UOp::index()
            .buffer(img_buf.clone())
            .indices(vec![new_idx_expr])
            .maybe_gate(gate.clone())
            .call()
            .ok()?
            .with_dtype(index.dtype());

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
pub fn pm_reduce() -> TypedPatternMatcher {
    crate::patterns! {
        // Match ALL REDUCEs - empty ranges handled by returning reduced value directly
        red @ Reduce(_, ..) => |red| reduce_to_acc(red),
    }
}

/// Horizontal reduce for accumulator pattern (devectorizer.py:283-289).
fn horizontal_reduce(inp: &Arc<UOp>, out_dtype: &DType) -> Vec<Arc<UOp>> {
    if inp.dtype() == *out_dtype {
        return vec![inp.clone()];
    }
    let horizontal_amount = inp.dtype().vcount() / out_dtype.vcount();
    debug_assert!(inp.dtype().vcount().is_multiple_of(out_dtype.vcount()), "horizontal mismatch");
    (0..horizontal_amount).map(|i| inp.gep((i..inp.dtype().vcount()).step_by(horizontal_amount).collect())).collect()
}

/// Convert REDUCE to explicit accumulator pattern (devectorizer.py:291-308).
fn reduce_to_acc(red: &Arc<UOp>) -> Option<Arc<UOp>> {
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
