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
pub fn devectorize(ast: &Arc<UOp>) -> Arc<UOp> {
    // Phase 1: Expand vector indices into grouped PTRCAT
    let phase1 = expand_index_patterns();
    let ast = graph_rewrite_bottom_up(&phase1, ast.clone(), &mut ());

    // Phase 2: Distribute PTRCAT through LOAD/STORE and split by divisibility
    // Include GEP(PTRCAT/CAT) patterns to handle reordering
    let phase2 = gep_ptrcat_patterns() + load_store_patterns();
    let ast = graph_rewrite_bottom_up(&phase2, ast, &mut ());

    // Phase 3: Convert bool LOAD/STORE to uint8 to avoid LLVM i1 garbage bits
    // LLVM's i1 type can have garbage in upper bits when stored to memory.
    // By casting to uint8 before store and after load, we ensure clean 0/1 values.
    // Based on Tinygrad's PTX/NIR bool→uint8 patterns.
    let phase3 = bool_storage_patterns();
    graph_rewrite_bottom_up(&phase3, ast, &mut ())
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

/// GEP patterns for PTRCAT/CAT reordering only.
/// More focused than full gep_pushing_patterns to avoid transforming unrelated GEPs.
fn gep_ptrcat_patterns() -> TypedPatternMatcher {
    fn is_gep_ptrcat(uop: &Arc<UOp>) -> bool {
        if let Op::Gep { vector, .. } = uop.op() {
            return matches!(vector.op(), Op::PtrCat { .. });
        }
        false
    }

    fn gep_ptrcat(gep: &Arc<UOp>) -> Option<Arc<UOp>> {
        let Op::Gep { vector, indices } = gep.op() else { return None };
        let Op::PtrCat { sources } = vector.op() else { return None };
        let reordered: Vec<_> = indices.iter().filter_map(|&idx| sources.get(idx).cloned()).collect();
        if reordered.len() != indices.len() {
            return None;
        }
        Some(UOp::ptrcat(reordered))
    }

    fn is_gep_cat(uop: &Arc<UOp>) -> bool {
        if let Op::Gep { vector, .. } = uop.op() {
            return matches!(vector.op(), Op::Cat { .. });
        }
        false
    }

    fn gep_cat(gep: &Arc<UOp>) -> Option<Arc<UOp>> {
        let Op::Gep { vector, indices } = gep.op() else { return None };
        let Op::Cat { sources } = vector.op() else { return None };
        let reordered: Vec<_> = indices.iter().filter_map(|&idx| sources.get(idx).cloned()).collect();
        if reordered.len() != indices.len() {
            return None;
        }
        Some(UOp::cat(reordered))
    }

    crate::patterns! {
        gep if is_gep_ptrcat(gep) => |gep| gep_ptrcat(gep),
        gep if is_gep_cat(gep) => |gep| gep_cat(gep),

        // Single-source Cat is identity: Cat([x]) → x
        cat if matches!(cat.op(), Op::Cat { .. }) => |cat| {
            let Op::Cat { sources } = cat.op() else { return None };
            (sources.len() == 1).then(|| Arc::clone(&sources[0]))
        },

        // Single-source PtrCat is identity: PtrCat([x]) → x
        ptrcat if matches!(ptrcat.op(), Op::PtrCat { .. }) => |ptrcat| {
            let Op::PtrCat { sources } = ptrcat.op() else { return None };
            (sources.len() == 1).then(|| Arc::clone(&sources[0]))
        },

        // Identity CAT reconstruction: CAT(GEP(x,[0]), GEP(x,[1]), ...) → x
        cat if matches!(cat.op(), Op::Cat { .. }) => |cat| {
            let Op::Cat { sources } = cat.op() else { return None };
            if sources.is_empty() { return None; }

            let Op::Gep { vector: first_vec, indices: first_idx } = sources[0].op() else { return None };
            if first_idx.len() != 1 || first_idx[0] != 0 { return None; }

            for (i, src) in sources.iter().enumerate() {
                let Op::Gep { vector, indices } = src.op() else { return None };
                if !Arc::ptr_eq(vector, first_vec) { return None; }
                if indices.len() != 1 || indices[0] != i { return None; }
            }

            if sources.len() != first_vec.dtype().vcount() { return None; }
            Some(Arc::clone(first_vec))
        },

        // Devectorize WHERE with vector condition.
        // Based on Tinygrad's no_vectorized_alu (devectorizer.py:219-223).
        // WHERE(<N x i1>, <N x T>, <N x T>) → VECTORIZE(WHERE(i1, T, T), ...)
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

/// Phase 1 patterns: expand vector INDEX into grouped PTRCAT.
fn expand_index_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // INDEX(buffer, vector_index) with vector index → expand and group
        index if is_vector_index(index) => |index| expand_vector_index(index),
    }
}

/// Phase 2 patterns: distribute PTRCAT and split LOAD/STORE.
fn load_store_patterns() -> TypedPatternMatcher {
    crate::patterns! {
        // Distribute PTRCAT through LOAD: LOAD(PTRCAT(a,b)) → CAT(LOAD(a), LOAD(b))
        load if is_ptrcat_load(load) => |load| distribute_ptrcat_load(load),

        // Distribute PTRCAT through STORE: STORE(PTRCAT(a,b), data) → GROUP(STORE(a, gep(data)), ...)
        store if is_ptrcat_store(store) => |store| distribute_ptrcat_store(store),

        // Split LOAD(CAST(INDEX)) by divisibility
        load if is_cast_index_load(load) => |load| split_load(load),

        // Split STORE(CAST(INDEX), ...) by divisibility
        store if is_cast_index_store(store) => |store| split_store(store),
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

/// Check if LOAD has PTRCAT as its index.
fn is_ptrcat_load(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Load { index, .. } if matches!(index.op(), Op::PtrCat { .. }))
}

/// Check if STORE has PTRCAT as its index.
fn is_ptrcat_store(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Store { index, .. } if matches!(index.op(), Op::PtrCat { .. }))
}

/// Check if LOAD has CAST(INDEX) as its index.
fn is_cast_index_load(uop: &Arc<UOp>) -> bool {
    if let Op::Load { index, .. } = uop.op()
        && let Op::Cast { src, .. } = index.op()
    {
        return matches!(src.op(), Op::Index { .. });
    }
    false
}

/// Check if STORE has CAST(INDEX) as its index.
fn is_cast_index_store(uop: &Arc<UOp>) -> bool {
    if let Op::Store { index, .. } = uop.op()
        && let Op::Cast { src, .. } = index.op()
    {
        return matches!(src.op(), Op::Index { .. });
    }
    false
}

// ============================================================================
// Root Key for Grouping (Tinygrad: offsets_rootsrc)
// ============================================================================

/// Key for grouping indices by root expression and validity.
///
/// Tinygrad groups by (validity, root) tuple. We use UOp IDs for hashing.
#[derive(Clone, Hash, Eq, PartialEq, Debug)]
enum RootKey {
    /// Expression-based root with optional gate (validity) ID
    Expr { valid_id: Option<u64>, root_id: u64 },
    /// Constant index
    Const,
}

impl RootKey {
    fn expr(root: &Arc<UOp>, gate: Option<&Arc<UOp>>) -> Self {
        RootKey::Expr { valid_id: gate.map(|g| g.id), root_id: root.id }
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
    let scalar_indices: Vec<Arc<UOp>> = (0..vec_count)
        .map(|i| {
            let scalar_idx = UOp::gep(vec_idx.clone(), vec![i]);
            UOp::new(
                Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![scalar_idx], gate: gate.clone() },
                index.dtype(),
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
/// Patterns:
/// - Add(root, CONST(offset)) → (Expr(root), offset)
/// - Add(CONST(offset), root) → (Expr(root), offset)
/// - CONST(offset) → (Const, offset)
/// - Invalid → (Invalid, 0)
/// - other → (Expr(other), 0)
fn extract_root_and_offset(idx: &Arc<UOp>, gate: Option<&Arc<UOp>>) -> (RootKey, i64) {
    // Extract total constant offset from the entire Add chain
    // E.g., Add(Add(base, 1), y) → (Add(base, y), 1)
    fn extract_const_sum(uop: &Arc<UOp>) -> (Option<Arc<UOp>>, i64) {
        match uop.op() {
            Op::Binary(BinaryOp::Add, left, right) => {
                // Check if right is const
                if let Op::Const(cv) = right.op()
                    && let ConstValue::Int(v) = cv.0
                {
                    let (inner, inner_sum) = extract_const_sum(left);
                    return (inner, inner_sum + v);
                }
                // Check if left is const
                if let Op::Const(cv) = left.op()
                    && let ConstValue::Int(v) = cv.0
                {
                    let (inner, inner_sum) = extract_const_sum(right);
                    return (inner, inner_sum + v);
                }
                // Recurse into both sides
                let (left_inner, left_sum) = extract_const_sum(left);
                let (right_inner, right_sum) = extract_const_sum(right);
                if left_sum != 0 || right_sum != 0 {
                    // Rebuild Add without the extracted constants
                    match (left_inner, right_inner) {
                        (Some(l), Some(r)) => {
                            if let Ok(new_add) = l.try_add(&r) {
                                return (Some(new_add), left_sum + right_sum);
                            }
                        }
                        (Some(l), None) => return (Some(l), left_sum + right_sum),
                        (None, Some(r)) => return (Some(r), left_sum + right_sum),
                        (None, None) => return (None, left_sum + right_sum),
                    }
                }
                (Some(Arc::clone(uop)), 0)
            }
            Op::Const(cv) => {
                if let ConstValue::Int(v) = cv.0 {
                    (None, v) // Pure constant - no root expression
                } else {
                    (Some(Arc::clone(uop)), 0)
                }
            }
            _ => (Some(Arc::clone(uop)), 0),
        }
    }

    let (root_opt, offset) = extract_const_sum(idx);
    match root_opt {
        Some(root) => (RootKey::expr(&root, gate), offset),
        None => (RootKey::Const, offset), // Pure constant expression
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
fn distribute_ptrcat_load(load: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Load { buffer, index } = load.op() else {
        return None;
    };

    let Op::PtrCat { sources } = index.op() else {
        return None;
    };

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
/// Based on Tinygrad's cat_after_store (devectorizer.py:97-104).
/// STORE(PTRCAT(a, b), data) → GROUP(STORE(a, gep(data, 0..n)), STORE(b, gep(data, n..)))
fn distribute_ptrcat_store(store: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Store { buffer, index, value, ranges } = store.op() else {
        return None;
    };

    let Op::PtrCat { sources } = index.op() else {
        return None;
    };

    tracing::debug!(num_sources = sources.len(), "distribute_ptrcat_store: distributing PTRCAT through STORE");

    // Create individual STOREs for each pointer
    let mut stores = Vec::new();
    let mut offset = 0usize;

    for ptr in sources.iter() {
        let ptr_count = ptr_element_count(ptr);

        // GEP to extract data elements for this store
        let gep_indices: Vec<usize> = (offset..offset + ptr_count).collect();
        let store_value = UOp::gep(value.clone(), gep_indices);

        // Create STORE
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

/// Split LOAD based on fold length divisibility.
///
/// Based on Tinygrad's split_load_store (devectorizer.py:130-174).
/// For LOAD(CAST(INDEX)), determine maximum fold length based on offset divisibility.
fn split_load(load: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Load { buffer, index } = load.op() else {
        return None;
    };

    let Op::Cast { src: inner_idx, dtype: _cast_dtype } = index.op() else {
        return None;
    };

    let Op::Index { indices, .. } = inner_idx.op() else {
        return None;
    };

    let sz = load.dtype().vcount();
    if sz <= 1 {
        return None;
    }

    tracing::debug!(sz = sz, "split_load: processing LOAD(CAST(INDEX))");

    // Determine fold lengths (based on device capability)
    // Default: [4, 2, 1] for float4 support
    let mut lengths: Vec<usize> = vec![4, 2, 1];

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
            let chunk_idx = if pos == 0 { inner_idx.clone() } else { offset_index(inner_idx, pos as i64) };

            // CAST to vector pointer if fold_len > 1
            let chunk_ptr = if fold_len > 1 {
                let vec_ptr_dtype = make_vec_ptr_dtype(buffer, fold_len);
                UOp::cast(chunk_idx, vec_ptr_dtype)
            } else {
                chunk_idx
            };

            // Create LOAD with appropriate dtype
            let chunk_dtype = if fold_len > 1 {
                DType::Vector { scalar: load.dtype().base(), count: fold_len }
            } else {
                DType::Scalar(load.dtype().base())
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
fn split_store(store: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Store { buffer, index, value, ranges } = store.op() else {
        return None;
    };

    let Op::Cast { src: inner_idx, .. } = index.op() else {
        return None;
    };

    let Op::Index { indices, .. } = inner_idx.op() else {
        return None;
    };

    let sz = value.dtype().vcount();
    if sz <= 1 {
        return None;
    }

    tracing::debug!(sz = sz, "split_store: processing STORE(CAST(INDEX), ...)");

    // Determine fold lengths
    let mut lengths: Vec<usize> = vec![4, 2, 1];

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
            let chunk_idx = if pos == 0 { inner_idx.clone() } else { offset_index(inner_idx, pos as i64) };

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

            // Create STORE
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

        // Test Add of two constants - should return total as offset
        let const_a = UOp::const_(DType::Index, ConstValue::Int(100));
        let const_b = UOp::const_(DType::Index, ConstValue::Int(3));
        let add_consts = UOp::new(Op::Binary(BinaryOp::Add, const_a, const_b), DType::Index);
        let (key, offset) = extract_root_and_offset(&add_consts, None);
        assert_eq!(offset, 103); // Sum of both constants
        assert!(matches!(key, RootKey::Const));
    }
}
