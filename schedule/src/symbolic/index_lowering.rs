//! Index dtype lowering patterns.
//!
//! Converts abstract Index dtype to concrete integer types (i32 or i64)
//! based on value bounds analysis. This runs as Stage 15 in the pipeline.
//!
//! Based on Tinygrad's pm_lower_index_dtype approach:
//! - Use i32 when value bounds fit within i32 range
//! - Use i64 otherwise (default for safety)
//!
//! This allows codegen to work with concrete types only.
//!
//! ## Pattern Coverage
//!
//! Tinygrad's pm_lower_index_dtype has 8 pattern categories:
//! 1. Binary ops with Index operands - cast operands, compute, cast back
//! 2. CONST/VCONST with Index dtype - select concrete type based on overflow
//! 3. WHERE with Index branches - cast branches to common type
//! 4. RANGE with Index end - propagate end's dtype
//! 5. VECTORIZE of Index elements - apply select_dtype to vector
//! 6. SPECIAL (gidx/lidx) - always i32
//! 7. DEFINE_VAR - always i32 (Morok uses bounds check)
//! 8. BIND - cast operands
//!
//! Plus cleanup patterns for Invalid INDEX and redundant casts.

use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::{BinaryOp, ConstValue};
use morok_ir::uop::cached_property::CachedProperty;
use morok_ir::uop::properties::VminVmaxProperty;
use morok_ir::{Op, UOp};

use crate::TypedPatternMatcher;

/// Determine the concrete integer dtype for an Index-typed operation.
///
/// Uses vmin/vmax bounds analysis:
/// - If both bounds fit in i32 range [-2^31, 2^31-1], use Int32
/// - Otherwise use Int64 (conservative default)
fn select_concrete_dtype(uop: &Arc<UOp>) -> DType {
    let (vmin, vmax) = VminVmaxProperty::get(uop);

    let fits_i32 = match (vmin, vmax) {
        (ConstValue::Int(min), ConstValue::Int(max)) => *min >= i32::MIN as i64 && *max <= i32::MAX as i64,
        (ConstValue::UInt(min), ConstValue::UInt(max)) => *min <= i32::MAX as u64 && *max <= i32::MAX as u64,
        _ => false, // Conservative: use i64 if bounds are unknown
    };

    if fits_i32 { DType::Scalar(ScalarDType::Int32) } else { DType::Scalar(ScalarDType::Int64) }
}

/// Compute least upper dtype for two concrete integer types.
/// Returns the wider type that can hold both.
fn least_upper_int_dtype(a: &DType, b: &DType) -> DType {
    // If either is i64, use i64; otherwise use i32
    match (a, b) {
        (DType::Scalar(ScalarDType::Int64), _) | (_, DType::Scalar(ScalarDType::Int64)) => {
            DType::Scalar(ScalarDType::Int64)
        }
        _ => DType::Scalar(ScalarDType::Int32),
    }
}

/// Rebuild a binary operation with new operands and dtype.
fn rebuild_binary(binary_op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>, dtype: DType) -> Arc<UOp> {
    UOp::new(Op::Binary(binary_op, lhs, rhs), dtype)
}

/// Pattern matcher for lowering Index dtype to concrete i32/i64.
///
/// Stage 15: Index dtype lowering (Tinygrad: pm_lower_index_dtype)
///
/// Patterns:
/// - Binary ops with Index result → lower operands, compute in concrete type
/// - CONST with Index dtype → CONST with i32/i64
/// - WHERE with Index result → lower branches to common concrete type
/// - SPECIAL with Index dtype → SPECIAL with i32
/// - DEFINE_VAR with Index dtype → DEFINE_VAR with i32/i64
/// - RANGE end → lowered based on bounds
/// - VECTORIZE of Index elements → lower each element
pub fn pm_lower_index_dtype() -> TypedPatternMatcher {
    crate::patterns! {
        // ====================================================================
        // Pattern 1: Binary ops with Index operands
        // ====================================================================
        // When a binary operation has Index dtype operands, we need to:
        // 1. Determine concrete types for operands
        // 2. Cast operands to a common concrete type
        // 3. Perform the operation in concrete type
        //
        // Match any node that is a Binary op with Index dtype
        node if node.dtype() == DType::Index && matches!(node.op(), Op::Binary(_, _, _)) => |node| {
            let Op::Binary(binary_op, lhs, rhs) = node.op() else { return None };

            // Both operands should be Index (or castable to Index)
            if lhs.dtype() != DType::Index || rhs.dtype() != DType::Index {
                return None;
            }

            // Select concrete dtypes for both operands
            let lhs_concrete = select_concrete_dtype(lhs);
            let rhs_concrete = select_concrete_dtype(rhs);
            // Use wider type that can hold both
            let common_dtype = least_upper_int_dtype(&lhs_concrete, &rhs_concrete);

            // Cast operands to common type
            let lhs_cast = lhs.cast(common_dtype.clone());
            let rhs_cast = rhs.cast(common_dtype.clone());

            // Rebuild the binary operation with concrete dtype
            Some(rebuild_binary(*binary_op, lhs_cast, rhs_cast, common_dtype))
        },

        // ====================================================================
        // Pattern 2: Index CONST → concrete int CONST
        // ====================================================================
        c @const(cv) if c.dtype() == DType::Index => |c, cv| {
            let target_dtype = select_concrete_dtype(c);
            Some(UOp::const_(target_dtype, cv))
        },

        // ====================================================================
        // Pattern 3: WHERE with Index result
        // ====================================================================
        // WHERE(cond, Index_x, Index_y) → WHERE(cond, concrete_x, concrete_y)
        Where(cond, x, y) if x.dtype() == DType::Index && y.dtype() == DType::Index => |cond, x, y| {
            let x_concrete = select_concrete_dtype(x);
            let y_concrete = select_concrete_dtype(y);
            let common_dtype = least_upper_int_dtype(&x_concrete, &y_concrete);

            let x_cast = x.cast(common_dtype.clone());
            let y_cast = y.cast(common_dtype);

            UOp::try_where(cond.clone(), x_cast, y_cast).ok()
        },

        // ====================================================================
        // Pattern 4: SPECIAL (gidx, lidx) - always i32
        // ====================================================================
        // GPU thread indices are always 32-bit
        special @ Special { name, end } if special.dtype() == DType::Index => |name, end| {
            let i32_dtype = DType::Scalar(ScalarDType::Int32);
            let new_end = end.cast(i32_dtype);
            Some(UOp::special(new_end, name.clone()))
        },

        // ====================================================================
        // Pattern 5: DEFINE_VAR - use i32 if bounds fit, else i64
        // ====================================================================
        dv @ DefineVar { name, min_val, max_val } if dv.dtype() == DType::Index => |name, min_val, max_val| {
            let fits_i32 = *min_val >= i32::MIN as i64 && *max_val <= i32::MAX as i64;
            let target_dtype = if fits_i32 {
                DType::Scalar(ScalarDType::Int32)
            } else {
                DType::Scalar(ScalarDType::Int64)
            };
            Some(UOp::new(
                Op::DefineVar { name: name.clone(), min_val: *min_val, max_val: *max_val },
                target_dtype,
            ))
        },

        // ====================================================================
        // Pattern 6: RANGE end - if end is Index, lower based on bounds
        // ====================================================================
        Range { end, axis_id, axis_type } if end.dtype() == DType::Index => |end, axis_id, axis_type| {
            let target_dtype = select_concrete_dtype(end);
            let lowered_end = end.cast(target_dtype);
            Some(UOp::range_axis(lowered_end, *axis_id, *axis_type))
        },

        // ====================================================================
        // Pattern 7: VECTORIZE of Index elements
        // ====================================================================
        // VECTORIZE(Index_a, Index_b, ...) → VECTORIZE(concrete_a, concrete_b, ...)
        vec @ Vectorize { elements } if vec.dtype() == DType::Index => |vec, elements| {
            // Select dtype based on entire vector's bounds
            let target_dtype = select_concrete_dtype(vec);

            // Cast each element to the target scalar type
            let lowered_elements: Vec<_> = elements
                .iter()
                .map(|e| e.cast(target_dtype.clone()))
                .collect();

            Some(UOp::vectorize(lowered_elements.into()))
        },

        // ====================================================================
        // Pattern 8: CAST to Index from concrete int - remove redundant cast
        // ====================================================================
        // If we're casting a concrete int to Index, and the result is only used
        // in contexts that will be lowered anyway, we can remove the cast.
        // This cleanup pattern runs after other lowering patterns.
        Cast { src, dtype } if *dtype == DType::Index && src.dtype().is_int() => |src| {
            // Just use the concrete int directly - Index dtype will be handled
            // by the operations that consume this value
            Some(src.clone())
        },

        // ====================================================================
        // Pattern 9: BIND with Index casts
        // ====================================================================
        // BIND(var.cast(index), val.cast(index)) → var.bind(val).cast(index)
        Bind { var: Cast { src: var_inner, dtype: idx_dtype }, value: Cast { src: val_inner, dtype: idx_dtype2 } }
            if *idx_dtype == DType::Index && *idx_dtype2 == DType::Index
            => |var_inner, val_inner| {
                let bound = var_inner.bind(val_inner.clone());
                Some(bound.cast(DType::Index))
            },

        // ====================================================================
        // Pattern 10: INDEX cleanup (ungated) - Invalid lowering + hanging cast
        // ====================================================================
        // Combines two patterns to avoid duplicate matching:
        // - buf.index(cond.where(idx, Invalid)) → INDEX(buf, idx, gate=cond)
        // - INDEX(buf, idx.cast(index)) → INDEX(buf, idx) when idx is concrete int
        Index { buffer, indices, gate: None }
            if indices.len() == 1 => |buffer, indices| {
                let idx_uop = &indices[0];
                // Derive element type from buffer's pointer type
                let result_dtype = match buffer.dtype() {
                    DType::Ptr { base, .. } => base.as_ref().clone(),
                    other => other,
                };

                // Try pattern A: WHERE(cond, idx, Invalid) → INDEX with gate
                if let Op::Ternary(morok_ir::TernaryOp::Where, cond, idx, false_val) = idx_uop.op()
                    && matches!(false_val.op(), Op::Invalid) {
                        return Some(UOp::new(
                            Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![idx.clone()], gate: Some(cond.clone()) },
                            result_dtype,
                        ));
                    }

                // Try pattern B: idx.cast(index) → idx when idx is concrete int
                if let Op::Cast { src: idx, dtype: idx_dtype } = idx_uop.op()
                    && *idx_dtype == DType::Index && idx.dtype().is_int() {
                        return Some(UOp::new(
                            Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![idx.clone()], gate: None },
                            result_dtype,
                        ));
                    }

                None
            },

        // ====================================================================
        // Pattern 12: Hanging cast cleanup (3-arg INDEX with valid)
        // ====================================================================
        // INDEX(buf, idx.cast(index), valid) where idx is concrete int → INDEX(buf, idx, valid)
        Index { buffer, indices, gate: Some(valid) }
            if indices.len() == 1 => |buffer, indices, valid| {
                let idx_uop = &indices[0];
                let Op::Cast { src: idx, dtype: idx_dtype } = idx_uop.op() else {
                    return None;
                };
                if *idx_dtype != DType::Index || !idx.dtype().is_int() {
                    return None;
                }
                // Derive element type from buffer's pointer type
                let result_dtype = match buffer.dtype() {
                    DType::Ptr { base, .. } => base.as_ref().clone(),
                    other => other,
                };
                Some(UOp::new(
                    Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![idx.clone()], gate: Some(valid.clone()) },
                    result_dtype,
                ))
            },

        // ====================================================================
        // Pattern 13: SINK cast strip
        // ====================================================================
        // Strip .cast(index) from sources in SINK
        Sink { sources } => |sources| {
            let mut changed = false;
            let new_sources: Vec<Arc<UOp>> = sources.iter()
                .map(|s| {
                    if let Op::Cast { src, dtype } = s.op()
                        && *dtype == DType::Index {
                            changed = true;
                            return src.clone();
                        }
                    s.clone()
                })
                .collect();
            if !changed { return None; }
            Some(UOp::sink(new_sources))
        },

        // ====================================================================
        // Pattern 14: END cast strip
        // ====================================================================
        // Strip .cast(index) from computation in END
        End { computation, ranges } => |computation, ranges| {
            if let Op::Cast { src, dtype } = computation.op()
                && *dtype == DType::Index {
                    return Some(UOp::new(Op::End { computation: src.clone(), ranges: ranges.clone() }, DType::Void));
                }
            None
        },
    }
}
