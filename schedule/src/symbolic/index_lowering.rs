//! Index dtype lowering patterns.
//!
//! Converts abstract Index dtype to concrete integer types (i32 or i64)
//! based on value bounds analysis. Follows Tinygrad's cascade approach.
//!
//! ## Cascade Pattern (from Tinygrad)
//!
//! Phase 1 - Create wrappers:
//!   CONST(Index) → CONST(concrete).cast(Index)
//!   DEFINE_VAR(Index) → DEFINE_VAR(concrete).cast(Index)
//!
//! Phase 2 - Process wrapped values:
//!   Binary(x.cast(Index), y.cast(Index)) → Binary(x, y, concrete).cast(Index)
//!   RANGE(end.cast(Index)) → RANGE(end, concrete).cast(Index)
//!
//! Phase 3 - Strip at terminals:
//!   INDEX(idx.cast(Index)) → INDEX(idx)
//!   SINK/END strip .cast(Index)

use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::types::ConstValue;
use morok_ir::uop::cached_property::CachedProperty;
use morok_ir::uop::properties::VminVmaxProperty;
use morok_ir::{Op, UOp};

use crate::TypedPatternMatcher;

/// Select concrete dtype based on bounds analysis.
fn select_dtype(uop: &Arc<UOp>) -> DType {
    let (vmin, vmax) = VminVmaxProperty::get(uop);
    let fits_i32 = match (vmin, vmax) {
        (ConstValue::Int(min), ConstValue::Int(max)) => *min >= i32::MIN as i64 && *max <= i32::MAX as i64,
        (ConstValue::UInt(min), ConstValue::UInt(max)) => *min <= i32::MAX as u64 && *max <= i32::MAX as u64,
        _ => false,
    };
    if fits_i32 { DType::Scalar(ScalarDType::Int32) } else { DType::Scalar(ScalarDType::Int64) }
}

/// Compute least upper dtype for integer types.
fn least_upper_dtype(a: &DType, b: &DType) -> DType {
    match (a, b) {
        (DType::Scalar(ScalarDType::Int64), _) | (_, DType::Scalar(ScalarDType::Int64)) => {
            DType::Scalar(ScalarDType::Int64)
        }
        _ => DType::Scalar(ScalarDType::Int32),
    }
}

/// Pattern matcher for lowering Index dtype to concrete i32/i64.
/// Based on Tinygrad's pm_lower_index_dtype.
pub fn pm_lower_index_dtype() -> TypedPatternMatcher {
    crate::patterns! {
        // ================================================================
        // PHASE 1: Create wrappers (leaf nodes)
        // ================================================================

        // CONST(Index) → CONST(concrete).cast(Index)
        // Tinygrad: u.replace(dtype=select_dtype(u)).cast(u.dtype)
        c @const(cv) if c.dtype() == DType::Index => |c, cv| {
            let dt = select_dtype(c);
            Some(UOp::const_(dt, cv).cast(DType::Index))
        },

        // VCONST(Vector<Index, N>) → VCONST(Vector<concrete, N>).cast(Vector<Index, N>)
        // Tinygrad: (CONST, VCONST) with dtype=index → u.replace(dtype=select_dtype(u)).cast(u.dtype)
        vc @ VConst { values } if vc.dtype().base() == ScalarDType::Index => |vc, values| {
            let dt = select_dtype(vc);
            let vcount = vc.dtype().vcount();
            let vec_dt = dt.vec(vcount);
            let vec_index_dt = DType::Vector { scalar: ScalarDType::Index, count: vcount };
            let new_vc = UOp::new(Op::VConst { values: values.clone() }, vec_dt);
            Some(new_vc.cast(vec_index_dt))
        },

        // DEFINE_VAR(Index) → DEFINE_VAR(concrete).cast(Index)
        dv @ DefineVar { name, min_val, max_val } if dv.dtype() == DType::Index => |dv, name, min_val, max_val| {
            let dt = select_dtype(dv);
            let var = UOp::new(Op::DefineVar { name: name.clone(), min_val: *min_val, max_val: *max_val }, dt);
            Some(var.cast(DType::Index))
        },

        // ================================================================
        // PHASE 2: Process wrapped values
        // ================================================================

        // Binary(x.cast(Index), y.cast(Index)) → x.cast(dt).alu(op, y.cast(dt)).cast(Index)
        // Tinygrad: x.cast(dt:=least_upper_dtype(select_dtype(u), x.dtype, y.dtype)).alu(u.op, y.cast(dt)).cast(u.dtype)
        node if node.dtype() == DType::Index && matches!(node.op(), Op::Binary(_, _, _)) => |node| {
            let Op::Binary(op, lhs, rhs) = node.op() else { return None };

            // Both operands must be .cast(Index) wrappers
            let (Op::Cast { src: x, dtype: lhs_dt }, Op::Cast { src: y, dtype: rhs_dt }) = (lhs.op(), rhs.op()) else {
                return None;
            };
            if *lhs_dt != DType::Index || *rhs_dt != DType::Index {
                return None;
            }

            // dt = least_upper_dtype(select_dtype(result), x.dtype, y.dtype)
            let result_dt = select_dtype(node);
            let dt = least_upper_dtype(&result_dt, &least_upper_dtype(&x.dtype(), &y.dtype()));

            // x.cast(dt).alu(op, y.cast(dt)).cast(Index)
            let result = UOp::new(Op::Binary(*op, x.cast(dt.clone()), y.cast(dt.clone())), dt);
            Some(result.cast(DType::Index))
        },

        // WHERE(cond, x.cast(Index), y.cast(Index)) → WHERE(cond, x.cast(dt), y.cast(dt)).cast(Index)
        Where(cond, true_val, false_val)
            if true_val.dtype() == DType::Index && false_val.dtype() == DType::Index => |cond, true_val, false_val| {
            let (Op::Cast { src: x, dtype: t_dt }, Op::Cast { src: y, dtype: f_dt }) = (true_val.op(), false_val.op()) else {
                return None;
            };
            if *t_dt != DType::Index || *f_dt != DType::Index {
                return None;
            }

            let dt = least_upper_dtype(&x.dtype(), &y.dtype());
            let result = UOp::try_where(cond.clone(), x.cast(dt.clone()), y.cast(dt)).ok()?;
            Some(result.cast(DType::Index))
        },

        // RANGE(end.cast(Index)) → RANGE(end, end.dtype).cast(Index)
        // Tinygrad: r.replace(dtype=end.dtype, src=(end,)).cast(dtypes.index)
        range @ Range { end, axis_id, axis_type } if range.dtype() == DType::Index => |end, axis_id, axis_type| {
            let Op::Cast { src: end_inner, dtype: end_dt } = end.op() else {
                return None;
            };
            if *end_dt != DType::Index {
                return None;
            }

            let dt = end_inner.dtype();
            let result = UOp::new(Op::Range { end: end_inner.clone(), axis_id: *axis_id, axis_type: *axis_type }, dt);
            Some(result.cast(DType::Index))
        },

        // SPECIAL(end.cast(Index)) → SPECIAL(end, i32).cast(Index)
        // Tinygrad: u.replace(dtype=dtypes.int, src=(var,)).cast(dtypes.index)
        special @ Special { name, end } if special.dtype() == DType::Index => |name, end| {
            let Op::Cast { src: end_inner, dtype: end_dt } = end.op() else {
                return None;
            };
            if *end_dt != DType::Index {
                return None;
            }

            let i32_dt = DType::Scalar(ScalarDType::Int32);
            let result = UOp::new(Op::Special { end: end_inner.clone(), name: name.clone() }, i32_dt);
            Some(result.cast(DType::Index))
        },

        // VECTORIZE(e0.cast(Index), ...) → VECTORIZE(e0.cast(dt), ...).cast(Vector<Index>)
        vec @ Vectorize { elements } if vec.dtype().base() == ScalarDType::Index => |vec, elements| {
            let inner: Option<Vec<_>> = elements.iter().map(|e| {
                match e.op() {
                    Op::Cast { src, dtype } if *dtype == DType::Index => Some(src.clone()),
                    _ => None,
                }
            }).collect();
            let inner = inner?;

            let dt = select_dtype(vec);
            let casted: Vec<_> = inner.iter().map(|e| e.cast(dt.clone())).collect();
            let vec_index_dt = DType::Vector { scalar: ScalarDType::Index, count: elements.len() };
            Some(UOp::vectorize(casted.into()).cast(vec_index_dt))
        },

        // BIND(var.cast(Index), val.cast(Index)) → var.bind(val).cast(Index)
        // Tinygrad: (UPat(Ops.BIND, src=(var.cast(index), val.cast(index))), lambda var,val: var.bind(val).cast(index))
        Bind { var, value } if var.dtype() == DType::Index => |var, value| {
            let Op::Cast { src: var_inner, dtype: var_dt } = var.op() else { return None };
            let Op::Cast { src: val_inner, dtype: val_dt } = value.op() else { return None };
            if *var_dt != DType::Index || *val_dt != DType::Index { return None; }

            // Compute common dtype for the binding
            let dt = least_upper_dtype(&var_inner.dtype(), &val_inner.dtype());
            let bound = var_inner.cast(dt.clone()).bind(val_inner.cast(dt));
            Some(bound.cast(DType::Index))
        },

        // ================================================================
        // PHASE 3: Cleanup - strip wrappers at terminal nodes
        // ================================================================

        // INDEX(buf, idx.cast(Index)) → INDEX(buf, idx)
        Index { buffer, indices, gate: None } if indices.len() == 1 => |buffer, indices| {
            let idx = &indices[0];
            let Op::Cast { src, dtype } = idx.op() else { return None };
            if *dtype != DType::Index || !src.dtype().is_int() { return None; }

            let result_dt = match buffer.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            Some(UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![src.clone()], gate: None }, result_dt))
        },

        // INDEX(buf, idx.cast(Index), valid) → INDEX(buf, idx, valid)
        Index { buffer, indices, gate: Some(valid) } if indices.len() == 1 => |buffer, indices, valid| {
            let idx = &indices[0];
            let Op::Cast { src, dtype } = idx.op() else { return None };
            if *dtype != DType::Index || !src.dtype().is_int() { return None; }

            let result_dt = match buffer.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            Some(UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![src.clone()], gate: Some(valid.clone()) }, result_dt))
        },

        // INDEX with Invalid: buf.index(cond.where(idx, Invalid)) → INDEX(buf, idx, gate=cond)
        Index { buffer, indices, gate: None } if indices.len() == 1 => |buffer, indices| {
            let idx_uop = &indices[0];
            let Op::Ternary(morok_ir::TernaryOp::Where, cond, idx, false_val) = idx_uop.op() else {
                return None;
            };
            if !matches!(false_val.op(), Op::Invalid) { return None; }

            let result_dt = match buffer.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            Some(UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![idx.clone()], gate: Some(cond.clone()) }, result_dt))
        },

        // SINK - strip .cast(Index) from sources
        Sink { sources } => |sources| {
            let mut changed = false;
            let new_sources: Vec<Arc<UOp>> = sources.iter().map(|s| {
                if let Op::Cast { src, dtype } = s.op() && *dtype == DType::Index {
                    changed = true;
                    src.clone()
                } else {
                    s.clone()
                }
            }).collect();
            if !changed { return None; }
            Some(UOp::sink(new_sources))
        },

        // END - strip .cast(Index) from computation
        End { computation, ranges } => |computation, ranges| {
            let Op::Cast { src, dtype } = computation.op() else { return None };
            if *dtype != DType::Index { return None; }
            Some(UOp::new(Op::End { computation: src.clone(), ranges: ranges.clone() }, DType::Void))
        },
    }
}
