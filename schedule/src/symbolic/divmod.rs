//! Div/Mod congruence folding (Tinygrad's `fold_divmod_congruence`).
//!
//! Decomposes `x` into `sum(factor_i * term_i) + const`, computes centered remainders,
//! and folds Mod/Idiv when the remainder range fits in one bucket.

use std::sync::Arc;

use morok_ir::types::{BinaryOp, ConstValue};
use morok_ir::uop::cached_property::CachedProperty;
use morok_ir::uop::properties::VminVmaxProperty;
use morok_ir::{IntoUOp, Op, UOp};

/// Tinygrad's fold_divmod_congruence (divandmod.py:46-50).
///
/// Guards with `x >= 0` so truncated div/mod == floor div/mod.
/// Handles partial fold for MOD: drops terms with `f % c == 0` (e.g., `(6*x + y) % 3 → y % 3`).
pub fn fold_divmod_congruence(x: &Arc<UOp>, c_uop: &Arc<UOp>, c_val: ConstValue, is_mod: bool) -> Option<Arc<UOp>> {
    let ConstValue::Int(c) = c_val else { return None };
    if c <= 0 {
        return None;
    }
    let dt = x.dtype();

    // Decompose: x = sum(factor_i * term_i) + k
    let (x_no_const, k_cv) = x.pop_const(BinaryOp::Add);
    let k = match k_cv {
        Some(ConstValue::Int(v)) => v,
        None => 0,
        _ => return None,
    };
    let uops: Vec<_> = x_no_const.split_uop(BinaryOp::Add);
    let decomp: Option<Vec<_>> = uops
        .iter()
        .map(|u| {
            let f = u.const_factor();
            if f == 0 {
                return None;
            }
            Some((u.divides_int(f)?, f))
        })
        .collect();
    let decomp = decomp?;

    // Guard: only fold when x >= 0 (truncated == floor for non-negatives).
    // Compute x_min from decomposition with checked arithmetic to avoid vmin overflow.
    let x_nonneg = decomp
        .iter()
        .try_fold(k, |acc, (t, f)| {
            let bound = if *f >= 0 {
                match t.vmin() {
                    ConstValue::Int(v) => *v,
                    _ => return None,
                }
            } else {
                match t.vmax() {
                    ConstValue::Int(v) => *v,
                    _ => return None,
                }
            };
            acc.checked_add(f.checked_mul(bound)?)
        })
        .is_some_and(|m| m >= 0);
    if !x_nonneg {
        return None;
    }

    // Centered remainders: min(f%c, f%c - c, key=abs)
    let rems: Vec<i64> = decomp
        .iter()
        .map(|(_, f)| {
            let r = f.rem_euclid(c);
            if (r - c).unsigned_abs() < r.unsigned_abs() { r - c } else { r }
        })
        .collect();

    // Build: rem = sum(rem_i * term_i) + k%c
    let kr = k.rem_euclid(c);
    let rem = rems
        .iter()
        .zip(decomp.iter())
        .filter(|(r, _)| **r != 0)
        .map(|(&r, (t, _))| if r == 1 { t.clone() } else { r.into_uop(dt.clone()).try_mul(t).ok().unwrap() })
        .reduce(|a, b| a.try_add(&b).ok().unwrap());
    let rem = match (rem, kr) {
        (Some(e), 0) => e,
        (Some(e), _) => e.try_add(&kr.into_uop(dt.clone())).ok()?,
        (None, _) => kr.into_uop(dt.clone()),
    };

    // Bucket check: rem.vmin//c == rem.vmax//c
    let (lo, hi) = match (rem.vmin(), rem.vmax()) {
        (ConstValue::Int(lo), ConstValue::Int(hi)) => (*lo, *hi),
        _ => return None,
    };
    if lo.div_euclid(c) != hi.div_euclid(c) {
        // Can't fully fold. For MOD: drop terms with f%c==0, re-wrap in Mod.
        if is_mod {
            let any_dropped = decomp.iter().any(|(_, f)| f.rem_euclid(c) == 0) || kr != k;
            if any_dropped {
                // Rebuild using positive remainders to keep sum non-negative
                let mut pos_rem: Option<Arc<UOp>> = if kr != 0 { Some(kr.into_uop(dt.clone())) } else { None };
                for (t, f) in &decomp {
                    let pr = f.rem_euclid(c);
                    if pr == 0 {
                        continue;
                    }
                    let v = if pr == 1 { t.clone() } else { pr.into_uop(dt.clone()).try_mul(t).ok()? };
                    pos_rem = Some(match pos_rem {
                        Some(a) => a.try_add(&v).ok()?,
                        None => v,
                    });
                }
                let pr = pos_rem.unwrap_or_else(|| 0i64.into_uop(dt.clone()));
                // Guard: pos_rem must be non-negative for truncated mod to equal floor mod
                if matches!(pr.vmin(), ConstValue::Int(v) if *v >= 0) {
                    return pr.try_mod(c_uop).ok();
                }
            }
        }
        return None;
    }
    let q = lo.div_euclid(c);

    if is_mod {
        // rem - floor(rem.vmin/c)*c
        if q == 0 { Some(rem) } else { rem.try_sub(&(q * c).into_uop(dt)).ok() }
    } else {
        // sum((f-r)/c * term) + (k - k%c + floor(rem.vmin/c)*c) / c
        let mut acc: Option<Arc<UOp>> = None;
        for (&r, (t, f)) in rems.iter().zip(decomp.iter()) {
            let fq = (f - r) / c;
            if fq == 0 {
                continue;
            }
            let v = if fq == 1 { t.clone() } else { fq.into_uop(dt.clone()).try_mul(t).ok()? };
            acc = Some(match acc {
                Some(a) => a.try_add(&v).ok()?,
                None => v,
            });
        }
        let kq = (k - kr + q * c) / c;
        if kq != 0 {
            let v = kq.into_uop(dt.clone());
            acc = Some(match acc {
                Some(a) => a.try_add(&v).ok()?,
                None => v,
            });
        }
        Some(acc.unwrap_or_else(|| 0i64.into_uop(dt)))
    }
}

/// Sum a list of UOps using the given template for dtype. Returns zero const if empty.
pub(crate) fn uop_sum(terms: &[Arc<UOp>], template: &Arc<UOp>) -> Arc<UOp> {
    if terms.is_empty() {
        return template.const_like(0i64);
    }
    terms.iter().cloned().reduce(|acc, t| acc.add(&t)).unwrap()
}

/// Unified divmod simplification function.
///
/// Based on Tinygrad's `fold_divmod_general` (divandmod.py:8-93).
/// Tries simplification rules in priority order, returning the first match.
///
/// Rules (in order):
/// 1. cancel_divmod — range lies in single denominator interval
/// 2. remove_nested_mod — `(a%4 + b)%2 → (a+b)%2` when 2|4
/// 3. fold_binary_numerator — single term with range of 2
/// 4. fold_divmod_congruence — factor congruence modular arithmetic
/// 5. gcd_with_remainder — factor out common GCD from numerator
/// 6. divide_by_gcd — variable denominator GCD factoring
/// 7. factor_remainder — `(d*x+y)//d → x+y//d` (last resort)
pub(crate) fn fold_divmod_general(op: BinaryOp, x: &Arc<UOp>, y: &Arc<UOp>) -> Option<Arc<UOp>> {
    let (x_vmin, x_vmax) = VminVmaxProperty::get(x);
    let (y_vmin, y_vmax) = VminVmaxProperty::get(y);
    let x_min = x_vmin.try_int()?;
    let x_max = x_vmax.try_int()?;
    let y_min = y_vmin.try_int()?;
    let y_max = y_vmax.try_int()?;

    // 0. Negative divisor/dividend normalization (Tinygrad divandmod.py:99-111).
    // Converts negative operands to positive, enabling downstream rules.
    if y_max < 0 {
        // x // d → -(x // (-d)) when d is always negative
        // x % d → x % (-d) when d is always negative
        let neg_y = y.neg();
        return if op == BinaryOp::Mod { x.try_mod(&neg_y).ok() } else { Some(x.try_div(&neg_y).ok()?.neg()) };
    }
    if x_max <= 0 {
        // x // d → -((-x) // d) when x is always non-positive
        // x % d → -((-x) % d) when x is always non-positive
        let neg_x = x.neg();
        return if op == BinaryOp::Mod {
            Some(neg_x.try_mod(y).ok()?.neg())
        } else {
            Some(neg_x.try_div(y).ok()?.neg())
        };
    }

    // 1. cancel_divmod: range of numerator lies within a single denominator interval
    if y_min * y_max > 0 {
        let corners =
            [x_min.checked_div(y_min), x_min.checked_div(y_max), x_max.checked_div(y_min), x_max.checked_div(y_max)];
        if let [Some(q1), Some(q2), Some(q3), Some(q4)] = corners
            && q1 == q2
            && q2 == q3
            && q3 == q4
        {
            let r = if op == BinaryOp::Mod {
                let qy = x.const_like(q1).try_mul(y).ok()?;
                x.try_sub(&qy).ok()?
            } else {
                x.const_like(q1)
            };

            return Some(r);
        }
    }

    // Peel constant from x
    let (x_peeled, pop_const) = x.pop_const(BinaryOp::Add);
    let const_val = match pop_const {
        Some(ConstValue::Int(v)) => v,
        None => 0,
        _ => return None,
    };
    let uops_no_const = x_peeled.split_uop(BinaryOp::Add);

    // ** Constant Denominator Rules ** (y is a scalar constant > 0)
    if let Op::Const(cv) = y.op()
        && let ConstValue::Int(c) = cv.0
        && c > 0
    // constant denom rules re-enabled
    {
        // 2. remove_nested_mod: (a%4 + b)%2 → (a+b)%2 when 2 divides 4
        if op == BinaryOp::Mod && x_min >= 0 {
            let mut new_xs = Vec::new();
            let mut changed = false;
            for u in &uops_no_const {
                if let Op::Binary(BinaryOp::Mod, inner_x, inner_y) = u.op()
                    && inner_y.divides_int(c).is_some()
                {
                    new_xs.push(Arc::clone(inner_x));
                    changed = true;
                } else {
                    new_xs.push(Arc::clone(u));
                }
            }
            if changed {
                let new_sum = uop_sum(&new_xs, y);
                let new_x = if const_val != 0 { new_sum.try_add(&x.const_like(const_val)).ok()? } else { new_sum };
                let (nv_min, _) = VminVmaxProperty::get(&new_x);
                if let ConstValue::Int(nv) = nv_min
                    && *nv >= 0
                {
                    let r = new_x.try_mod(y).ok()?;
                    return Some(r);
                }
            }
        }

        // Shared decomposition: factor each term as term * const_factor
        let decomp: Option<Vec<(Arc<UOp>, i64)>> = uops_no_const
            .iter()
            .map(|u| {
                let f = u.const_factor();
                u.divides_int(f).map(|t| (t, f))
            })
            .collect();
        let decomp = decomp?;
        let terms: Vec<Arc<UOp>> = decomp.iter().map(|(t, _)| t.clone()).collect();
        let factors: Vec<i64> = decomp.iter().map(|(_, f)| *f).collect();

        // 3. fold_binary_numerator: single non-const term with range of 2.
        // Guard: x_min >= 0 to avoid truncated/floor div mismatch.
        if terms.len() == 1 && x_min >= 0 {
            let v = &terms[0];
            let (vmin_cv, vmax_cv) = VminVmaxProperty::get(v);
            if let (ConstValue::Int(v_min), ConstValue::Int(v_max)) = (vmin_cv, vmax_cv)
                && v_max.checked_sub(*v_min) == Some(1)
            {
                let f = factors[0];
                let fv_min = f.checked_mul(*v_min)?.checked_add(const_val)?;
                let fv_max = f.checked_mul(*v_max)?.checked_add(const_val)?;
                let (y1, y2) = if op == BinaryOp::Mod { (fv_min % c, fv_max % c) } else { (fv_min / c, fv_max / c) };
                // (y2 - y1) * (v - v_min) + y1
                let v_shifted = v.try_sub(&v.const_like(*v_min)).ok()?;
                let r = v_shifted.try_mul(&v.const_like(y2 - y1)).ok()?.try_add(&v.const_like(y1)).ok()?;
                return Some(r);
            }
        }

        // 4. fold_divmod_congruence: fold if congruent to expression in [0, c)
        if x_min >= 0 {
            // rems = [min(f%c, f%c - c, key=abs) for f in factors]
            let rems: Vec<i64> = factors
                .iter()
                .map(|&f| {
                    let r = f.rem_euclid(c);
                    if (r - c).unsigned_abs() < r.unsigned_abs() { r - c } else { r }
                })
                .collect();

            // rem = sum(r*v for r,v in zip(rems, terms)) + const%c
            let mut rem_parts: Vec<Arc<UOp>> = Vec::new();
            for (&r, v) in rems.iter().zip(terms.iter()) {
                if r == 0 {
                    continue;
                }
                if r == 1 {
                    rem_parts.push(v.clone());
                } else {
                    rem_parts.push(v.try_mul(&v.const_like(r)).ok()?);
                }
            }
            let const_rem = const_val.rem_euclid(c);
            if const_rem != 0 {
                rem_parts.push(x.const_like(const_rem));
            }

            let rem = uop_sum(&rem_parts, x);
            let (rem_vmin, rem_vmax) = VminVmaxProperty::get(&rem);
            if let (ConstValue::Int(rem_min), ConstValue::Int(rem_max)) = (rem_vmin, rem_vmax) {
                // Python's // is floor division; use div_euclid for same semantics
                if rem_min.div_euclid(c) == rem_max.div_euclid(c) {
                    if op == BinaryOp::Mod {
                        let offset = rem_min.div_euclid(c) * c;
                        let r = if offset != 0 { rem.try_sub(&rem.const_like(offset)).ok()? } else { rem };
                        return Some(r);
                    } else {
                        let mut quo_parts: Vec<Arc<UOp>> = Vec::new();
                        for ((&f, &r), v) in factors.iter().zip(rems.iter()).zip(terms.iter()) {
                            let coeff = (f - r) / c;
                            if coeff == 0 {
                                continue;
                            }
                            if coeff == 1 {
                                quo_parts.push(v.clone());
                            } else {
                                quo_parts.push(v.try_mul(&v.const_like(coeff)).ok()?);
                            }
                        }
                        let const_quo = (const_val - const_rem + rem_min.div_euclid(c) * c) / c;
                        if const_quo != 0 {
                            quo_parts.push(x.const_like(const_quo));
                        }
                        let r = uop_sum(&quo_parts, x);
                        return Some(r);
                    }
                }
            }
        }

        // 5. gcd_with_remainder: factor out common GCD from numerator
        // Uses symbolic GCD matching Tinygrad's UOp.gcd(*uops_no_const, y)
        if x_min >= 0 {
            let mut gcd_inputs: Vec<Arc<UOp>> = uops_no_const.clone();
            gcd_inputs.push(Arc::clone(y));
            let g_uop = UOp::symbolic_gcd(&gcd_inputs);

            if let Op::Const(cv) = g_uop.op()
                && let ConstValue::Int(g) = cv.0
                && g > 1
                && let Some(new_x_base) = x_peeled.divide_exact(&g_uop)
            {
                let const_rem_div_g = (const_val.rem_euclid(c)) / g;
                let new_x = if const_rem_div_g != 0 {
                    new_x_base.try_add(&x.const_like(const_rem_div_g)).ok()?
                } else {
                    new_x_base
                };

                let (new_vmin, _) = VminVmaxProperty::get(&new_x);
                if let ConstValue::Int(nv) = new_vmin
                    && *nv >= 0
                {
                    let new_c_uop = x.const_like(c / g);
                    if op == BinaryOp::Mod {
                        let ret = new_x.try_mod(&new_c_uop).ok()?;
                        let result = ret.try_mul(&x.const_like(g)).ok()?;
                        let const_mod_g = const_val.rem_euclid(g);
                        let r =
                            if const_mod_g != 0 { result.try_add(&x.const_like(const_mod_g)).ok()? } else { result };
                        return Some(r);
                    } else {
                        let ret = new_x.try_div(&new_c_uop).ok()?;
                        let const_div_c = const_val / c;
                        let r = if const_div_c != 0 { ret.try_add(&x.const_like(const_div_c)).ok()? } else { ret };
                        return Some(r);
                    }
                }
            }
        }

        // 5b. nest_div_by_smallest_factor (Tinygrad divandmod.py:62-67)
        // For IDIV only: recursively divide c by the smallest factor found in numerator terms.
        // Each recursive call divides c by at least 2, so depth <= log2(c).
        if op == BinaryOp::Idiv && x_min >= 0 {
            let smallest = factors.iter().filter(|&&f| f.abs() > 1 && c % f == 0).map(|f| f.unsigned_abs()).min();
            if let Some(div) = smallest {
                let div = div.min(c as u64) as i64;
                if div > 1
                    && div < c
                    && let Some(inner) = x.divides_int(div)
                {
                    let remaining = c / div;
                    if let Some(result) = fold_divmod_general(BinaryOp::Idiv, &inner, &x.const_like(remaining)) {
                        let (smin, _) = VminVmaxProperty::get(&result);
                        if let ConstValue::Int(sv) = smin
                            && *sv >= 0
                        {
                            return Some(result);
                        }
                    }
                }
            }
        }
    }

    // ** Variable Denominator / Fallback Rules **
    let mut all_uops = uops_no_const;
    if const_val != 0 {
        all_uops.push(x.const_like(const_val));
    }

    // 6. divide_by_gcd: x//y → (x//gcd)//(y//gcd)
    // Uses symbolic GCD matching Tinygrad's UOp.gcd(*all_uops, y)
    {
        let mut gcd_inputs: Vec<Arc<UOp>> = all_uops.clone();
        gcd_inputs.push(Arc::clone(y));
        let g_uop = UOp::symbolic_gcd(&gcd_inputs);

        let is_trivial = matches!(g_uop.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Int(1)));
        if !is_trivial
            && let Some(x_div) = x.divide_exact(&g_uop)
            && let Some(y_div) = y.divide_exact(&g_uop)
        {
            let r = if op == BinaryOp::Mod {
                let ret = x_div.try_mod(&y_div).ok()?;
                ret.try_mul(&g_uop).ok()?
            } else {
                x_div.try_div(&y_div).ok()?
            };
            return Some(r);
        }
    }

    // 7. factor_remainder: (d*x+y)//d → x+y//d
    if y_min < 0 || x_min < 0 {
        return None;
    }

    let mut quo = Vec::new();
    let mut rem = Vec::new();
    for u in &all_uops {
        if let Some(q) = u.divide_exact(y) {
            quo.push(q);
        } else if op == BinaryOp::Mod
            && let Op::Const(cv) = y.op()
            && let ConstValue::Int(y_arg) = cv.0
        {
            let cf = u.const_factor();
            if cf.rem_euclid(y_arg) != cf {
                let reduced = u.divides_int(cf)?.try_mul(&u.const_like(cf.rem_euclid(y_arg))).ok()?;
                rem.push(reduced);
                quo.push(u.const_like(0i64));
            } else {
                rem.push(Arc::clone(u));
            }
        } else {
            rem.push(Arc::clone(u));
        }
    }

    if quo.is_empty() {
        return None;
    }

    let new_x = uop_sum(&rem, x);
    let (new_x_vmin, _) = VminVmaxProperty::get(&new_x);
    let ConstValue::Int(nv) = new_x_vmin else {
        return None;
    };
    if *nv < 0 {
        return None;
    }

    let r = if op == BinaryOp::Mod {
        new_x.try_mod(y).ok()?
    } else {
        let quo_sum = uop_sum(&quo, x);
        new_x.try_div(y).ok()?.try_add(&quo_sum).ok()?
    };
    Some(r)
}
