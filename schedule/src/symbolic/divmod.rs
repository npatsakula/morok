//! Conservative div/mod congruence folding.
//!
//! Candidate construction uses mathematical integer algebra, but callers must
//! reject it unless both the original and replacement arithmetic trees are
//! proven not to wrap under their concrete dtype.

use std::sync::Arc;

use smallvec::SmallVec;

use svod_ir::UOp;
use svod_ir::ops;
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::helpers::gcd;
use svod_ir::uop::properties::SoundVminVmaxProperty;

pub(crate) fn uop_sum(terms: &[Arc<UOp>], template: &Arc<UOp>) -> Arc<UOp> {
    terms.iter().cloned().reduce(|sum, term| sum.add(&term)).unwrap_or_else(|| template.const_like(0i64))
}

fn scaled(term: &Arc<UOp>, coefficient: i64) -> Option<Arc<UOp>> {
    match coefficient {
        0 => Some(term.const_like(0i64)),
        1 => Some(term.clone()),
        _ => term.try_mul(&term.const_like(coefficient)).ok(),
    }
}

fn try_uop_sum(terms: &[Arc<UOp>], template: &Arc<UOp>) -> Option<Arc<UOp>> {
    let mut sum: Option<Arc<UOp>> = None;
    for term in terms {
        sum = Some(if let Some(sum) = sum { sum.try_add(term).ok()? } else { term.clone() });
    }
    Some(sum.unwrap_or_else(|| template.const_like(0i64)))
}

/// Fold an affine numerator modulo a positive constant divisor
/// (`divandmod.py:38-48`).
///
/// For `x = sum(f_i*t_i) + k`, choose `r_i == f_i (mod c)` and construct
/// `rem = sum(r_i*t_i) + (k mod c)`. Then `x = rem + c*Q` exactly, so if `rem`
/// stays in one quotient bucket:
///
/// * `x % c = rem - floor(rem/c)*c`
/// * `x // c = sum((f_i-r_i)/c*t_i) + (k-k%c+floor(rem/c)*c)/c`
///
/// The identity is sign-agnostic under floor division, and upstream carries no
/// numerator sign guard here (unlike `factor_remainder`, `divandmod.py:84`).
/// Upstream also searches both remainder signs
/// (`rem_choices`, `divandmod.py:41-45`).
///
/// This function only constructs the candidate. `exact_integer_rewrite` is the
/// mandatory typed no-wrap proof at the pattern call site.
pub fn fold_divmod_congruence(x: &Arc<UOp>, _c_uop: &Arc<UOp>, c_val: ConstValue, is_mod: bool) -> Option<Arc<UOp>> {
    // Hardware vectors need lane-wise constants and scaling. Keep this
    // indexing rewrite scalar rather than constructing a partial candidate.
    if x.dtype().vcount() != 1 {
        return None;
    }
    let ConstValue::Int(c) = c_val else { return None };
    if c <= 0 {
        return None;
    }
    let c128 = c as i128;

    let (without_const, constant) = x.pop_const(BinaryOp::Add);
    let ConstValue::Int(constant) = constant else { return None };
    let terms = without_const.split_uop(BinaryOp::Add);
    let decomposition: Option<Vec<_>> = terms
        .iter()
        .map(|term| {
            let factor = term.const_factor();
            (factor != 0).then(|| term.divides(factor)).flatten().map(|base| (base, factor))
        })
        .collect();
    let decomposition = decomposition?;

    // Both signs of the remainder for a lone term (it covers a binary numerator
    // that crosses one period) or on an exact `f%c == c//2` tie; otherwise the
    // smaller one, to keep the product over terms small (`divandmod.py:41-43`).
    let choices: Vec<SmallVec<[i64; 2]>> = decomposition
        .iter()
        .map(|(_, factor)| {
            let positive = i64::try_from((*factor as i128).rem_euclid(c128)).ok()?;
            let negative = positive.checked_sub(c)?;
            Some(if positive.checked_mul(2) == Some(c) || decomposition.len() == 1 {
                SmallVec::from_slice(&[positive, negative])
            } else if negative.unsigned_abs() < positive.unsigned_abs() {
                SmallVec::from_slice(&[negative])
            } else {
                SmallVec::from_slice(&[positive])
            })
        })
        .collect::<Option<_>>()?;

    // `itertools.product` order: the last choice varies fastest.
    let combinations = choices.iter().try_fold(1usize, |count, choice| count.checked_mul(choice.len()))?;
    let mut remainders = vec![0i64; choices.len()];
    (0..combinations).find_map(|mut code| {
        for (remainder, choice) in remainders.iter_mut().zip(&choices).rev() {
            *remainder = choice[code % choice.len()];
            code /= choice.len();
        }
        congruence_candidate(x, c, &decomposition, &remainders, constant, is_mod)
    })
}

/// One `rems` combination of [`fold_divmod_congruence`]'s remainder search.
fn congruence_candidate(
    x: &Arc<UOp>,
    c: i64,
    decomposition: &[(Arc<UOp>, i64)],
    remainders: &[i64],
    constant: i64,
    is_mod: bool,
) -> Option<Arc<UOp>> {
    let c128 = c as i128;
    let constant_remainder = constant.rem_euclid(c);

    let mut remainder_terms = Vec::new();
    for ((base, _), coefficient) in decomposition.iter().zip(remainders) {
        if *coefficient != 0 {
            remainder_terms.push(scaled(base, *coefficient)?);
        }
    }
    if constant_remainder != 0 {
        remainder_terms.push(x.const_like(constant_remainder));
    }
    let remainder = try_uop_sum(&remainder_terms, x)?;
    let (ConstValue::Int(rem_min), ConstValue::Int(rem_max)) = SoundVminVmaxProperty::get(&remainder).as_ref()? else {
        return None;
    };
    let quotient_bucket = rem_min.div_euclid(c);
    if quotient_bucket != rem_max.div_euclid(c) {
        return None;
    }

    if is_mod {
        let offset = (quotient_bucket as i128).checked_mul(c128)?;
        return if offset == 0 {
            Some(remainder)
        } else {
            remainder.try_sub(&x.const_like(i64::try_from(offset).ok()?)).ok()
        };
    }

    let mut quotient_terms = Vec::new();
    for ((base, factor), remainder) in decomposition.iter().zip(remainders) {
        let coefficient = (*factor as i128).checked_sub(*remainder as i128)?.checked_div(c128)?;
        if coefficient != 0 {
            quotient_terms.push(scaled(base, i64::try_from(coefficient).ok()?)?);
        }
    }
    let bucket_offset = (quotient_bucket as i128).checked_mul(c128)?;
    let constant_quotient =
        (constant as i128).checked_sub(constant_remainder as i128)?.checked_add(bucket_offset)?.checked_div(c128)?;
    if constant_quotient != 0 {
        quotient_terms.push(x.const_like(i64::try_from(constant_quotient).ok()?));
    }
    try_uop_sum(&quotient_terms, x)
}

/// Faithful port of tinygrad's `fold_divmod_general` (`uop/divandmod.py:8-96`).
///
/// Rules, in upstream order: `cancel_divmod` (:13), the PARAM-multiple guard
/// (:15), then the constant-denominator half — `nested_div` (:26),
/// `remove_nested_mod` (:29-36), the congruence fold (:38-48, delegated to
/// [`fold_divmod_congruence`]), `gcd_with_remainder` (:50-55) and the recursive
/// `nest_by_factor` (:57-70) with upstream's backward-slice cost minimisation —
/// and finally the denominator-agnostic fallback, `divide_by_gcd` (:79-83) and
/// `factor_remainder` (:85-96).
///
/// Divergences from upstream, both in the safe direction:
/// * a divisor whose range is exactly `{0}` declines the rewrite instead of
///   raising, so the caller leaves the original division in place;
/// * `nested_div` additionally requires `k.vmin > 0`. Upstream writes `k > 0`
///   on a UOp, which is vacuously true in Python and would mis-rewrite a
///   non-positive `k`.
///
/// Like [`fold_divmod_congruence`] this only constructs a candidate; the caller
/// must still run the typed no-wrap proof (`exact_integer_rewrite`).
pub(crate) fn fold_divmod_general(op: BinaryOp, x: &Arc<UOp>, y: &Arc<UOp>) -> Option<Arc<UOp>> {
    if x.dtype().vcount() != 1 {
        return None;
    }
    let is_mod = op == BinaryOp::FloorMod;
    let (y_min, y_max) = int_range(y)?;
    // Upstream raises ZeroDivisionError here; declining the rewrite keeps the
    // original node and leaves the diagnosis to the backend.
    if y_min == 0 && y_max == 0 {
        return None;
    }

    // cancel_divmod: the quotient lands in a single bucket, so it is constant.
    let (quotient_min, quotient_max) = int_range(&x.try_div(y).ok()?)?;
    if quotient_min == quotient_max {
        return if is_mod { x.try_sub(&scaled(y, quotient_min)?).ok() } else { Some(x.const_like(quotient_min)) };
    }

    // A parameter that is a known multiple of a constant divisor is irreducible.
    if let svod_ir::Op::Param(ops::Param { arg, .. }) = x.op()
        && let Some(multiple_of) = arg.multiple_of.and_then(|m| i64::try_from(m).ok())
        && let Some(c) = const_int(y).filter(|c| *c != 0)
        && multiple_of.checked_rem(c) == Some(0)
    {
        return is_mod.then(|| x.const_like(0i64));
    }

    let (peeled, constant) = x.pop_const(BinaryOp::Add);
    let ConstValue::Int(constant) = constant else { return None };
    let terms = peeled.split_uop(BinaryOp::Add);

    const_denominator_rules(op, x, y, &peeled, constant, &terms).or_else(|| variable_denominator_rules(op, x, y))
}

/// The constant-denominator half of `fold_divmod_general` (`divandmod.py:22-74`).
///
/// Only fires for a scalar constant divisor `> 0`. Returning `None` falls
/// through to [`variable_denominator_rules`], exactly as upstream does.
fn const_denominator_rules(
    op: BinaryOp,
    x: &Arc<UOp>,
    y: &Arc<UOp>,
    peeled: &Arc<UOp>,
    constant: i64,
    terms: &[Arc<UOp>],
) -> Option<Arc<UOp>> {
    let c = const_int(y).filter(|c| *c > 0)?;
    let is_mod = op == BinaryOp::FloorMod;

    // nested_div: (a % (k*c)) // c -> (a // c) % k, for k > 0.
    if !is_mod
        && let svod_ir::Op::Binary(BinaryOp::FloorMod, inner, modulus) = x.op()
        && let Some(k) = modulus.divides(c)
        && matches!(int_range(&k), Some((k_min, _)) if k_min > 0)
    {
        return inner.try_div(y).ok()?.try_mod(&k).ok();
    }

    // remove_nested_mod: (a % (k*c) + b) % c -> (a + b) % c.
    if is_mod {
        let stripped: Vec<Arc<UOp>> = terms
            .iter()
            .map(|term| match term.op() {
                svod_ir::Op::Binary(BinaryOp::FloorMod, inner, modulus) if modulus.divides(c).is_some() => {
                    inner.clone()
                }
                _ => term.clone(),
            })
            .collect();
        if stripped.iter().zip(terms).any(|(new, old)| !Arc::ptr_eq(new, old)) {
            let sum = try_uop_sum(&stripped, x)?;
            let sum = if constant == 0 { sum } else { sum.try_add(&x.const_like(constant)).ok()? };
            return sum.try_mod(y).ok();
        }
    }

    if let Some(folded) = fold_divmod_congruence(x, y, ConstValue::Int(c), is_mod) {
        return Some(folded);
    }

    let factors: Vec<i64> = terms.iter().map(|term| term.const_factor()).collect();
    let bases: Vec<Arc<UOp>> =
        terms.iter().zip(&factors).map(|(term, factor)| term.divides(*factor)).collect::<Option<_>>()?;

    // gcd_with_remainder: factor the common gcd of every coefficient out of both sides.
    let g = factors.iter().fold(c, |acc, factor| gcd(acc, *factor));
    if g > 1
        && let Some(reduced) = peeled.divides(g)
    {
        let shift = constant.div_euclid(g).rem_euclid(c / g);
        let reduced = if shift == 0 { reduced } else { reduced.try_add(&x.const_like(shift)).ok()? };
        if matches!(int_range(&reduced), Some((min, _)) if min >= 0) {
            let inner = x.const_like(c / g);
            return if is_mod {
                let scaled_up = reduced.try_mod(&inner).ok()?.try_mul(&x.const_like(g)).ok()?;
                offset_by(&scaled_up, constant.rem_euclid(g))
            } else {
                offset_by(&reduced.try_div(&inner).ok()?, constant.div_euclid(c))
            };
        }
    }

    nest_by_factor(op, x, c, constant, &bases, &factors, terms)
}

/// nest_by_factor (`divandmod.py:57-70`): `x//c -> (x//f)//(c//f)` and
/// `x%c -> (x//f % (c//f))*f + x%f`, over every coefficient `f` that properly
/// divides `c`. Upstream keeps the candidate with the smallest backward slice.
fn nest_by_factor(
    op: BinaryOp,
    x: &Arc<UOp>,
    c: i64,
    constant: i64,
    bases: &[Arc<UOp>],
    factors: &[i64],
    terms: &[Arc<UOp>],
) -> Option<Arc<UOp>> {
    let is_mod = op == BinaryOp::FloorMod;
    let (x_min, _) = int_range(x)?;

    let mut divisors: Vec<i64> = terms
        .iter()
        .zip(factors)
        .filter(|(term, factor)| {
            !matches!(term.op(), svod_ir::Op::Const(_)) && (2..c).contains(&factor.abs()) && c % factor.abs() == 0
        })
        .map(|(_, factor)| factor.abs())
        .collect();
    divisors.sort_unstable();
    divisors.dedup();

    let mut best: Option<(usize, Arc<UOp>)> = None;
    for divisor in divisors {
        let divisor_uop = x.const_like(divisor);
        let Some(nested) = fold_divmod_general(BinaryOp::FloorDiv, x, &divisor_uop) else { continue };
        let outer = x.const_like(c / divisor);
        let candidate = if !is_mod {
            let Ok(candidate) = nested.try_div(&outer) else { continue };
            candidate
        } else {
            // Reconstructing x from x//divisor needs the low digit x%divisor,
            // which is only the coefficient residues when it stays in [0, divisor).
            if x_min < 0 || !matches!(int_range(&nested), Some((min, _)) if min >= 0) {
                continue;
            }
            let mut low: Vec<Arc<UOp>> = bases
                .iter()
                .zip(factors)
                .filter(|(_, factor)| factor.rem_euclid(divisor) != 0)
                .map(|(base, factor)| scaled(base, factor.rem_euclid(divisor)))
                .collect::<Option<_>>()?;
            if constant.rem_euclid(divisor) != 0 {
                low.push(x.const_like(constant.rem_euclid(divisor)));
            }
            let digit = try_uop_sum(&low, x)?;
            if !matches!(int_range(&digit), Some((min, max)) if min >= 0 && max < divisor) {
                continue;
            }
            let Ok(high) = nested.try_mod(&outer).and_then(|rest| rest.try_mul(&divisor_uop)) else { continue };
            if low.is_empty() { high } else { high.try_add(&digit).ok()? }
        };
        let cost = candidate.node_count();
        if best.as_ref().is_none_or(|(best_cost, _)| cost < *best_cost) {
            best = Some((cost, candidate));
        }
    }
    best.map(|(_, candidate)| candidate)
}

/// The denominator-agnostic fallback (`divandmod.py:76-96`): `divide_by_gcd`
/// then `factor_remainder`. This is what folds `(N*i + j) // N` for symbolic `N`.
fn variable_denominator_rules(op: BinaryOp, x: &Arc<UOp>, y: &Arc<UOp>) -> Option<Arc<UOp>> {
    let is_mod = op == BinaryOp::FloorMod;
    let terms = x.split_uop(BinaryOp::Add);

    // divide_by_gcd: x op y -> (x/g) op (y/g), rescaled by g for the remainder.
    let mut with_divisor = terms.clone();
    with_divisor.push(y.clone());
    let divisor_gcd = UOp::symbolic_gcd(&with_divisor);
    if const_int(&divisor_gcd) != Some(1) {
        let folded = binary(op, &x.divide_exact(&divisor_gcd)?, &y.divide_exact(&divisor_gcd)?)?;
        return if is_mod { folded.try_mul(&divisor_gcd).ok() } else { Some(folded) };
    }

    // factor_remainder: (y*a + b) op y -> a + b//y / b%y, in the non-negative domain.
    non_negative(x)?;
    non_negative(y)?;
    let divisor = const_int(y);
    let (mut quotient, mut remainder) = (Vec::new(), Vec::new());
    for term in &terms {
        if let Some(exact) = term.divide_exact(y) {
            quotient.push(exact);
            continue;
        }
        // A constant divisor also splits a coefficient that is not already reduced.
        let split = divisor.and_then(|divisor| {
            let factor = term.const_factor();
            let residue = factor.rem_euclid(divisor);
            (residue != factor).then(|| term.divides(factor).map(|base| (base, residue, factor.div_euclid(divisor))))?
        });
        match split {
            Some((base, residue, carry)) => {
                remainder.push(scaled(&base, residue)?);
                quotient.push(if is_mod { x.const_like(0i64) } else { scaled(&base, carry)? });
            }
            None => remainder.push(term.clone()),
        }
    }
    if quotient.is_empty() {
        return None;
    }
    let new_x = try_uop_sum(&remainder, x)?;
    non_negative(&new_x)?;
    let folded = binary(op, &new_x, y)?;
    if is_mod { Some(folded) } else { folded.try_add(&try_uop_sum(&quotient, x)?).ok() }
}

fn offset_by(value: &Arc<UOp>, offset: i64) -> Option<Arc<UOp>> {
    if offset == 0 { Some(value.clone()) } else { value.try_add(&value.const_like(offset)).ok() }
}

fn binary(op: BinaryOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> Option<Arc<UOp>> {
    match op {
        BinaryOp::FloorMod => lhs.try_mod(rhs).ok(),
        BinaryOp::FloorDiv => lhs.try_div(rhs).ok(),
        _ => None,
    }
}

fn int_range(u: &Arc<UOp>) -> Option<(i64, i64)> {
    match SoundVminVmaxProperty::get(u).as_ref()? {
        (ConstValue::Int(min), ConstValue::Int(max)) => Some((*min, *max)),
        _ => None,
    }
}

fn const_int(u: &Arc<UOp>) -> Option<i64> {
    match u.op() {
        svod_ir::Op::Const(value) => value.0.try_int(),
        _ => None,
    }
}

fn non_negative(u: &Arc<UOp>) -> Option<()> {
    (int_range(u)?.0 >= 0).then_some(())
}
