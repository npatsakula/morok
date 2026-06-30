//! Pure tests for ST swizzles.

use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::uop::eval::eval_binary_op;
use svod_ir::{ConstValue, Op, UOp};

use crate::swizzle::Swizzle;

/// Fold a pure constant `Index` expression tree (the swizzle is `Const`s under
/// `Binary` ops only) to its `i64` value.
fn eval_const(u: &Arc<UOp>) -> i64 {
    match u.op() {
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(i) => i,
            other => panic!("eval_const: non-int const {other:?}"),
        },
        Op::Binary(op, a, b) => {
            let (av, bv) = (eval_const(a), eval_const(b));
            match eval_binary_op(*op, ConstValue::Int(av), ConstValue::Int(bv)) {
                Some(ConstValue::Int(r)) => r,
                other => panic!("eval_const: {op:?}({av},{bv}) folded to {other:?}"),
            }
        }
        other => panic!("eval_const: unexpected op {other:?}"),
    }
}

/// The whole-tile element offset for in-tile `(row, col)` under `sw`.
fn offset(sw: Swizzle, row: usize, col: usize, rows: usize, cols: usize, scalar: ScalarDType) -> i64 {
    let cidx = |v: usize| UOp::const_(DType::Index, ConstValue::Int(v as i64));
    eval_const(&sw.tile_offset(cidx(row), cidx(col), rows, cols, scalar))
}

/// A swizzle must be a bijection of `[0,rows)×[0,cols)` onto the flat tile slots
/// `[0, rows*cols)` (else LDS round-trips corrupt), for every store/load to land
/// on the same slot.
fn assert_bijection(sw: Swizzle, rows: usize, cols: usize, scalar: ScalarDType) {
    let mut seen = vec![false; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let slot = offset(sw, r, c, rows, cols, scalar);
            assert!((0..(rows * cols) as i64).contains(&slot), "{sw:?}: ({r},{c}) -> {slot} out of range");
            assert!(!seen[slot as usize], "{sw:?}: collision at slot {slot} — not a bijection");
            seen[slot as usize] = true;
        }
    }
    assert!(seen.iter().all(|&b| b), "{sw:?}: not surjective");
}

#[test]
fn test_swizzle_is_bijection() {
    for &(rows, cols) in &[(16, 16), (16, 64), (64, 64), (256, 64), (128, 128), (64, 32)] {
        assert_bijection(Swizzle::Sw16x16, rows, cols, ScalarDType::BFloat16);
    }
    // f32 LDS tiles (e.g. the FA accumulator scratch).
    assert_bijection(Swizzle::Sw16x16, 64, 64, ScalarDType::Float32);
}

/// The bank-conflict fix (gfx942): a single warp's `ds_read_b64` MFMA gather of a
/// 16×16 bf16 fragment must touch 32 distinct LDS banks (no conflict). Lane `l`
/// reads `row = l%16`, the 4-wide K-run at `col = (l/16)*4`; the 16 lanes of one
/// column group must cover all 32 banks (16 reads × 2 words). The OLD per-16-col
/// swizzle collapsed rows `r, r+4, r+8, r+12` onto the same bank (4-way conflict).
#[test]
fn test_sw16x16_gather_no_bank_conflict() {
    let scalar = ScalarDType::BFloat16;
    let itemsize = 2usize;
    for &(rows, cols) in &[(64, 64), (256, 64), (128, 128)] {
        for fh in 0..rows / 16 {
            for fw in 0..cols / 16 {
                for g in 0..4usize {
                    // The 16 lanes of column group `g`, each a 2-word (b64) read.
                    let mut banks = std::collections::HashSet::new();
                    for r in 0..16usize {
                        let (full_row, full_col) = (fh * 16 + r, fw * 16 + g * 4);
                        let elem = offset(Swizzle::Sw16x16, full_row, full_col, rows, cols, scalar);
                        let word = (elem * itemsize as i64) / 4; // 4-byte LDS bank word
                        banks.insert(word % 32);
                        banks.insert((word + 1) % 32);
                    }
                    assert_eq!(banks.len(), 32, "{rows}x{cols} frag({fh},{fw}) group {g}: bank conflict");
                }
            }
        }
    }
}

/// The 4-wide `ds_read_b64`/`ds_write_b64` K-run must land on 4 contiguous slots
/// with a constant swizzle delta (else the vectorized gather/commit reorders the
/// run). Holds because the XOR delta only changes every 128 bytes (`>>7`).
#[test]
fn test_sw16x16_vec4_run_contiguous() {
    let scalar = ScalarDType::BFloat16;
    for &(rows, cols) in &[(64, 64), (256, 64)] {
        for r in 0..rows {
            for c0 in (0..cols).step_by(4) {
                let base = offset(Swizzle::Sw16x16, r, c0, rows, cols, scalar);
                for k in 1..4 {
                    let here = offset(Swizzle::Sw16x16, r, c0 + k, rows, cols, scalar);
                    assert_eq!(here, base + k as i64, "({r},{c0}) run broke at +{k}");
                }
            }
        }
    }
}

#[test]
fn test_base_shape_arithmetic() {
    use crate::tiles::{RT_16X16, RT_32X32, ST_16X16};
    // 16x16 over wave64 -> 4 elements/thread; RT stride 4 -> 1 stride-group.
    assert_eq!(ST_16X16.base.elements_per_thread(), 4);
    assert_eq!(RT_16X16.elements_per_thread(), 4);
    assert_eq!(RT_16X16.num_strides(), 1);
    // 32x32 over wave64 -> 16 elements/thread; stride 4 -> 4 stride-groups.
    assert_eq!(RT_32X32.elements_per_thread(), 16);
    assert_eq!(RT_32X32.num_strides(), 4);
}
