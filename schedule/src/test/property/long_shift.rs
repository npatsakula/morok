//! Property tests for the 64-bit word split in [`pm_long_decomp`].
//!
//! `pm_long_decomp` has no tinygrad counterpart — tinygrad targets only backends
//! with native 64-bit integers, so the shift rules in `decompositions.py` are the
//! `MUL`/`IDIV` -> shift strengthening, not a word split. Svod emits the split for
//! backends without native i64 (WebGPU), so each word must reproduce the native
//! `<<` / `>>` result bit for bit, including shifts of 32 or more.

use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{DType, ScalarDType};
use svod_ir::rewrite::graph_rewrite_bottom_up;
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::uop::eval::{eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};
use svod_ir::{Op, UOp};

use crate::devectorize::pm_long_decomp;
use crate::test::unit::devectorize::helpers::create_buffer_typed;
use svod_ir::ops;

/// Fold a fully constant word expression, mirroring what the backend would compute.
fn eval_word(expr: &Arc<UOp>) -> Option<ConstValue> {
    let dtype = expr.dtype().base();
    match expr.op() {
        Op::Const(constant) => Some(constant.0),
        Op::Unary(op, src) => eval_unary_op_typed(*op, eval_word(src)?, dtype),
        Op::Binary(op, lhs, rhs) => eval_binary_op_typed(*op, eval_word(lhs)?, eval_word(rhs)?, dtype),
        Op::Ternary(op, a, b, c) => eval_ternary_op_typed(*op, eval_word(a)?, eval_word(b)?, eval_word(c)?, dtype),
        Op::Cast(ops::Cast { src, .. }) | Op::BitCast(ops::BitCast { src, .. }) => {
            Some(reinterpret(eval_word(src)?, dtype))
        }
        _ => None,
    }
}

fn reinterpret(value: ConstValue, dtype: ScalarDType) -> ConstValue {
    let bits = match value {
        ConstValue::Int(v) => v as u32,
        ConstValue::UInt(v) => v as u32,
        ConstValue::Bool(v) => v as u32,
        other => return other,
    };
    if dtype.is_signed() { ConstValue::Int(bits as i32 as i64) } else { ConstValue::UInt(bits as u64) }
}

fn word_bits(value: ConstValue) -> u32 {
    match value {
        ConstValue::Int(v) => v as u32,
        ConstValue::UInt(v) => v as u32,
        other => panic!("word is not an integer: {other:?}"),
    }
}

/// The 64-bit constant `value` at `from`.
pub fn long_const(from: ScalarDType, value: u64) -> Arc<UOp> {
    let long = DType::Scalar(from);
    UOp::const_(long, if from == ScalarDType::Int64 { ConstValue::Int(value as i64) } else { ConstValue::UInt(value) })
}

/// A leftover 64-bit node means the rewrite stalled: `graph_rewrite_bottom_up`
/// then returns the input untouched instead of the word split.
pub fn assert_fully_split(root: &Arc<UOp>) {
    for node in root.toposort() {
        assert!(
            !matches!(node.dtype().base(), ScalarDType::Int64 | ScalarDType::UInt64),
            "64-bit node survived decomposition: {}",
            root.tree()
        );
    }
}

/// Split `STORE(buffer[at], value)` with `pm_long_decomp` and return each word's
/// stored bits and the element of the doubled 32-bit buffer it lands on.
pub fn split_store(from: ScalarDType, at: i64, value: Arc<UOp>) -> [(u32, i64); 2] {
    let index = UOp::index()
        .buffer(create_buffer_typed(8, from))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(at))])
        .call()
        .unwrap();
    let decomposed = graph_rewrite_bottom_up(&pm_long_decomp(), index.store(value), &mut ());
    assert_fully_split(&decomposed);

    let mut words = [None; 2];
    for node in decomposed.toposort() {
        let Op::Store(ops::Store { index, value, .. }) = node.op() else { continue };
        let Op::Index(ops::Index { indices, .. }) = index.op() else { panic!("a split store addresses an INDEX") };
        let address = eval_word(indices.last().expect("INDEX carries an index")).expect("address must fold");
        let word = node.tag().as_ref().expect("split store is word-tagged")[1];
        words[word] = Some((
            word_bits(eval_word(value).expect("word expression must fold")),
            address.try_int().expect("address"),
        ));
    }
    [words[0].expect("low word"), words[1].expect("high word")]
}

/// Run `pm_long_decomp` over `STORE(index, value)` and return `[low, high]`. The two
/// words must land on adjacent elements, never both on the same one.
pub fn split_long(from: ScalarDType, value: Arc<UOp>) -> [u32; 2] {
    let [(low, low_at), (high, high_at)] = split_store(from, 1, value);
    assert_eq!([low_at, high_at], [2, 3], "{from:?} word addresses");
    [low, high]
}

/// Run `pm_long_decomp` over `STORE(index, value <op> shift)` and return `[low, high]`.
pub fn split_shift(op: BinaryOp, value: u64, shift: u64, from: ScalarDType) -> [u32; 2] {
    let operands = Op::Binary(op, long_const(from, value), long_const(from, shift));
    split_long(from, UOp::new(operands, DType::Scalar(from)))
}

/// Native reference for `value <op> shift` at 64 bits.
pub fn native_shift(op: BinaryOp, value: u64, shift: u64, from: ScalarDType) -> u64 {
    match op {
        BinaryOp::Shl => value << shift,
        BinaryOp::Shr if from == ScalarDType::Int64 => ((value as i64) >> shift) as u64,
        BinaryOp::Shr => value >> shift,
        other => panic!("not a shift: {other:?}"),
    }
}

pub fn assert_shift_words(op: BinaryOp, value: u64, shift: u64, from: ScalarDType) {
    let expected = native_shift(op, value, shift, from);
    assert_eq!(
        split_shift(op, value, shift, from),
        [expected as u32, (expected >> 32) as u32],
        "{from:?} {value:#018x} {op:?} {shift}"
    );
}

fn assert_words(from: ScalarDType, value: Arc<UOp>, expected: u64, what: &str) {
    assert_eq!(split_long(from, value), [expected as u32, (expected >> 32) as u32], "{from:?} {what}");
}

/// `a * b` and `-a` at 64 bits, plus the float cast that reads both words.
pub fn assert_long_arithmetic(a: u64, b: u64, from: ScalarDType) {
    let mul = Op::Binary(BinaryOp::Mul, long_const(from, a), long_const(from, b));
    assert_words(from, UOp::new(mul, DType::Scalar(from)), a.wrapping_mul(b), "mul");
    let neg = Op::Unary(svod_ir::UnaryOp::Neg, long_const(from, a));
    assert_words(from, UOp::new(neg, DType::Scalar(from)), a.wrapping_neg(), "neg");
    // A cast away from a long is not itself split, so only the stall is observable.
    let cast = long_const(from, a).cast(DType::Float32);
    assert_fully_split(&graph_rewrite_bottom_up(&pm_long_decomp(), cast, &mut ()));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The word split of `x << s` / `x >> s` must equal the native 64-bit shift for every `s < 64`.
    #[test]
    fn long_shift_words_match_native(
        value in any::<u64>(),
        shift in 0u64..64,
        signed in any::<bool>(),
        right in any::<bool>(),
    ) {
        let from = if signed { ScalarDType::Int64 } else { ScalarDType::UInt64 };
        let op = if right { BinaryOp::Shr } else { BinaryOp::Shl };
        let expected = native_shift(op, value, shift, from);
        prop_assert_eq!(split_shift(op, value, shift, from), [expected as u32, (expected >> 32) as u32]);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// The word split of `a * b` and `-a` must equal the native 64-bit result.
    #[test]
    fn long_arithmetic_words_match_native(a in any::<u64>(), b in any::<u64>(), signed in any::<bool>()) {
        assert_long_arithmetic(a, b, if signed { ScalarDType::Int64 } else { ScalarDType::UInt64 });
    }
}
