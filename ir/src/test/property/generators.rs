//! Generators for property-based testing.
//!
//! Provides Arbitrary implementations and custom strategies for generating
//! UOp graphs, constants, operations, and dtype families.

use std::rc::Rc;

use half::bf16;
use proptest::prelude::*;

use morok_dtype::{DType, ScalarDType};

use crate::UOp;
use crate::types::{BinaryOp, ConstValue, TernaryOp, UnaryOp};

use morok_dtype::ScalarDType as Scalar;
static NON_SUPPORTED: &[Scalar] = &[Scalar::FP8E4M3, Scalar::FP8E5M2, Scalar::Index, Scalar::Void];

pub fn const_(dtype: ScalarDType) -> impl Strategy<Value = ConstValue> {
    use ScalarDType::*;
    match dtype {
        Int8 => any::<i8>().prop_map(|i| ConstValue::Int(i as i64)).boxed(),
        Int16 => any::<i16>().prop_map(|i| ConstValue::Int(i as i64)).boxed(),
        Int32 => any::<i32>().prop_map(|i| ConstValue::Int(i as i64)).boxed(),
        Int64 => any::<i64>().prop_map(ConstValue::Int).boxed(),
        UInt8 => any::<u8>().prop_map(|i| ConstValue::UInt(i as u64)).boxed(),
        UInt16 => any::<u16>().prop_map(|i| ConstValue::UInt(i as u64)).boxed(),
        UInt32 => any::<u32>().prop_map(|i| ConstValue::UInt(i as u64)).boxed(),
        UInt64 => any::<u64>().prop_map(ConstValue::UInt).boxed(),
        Bool => any::<bool>().prop_map(ConstValue::Bool).boxed(),
        Float16 => any::<f32>().prop_map(|i| ConstValue::Float(half::f16::from_f32(i).to_f64())).boxed(),
        BFloat16 => any::<f32>().prop_map(|i| ConstValue::Float(bf16::from_f32(i).to_f64())).boxed(),
        Float32 => any::<f32>().prop_map(|i| ConstValue::Float(i as f64)).boxed(),
        Float64 => any::<f64>().prop_map(ConstValue::Float).boxed(),
        _ => unreachable!(),
    }
}

/// Generate scalar DType suitable for arithmetic operations.
pub fn arithmetic_sdtype() -> impl Strategy<Value = ScalarDType> {
    morok_dtype::test::proptests::generators::scalar_generator()
        .prop_filter("only supported types", |sdtype| !NON_SUPPORTED.contains(sdtype))
}

pub fn const_pair() -> impl Strategy<Value = (DType, ConstValue)> {
    arithmetic_sdtype().prop_flat_map(|sdtype| const_(sdtype).prop_map(move |value| (DType::Scalar(sdtype), value)))
}

// ============================================================================
// ConstValue Generators
// ============================================================================

// /// Generate arbitrary ConstValue with reasonable bounds.
// pub fn arb_const_value() -> impl Strategy<Value = ConstValue> {
//     prop_oneof![
//         (-1000i64..=1000).prop_map(ConstValue::Int),
//         (0u64..=1000).prop_map(ConstValue::UInt),
//         (-100.0..=100.0).prop_map(ConstValue::Float),
//         any::<bool>().prop_map(ConstValue::Bool),
//     ]
// }

/// Generate small integer constants (useful for arithmetic tests).
pub fn arb_small_int() -> impl Strategy<Value = ConstValue> {
    (-10i64..=10).prop_map(ConstValue::Int)
}

/// Generate non-zero constants (useful for division/mod tests).
pub fn nonzero_int() -> impl Strategy<Value = ConstValue> {
    any::<i64>().prop_filter("non zer", |&x| x != 0).prop_map(ConstValue::Int)
}

// ============================================================================
// DType Generators
// ============================================================================

/// Generate integer DType (signed and unsigned).
pub fn arb_int_dtype() -> impl Strategy<Value = DType> {
    morok_dtype::test::proptests::generators::int_dtype().prop_map(Into::into)
}

/// Generate float DType.
pub fn arb_float_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![Just(DType::Float16), Just(DType::Float32), Just(DType::Float64),]
}

// ============================================================================
// DType Family Generators (for widening tests)
// ============================================================================

/// DType family for testing widening transformations.
#[derive(Debug, Clone)]
pub enum DTypeFamily {
    SignedInt,
    UnsignedInt,
    Float,
}

impl DTypeFamily {
    /// Get all dtypes in this family, in widening order.
    pub fn widening_sequence(&self) -> Vec<DType> {
        match self {
            Self::SignedInt => vec![DType::Int8, DType::Int16, DType::Int32, DType::Int64],
            Self::UnsignedInt => vec![DType::UInt8, DType::UInt16, DType::UInt32, DType::UInt64],
            Self::Float => vec![DType::Float16, DType::Float32, DType::Float64],
        }
    }

    /// Get the narrowest dtype in this family.
    pub fn narrowest(&self) -> DType {
        self.widening_sequence()[0].clone()
    }

    /// Get the widest dtype in this family.
    pub fn widest(&self) -> DType {
        let seq = self.widening_sequence();
        seq[seq.len() - 1].clone()
    }
}

pub fn arb_dtype_family() -> impl Strategy<Value = DTypeFamily> {
    prop_oneof![Just(DTypeFamily::SignedInt), Just(DTypeFamily::UnsignedInt), Just(DTypeFamily::Float),]
}

// ============================================================================
// Operation Generators
// ============================================================================

/// Generate arbitrary BinaryOp, weighted towards common operations.
pub fn arb_binary_op() -> impl Strategy<Value = BinaryOp> {
    prop_oneof![
        5 => Just(BinaryOp::Add),
        5 => Just(BinaryOp::Mul),
        4 => Just(BinaryOp::Sub),
        2 => Just(BinaryOp::Idiv),
        2 => Just(BinaryOp::Mod),
        3 => Just(BinaryOp::Max),
        1 => Just(BinaryOp::Pow),
        3 => Just(BinaryOp::Lt),
        3 => Just(BinaryOp::Eq),
        3 => Just(BinaryOp::Ne),
        2 => Just(BinaryOp::And),
        2 => Just(BinaryOp::Or),
        1 => Just(BinaryOp::Xor),
    ]
}

/// Generate arithmetic BinaryOp only.
pub fn arb_arithmetic_binary_op() -> impl Strategy<Value = BinaryOp> {
    prop_oneof![Just(BinaryOp::Add), Just(BinaryOp::Mul), Just(BinaryOp::Sub), Just(BinaryOp::Max),]
}

/// Generate commutative BinaryOp.
pub fn arb_commutative_binary_op() -> impl Strategy<Value = BinaryOp> {
    prop_oneof![
        Just(BinaryOp::Add),
        Just(BinaryOp::Mul),
        Just(BinaryOp::Eq),
        Just(BinaryOp::Ne),
        Just(BinaryOp::And),
        Just(BinaryOp::Or),
        Just(BinaryOp::Xor),
        Just(BinaryOp::Max),
    ]
}

/// Generate associative BinaryOp.
pub fn arb_associative_binary_op() -> impl Strategy<Value = BinaryOp> {
    prop_oneof![Just(BinaryOp::Add), Just(BinaryOp::Mul), Just(BinaryOp::And), Just(BinaryOp::Or), Just(BinaryOp::Max),]
}

/// Generate arbitrary UnaryOp.
pub fn arb_unary_op() -> impl Strategy<Value = UnaryOp> {
    prop_oneof![
        Just(UnaryOp::Neg),
        Just(UnaryOp::Sqrt),
        Just(UnaryOp::Exp2),
        Just(UnaryOp::Log2),
        Just(UnaryOp::Sin),
        Just(UnaryOp::Reciprocal),
        Just(UnaryOp::Trunc),
    ]
}

/// Generate arbitrary TernaryOp.
pub fn arb_ternary_op() -> impl Strategy<Value = TernaryOp> {
    prop_oneof![Just(TernaryOp::Where), Just(TernaryOp::MulAcc),]
}

// ============================================================================
// UOp Generators
// ============================================================================

/// Generate a constant UOp with given dtype.
pub fn arb_const_uop(dtype: DType) -> impl Strategy<Value = Rc<UOp>> {
    const_(dtype.scalar().unwrap()).prop_map(move |cv| UOp::const_(dtype.clone(), cv))
}

/// Generate a variable UOp with bounded range.
pub fn arb_var_uop(dtype: DType) -> impl Strategy<Value = Rc<UOp>> {
    ("[a-z]", 0i64..100, 1i64..100).prop_map(move |(name, min, size)| UOp::var(name, dtype.clone(), min, min + size))
}

/// Generate a simple UOp (constant or variable).
pub fn arb_simple_uop(dtype: DType) -> impl Strategy<Value = Rc<UOp>> {
    prop_oneof![arb_const_uop(dtype.clone()), arb_var_uop(dtype),]
}

/// Generate an arithmetic expression tree of given depth.
///
/// Depth 0: constant or variable
/// Depth N: binary/unary operation over depth N-1 expressions
pub fn arb_arithmetic_tree(dtype: DType, depth: usize) -> impl Strategy<Value = Rc<UOp>> {
    let leaf = arb_simple_uop(dtype.clone());

    leaf.prop_recursive(depth as u32, depth as u32 * 4, 3, move |inner| {
        let _dtype = dtype.clone();
        prop_oneof![
            // Binary operation
            (arb_arithmetic_binary_op(), inner.clone(), inner.clone()).prop_map(move |(op, lhs, rhs)| {
                match op {
                    BinaryOp::Add => lhs.try_add_op(&rhs).unwrap(),
                    BinaryOp::Mul => lhs.try_mul_op(&rhs).unwrap(),
                    BinaryOp::Sub => lhs.try_sub_op(&rhs).unwrap(),
                    BinaryOp::Max => lhs.try_max_op(&rhs).unwrap(),
                    _ => unreachable!("arb_arithmetic_binary_op only generates Add, Mul, Sub, Max"),
                }
            }),
            // Unary operation (only Neg for arithmetic)
            inner.clone().prop_map(move |src| src.neg()),
        ]
    })
}

/// Generate an arithmetic tree with depth up to max_depth.
pub fn arb_arithmetic_tree_up_to(dtype: DType, max_depth: usize) -> impl Strategy<Value = Rc<UOp>> {
    (0..=max_depth).prop_flat_map(move |depth| arb_arithmetic_tree(dtype.clone(), depth))
}

// ============================================================================
// Bounded Generators (for Z3 verification tests)
// ============================================================================

/// Generate bounded constant for Z3 verification tests.
/// Uses small values to avoid overflow when combined in arithmetic trees.
/// Z3 uses unbounded integers, so we need to avoid values that would overflow.
pub fn arb_bounded_const(dtype: DType) -> impl Strategy<Value = Rc<UOp>> {
    use morok_dtype::ScalarDType::*;
    (-100i64..=100).prop_map(move |v| {
        let cv = match dtype.scalar().unwrap() {
            Int8 | Int16 | Int32 | Int64 | Index => ConstValue::Int(v),
            UInt8 | UInt16 | UInt32 | UInt64 => ConstValue::UInt(v.unsigned_abs()),
            _ => ConstValue::Int(v),
        };
        UOp::const_(dtype.clone(), cv)
    })
}

/// Generate simple UOp with bounded constants (for Z3 tests).
pub fn arb_simple_uop_bounded(dtype: DType) -> impl Strategy<Value = Rc<UOp>> {
    prop_oneof![arb_bounded_const(dtype.clone()), arb_var_uop(dtype),]
}

/// Generate arithmetic tree with bounded constants (for Z3 verification).
pub fn arb_arithmetic_tree_bounded(dtype: DType, depth: usize) -> impl Strategy<Value = Rc<UOp>> {
    let leaf = arb_simple_uop_bounded(dtype.clone());

    leaf.prop_recursive(depth as u32, depth as u32 * 4, 3, move |inner| {
        prop_oneof![
            (arb_arithmetic_binary_op(), inner.clone(), inner.clone()).prop_map(move |(op, lhs, rhs)| {
                match op {
                    BinaryOp::Add => lhs.try_add_op(&rhs).unwrap(),
                    BinaryOp::Mul => lhs.try_mul_op(&rhs).unwrap(),
                    BinaryOp::Sub => lhs.try_sub_op(&rhs).unwrap(),
                    BinaryOp::Max => lhs.try_max_op(&rhs).unwrap(),
                    _ => unreachable!("arb_arithmetic_binary_op only generates Add, Mul, Sub, Max"),
                }
            }),
            inner.clone().prop_map(move |src| src.neg()),
        ]
    })
}

/// Generate bounded arithmetic tree with depth up to max_depth.
pub fn arb_arithmetic_tree_bounded_up_to(dtype: DType, max_depth: usize) -> impl Strategy<Value = Rc<UOp>> {
    (0..=max_depth).prop_flat_map(move |depth| arb_arithmetic_tree_bounded(dtype.clone(), depth))
}

// ============================================================================
// Known Property Graph Generators
// ============================================================================

/// Graph with known algebraic property.
#[derive(Debug, Clone)]
pub enum KnownPropertyGraph {
    /// x + 0 (should simplify to x)
    AddZero { x: Rc<UOp>, dtype: DType },
    /// x * 1 (should simplify to x)
    MulOne { x: Rc<UOp>, dtype: DType },
    /// x - 0 (should simplify to x)
    SubZero { x: Rc<UOp>, dtype: DType },
    /// x * 0 (should simplify to 0)
    MulZero { x: Rc<UOp>, dtype: DType },
    /// x - x (should simplify to 0)
    SubSelf { x: Rc<UOp>, dtype: DType },
    /// x + x (should be equivalent to 2 * x)
    AddSelf { x: Rc<UOp>, dtype: DType },
}

impl KnownPropertyGraph {
    /// Build the UOp graph for this known property.
    pub fn build(&self) -> Rc<UOp> {
        match self {
            Self::AddZero { x, dtype } => {
                let zero = ConstValue::zero(dtype.scalar().unwrap());
                x.try_add_op(&UOp::const_(dtype.clone(), zero)).unwrap()
            }
            Self::MulOne { x, dtype } => {
                let one = ConstValue::one(dtype.scalar().unwrap());
                x.try_mul_op(&UOp::const_(dtype.clone(), one)).unwrap()
            }
            Self::SubZero { x, dtype } => {
                let zero = ConstValue::zero(dtype.scalar().unwrap());
                x.try_sub_op(&UOp::const_(dtype.clone(), zero)).unwrap()
            }
            Self::MulZero { x, dtype } => {
                let zero = ConstValue::zero(dtype.scalar().unwrap());
                x.try_mul_op(&UOp::const_(dtype.clone(), zero)).unwrap()
            }
            Self::SubSelf { x, .. } => x.try_sub_op(x).unwrap(),
            Self::AddSelf { x, .. } => x.try_add_op(x).unwrap(),
        }
    }

    /// Get the expected simplified result (if deterministic).
    pub fn expected_result(&self) -> Option<Rc<UOp>> {
        match self {
            Self::AddZero { x, .. } | Self::MulOne { x, .. } | Self::SubZero { x, .. } => Some(Rc::clone(x)),
            Self::MulZero { dtype, .. } | Self::SubSelf { dtype, .. } => {
                Some(UOp::const_(dtype.clone(), ConstValue::Int(0)))
            }
            Self::AddSelf { .. } => None, // Could be 2*x or x+x
        }
    }
}

/// Generate a graph with known algebraic property.
pub fn arb_known_property_graph() -> impl Strategy<Value = KnownPropertyGraph> {
    arb_int_dtype()
        .prop_flat_map(|dtype| {
            arb_var_uop(dtype.clone()).prop_flat_map(move |x| {
                let dtype = dtype.clone();
                prop_oneof![
                    Just(KnownPropertyGraph::AddZero { x: Rc::clone(&x), dtype: dtype.clone() }),
                    Just(KnownPropertyGraph::MulOne { x: Rc::clone(&x), dtype: dtype.clone() }),
                    Just(KnownPropertyGraph::SubZero { x: Rc::clone(&x), dtype: dtype.clone() }),
                    Just(KnownPropertyGraph::MulZero { x: Rc::clone(&x), dtype: dtype.clone() }),
                    Just(KnownPropertyGraph::SubSelf { x: Rc::clone(&x), dtype: dtype.clone() }),
                    Just(KnownPropertyGraph::AddSelf { x, dtype }),
                ]
            })
        })
        .boxed()
}
