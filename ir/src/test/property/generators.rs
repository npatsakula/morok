//! Generators for property-based testing.
//!
//! Provides Arbitrary implementations and custom strategies for generating
//! UOp graphs, constants, operations, and dtype families.

use std::rc::Rc;

use proptest::prelude::*;

use morok_dtype::DType;

use crate::types::{BinaryOp, ConstValue, TernaryOp, UnaryOp};
use crate::{Op, UOp};

// ============================================================================
// ConstValue Generators
// ============================================================================

/// Generate arbitrary ConstValue with reasonable bounds.
pub fn arb_const_value() -> impl Strategy<Value = ConstValue> {
    prop_oneof![
        (-1000i64..=1000).prop_map(ConstValue::Int),
        (0u64..=1000).prop_map(ConstValue::UInt),
        (-100.0..=100.0).prop_map(ConstValue::Float),
        any::<bool>().prop_map(ConstValue::Bool),
    ]
}

/// Generate small integer constants (useful for arithmetic tests).
pub fn arb_small_int() -> impl Strategy<Value = ConstValue> {
    (-10i64..=10).prop_map(ConstValue::Int)
}

/// Generate non-zero constants (useful for division/mod tests).
pub fn arb_nonzero_int() -> impl Strategy<Value = ConstValue> {
    prop_oneof![(-1000i64..=-1).prop_map(ConstValue::Int), (1i64..=1000).prop_map(ConstValue::Int),]
}

// ============================================================================
// DType Generators
// ============================================================================

/// Generate scalar DType suitable for arithmetic operations.
pub fn arb_arithmetic_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::Int8),
        Just(DType::Int16),
        Just(DType::Int32),
        Just(DType::Int64),
        Just(DType::UInt8),
        Just(DType::UInt16),
        Just(DType::UInt32),
        Just(DType::UInt64),
        Just(DType::Float32),
        Just(DType::Float64),
    ]
}

/// Generate integer DType (signed and unsigned).
pub fn arb_int_dtype() -> impl Strategy<Value = DType> {
    prop_oneof![
        Just(DType::Int8),
        Just(DType::Int16),
        Just(DType::Int32),
        Just(DType::Int64),
        Just(DType::UInt8),
        Just(DType::UInt16),
        Just(DType::UInt32),
        Just(DType::UInt64),
    ]
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
    arb_const_value().prop_map(move |cv| {
        // Cast to target dtype if needed
        let final_cv =
            if let Some(scalar_dt) = dtype.scalar() { cv.cast(&DType::Scalar(scalar_dt)).unwrap_or(cv) } else { cv };
        UOp::const_(dtype.clone(), final_cv)
    })
}

/// Generate a variable UOp with bounded range.
pub fn arb_var_uop(dtype: DType) -> impl Strategy<Value = Rc<UOp>> {
    ("[a-z]", 0i64..100, 1i64..100).prop_map(move |(name, min_offset, range_size)| {
        let min_val = min_offset;
        let max_val = min_val + range_size;
        UOp::var(&name, dtype.clone(), min_val, max_val)
    })
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
        let dtype = dtype.clone();
        let dtype_for_binary = dtype.clone();
        let dtype_for_unary = dtype;
        prop_oneof![
            // Binary operation
            (arb_arithmetic_binary_op(), inner.clone(), inner.clone())
                .prop_map(move |(op, lhs, rhs)| { UOp::new(Op::Binary(op, lhs, rhs), dtype_for_binary.clone()) }),
            // Unary operation (only Neg for arithmetic)
            inner.clone().prop_map(move |src| { UOp::new(Op::Unary(UnaryOp::Neg, src), dtype_for_unary.clone()) }),
        ]
    })
}

/// Generate an arithmetic tree with depth up to max_depth.
pub fn arb_arithmetic_tree_up_to(dtype: DType, max_depth: usize) -> impl Strategy<Value = Rc<UOp>> {
    (0..=max_depth).prop_flat_map(move |depth| arb_arithmetic_tree(dtype.clone(), depth))
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
                let zero = UOp::const_(dtype.clone(), ConstValue::Int(0));
                UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(x), zero), dtype.clone())
            }
            Self::MulOne { x, dtype } => {
                let one = UOp::const_(dtype.clone(), ConstValue::Int(1));
                UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(x), one), dtype.clone())
            }
            Self::SubZero { x, dtype } => {
                let zero = UOp::const_(dtype.clone(), ConstValue::Int(0));
                UOp::new(Op::Binary(BinaryOp::Sub, Rc::clone(x), zero), dtype.clone())
            }
            Self::MulZero { x, dtype } => {
                let zero = UOp::const_(dtype.clone(), ConstValue::Int(0));
                UOp::new(Op::Binary(BinaryOp::Mul, Rc::clone(x), zero), dtype.clone())
            }
            Self::SubSelf { x, dtype } => {
                UOp::new(Op::Binary(BinaryOp::Sub, Rc::clone(x), Rc::clone(x)), dtype.clone())
            }
            Self::AddSelf { x, dtype } => {
                UOp::new(Op::Binary(BinaryOp::Add, Rc::clone(x), Rc::clone(x)), dtype.clone())
            }
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
