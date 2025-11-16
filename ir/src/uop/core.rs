//! Core UOp struct and fundamental operations.
//!
//! This module contains the [`UOp`] struct definition and its core methods
//! for accessing operation data, dtype, shape, and graph traversal.

use std::collections::HashSet;
use std::rc::Rc;

use crate::op::Op;
use crate::shape;
use crate::types::ConstValue;
use morok_dtype::DType;

/// Micro-operation node in the computation graph.
///
/// UOps form a DAG where operations reference their inputs through the Op enum.
/// Hash consing ensures that structurally identical UOps share the same allocation.
///
/// Shape inference is lazy and cached - computed on first access via `shape()` method.
#[derive(Debug)]
pub struct UOp {
    /// Unique stable ID for this UOp instance.
    /// Used for identity-based caching instead of fragile raw pointers.
    pub(crate) id: u64,
    pub(crate) op: Op,
    pub(crate) dtype: DType,
    /// Cached shape - computed lazily on first access.
    /// OnceCell provides thread-safe lazy initialization.
    pub(crate) shape_cache: std::cell::OnceCell<Option<shape::Shape>>,
}

impl UOp {
    /// Get the operation.
    pub fn op(&self) -> &Op {
        &self.op
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the shape of this UOp.
    ///
    /// Shape is computed lazily on first access and cached.
    /// Returns None if shape cannot be determined (e.g., for control flow ops).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{UOp, ConstValue};
    /// # use morok_dtype::DType;
    /// let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    /// assert_eq!(scalar.shape().map(|s| s.len()), Some(0)); // Scalar has empty shape
    /// ```
    pub fn shape(&self) -> Option<&shape::Shape> {
        self.shape_cache.get_or_init(|| shape::infer_shape_from_op(self)).as_ref()
    }

    /// Topological sort of the computation graph.
    ///
    /// Returns nodes in an order where all dependencies come before their dependents.
    pub fn toposort(self: &Rc<Self>) -> Vec<Rc<Self>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Rc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else {
                stack.push((node.clone(), true));

                // Use for_each_child for zero-allocation traversal
                let mut children = Vec::new();
                node.op.map_child(|child| {
                    if !visited.contains(&Rc::as_ptr(child)) {
                        children.push(child.clone());
                    }
                });

                // Push in reverse order for proper traversal
                for child in children.into_iter().rev() {
                    stack.push((child, false));
                }
            }
        }

        result
    }
}

impl Clone for UOp {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            op: self.op.clone(),
            dtype: self.dtype.clone(),
            shape_cache: std::cell::OnceCell::new(),
        }
    }
}

/// Trait for converting scalar values into UOps.
///
/// This allows operator overloading to work with mixed scalar/UOp operands.
/// For example: `uop + 5.0` or `5.0 + uop`.
pub trait IntoUOp {
    fn into_uop(self, dtype: DType) -> Rc<UOp>;
}

impl IntoUOp for f32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self as f64))
    }
}

impl IntoUOp for f64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self))
    }
}

impl IntoUOp for i32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self as i64))
    }
}

impl IntoUOp for i64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self))
    }
}

impl IntoUOp for u32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self as u64))
    }
}

impl IntoUOp for u64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self))
    }
}

impl IntoUOp for bool {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Bool(self))
    }
}
