//! SIMD/vector operations (vectorize, gep, vconst).

use std::rc::Rc;

use morok_dtype::DType;
use smallvec::SmallVec;

use super::super::{Op, UOp};

impl UOp {
    /// Create vector from scalar elements.
    pub fn vectorize(elements: SmallVec<[Rc<Self>; 4]>) -> Rc<Self> {
        let base_dtype = if let Some(first) = elements.first() {
            first.dtype()
        } else {
            DType::Float32 // Default for empty vectors
        };
        let vec_dtype = base_dtype.vec(elements.len());
        Self::new(Op::Vectorize { elements }, vec_dtype)
    }

    /// Get element pointer (extract element(s) from vector).
    pub fn gep(vector: Rc<Self>, indices: Vec<usize>) -> Rc<Self> {
        let vector_dtype = vector.dtype();

        // Extract scalar if single element, keep vector if multiple
        let dtype = if indices.len() == 1 {
            // Extract single element -> scalar
            match vector_dtype.scalar() {
                Some(s) => DType::Scalar(s),
                None => vector_dtype.clone(),
            }
        } else {
            // Extract multiple elements -> vector
            match vector_dtype.scalar() {
                Some(s) => DType::Scalar(s).vec(indices.len()),
                None => vector_dtype.clone(),
            }
        };

        Self::new(Op::Gep { vector, indices }, dtype)
    }

    /// Vector constant.
    pub fn vconst(values: Vec<super::super::ConstValue>) -> Rc<Self> {
        let scalar_dtype = match values.first() {
            Some(super::super::ConstValue::Int(_)) => DType::Int64,
            Some(super::super::ConstValue::UInt(_)) => DType::UInt64,
            Some(super::super::ConstValue::Float(_)) => DType::Float64,
            Some(super::super::ConstValue::Bool(_)) => DType::Bool,
            None => DType::Float32,
        };
        let vec_dtype = scalar_dtype.vec(values.len());
        Self::new(Op::VConst { values }, vec_dtype)
    }

    /// Define symbolic variable with bounds.
    pub fn define_var(name: String, min_val: i64, max_val: i64) -> Rc<Self> {
        Self::new(Op::DefineVar { name, min_val, max_val }, DType::Index)
    }

    /// Bind concrete value to symbolic variable.
    pub fn bind(var: Rc<Self>, value: Rc<Self>) -> Rc<Self> {
        let dtype = var.dtype();
        Self::new(Op::Bind { var, value }, dtype)
    }

    /// Define register memory.
    pub fn define_reg(size: usize) -> Rc<Self> {
        use morok_dtype::AddrSpace;
        let ptr_dtype = DType::Void.ptr(Some(size), AddrSpace::Reg);
        Self::new(Op::DefineReg { size }, ptr_dtype)
    }
}
