//! UOp → Z3 conversion.
//!
//! Converts Morok IR (UOps) to Z3 expressions for verification.
//! Uses z3 crate v0.19.4's global context model.

use std::collections::HashMap;
use std::rc::Rc;

use morok_dtype::DType;
use morok_ir::types::{BinaryOp, ConstValue, TernaryOp, UnaryOp};
use morok_ir::{Op, UOp};
use z3::Solver;
use z3::ast::{Bool, Dynamic, Int};

use crate::z3::alu::{z3_cdiv, z3_cmod};

/// Z3 conversion context with solver.
pub struct Z3Context {
    solver: Solver,
}

impl Z3Context {
    /// Create a new Z3 context with a solver.
    pub fn new() -> Self {
        let solver = Solver::new();
        Self { solver }
    }

    /// Get mutable reference to the solver.
    pub fn solver(&mut self) -> &mut Solver {
        &mut self.solver
    }

    /// Convert a UOp graph to Z3 expression.
    ///
    /// Processes UOps in topological order (bottom-up) to ensure dependencies
    /// are converted before they're used.
    pub fn convert_uop(&mut self, uop: &Rc<UOp>) -> Result<Dynamic, ConversionError> {
        let mut cache = HashMap::new();
        self.convert_uop_cached(uop, &mut cache)
    }

    /// Convert UOp with caching to avoid redundant conversion.
    fn convert_uop_cached(
        &mut self,
        uop: &Rc<UOp>,
        cache: &mut HashMap<usize, Dynamic>,
    ) -> Result<Dynamic, ConversionError> {
        // Use pointer address as cache key
        let key = Rc::as_ptr(uop) as usize;

        // Check cache first
        if let Some(z3_expr) = cache.get(&key) {
            return Ok(z3_expr.clone());
        }

        // Convert based on operation type
        let z3_expr = match uop.op() {
            Op::Const(cv) => Self::convert_const(&cv.0)?,

            Op::DefineVar { name, min_val, max_val } => self.convert_var(name, *min_val, *max_val)?,

            Op::Range { end, .. } => {
                // Range represents loop variable: [0, end)
                let end_z3 = self.convert_uop_cached(end, cache)?;

                // Create a fresh variable for this range
                let range_var = Int::fresh_const("range");
                let zero = Int::from_i64(0);

                // Add constraints: 0 <= range_var < end
                self.solver.assert(range_var.ge(&zero));
                if let Some(end_int) = end_z3.as_int() {
                    self.solver.assert(range_var.lt(end_int));
                }

                Dynamic::from_ast(&range_var)
            }

            Op::Unary(op, src) => {
                let src_z3 = self.convert_uop_cached(src, cache)?;
                Self::convert_unary(*op, &src_z3)?
            }

            Op::Binary(op, lhs, rhs) => {
                let lhs_z3 = self.convert_uop_cached(lhs, cache)?;
                let rhs_z3 = self.convert_uop_cached(rhs, cache)?;
                Self::convert_binary(*op, &lhs_z3, &rhs_z3)?
            }

            Op::Ternary(TernaryOp::Where, cond, true_val, false_val) => {
                let cond_z3 = self.convert_uop_cached(cond, cache)?;
                let true_z3 = self.convert_uop_cached(true_val, cache)?;
                let false_z3 = self.convert_uop_cached(false_val, cache)?;

                if let Some(cond_bool) = cond_z3.as_bool() {
                    if let (Some(true_int), Some(false_int)) = (true_z3.as_int(), false_z3.as_int()) {
                        Dynamic::from_ast(&cond_bool.ite(&true_int, &false_int))
                    } else {
                        return Err(ConversionError::UnsupportedOperation(
                            "WHERE with non-integer branches".to_string(),
                        ));
                    }
                } else {
                    return Err(ConversionError::UnsupportedOperation("WHERE with non-boolean condition".to_string()));
                }
            }

            Op::Ternary(TernaryOp::MulAcc, a, b, c) => {
                let a_z3 = self.convert_uop_cached(a, cache)?;
                let b_z3 = self.convert_uop_cached(b, cache)?;
                let c_z3 = self.convert_uop_cached(c, cache)?;

                if let (Some(a_int), Some(b_int), Some(c_int)) = (a_z3.as_int(), b_z3.as_int(), c_z3.as_int()) {
                    Dynamic::from_ast(&(a_int * b_int + c_int))
                } else {
                    return Err(ConversionError::UnsupportedOperation("MULACC with non-integer operands".to_string()));
                }
            }

            Op::Cast { src, dtype } => {
                // Conservative approximation: create a fresh bounded variable
                let _src_z3 = self.convert_uop_cached(src, cache)?;
                self.convert_bounded_from_dtype(dtype.clone())?
            }

            _ => {
                return Err(ConversionError::UnsupportedOperation(format!("{:?}", uop.op())));
            }
        };

        // Cache the result
        cache.insert(key, z3_expr.clone());
        Ok(z3_expr)
    }

    /// Convert a constant value to Z3.
    fn convert_const(cv: &ConstValue) -> Result<Dynamic, ConversionError> {
        match cv {
            ConstValue::Int(v) => Ok(Dynamic::from_ast(&Int::from_i64(*v))),
            ConstValue::UInt(v) => {
                // Represent as signed int; may overflow for very large u64
                Ok(Dynamic::from_ast(&Int::from_u64(*v)))
            }
            ConstValue::Bool(v) => Ok(Dynamic::from_ast(&Bool::from_bool(*v))),
            ConstValue::Float(_) => {
                Err(ConversionError::UnsupportedType("Float constants not fully supported".to_string()))
            }
        }
    }

    /// Convert a variable with bounds to Z3.
    fn convert_var(&mut self, name: &str, min_val: i64, max_val: i64) -> Result<Dynamic, ConversionError> {
        let var = Int::new_const(name);
        let min_z3 = Int::from_i64(min_val);
        let max_z3 = Int::from_i64(max_val);

        // Add constraints: min_val <= var <= max_val
        self.solver.assert(var.ge(&min_z3));
        self.solver.assert(var.le(&max_z3));

        Ok(Dynamic::from_ast(&var))
    }

    /// Create a fresh bounded variable from dtype.
    fn convert_bounded_from_dtype(&mut self, dtype: DType) -> Result<Dynamic, ConversionError> {
        let (min_val, max_val) = dtype_bounds(dtype);
        let var = Int::fresh_const("cast");
        let min_z3 = Int::from_i64(min_val);
        let max_z3 = Int::from_i64(max_val);

        self.solver.assert(var.ge(&min_z3));
        self.solver.assert(var.le(&max_z3));

        Ok(Dynamic::from_ast(&var))
    }

    /// Convert unary operation.
    fn convert_unary(op: UnaryOp, src: &Dynamic) -> Result<Dynamic, ConversionError> {
        let src_int =
            src.as_int().ok_or_else(|| ConversionError::TypeMismatch("Expected int for unary op".to_string()))?;

        match op {
            UnaryOp::Neg => Ok(Dynamic::from_ast(&-src_int)),
            _ => Err(ConversionError::UnsupportedOperation(format!("Unary op: {:?}", op))),
        }
    }

    /// Convert binary operation.
    fn convert_binary(op: BinaryOp, lhs: &Dynamic, rhs: &Dynamic) -> Result<Dynamic, ConversionError> {
        match op {
            // Arithmetic operations (require integers)
            BinaryOp::Add => {
                let l = lhs.as_int().ok_or(ConversionError::TypeMismatch("ADD: expected int".to_string()))?;
                let r = rhs.as_int().ok_or(ConversionError::TypeMismatch("ADD: expected int".to_string()))?;
                Ok(Dynamic::from_ast(&(l + r)))
            }
            BinaryOp::Sub => {
                let l = lhs.as_int().ok_or(ConversionError::TypeMismatch("SUB: expected int".to_string()))?;
                let r = rhs.as_int().ok_or(ConversionError::TypeMismatch("SUB: expected int".to_string()))?;
                Ok(Dynamic::from_ast(&(l - r)))
            }
            BinaryOp::Mul => {
                let l = lhs.as_int().ok_or(ConversionError::TypeMismatch("MUL: expected int".to_string()))?;
                let r = rhs.as_int().ok_or(ConversionError::TypeMismatch("MUL: expected int".to_string()))?;
                Ok(Dynamic::from_ast(&(l * r)))
            }
            BinaryOp::Idiv => {
                let l = lhs.as_int().ok_or(ConversionError::TypeMismatch("IDIV: expected int".to_string()))?;
                let r = rhs.as_int().ok_or(ConversionError::TypeMismatch("IDIV: expected int".to_string()))?;
                // Use truncated division (C-style)
                Ok(Dynamic::from_ast(&z3_cdiv(&l, &r)))
            }
            BinaryOp::Mod => {
                let l = lhs.as_int().ok_or(ConversionError::TypeMismatch("MOD: expected int".to_string()))?;
                let r = rhs.as_int().ok_or(ConversionError::TypeMismatch("MOD: expected int".to_string()))?;
                // Use C-style modulo
                Ok(Dynamic::from_ast(&z3_cmod(&l, &r)))
            }
            BinaryOp::Max => {
                let l = lhs.as_int().ok_or(ConversionError::TypeMismatch("MAX: expected int".to_string()))?;
                let r = rhs.as_int().ok_or(ConversionError::TypeMismatch("MAX: expected int".to_string()))?;
                // max(a, b) = if a > b then a else b
                Ok(Dynamic::from_ast(&l.gt(&r).ite(&l, &r)))
            }

            // Comparison operations (return boolean)
            BinaryOp::Lt => {
                let l = lhs.as_int().ok_or(ConversionError::TypeMismatch("LT: expected int".to_string()))?;
                let r = rhs.as_int().ok_or(ConversionError::TypeMismatch("LT: expected int".to_string()))?;
                Ok(Dynamic::from_ast(&l.lt(r)))
            }
            BinaryOp::Eq => {
                // Try int first, then bool
                if let (Some(l), Some(r)) = (lhs.as_int(), rhs.as_int()) {
                    Ok(Dynamic::from_ast(&l.eq(r)))
                } else if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&l.eq(r)))
                } else {
                    Err(ConversionError::TypeMismatch("EQ: type mismatch".to_string()))
                }
            }
            BinaryOp::Ne => {
                // Try int first, then bool
                if let (Some(l), Some(r)) = (lhs.as_int(), rhs.as_int()) {
                    Ok(Dynamic::from_ast(&l.eq(r).not()))
                } else if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&l.eq(r).not()))
                } else {
                    Err(ConversionError::TypeMismatch("NE: type mismatch".to_string()))
                }
            }

            // Bitwise operations
            BinaryOp::And => {
                // Can be int (bitwise) or bool (logical)
                if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&Bool::and(&[l, r])))
                } else {
                    Err(ConversionError::UnsupportedOperation("Bitwise AND not implemented".to_string()))
                }
            }
            BinaryOp::Or => {
                // Can be int (bitwise) or bool (logical)
                if let (Some(l), Some(r)) = (lhs.as_bool(), rhs.as_bool()) {
                    Ok(Dynamic::from_ast(&Bool::or(&[l, r])))
                } else {
                    Err(ConversionError::UnsupportedOperation("Bitwise OR not implemented".to_string()))
                }
            }

            _ => Err(ConversionError::UnsupportedOperation(format!("Binary op: {:?}", op))),
        }
    }
}

impl Default for Z3Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Get conservative bounds for a dtype.
fn dtype_bounds(dtype: DType) -> (i64, i64) {
    use morok_dtype::ScalarDType;

    match dtype {
        DType::Scalar(sdt) => match sdt {
            ScalarDType::Bool => (0, 1),
            ScalarDType::Int8 => (i8::MIN as i64, i8::MAX as i64),
            ScalarDType::Int16 => (i16::MIN as i64, i16::MAX as i64),
            ScalarDType::Int32 => (i32::MIN as i64, i32::MAX as i64),
            ScalarDType::Int64 => (i64::MIN, i64::MAX),
            ScalarDType::UInt8 => (0, u8::MAX as i64),
            ScalarDType::UInt16 => (0, u16::MAX as i64),
            ScalarDType::UInt32 => (0, u32::MAX as i64),
            ScalarDType::UInt64 => (0, i64::MAX),    // Conservative
            _ => (i32::MIN as i64, i32::MAX as i64), // Float types
        },
        DType::Ptr { .. } => (0, i64::MAX),
        DType::Vector { scalar, .. } => dtype_bounds(DType::Scalar(scalar)),
        DType::Image { .. } => (0, i64::MAX), // Conservative bounds for image types
    }
}

/// Z3 conversion error.
#[derive(Debug, Clone)]
pub enum ConversionError {
    UnsupportedOperation(String),
    UnsupportedType(String),
    TypeMismatch(String),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedOperation(s) => write!(f, "Unsupported operation: {}", s),
            Self::UnsupportedType(s) => write!(f, "Unsupported type: {}", s),
            Self::TypeMismatch(s) => write!(f, "Type mismatch: {}", s),
        }
    }
}

impl std::error::Error for ConversionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_const_int() {
        let mut z3ctx = Z3Context::new();

        let uop = UOp::const_(DType::Int32, ConstValue::Int(42));
        let z3_expr = z3ctx.convert_uop(&uop).expect("Should convert");

        assert!(z3_expr.as_int().is_some());
    }

    #[test]
    fn test_convert_simple_add() {
        let mut z3ctx = Z3Context::new();

        let a = UOp::const_(DType::Int32, ConstValue::Int(10));
        let b = UOp::const_(DType::Int32, ConstValue::Int(20));
        let add = UOp::new(Op::Binary(BinaryOp::Add, a, b), DType::Int32);

        let z3_expr = z3ctx.convert_uop(&add).expect("Should convert");
        assert!(z3_expr.as_int().is_some());
    }

    #[test]
    fn test_convert_variable() {
        let mut z3ctx = Z3Context::new();

        let var = UOp::var("x", DType::Int32, 0, 100);
        let z3_expr = z3ctx.convert_uop(&var).expect("Should convert");

        assert!(z3_expr.as_int().is_some());

        // Solver should have constraints for variable bounds
        assert_eq!(z3ctx.solver.check(), z3::SatResult::Sat);
    }
}
