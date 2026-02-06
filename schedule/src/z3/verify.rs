//! Pattern verification using Z3.
//!
//! Provides equivalence checking for symbolic rewrites with counterexample extraction.

use std::sync::Arc;

use morok_ir::UOp;

use crate::z3::convert::Z3Context;

/// Result of equivalence verification.
pub type VerificationResult = Result<(), CounterExample>;

/// Counterexample when verification fails.
#[derive(Debug, Clone)]
pub enum CounterExample {
    /// Z3 found a concrete input where expressions differ.
    Found { message: String, model: String },
    /// Z3 timed out or returned unknown.
    Timeout,
    /// Conversion to Z3 failed.
    ConversionFailed(String),
}

impl std::fmt::Display for CounterExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Found { message, model } => {
                write!(f, "Counterexample found: {}\nModel: {}", message, model)
            }
            Self::Timeout => write!(f, "Z3 timeout or unknown result"),
            Self::ConversionFailed(s) => write!(f, "Conversion failed: {}", s),
        }
    }
}

impl std::error::Error for CounterExample {}

/// Verify that two UOp expressions are semantically equivalent.
///
/// Uses Z3 to check if `original ≡ simplified` by proving that
/// `NOT(original == simplified)` is UNSAT.
///
/// # Errors
///
/// Returns a counterexample if:
/// - Expressions are not equivalent (Z3 returns SAT)
/// - Z3 times out (returns UNKNOWN)
/// - Conversion to Z3 fails
pub fn verify_equivalence(original: &Arc<UOp>, simplified: &Arc<UOp>) -> VerificationResult {
    // Create Z3 context
    let mut z3ctx = Z3Context::new();

    // Convert both expressions to Z3
    let z3_original = match z3ctx.convert_uop(original) {
        Ok(expr) => expr,
        Err(e) => return Err(CounterExample::ConversionFailed(format!("Failed to convert original: {}", e))),
    };

    let z3_simplified = match z3ctx.convert_uop(simplified) {
        Ok(expr) => expr,
        Err(e) => return Err(CounterExample::ConversionFailed(format!("Failed to convert simplified: {}", e))),
    };

    // Try to cast to same type for comparison
    let (z3_original, z3_simplified) = match (z3_original.as_int(), z3_simplified.as_int()) {
        (Some(o), Some(s)) => (o, s),
        _ => {
            // Try bool
            match (z3_original.as_bool(), z3_simplified.as_bool()) {
                (Some(o), Some(s)) => {
                    // For bools, convert to assertion
                    let solver = z3ctx.solver();
                    solver.assert(o.eq(s).not());

                    match solver.check() {
                        z3::SatResult::Unsat => return Ok(()),
                        z3::SatResult::Sat => {
                            let model = solver
                                .get_model()
                                .map(|m| m.to_string())
                                .unwrap_or_else(|| "No model available".to_string());

                            return Err(CounterExample::Found {
                                message: "Boolean expressions not equivalent".to_string(),
                                model,
                            });
                        }
                        z3::SatResult::Unknown => return Err(CounterExample::Timeout),
                    }
                }
                _ => {
                    return Err(CounterExample::ConversionFailed(
                        "Type mismatch: cannot compare expressions".to_string(),
                    ));
                }
            }
        }
    };

    // Assert that the expressions are NOT equal
    // If UNSAT, they're always equal (proven equivalent)
    // If SAT, we found a counterexample
    let solver = z3ctx.solver();
    solver.assert(z3_original.eq(z3_simplified).not());

    match solver.check() {
        z3::SatResult::Unsat => {
            // Proven equivalent!
            Ok(())
        }
        z3::SatResult::Sat => {
            // Found counterexample
            let model = solver.get_model().map(|m| m.to_string()).unwrap_or_else(|| "No model available".to_string());

            Err(CounterExample::Found {
                message: format!(
                    "Expressions not equivalent:\nOriginal: {:?}\nSimplified: {:?}",
                    original.op(),
                    simplified.op()
                ),
                model,
            })
        }
        z3::SatResult::Unknown => Err(CounterExample::Timeout),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;

    use morok_ir::types::ConstValue;

    #[test]
    fn test_verify_identity_add_zero() {
        // x + 0 = x
        let x = UOp::var("x", DType::Int32, 0, 100);
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));
        let x_plus_zero = x.try_add(&zero).unwrap();

        verify_equivalence(&x_plus_zero, &x).expect("x + 0 should equal x");
    }

    #[test]
    fn test_verify_commutativity() {
        // x + y = y + x
        let x = UOp::var("x", DType::Int32, 0, 100);
        let y = UOp::var("y", DType::Int32, 0, 100);
        let x_plus_y = x.try_add(&y).unwrap();
        let y_plus_x = y.try_add(&x).unwrap();

        verify_equivalence(&x_plus_y, &y_plus_x).expect("x + y should equal y + x");
    }

    #[test]
    fn test_verify_detect_inequality() {
        // x + 1 ≠ x (should find counterexample)
        let x = UOp::var("x", DType::Int32, 0, 100);
        let one = UOp::const_(DType::Int32, ConstValue::Int(1));
        let x_plus_one = x.try_add(&one).unwrap();

        let result = verify_equivalence(&x_plus_one, &x);
        assert!(result.is_err(), "x + 1 should not equal x");

        if let Err(CounterExample::Found { message, model }) = result {
            tracing::debug!(message = %message, model = %model, "z3 counterexample found");
        }
    }

    #[test]
    fn test_verify_self_folding() {
        // x - x = 0
        let x = UOp::var("x", DType::Int32, 0, 100);
        let x_minus_x = x.try_sub(&x).unwrap();
        let zero = UOp::const_(DType::Int32, ConstValue::Int(0));

        verify_equivalence(&x_minus_x, &zero).expect("x - x should equal 0");
    }
}
