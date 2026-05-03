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
#[path = "../test/unit/z3/verify_internal.rs"]
mod tests;
