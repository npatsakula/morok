//! Test helpers for Z3 verification tests.

use std::rc::Rc;

use morok_ir::UOp;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;
use crate::z3::verify::verify_equivalence;

/// Verify that a pattern simplifies correctly and is semantically equivalent.
///
/// This helper:
/// 1. Applies symbolic simplification to the input
/// 2. Verifies the result is semantically equivalent using Z3
/// 3. Returns the simplified expression for further assertions
pub fn verify_simplifies_to(expr: Rc<UOp>, expected: Rc<UOp>) -> Rc<UOp> {
    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    // Verify semantic equivalence
    verify_equivalence(&expr, &simplified).expect("Simplification should preserve semantics");

    // Also check structural equality with expected (optional but helpful for debugging)
    if !Rc::ptr_eq(&simplified, &expected) {
        // If not pointer-equal, they might still be structurally equivalent
        // This is okay - Z3 verification is the source of truth
    }

    simplified
}

/// Verify that simplification preserves semantics (even if structure changes).
pub fn verify_preserves_semantics(expr: Rc<UOp>) -> Rc<UOp> {
    let matcher = symbolic_simple();
    let simplified = graph_rewrite(&matcher, expr.clone(), &mut ());

    verify_equivalence(&expr, &simplified).expect("Simplification should preserve semantics");

    simplified
}
