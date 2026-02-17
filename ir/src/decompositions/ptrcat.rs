//! PtrCat decomposition for backends that don't support vector-of-pointer types.
//!
//! MLIR's LLVM dialect doesn't support `vector<N x ptr>` types, even though
//! LLVM IR text does. This module provides decomposition to eliminate bare
//! PtrCat operations that aren't consumed by Gep (which should already be
//! optimized by symbolic patterns).

use std::sync::Arc;

use crate::prelude::*;

/// Eliminate bare PtrCat by returning first pointer.
///
/// PtrCat without a consuming Gep is dead code - the vector of pointers
/// is created but never used. This pattern returns the first pointer
/// as a placeholder (the actual value doesn't matter since it's dead code).
///
/// # Rationale
///
/// `Gep(PtrCat([...]), indices)` patterns are already optimized by symbolic
/// patterns in `schedule/src/symbolic/patterns.rs`. Any remaining PtrCat
/// operations are not consumed and thus dead code. Backends that can't
/// represent vector-of-pointer types (like MLIR LLVM dialect) need to
/// eliminate these before codegen.
///
/// # Example
///
/// ```ignore
/// Before: PtrCat([p0, p1, p2, p3])  // Dead code, never consumed
/// After:  p0                        // Placeholder, still dead but representable
/// ```
pub fn eliminate_ptrcat(sources: &[Arc<UOp>]) -> Arc<UOp> {
    // Return first pointer as placeholder for dead PtrCat
    // This works because if PtrCat isn't consumed by Gep, it's dead code
    sources[0].clone()
}
