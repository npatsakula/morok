//! Static LLVM context storage for codegen.

use inkwell::context::Context;
use std::cell::RefCell;

thread_local! {
    static CONTEXT: RefCell<Option<Box<Context>>> = const { RefCell::new(None) };
}

/// Get or create the thread-local LLVM context.
///
/// The context is lazily initialized and reused within a thread.
/// This avoids creating multiple contexts and ensures proper lifetime management.
pub fn with_context<F, R>(f: F) -> R
where
    F: FnOnce(&Context) -> R,
{
    CONTEXT.with(|cell| {
        let mut opt = cell.borrow_mut();
        if opt.is_none() {
            *opt = Some(Box::new(Context::create()));
        }

        // SAFETY: Context is boxed (stable address) and we control the borrow
        // The reference doesn't escape this function
        let context_ref = unsafe { &**(opt.as_ref().unwrap() as *const Box<Context>) };
        f(context_ref)
    })
}

/// Clear the thread-local context (for testing or cleanup).
pub fn clear_context() {
    CONTEXT.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_reuse() {
        let ptr1 = with_context(|ctx| ctx as *const Context);
        let ptr2 = with_context(|ctx| ctx as *const Context);
        assert_eq!(ptr1, ptr2, "Context should be reused");
    }
}
