//! Process-global arbiter for the single in-process compiler LLVM slot.
//!
//! Two backends can run an LLVM inside this process: the CPU LLVM backend loads
//! a libLLVM via `libloading`, and the Metal private compiler
//! (`MTLCompiler.framework`) loads its own libLLVM with `RTLD_GLOBAL`. Two
//! independent libLLVM images share managed statics and command-line-option
//! registries, and LLVM's verifier segfaults when both are resident. Only the
//! first claimant may use in-process LLVM; the loser falls back to an
//! out-of-process path (the CPU backend's `clang` subprocess, or Metal's
//! `newLibraryWithSource:` compiler daemon). The decision is process-global and
//! permanent, and order-independent: whichever backend compiles first wins.

use std::sync::Mutex;

static OWNER: Mutex<Option<&'static str>> = Mutex::new(None);

/// Try to claim the in-process LLVM slot for `who`. Returns `true` if `who` may
/// use in-process LLVM (it claimed the slot, or already holds it), `false` if a
/// different backend holds it.
pub fn claim_inprocess_llvm(who: &'static str) -> bool {
    let mut owner = OWNER.lock().expect("in-process LLVM arbiter mutex poisoned");
    match *owner {
        None => {
            *owner = Some(who);
            true
        }
        Some(current) => current == who,
    }
}
