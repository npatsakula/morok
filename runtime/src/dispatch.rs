//! Kernel dispatch via libffi.
//!
//! `KernelCif` wraps libffi's `Cif` with Send+Sync so it can be stored on
//! kernel structs and shared across rayon threads. A thread-local buffer
//! avoids per-call allocation for the packed u64 args.

use std::cell::RefCell;

use libffi::low::CodePtr;
use libffi::middle::{self, Cif, Type};
use smallvec::SmallVec;

/// Send+Sync wrapper for libffi Cif.
///
/// # Safety
///
/// `Cif` is `!Send + !Sync` due to raw pointer fields (conservative auto-trait).
/// Once prepared, a CIF is immutable — `Cif::call(&self)` only reads the
/// descriptor and `ffi_call` does not mutate it for non-closure calls.
/// All our CIFs describe stateless kernel signatures (N × u64 → void).
pub(crate) struct KernelCif {
    cif: Cif,
    arg_count: usize,
}

unsafe impl Send for KernelCif {}
unsafe impl Sync for KernelCif {}

impl KernelCif {
    /// Create a CIF for a kernel with `arg_count` u64 arguments returning void.
    pub fn new(arg_count: usize) -> Self {
        let types = (0..arg_count).map(|_| Type::u64()).collect::<Vec<_>>();
        Self { cif: Cif::new(types, Type::void()), arg_count }
    }

    /// Call the kernel, packing buffers + vals as u64 args.
    ///
    /// Uses a thread-local buffer for the packed args — zero allocation
    /// after warmup. The `SmallVec<[Arg; 32]>` avoids heap allocation for
    /// kernels with ≤32 arguments (the common case); kernels above that cap
    /// fall back to a heap allocation per dispatch.
    ///
    /// `var_patch`: if `Some((var_idx, value))`, patches
    /// `vals[var_idx]` to `value` before calling.
    #[inline]
    pub unsafe fn dispatch(
        &self,
        fn_ptr: *const (),
        buffers: &[*mut u8],
        vals: &[i64],
        var_patch: Option<(usize, usize)>,
    ) {
        assert_eq!(
            buffers.len() + vals.len(),
            self.arg_count,
            "kernel dispatch: expected {} args, got {} bufs + {} vals",
            self.arg_count,
            buffers.len(),
            vals.len()
        );

        thread_local! {
            static PACKED: RefCell<SmallVec<[u64; 32]>> = RefCell::new(SmallVec::new());
        }

        PACKED.with_borrow_mut(|packed| {
            if packed.len() != self.arg_count {
                packed.resize(self.arg_count, 0);
            }

            for (idx, &ptr) in buffers.iter().enumerate() {
                packed[idx] = ptr as u64;
            }
            for (idx, &val) in vals.iter().enumerate() {
                packed[buffers.len() + idx] = val as u64;
            }

            if let Some((var_idx, value)) = var_patch {
                packed[buffers.len() + var_idx] = value as u64;
            }

            let mut ffi_args: SmallVec<[middle::Arg; 32]> = SmallVec::with_capacity(self.arg_count);
            for value in packed.iter() {
                ffi_args.push(middle::arg(value));
            }

            unsafe {
                self.cif.call::<()>(CodePtr(fn_ptr as *mut _), &ffi_args);
            }
        });
    }
}
