//! Kernel-scaffold helpers — the declarative preamble every builder re-typed by
//! hand: typed GL binding, role-based tile shortcuts, grid-index accessors, and
//! divisibility checks.
//!
//! Each helper is a thin, **allocation-order-preserving** forwarder over the
//! [`Kernel`]/[`crate::ArchCaps`] primitives, so a kernel migrated to the scaffold
//! emits the *identical* UOp graph (same `Param`/`DefineReg`/`DefineLocal` slot ids
//! → same content hash; [`crate::kernel_fingerprint`] diffs it if needed). The point
//! is to make the load-bearing invariants
//! safe by construction instead of by comment: the ABI slot order ([`Kernel::bind_abi`]
//! binds outputs-then-inputs by parameter structure) and the role→fragment
//! resolution (the tile shortcuts resolve [`crate::arch::FragRole`] via `caps`, so a
//! kernel never names a physical `RT_16X16`-family constant).

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::UOp;

use crate::arch::FragRole;
use crate::kernel::Kernel;
use crate::tile::{GL, RT, RV, ST};
use crate::tiles::{RT_16X16, TileLayout, VecLayout};

/// A global-buffer binding spec for [`Kernel::bind_abi`] (logical shape + element
/// dtype). The concrete buffer's dtype governs; `dtype` carries the author's intent.
#[derive(Clone, Debug)]
pub struct GlSpec {
    shape: Vec<usize>,
    dtype: DType,
}

impl GlSpec {
    /// A GLOBAL-buffer spec (logical `shape` + element `dtype`) for ABI binding
    /// via [`Kernel::bind_abi`].
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        Self { shape: shape.to_vec(), dtype }
    }
}

impl Kernel {
    /// Bind `outputs` then `inputs` as GL tiles, **in that order** — so the ABI slot
    /// order (the kernel's buffer / `Param` order) is fixed by the parameter
    /// structure rather than by the order of free-standing `gl` calls + a comment.
    /// Calls [`Kernel::gl`] in slice order, so the `Param` slots are identical to the
    /// hand-written sequence. A conditional/optional buffer binds with a plain
    /// [`Kernel::gl`] *after* this call (trailing-only — never interleaved).
    pub fn bind_abi(&self, outputs: &[GlSpec], inputs: &[GlSpec]) -> (Vec<GL>, Vec<GL>) {
        let outs = outputs.iter().map(|s| self.gl(&s.shape, s.dtype.clone())).collect();
        let ins = inputs.iter().map(|s| self.gl(&s.shape, s.dtype.clone())).collect();
        (outs, ins)
    }

    /// The grid block index on axis 0, named for readability (`block_idx[0]`).
    pub fn grid_x(&self) -> Arc<UOp> {
        self.block_idx[0].clone()
    }
    /// The grid block index on axis 1 (`block_idx[1]`).
    pub fn grid_y(&self) -> Arc<UOp> {
        self.block_idx[1].clone()
    }
    /// The grid block index on axis 2 (`block_idx[2]`).
    pub fn grid_z(&self) -> Arc<UOp> {
        self.block_idx[2].clone()
    }

    /// An f32 accumulator register tile ([`FragRole::Accumulator`]), arch-resolved.
    pub fn acc(&self, dims: (usize, usize), layout: TileLayout) -> RT<'_> {
        self.rt(dims, DType::Float32, layout, self.caps.frag(FragRole::Accumulator))
    }
    /// An f32 **transposed** accumulator ([`FragRole::AccumulatorT`]) — the layout for
    /// an N-major store (e.g. the FA output `O[q,d]` from the `[d,q]` PV accumulator).
    pub fn acc_t(&self, dims: (usize, usize), layout: TileLayout) -> RT<'_> {
        self.rt(dims, DType::Float32, layout, self.caps.frag(FragRole::AccumulatorT))
    }
    /// A WMMA input-operand register tile ([`FragRole::Operand`]) of dtype `dt`.
    pub fn operand(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> RT<'_> {
        self.rt(dims, dt, layout, self.caps.frag(FragRole::Operand))
    }
    /// An f32 ortho register-vector — the softmax/reduce accumulator vectors. Uses
    /// [`RT_16X16`] on both arches (the vectors are not arch-fragment-resolved today).
    pub fn acc_vec(&self, length: usize) -> RV<'_> {
        self.rv(length, DType::Float32, VecLayout::Ortho, RT_16X16)
    }
    /// A shared (LDS) tile with the arch's canonical strip ([`crate::ArchCaps::shared_default`]).
    pub fn shared(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> ST {
        self.st(dims, dt, layout, self.caps.shared_default())
    }
    /// A 2×-size double-buffered shared tile with the canonical strip.
    pub fn shared_db(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> ST {
        self.st_db(dims, dt, layout, self.caps.shared_default())
    }
    /// A shared (LDS) tile with the XOR-swizzled strip ([`crate::ArchCaps::shared_swizzled`]),
    /// for kernels that swizzle to avoid LDS bank conflicts (the matmul A/B strips).
    pub fn shared_sw(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> ST {
        self.st(dims, dt, layout, self.caps.shared_swizzled())
    }
    /// A 2×-size double-buffered XOR-swizzled shared tile: [`Self::shared_sw`] in a
    /// ping/pong ring (the software-pipelined matmul A/B strips). Select a half with
    /// `with_base_offset(parity * half_elems())`.
    pub fn shared_db_sw(&self, dims: (usize, usize), dt: DType, layout: TileLayout) -> ST {
        self.st_db(dims, dt, layout, self.caps.shared_swizzled())
    }

    /// Build-time divisibility check with a uniform message (emits no UOps, so it is
    /// invisible to the graph fingerprint).
    ///
    /// # Panics
    /// Panics if `value` is not divisible by `by` (its whole job — the message
    /// names `what`). `by == 0` is a divide-by-zero.
    pub fn assert_divisible(value: usize, by: usize, what: &str) {
        assert_eq!(value % by, 0, "{what}: {value} must be a multiple of {by}");
    }
}
