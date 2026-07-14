//! **The declarative tile-type layer** (`scratchpad/tile_layer_design.md` §1) — a deriving device that
//! generalizes [`crate::shape::MfmaShape`] from "the matrix-core shape" to "the whole tile". A tile is a
//! zero-sized *type* whose parameters (dtype, dims, operand role / swizzle, MFMA shape) *compute* a
//! [`TileDesc`] — plain DATA, exactly the fields [`crate::tile_move::LdsView`]/`LdsStage` thread by hand
//! today. No [`crate::ir::Node`] gains a type parameter (Recommendation C, `shape.rs:8-14`): the type
//! computes the constants, the IR stays data-driven, so the arena/lowering/verifier are untouched.
//!
//! **Migration step 1 (this file): the deriving device only** — the tile types + [`Tile`] handle + the
//! `desc()` derivation, proven against the hand-written [`MfmaShape`] data in `test::tile`. No IR node,
//! no `load`/`store`/`mma` vocabulary, no kernel migration (those are steps 2–5); `movement.rs` and the
//! kernels are unchanged, so the `test::byte_identity` gate stays green.
//!
//! **Honesty note (this session's measurements):** the design's §3.3 "register diet → 388→256 is a
//! one-line choice" is FALSIFIED — the fill is only ~40 combined regs and d128 FA is register-file
//! *capacity*-bound (see memory `mfma-shape-generic-design`). The tile layer's standing value is
//! ergonomics + consolidation (delete `movement.rs`, unify GEMM/FA/sort, delete the `const EPT`
//! bug-class); direct-to-LDS is an honest residency *choice* (a step of the perf stack), not a claimed
//! occupancy win. The [`TileDesc::vgprs`] ledger is kept as a legibility aid, not a magic lever.

use std::marker::PhantomData;

use crate::build::Elem;
use crate::ir::{AccDist, FragMap, Layout, Residency, TileId, Transform};
use crate::shape::MfmaShape;

/// The runtime descriptor a tile *type* derives — the DATA the ops address with (the exact set of
/// fields `LdsView`/`LdsStage` thread by hand today, computed ONCE from the type params). It rides on
/// the [`Tile`] handle and feeds the existing data-driven builder methods unchanged. Not `Copy`
/// (`swizzle: Layout` holds a `SmallVec`); the handle clones cheaply.
#[derive(Clone, Debug)]
pub struct TileDesc {
    /// Logical tile rows.
    pub rows: usize,
    /// Logical tile cols.
    pub cols: usize,
    /// Where the tile physically lives ([`Residency::Reg`] | `Lds` | `Global`).
    pub residency: Residency,
    /// Register operand map (`ir.rs` `FragMap`) for a Row/Col operand tile; `None` for LDS/global and
    /// for the accumulator (which addresses via [`Self::acc`], not a `FragMap`).
    pub frag: Option<FragMap>,
    /// Accumulator lane→(row,col) distribution — `Some` only for an `Acc` register tile.
    pub acc: Option<AccDist>,
    /// Fragments along the tile's axes (`movement.rs` `n_frags`) — the number of MFMA base fragments
    /// an `R×C` tile spans in its operand role. `0` for LDS/global.
    pub n_frags: usize,
    /// LDS inner width / row stride (`movement.rs` `inner`/`cols`); `= cols` for the contiguous case.
    pub inner: usize,
    /// The LDS bank swizzle (`Xor{cols}` | contiguous) — a property of the *type* (`LdsTile`'s `Sw`),
    /// so fill and gather cannot disagree on the XOR the way the hand-minted handles could.
    pub swizzle: Layout,
    /// Per-lane element run (the shape `EPT_A`/`EPT_B`/`EPT_C`); `0` for LDS/global.
    pub ept: usize,
    /// Live VGPRs this tile occupies (`0` for LDS/global) — `n_frags · ept · sizeof(E) / 4` (dwords).
    /// The register-footprint ledger (`TilePool::vgprs`, `schedule.rs:161`, promoted to the type).
    pub vgprs: usize,
}

/// Every tile type implements this. The ops read the TYPE for compile-time dispatch (which residency
/// path, which MFMA opcode) and [`Self::desc`] for addressing.
pub trait TileType: Copy + 'static {
    /// The element dtype (the one thing the *handle* already carries today, `Frag<E>`).
    type Dtype: Elem;
    /// Logical rows.
    const ROWS: usize;
    /// Logical cols.
    const COLS: usize;
    /// Physical residency.
    const RES: Residency;
    /// Derive the runtime descriptor from the type params.
    fn desc() -> TileDesc;
    /// The register footprint — the diet ledger, summed at authoring time over a kernel's reg tiles.
    fn vgprs() -> usize {
        Self::desc().vgprs
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// Register-tile operand roles (HK's `rt_layout` row/col/accumulator) — each derives its FragMap /
// AccDist / ept / n_frags from the MFMA shape, so the role marker is all the kernel names.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// A register tile's operand role, deriving the shape-dependent fragment data.
pub trait RegLayout: Copy + 'static {
    /// The operand `FragMap` — `Some` for Row/Col operands, `None` for the accumulator.
    fn frag<S: MfmaShape>() -> Option<FragMap>;
    /// The accumulator distribution — `Some` only for [`Acc`].
    fn acc<S: MfmaShape>() -> Option<AccDist>;
    /// Per-lane element run in this role (`EPT_A`/`EPT_B`/`EPT_C`).
    fn ept<S: MfmaShape>() -> usize;
    /// MFMA base fragments an `R×C` tile spans in this role.
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize;
}

/// The A-operand (Row) role — a straight `M×K` fragment (`ds_read_b64`, no register transpose).
#[derive(Copy, Clone, Debug)]
pub struct ARow;
/// The B-operand (Col) role — a transposed `K×N` fragment.
#[derive(Copy, Clone, Debug)]
pub struct BCol;
/// The C-accumulator role — an `M×N` fragment addressed by the [`AccDist`] distribution.
#[derive(Copy, Clone, Debug)]
pub struct Acc;

impl RegLayout for ARow {
    fn frag<S: MfmaShape>() -> Option<FragMap> {
        Some(S::a_map())
    }
    fn acc<S: MfmaShape>() -> Option<AccDist> {
        None
    }
    fn ept<S: MfmaShape>() -> usize {
        S::EPT_A
    }
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize {
        (rows / S::M) * (cols / S::K)
    }
}

impl RegLayout for BCol {
    fn frag<S: MfmaShape>() -> Option<FragMap> {
        Some(S::b_map())
    }
    fn acc<S: MfmaShape>() -> Option<AccDist> {
        None
    }
    fn ept<S: MfmaShape>() -> usize {
        S::EPT_B
    }
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize {
        (rows / S::K) * (cols / S::N)
    }
}

impl RegLayout for Acc {
    fn frag<S: MfmaShape>() -> Option<FragMap> {
        None
    }
    fn acc<S: MfmaShape>() -> Option<AccDist> {
        Some(S::acc_dist())
    }
    fn ept<S: MfmaShape>() -> usize {
        S::EPT_C
    }
    fn n_frags<S: MfmaShape>(rows: usize, cols: usize) -> usize {
        (rows / S::M) * (cols / S::N)
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// LDS-tile swizzle (HK's `st_shape::swizzle`) — a property of the *type*, so one `LdsTile` owns the
// bank XOR and fill/gather cannot disagree.
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// An LDS tile's bank-swizzle policy.
pub trait Swizzle: Copy + 'static {
    /// The layout for a `cols`-wide tile.
    fn layout(cols: usize) -> Layout;
}

/// Contiguous (no XOR) — the padded / direct-to-LDS-compatible layout. A direct-to-LDS fill REQUIRES
/// this: the hardware pins lane `L`→`m0 + L·4`, so a per-lane XOR cannot be applied on the fill side
/// (this session's finding). The [`crate::tile_move`] register-staged fill can use either.
#[derive(Copy, Clone, Debug)]
pub struct Plain;
/// The HK/CK bank-conflict XOR swizzle (`col ^ delta(row)`). Only reachable via the register-staged
/// fill; not compatible with a direct-to-LDS `(Lds,Global)` move.
#[derive(Copy, Clone, Debug)]
pub struct Xor;

impl Swizzle for Plain {
    fn layout(_cols: usize) -> Layout {
        Layout::contiguous()
    }
}

impl Swizzle for Xor {
    fn layout(cols: usize) -> Layout {
        let mut transforms = smallvec::SmallVec::new();
        transforms.push(Transform::Xor { cols });
        Layout { transforms }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// The three residency-specialized tile types (HK's `rt` / `st` / `gl`).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// Register tile — HK's `rt<T,R,C,Layout,Shape>`. dtype + logical dims + operand role + MFMA shape,
/// all in the type; derives its `FragMap`/`AccDist`/`n_frags`/`ept`/`vgprs` from `L` and `S`.
#[derive(Copy, Clone, Debug)]
pub struct RegTile<E: Elem, const R: usize, const C: usize, L: RegLayout, S: MfmaShape>(PhantomData<(E, L, S)>);

impl<E: Elem, const R: usize, const C: usize, L: RegLayout, S: MfmaShape> TileType for RegTile<E, R, C, L, S> {
    type Dtype = E;
    const ROWS: usize = R;
    const COLS: usize = C;
    const RES: Residency = Residency::Reg;
    fn desc() -> TileDesc {
        let n_frags = L::n_frags::<S>(R, C);
        let ept = L::ept::<S>();
        // per-lane bytes → dwords (VGPRs): `<4×bf16>` = 2 VGPRs; an `EPT_C=16` f32 acc = 16 VGPRs.
        let vgprs = n_frags * ept * E::dtype().bytes() / 4;
        TileDesc {
            rows: R,
            cols: C,
            residency: Residency::Reg,
            frag: L::frag::<S>(),
            acc: L::acc::<S>(),
            n_frags,
            inner: C,
            swizzle: Layout::contiguous(),
            ept,
            vgprs,
        }
    }
}

/// Shared/LDS tile — HK's `st<T,R,C,Shape>`. The swizzle is a property of the type (`Sw`), so fill and
/// gather cannot disagree on the bank XOR (the invariant `SharedTile` enforces today by convention).
#[derive(Copy, Clone, Debug)]
pub struct LdsTile<E: Elem, const R: usize, const C: usize, Sw: Swizzle>(PhantomData<(E, Sw)>);

impl<E: Elem, const R: usize, const C: usize, Sw: Swizzle> TileType for LdsTile<E, R, C, Sw> {
    type Dtype = E;
    const ROWS: usize = R;
    const COLS: usize = C;
    const RES: Residency = Residency::Lds;
    fn desc() -> TileDesc {
        TileDesc {
            rows: R,
            cols: C,
            residency: Residency::Lds,
            frag: None,
            acc: None,
            n_frags: 0,
            inner: C,
            swizzle: Sw::layout(C),
            ept: 0,
            vgprs: 0,
        }
    }
}

/// Global tile — HK's `gl<T,…>`. dtype + logical dims; residency/addressing derived from the type.
#[derive(Copy, Clone, Debug)]
pub struct GlobalTile<E: Elem, const R: usize, const C: usize>(PhantomData<E>);

impl<E: Elem, const R: usize, const C: usize> TileType for GlobalTile<E, R, C> {
    type Dtype = E;
    const ROWS: usize = R;
    const COLS: usize = C;
    const RES: Residency = Residency::Global;
    fn desc() -> TileDesc {
        TileDesc {
            rows: R,
            cols: C,
            residency: Residency::Global,
            frag: None,
            acc: None,
            n_frags: 0,
            inner: C,
            swizzle: Layout::contiguous(),
            ept: 0,
            vgprs: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────────────────────────
// The typed handle the ops pass around — the IR handle(s) + the derived desc as DATA (mirroring how
// `Frag<E>` carries `id` + a `FragMap`). Replaces `LdsView`/`LdsStage`/`SharedTile` (steps 3–5).
// ─────────────────────────────────────────────────────────────────────────────────────────────────

/// The underlying IR handle(s) a [`Tile`] wraps — the same ids used today (a `Frag`/`Lds`/`Buf` id, or
/// a small run of frag ids for a multi-fragment operand tile).
#[derive(Clone, Debug)]
pub enum TileInner {
    /// One or more register-fragment ids (a multi-fragment operand tile is a run).
    Reg(smallvec::SmallVec<[TileId; 4]>),
    /// A `DefineLocal` LDS id.
    Lds(TileId),
    /// An ABI-bound global buffer id.
    Global(TileId),
}

/// The typed tile handle: the IR `inner` handle(s) + the DERIVED `desc`. The ops dispatch on the type
/// `T` (compile-time: residency path / MFMA opcode) and address with `desc`.
#[derive(Clone, Debug)]
pub struct Tile<T: TileType> {
    /// The underlying IR handle(s).
    pub inner: TileInner,
    /// The descriptor derived from `T` (computed once by `T::desc()`).
    pub desc: TileDesc,
    _t: PhantomData<T>,
}

impl<T: TileType> Tile<T> {
    /// Wrap raw IR handle(s) as a typed tile, attaching the type-derived descriptor.
    pub fn new(inner: TileInner) -> Self {
        Tile { inner, desc: T::desc(), _t: PhantomData }
    }
}
