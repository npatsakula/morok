//! The typed builder — a thin, ergonomic front-end that emits interned [`TileId`]
//! handles, never eager UOp (DESIGN.md §A, §0).
//!
//! Typing is deliberately *gentle* (DESIGN.md §OPEN-2, ratified): the element
//! dtype rides in the types via a **sealed** [`Elem`] trait (so `add(a, b)`
//! type-checks that both operands share a dtype, and "no impl ⇒ no method"), while
//! shape/register-range/schedule validity stay in data + the verifier — never in
//! surface types (HK's single largest unreadability source). We do NOT push
//! typestate: handles are plain `Copy` wrappers over `TileId`.

use std::collections::HashMap;
use std::marker::PhantomData;

use svod_dtype::DType;

use crate::ir::{AccDist, BinOp, FragMap, IndexOp, Node, Scalar, ScopeAxis, TileId, TileIr};

mod sealed {
    pub trait Sealed {}
}

/// A legal tile element dtype (sealed — the DSL's dtype whitelist). Adding a dtype
/// is one `impl`; there is no way for a caller to invent an unsupported element.
pub trait Elem: sealed::Sealed + Copy + 'static {
    fn dtype() -> DType;
}

/// 32-bit float — the accumulation / elementwise dtype the skeleton exercises.
#[derive(Copy, Clone, Debug)]
pub struct F32;
impl sealed::Sealed for F32 {}
impl Elem for F32 {
    fn dtype() -> DType {
        DType::Float32
    }
}

/// bfloat16 — the matrix-core operand dtype (the matmul's A/B inputs; f32 accumulate).
#[derive(Copy, Clone, Debug)]
pub struct BF16;
impl sealed::Sealed for BF16 {}
impl Elem for BF16 {
    fn dtype() -> DType {
        DType::BFloat16
    }
}

/// An `Elem`-typed value handle (a register-resident scalar/tile). The phantom
/// dtype is what makes [`Builder::add`] & friends reject mismatched operands at
/// compile time without any runtime dtype check.
#[derive(Copy, Clone, Debug)]
pub struct Val<E: Elem> {
    pub id: TileId,
    _e: PhantomData<E>,
}

/// A **shape-labelled view** over a [`Val`] — an `R×C` MMA tile whose dims live only in the type
/// (const generics) and ERASE at the matrix-core op. It exists solely in the straight-line
/// `gather → mma` window and is NEVER carried in a slot, so the interned IR and the [`Frag`]/
/// `AccSlot` carry path stay data-driven (no monomorphisation of the movement layer). The shape
/// match `Tile<M,K> · Tile<K,N> → Tile<M,N>` — a *shared* `const K` — is what type-checks operand
/// composition with no type-level arithmetic; the accumulator width still rides as data via
/// [`crate::shape::MfmaShape`]. See [`Builder::mma_of`] / [`Builder::mma_asm_of`].
#[derive(Copy, Clone, Debug)]
pub struct Tile<E: Elem, const R: usize, const C: usize> {
    val: Val<E>,
}

impl<E: Elem, const R: usize, const C: usize> Tile<E, R, C> {
    /// Drop the shape label back to the underlying value (the erasure at the op boundary).
    pub fn erase(self) -> Val<E> {
        self.val
    }
}

/// An index-typed value handle (addressing arithmetic / loop counters / axes).
#[derive(Copy, Clone, Debug)]
pub struct Idx(pub TileId);

/// A global buffer handle bound to an ABI slot.
#[derive(Copy, Clone, Debug)]
pub struct Buf<E: Elem> {
    pub id: TileId,
    pub len: usize,
    _e: PhantomData<E>,
}

/// A global buffer view with a runtime logical length and a static maximum allocation.
#[derive(Copy, Clone, Debug)]
pub struct BoundedBuf<E: Elem> {
    pub buf: Buf<E>,
    pub len: Idx,
}

/// A register-accumulator handle.
#[derive(Copy, Clone, Debug)]
pub struct Reg<E: Elem> {
    pub id: TileId,
    pub len: usize,
    _e: PhantomData<E>,
}

/// A shared-memory (LDS) buffer handle. Unlike [`Buf`] it binds no ABI slot — it is a
/// per-kernel `DefineLocal` allocation staged from global by the workgroup and read
/// back (cross-lane) after a [`Builder::barrier`]. The block-tiling reuse lever lives
/// here: one LDS strip fill feeds many MFMAs.
#[derive(Copy, Clone, Debug)]
pub struct Lds<E: Elem> {
    pub id: TileId,
    pub len: usize,
    _e: PhantomData<E>,
}

/// A register-**fragment** handle: a per-lane MFMA fragment carrying its
/// [`FragMap`] lane→(row,col) layout as data. The map drives the `lane_rc`
/// addressing in the fragment gather/scatter and the `ept` of the MMA operands.
#[derive(Copy, Clone, Debug)]
pub struct Frag<E: Elem> {
    pub id: TileId,
    pub map: FragMap,
    _e: PhantomData<E>,
}

/// A loop handle: its counter is index-typed, its closing edge is [`Builder::end`].
#[derive(Copy, Clone, Debug)]
pub struct Range {
    pub id: TileId,
}

/// A reusable runtime iteration domain. Multiple loops may consume the same bounded trip expression
/// while receiving distinct range IDs.
#[derive(Copy, Clone, Debug)]
pub struct IterDomain {
    pub trips: Idx,
}

/// A lexical authoring scope. Rebinding a reusable leaf through a scope gives downstream expressions
/// region identity without emitting an instruction or inventing a memory-ordering dependency.
#[derive(Copy, Clone, Debug)]
pub struct Scope {
    pub id: Option<TileId>,
}

impl Scope {
    /// Identity/no-op scope used by schedules that do not need lexical disambiguation.
    pub const ROOT: Self = Self { id: None };
}

impl Range {
    /// This loop's range as an ordering edge (keeps a routed read in the loop body).
    pub fn dep(self) -> Edge {
        Edge(self.id)
    }
}

/// An effect handle (a store, an ended store, an after-wrapped buffer).
#[derive(Copy, Clone, Debug)]
#[must_use]
pub struct Effect(pub TileId);

impl Effect {
    /// This effect as an ordering edge (a happens-before token).
    pub fn dep(self) -> Edge {
        Edge(self.0)
    }
}

/// A happens-before ordering token — an [`Effect`] or [`Range`] completion routed as a scheduling
/// edge. A newtype over `TileId`: a payload value id cannot masquerade as an ordering token, and the
/// one sanctioned value→edge cast is the explicit [`Edge::anchor`]. Erases to the same `TileId`, so
/// the interned IR is unchanged.
#[derive(Copy, Clone, Debug)]
#[must_use]
pub struct Edge(TileId);

impl Edge {
    /// The ONE sanctioned value-id → ordering-edge cast: pin a scheduling hint on a *value* whose
    /// completion positions it (a load result used as a `sched_fence` anchor, a `set_prio` op anchor,
    /// a scope marker, the `wave_barrier` warp-row operand's neighbours).
    pub fn anchor(id: TileId) -> Self {
        Self(id)
    }
    /// The underlying id, unpacked only at the interning boundary.
    pub(crate) fn raw(self) -> TileId {
        self.0
    }
}

/// An effect/batch anchor that positions a queue-wide VMEM wait. Constructed from [`Effect`], not a
/// payload value, so a wait cannot accidentally claim readiness for one arbitrary transfer result.
pub struct VmemWaitAnchor(TileId);

impl From<Effect> for VmemWaitAnchor {
    fn from(effect: Effect) -> Self {
        Self(effect.0)
    }
}

// Qualification probes predate the typed wait boundary. Keep their test-only source compatibility;
// production code must supply an Effect or combined commit batch.
#[cfg(test)]
impl From<TileId> for VmemWaitAnchor {
    fn from(id: TileId) -> Self {
        Self(id)
    }
}

impl<E: Elem> Val<E> {
    fn wrap(id: TileId) -> Self {
        Val { id, _e: PhantomData }
    }

    /// Label this value's logical `R×C` MMA-tile shape (a zero-cost view; erases at the op).
    pub fn tile<const R: usize, const C: usize>(self) -> Tile<E, R, C> {
        Tile { val: self }
    }
}

/// The staged builder. Owns the [`TileIr`] arena; every method interns and returns
/// a `Copy` handle.
pub struct Builder {
    pub ir: TileIr,
    pub name: String,
    runtime_params: HashMap<String, (i64, i64, TileId)>,
}

impl Builder {
    pub fn new(name: impl Into<String>) -> Self {
        Builder { ir: TileIr::new(), name: name.into(), runtime_params: HashMap::new() }
    }

    /// Consume the builder, yielding the arena (for lowering / passes).
    pub fn into_ir(self) -> TileIr {
        self.ir
    }

    // ── leaves ───────────────────────────────────────────────────────────────

    /// Bind the next global buffer of `len` `E`-elements to a fresh ABI slot
    /// (outputs must be bound before inputs — the slot order is the launch ABI).
    pub fn global<E: Elem>(&mut self, len: usize) -> Buf<E> {
        let slot = self.ir.fresh_slot();
        let id = self.ir.intern(Node::Global { slot, dtype: E::dtype(), len });
        Buf { id, len, _e: PhantomData }
    }

    /// Allocate a register accumulator of `len` `E`-elements (a fresh, disambiguated
    /// `DefineReg` slot).
    pub fn define_reg<E: Elem>(&mut self, len: usize) -> Reg<E> {
        let id = self.ir.fresh_reg_id();
        let id = self.ir.intern(Node::DefineReg { id, dtype: E::dtype(), len });
        Reg { id, len, _e: PhantomData }
    }

    /// Allocate a shared-memory (LDS) buffer of `len` `E`-elements (a fresh,
    /// disambiguated `DefineLocal` slot). Stage into it with [`Self::store_lds`], fence
    /// with [`Self::barrier`], and read back (cross-lane) with [`Self::load_lds_after`].
    pub fn define_local<E: Elem>(&mut self, len: usize) -> Lds<E> {
        let id = self.ir.fresh_local_id();
        let id = self.ir.intern(Node::DefineLocal { id, dtype: E::dtype(), len });
        Lds { id, len, _e: PhantomData }
    }

    /// The grid index along `axis` with the given `bound` (the launch geometry
    /// rides on these — global size is the product of grid bounds).
    pub fn grid_axis(&mut self, axis: u8, bound: i64) -> Idx {
        Idx(self.ir.intern(Node::Axis { axis: ScopeAxis::Grid(axis), bound }))
    }

    /// A grid index whose launch extent is evaluated from runtime scalar arguments.
    pub fn grid_axis_dyn(&mut self, axis: u8, bound: Idx) -> Idx {
        Idx(self.ir.intern(Node::AxisDyn { axis: ScopeAxis::Grid(axis), bound: bound.0 }))
    }

    /// The block (thread) index with `bound` threads.
    pub fn block_axis(&mut self, bound: i64) -> Idx {
        Idx(self.ir.intern(Node::Axis { axis: ScopeAxis::Block, bound }))
    }

    /// A block index with a runtime local-size bound. Prefer a static block size for tuned GPU kernels.
    pub fn block_axis_dyn(&mut self, bound: Idx) -> Idx {
        Idx(self.ir.intern(Node::AxisDyn { axis: ScopeAxis::Block, bound: bound.0 }))
    }

    /// Open a loop over `trips` iterations; nesting is emergent from the resulting
    /// [`Range`]/`End` edges — never authored (DESIGN.md §D boundary).
    pub fn range(&mut self, trips: i64) -> Range {
        let rid = self.ir.fresh_range_id();
        Range { id: self.ir.intern(Node::Range { id: rid, trips }) }
    }

    /// Open a statically-sized loop after a peeled prologue or other required incoming effects.
    /// Unlike attaching `deps` to the loop body, this makes those effects dominate the Range itself.
    pub fn range_after(&mut self, trips: i64, deps: &[Edge]) -> Range {
        let rid = self.ir.fresh_range_id();
        let deps = deps.iter().map(|e| e.raw()).collect();
        Range { id: self.ir.intern(Node::RangeAfter { id: rid, trips, deps }) }
    }

    /// Package a runtime trip expression as a reusable iteration domain.
    pub fn iter_domain(&self, trips: Idx) -> IterDomain {
        IterDomain { trips }
    }

    /// Open a loop over a runtime iteration domain.
    pub fn range_dyn(&mut self, domain: IterDomain) -> Range {
        self.range_dyn_after(domain, &[])
    }

    /// Open a runtime-sized loop after a peeled prologue or other incoming effects.
    pub fn range_dyn_after(&mut self, domain: IterDomain, deps: &[Edge]) -> Range {
        let rid = self.ir.fresh_range_id();
        let deps = deps.iter().map(|e| e.raw()).collect();
        Range { id: self.ir.intern(Node::RangeDyn { id: rid, trips: domain.trips.0, deps }) }
    }

    /// Mint a zero-instruction lexical scope marker. Use [`Self::scope_idx`] / [`Self::scope_lds`] on
    /// reusable leaves before deriving addresses in warmup, loop, epilogue, or drain regions.
    pub fn scope(&mut self, deps: &[Edge]) -> Scope {
        let id = self.ir.fresh_scope_id();
        let deps = deps.iter().map(|e| e.raw()).collect();
        Scope { id: Some(self.ir.intern(Node::Scope { id, deps })) }
    }

    /// The loop counter as an index value.
    pub fn counter(&self, r: Range) -> Idx {
        Idx(r.id)
    }

    /// An index-typed integer constant.
    pub fn idx_const(&mut self, v: i64) -> Idx {
        Idx(self.ir.intern(Node::Const { scalar: Scalar::Int(v), dtype: DType::Index }))
    }

    /// Declare an i32-compatible bounded runtime integer kernel argument. Reusing a name requires the
    /// same inclusive bounds and returns the same value handle.
    pub fn scalar_param(&mut self, name: impl Into<String>, min: i64, max: i64) -> Idx {
        let name = name.into();
        assert!(min <= max, "runtime scalar {name:?} has inverted bounds [{min}, {max}]");
        assert!(
            min >= i32::MIN as i64 && max <= i32::MAX as i64,
            "runtime scalar {name:?} bounds must fit the current i32 device ABI"
        );
        let mut chars = name.chars();
        assert!(
            chars.next().is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
                && chars.all(|c| c == '_' || c.is_ascii_alphanumeric()),
            "runtime scalar name {name:?} is not an LLVM identifier"
        );
        if let Some(&(old_min, old_max, id)) = self.runtime_params.get(&name) {
            assert_eq!((old_min, old_max), (min, max), "runtime scalar {name:?} reused with different bounds");
            return Idx(id);
        }
        let id = self.ir.intern(Node::ScalarParam { name: name.clone(), min, max });
        self.runtime_params.insert(name, (min, max, id));
        Idx(id)
    }

    /// Bind a global buffer to a runtime logical element length. The length's declared maximum must not
    /// exceed the statically allocated capacity when it is a direct scalar parameter or constant.
    pub fn bounded<E: Elem>(&self, buf: Buf<E>, len: Idx) -> BoundedBuf<E> {
        let bounds = match self.ir.node(len.0) {
            Node::ScalarParam { min, max, .. } => Some((*min, *max)),
            Node::Const { scalar: Scalar::Int(v), dtype } if *dtype == DType::Index => Some((*v, *v)),
            _ => None,
        };
        let (min, max) = bounds.expect("bounded view length must be a direct bounded scalar or index constant");
        assert!(min >= 0, "bounded view length must be non-negative");
        assert!(max as usize <= buf.len, "bounded view maximum {max} exceeds buffer capacity {}", buf.len);
        BoundedBuf { buf, len }
    }

    /// An `E`-typed scalar constant.
    pub fn scalar<E: Elem>(&mut self, bits: Scalar) -> Val<E> {
        Val::wrap(self.ir.intern(Node::Const { scalar: bits, dtype: E::dtype() }))
    }

    /// An f32 constant.
    pub fn f32(&mut self, v: f32) -> Val<F32> {
        self.scalar::<F32>(Scalar::f32(v))
    }

    // ── addressing (index band) ──────────────────────────────────────────────

    pub fn idx_add(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Add, a: a.0, b: b.0 }))
    }
    pub fn idx_sub(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Sub, a: a.0, b: b.0 }))
    }
    pub fn idx_mul(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Mul, a: a.0, b: b.0 }))
    }
    pub fn idx_mod(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Mod, a: a.0, b: b.0 }))
    }
    pub fn idx_div(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Div, a: a.0, b: b.0 }))
    }
    pub fn idx_xor(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Xor, a: a.0, b: b.0 }))
    }
    pub fn idx_shr(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Shr, a: a.0, b: b.0 }))
    }
    pub fn idx_shl(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Shl, a: a.0, b: b.0 }))
    }
    pub fn idx_min(&mut self, a: Idx, b: Idx) -> Idx {
        Idx(self.ir.intern(Node::IndexAlu { op: IndexOp::Min, a: a.0, b: b.0 }))
    }

    /// Integer ceil-division for non-negative runtime dimensions.
    pub fn idx_ceil_div(&mut self, value: Idx, divisor: Idx) -> Idx {
        let one = self.idx_const(1);
        let adjustment = self.idx_sub(divisor, one);
        let numerator = self.idx_add(value, adjustment);
        self.idx_div(numerator, divisor)
    }

    /// Compute a flat element offset from runtime/static coordinates and strides.
    pub fn strided_offset<const R: usize>(&mut self, indices: [Idx; R], strides: [Idx; R]) -> Idx {
        let mut offset = self.idx_const(0);
        for axis in 0..R {
            let term = self.idx_mul(indices[axis], strides[axis]);
            offset = self.idx_add(offset, term);
        }
        offset
    }

    /// A **layout-application point** for the column of a `cols`-wide LDS tile access at
    /// logical `(row, col)` — [`Node::LdsCol`]. It lowers to `col` (flat) unless
    /// `SwizzlePass` has rewritten it to the bank XOR. The full LDS offset is
    /// `row·cols + lds_col(row, col, cols)`; emit it wherever an LDS tile is addressed so
    /// the swizzle stays a composable `.apply` refinement, not hand-woven arithmetic.
    pub fn lds_col(&mut self, row: Idx, col: Idx, cols: usize) -> Idx {
        Idx(self.ir.intern(Node::LdsCol { row: row.0, col: col.0, cols }))
    }

    /// The per-lane `(row, col)` within a base fragment — tk's `lane_rc`
    /// (`tk/src/group/mod.rs`) for the gfx942 non-interleaved MFMA fragment, built
    /// from explicit `IndexAlu` div/mod (the const-foldable index band, §2.4):
    /// - Row (`transpose = false`, the A operand): `row = lane % rows`,
    ///   `col = (lane / rows)·stride + inner`.
    /// - Col (`transpose = true`, the B / C operands): `row = (lane / cols)·stride + inner`,
    ///   `col = lane % cols`.
    pub fn lane_rc(&mut self, map: FragMap, lane: Idx, inner: Idx) -> (Idx, Idx) {
        let stride = self.idx_const(map.stride as i64);
        if map.transpose {
            let cols = self.idx_const(map.cols as i64);
            let lg = self.idx_div(lane, cols);
            let lg = self.idx_mul(lg, stride);
            let row = self.idx_add(lg, inner);
            let col = self.idx_mod(lane, cols);
            (row, col)
        } else {
            let rows = self.idx_const(map.rows as i64);
            let row = self.idx_mod(lane, rows);
            let lg = self.idx_div(lane, rows);
            let lg = self.idx_mul(lg, stride);
            let col = self.idx_add(lg, inner);
            (row, col)
        }
    }

    /// The per-lane `(row, col)` of accumulator element `i` under an [`AccDist`] (§migration) — the
    /// MFMA **accumulator** distribution the [`FragMap`]'s single `lane_rc` run cannot express. For the
    /// 16×16×16 degenerate case (`m_blocks == 1`) this emits the SAME index nodes as
    /// [`Self::lane_rc`]'s `transpose` branch (`row = (lane/16)·4 + i`, `col = lane%16`) — the consts are
    /// created in the same order (`lane_m_stride` then `n_lanes`) and the block term is folded away, so
    /// interning collapses the two to identical [`TileId`]s (proven in `test::byte_identity`). For
    /// 32×32×8 the `m_blk·m_block_stride` term adds the four row-blocks (8 apart) the single run lacked.
    pub fn acc_rc(&mut self, dist: AccDist, lane: Idx, i: usize) -> (Idx, Idx) {
        let m_blk = i / dist.m_inner;
        let m_in = i % dist.m_inner;
        // `lane_m_stride` first, then `n_lanes` — mirrors `lane_rc(transpose)`'s `stride`-then-`cols`
        // const order so a brand-new const lands at the same id (the byte-identity requirement).
        let lane_stride = self.idx_const(dist.lane_m_stride as i64);
        let n_lanes = self.idx_const(dist.n_lanes as i64);
        let lg = self.idx_div(lane, n_lanes);
        let lg = self.idx_mul(lg, lane_stride);
        let inner = self.idx_const(m_in as i64);
        let row = self.idx_add(lg, inner);
        // The outer M-block offset (`i/m_inner`·`m_block_stride`) — absent for 16×16×16 (`m_blocks==1`,
        // so `m_blk == 0`), so no node is emitted there and the 16×16 path stays byte-identical.
        let row = if m_blk == 0 {
            row
        } else {
            let blk = self.idx_const((m_blk * dist.m_block_stride) as i64);
            self.idx_add(row, blk)
        };
        let col = self.idx_mod(lane, n_lanes);
        (row, col)
    }

    // ── loads / stores (movement, lowered to INDEX + LOAD/STORE) ─────────────

    /// Load an `E`-element from a global buffer at flat `offset`.
    pub fn load<E: Elem>(&mut self, buf: Buf<E>, offset: Idx) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: buf.id, offset: offset.0, dtype: E::dtype() }))
    }

    /// Gated scalar load from a bounded view. Out-of-range lanes produce `alt` and never dereference the
    /// underlying pointer.
    pub fn load_bounded<E: Elem>(&mut self, view: BoundedBuf<E>, offset: Idx, alt: Val<E>) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadGlobalBounded {
            buf: view.buf.id,
            offset: offset.0,
            bound: view.len.0,
            alt: alt.id,
            dtype: E::dtype(),
        }))
    }

    /// Build a bounded vector load from independently gated scalar elements. This defines safe partial
    /// tail semantics without relying on target-specific raw-buffer straddle behavior.
    pub fn load_vec_bounded<E: Elem>(&mut self, view: BoundedBuf<E>, base: Idx, ept: usize, alt: Val<E>) -> Val<E> {
        let values: Vec<Val<E>> = (0..ept)
            .map(|i| {
                let i = self.idx_const(i as i64);
                let offset = self.idx_add(base, i);
                self.load_bounded(view, offset, alt)
            })
            .collect();
        self.vec_build(&values)
    }

    /// Bounded scalar load through an R-dimensional strided view.
    pub fn load_strided_bounded<E: Elem, const R: usize>(
        &mut self,
        view: BoundedBuf<E>,
        indices: [Idx; R],
        strides: [Idx; R],
        alt: Val<E>,
    ) -> Val<E> {
        let offset = self.strided_offset(indices, strides);
        self.load_bounded(view, offset, alt)
    }

    /// Load an `E`-element from a register cell at flat `offset` (with an optional
    /// ordering edge — the loop-carry read routes through the prior store + range).
    pub fn load_reg<E: Elem>(&mut self, reg: Reg<E>, offset: Idx) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: reg.id, offset: offset.0, dtype: E::dtype() }))
    }

    /// Load from a register cell whose backing handle has been ordering-wrapped
    /// (`reg.after([prev_store, range, …])`), so the read observes the
    /// routed-through effects/ranges — the loop-carry read (DESIGN.md §2.1). `deps`
    /// may mix effect and range handles (see [`Effect::dep`] / [`Range::dep`]).
    pub fn load_reg_after<E: Elem>(&mut self, reg: Reg<E>, offset: Idx, deps: &[Edge]) -> Val<E> {
        let after = self.after_buf(reg.id, deps);
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: after, offset: offset.0, dtype: E::dtype() }))
    }

    /// Store an `E` value into a global buffer at flat `offset` (a terminal effect).
    pub fn store<E: Elem>(&mut self, buf: Buf<E>, offset: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreGlobal { buf: buf.id, offset: offset.0, value: value.id }))
    }

    /// Gated scalar store to a bounded view. Out-of-range lanes leave memory untouched.
    pub fn store_bounded<E: Elem>(&mut self, view: BoundedBuf<E>, offset: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreGlobalBounded {
            buf: view.buf.id,
            offset: offset.0,
            bound: view.len.0,
            value: value.id,
        }))
    }

    /// Bounded scalar store through an R-dimensional strided view.
    pub fn store_strided_bounded<E: Elem, const R: usize>(
        &mut self,
        view: BoundedBuf<E>,
        indices: [Idx; R],
        strides: [Idx; R],
        value: Val<E>,
    ) -> Effect {
        let offset = self.strided_offset(indices, strides);
        self.store_bounded(view, offset, value)
    }

    /// Store an `E` value into a register cell at flat `offset`.
    pub fn store_reg<E: Elem>(&mut self, reg: Reg<E>, offset: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreGlobal { buf: reg.id, offset: offset.0, value: value.id }))
    }

    // ── shared-memory (LDS) staging + barrier (§2.5, the reuse lever) ─────────

    /// Store an `E` value into an LDS buffer at flat `offset` — the global→LDS stage
    /// write (a lane fills its share of the shared strip).
    pub fn store_lds<E: Elem>(&mut self, lds: Lds<E>, offset: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreGlobal { buf: lds.id, offset: offset.0, value: value.id }))
    }

    /// Store into an LDS buffer **ordered after** `deps` (the buffer handle observes them):
    /// the pipeline's commit-after-WAR — the write that overwrites the strip must follow the
    /// previous iteration's gather (the WAR barrier), carried through `[seed, range]`. The
    /// `After` edge on the destination buffer is the store analog of [`Self::load_lds_after`].
    pub fn store_lds_after<E: Elem>(&mut self, lds: Lds<E>, offset: Idx, value: Val<E>, deps: &[Edge]) -> Effect {
        let after = self.after_buf(lds.id, deps);
        Effect(self.ir.intern(Node::StoreGlobal { buf: after, offset: offset.0, value: value.id }))
    }

    /// An LDS handle re-bound so **every** write through it observes `deps` (the WAR barrier):
    /// the whole-strip commit-after-WAR for the `stages=2` pipeline. Wrapping the buffer once
    /// (vs. threading `deps` into each `store_lds*`) keeps the vectorised/scalar fill functions
    /// dep-agnostic — the ordering rides on the destination handle they already write to. The
    /// buffer analog of [`Self::frag_after`].
    pub fn lds_after<E: Elem>(&mut self, lds: Lds<E>, deps: &[Edge]) -> Lds<E> {
        let id = self.after_buf(lds.id, deps);
        Lds { id, len: lds.len, _e: PhantomData }
    }

    /// ONE `<ept×E>` vector load of a contiguous **global** run at flat `base` (the
    /// coalesced, vectorised fill read → `buffer_load_dwordx*`) — the global mirror of
    /// [`Self::load_lds_vec_after`], no ordering edge (a plain source read).
    pub fn load_vec<E: Elem>(&mut self, buf: Buf<E>, base: Idx, ept: usize) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadVecAt { buf: buf.id, base: base.0, ept, dtype: E::dtype() }))
    }

    /// A global `load_vec` ordered after `deps` — the ordering rides on the buffer via `After` (as the
    /// LDS reads carry their RAW ordering), so the linearizer emits the load in `deps`' cluster instead
    /// of floating it to the loop top. Used to PIN the split prefetch (HK's A@C0 / B@C4): the load nodes
    /// hash-cons identically regardless of authoring cluster, so without this edge the split is a no-op.
    pub fn load_vec_after<E: Elem>(&mut self, buf: Buf<E>, base: Idx, ept: usize, deps: &[Edge]) -> Val<E> {
        let buf = if deps.is_empty() { buf.id } else { self.after_buf(buf.id, deps) };
        Val::wrap(self.ir.intern(Node::LoadVecAt { buf, base: base.0, ept, dtype: E::dtype() }))
    }

    /// The **buffer-resource descriptor** (`srsrc`) of a global `buf`, based at element `base_off`
    /// ([`Node::MakeBufferRsrc`] → `make.buffer.rsrc.p0` of `&buf[base_off]`). `base_off` is the
    /// workgroup-uniform tile origin (`origin·K + k_base`), so the descriptor base advances per tile in
    /// SCALAR — the escape from FLAT `global_load`'s per-iteration 64-bit VGPR address. Feeds
    /// [`Self::buffer_load_raw`] with `soffset = 0` (HK's scheme; a non-zero soffset is mishandled by the
    /// raw config `0x110000`). `num_bytes` = the whole-buffer extent (the SRD bound is base-relative;
    /// in-tile voffsets never exceed it, and valid tiling keeps every access in-buffer).
    pub fn make_buffer_rsrc<E: Elem>(&mut self, buf: Buf<E>, base_off: Idx) -> Idx {
        let num_bytes = (buf.len * E::dtype().bytes()) as i64;
        Idx(self.ir.intern(Node::MakeBufferRsrc { buf: buf.id, base_off: base_off.0, num_bytes }))
    }

    /// Build a modern raw-buffer descriptor over a bounded view. The descriptor stays based at element
    /// zero so its runtime byte extent is exact and does not require saturating base-relative arithmetic.
    pub fn make_buffer_rsrc_bounded<E: Elem>(&mut self, view: BoundedBuf<E>) -> Idx {
        let zero = self.idx_const(0);
        let elem_bytes = self.idx_const(E::dtype().bytes() as i64);
        let num_bytes = self.idx_mul(view.len, elem_bytes);
        Idx(self.ir.intern(Node::MakeBufferRsrcDyn { buf: view.buf.id, base_off: zero.0, num_bytes: num_bytes.0 }))
    }

    /// ONE MUBUF `raw.buffer.load` ([`Node::BufferLoadRaw`]) reading the `ept`-element run at
    /// `rsrc[voffset]` bytes (`soffset = 0`), the SGPR-descriptor DRAM prefetch over FLAT `global_load`. The
    /// descriptor base (from [`Self::make_buffer_rsrc`]) advances per K-tile in scalar; `voffset` is the
    /// per-lane within-tile byte offset (loop-invariant). `order` pins the load into its authoring cluster
    /// (ordering-only, as [`Self::load_vec_after`]'s `deps`).
    pub fn buffer_load_raw<E: Elem>(&mut self, rsrc: Idx, voffset: Idx, ept: usize, order: &[Edge]) -> Val<E> {
        let order = order.iter().map(|e| e.raw()).collect();
        Val::wrap(self.ir.intern(Node::BufferLoadRaw {
            rsrc: rsrc.0,
            voffset: voffset.0,
            ept,
            dtype: E::dtype(),
            order,
        }))
    }

    /// ONE `<ept×E>` vector store of a contiguous LDS run at flat `base` ([`Node::StoreVecAt`]
    /// → `ds_write_b64`/`b128`) — the store mirror of [`Self::load_lds_vec_after`], replacing
    /// `ept` scalar `store_lds` for a contiguous, aligned run (the vectorised fill).
    pub fn store_lds_vec<E: Elem>(&mut self, lds: Lds<E>, base: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreVecAt { buf: lds.id, base: base.0, value: value.id }))
    }

    /// Issue one gfx942 hardware direct GLOBAL→LDS dword transfer. Both offsets are bf16-element
    /// offsets; the source and destination must therefore be even. The returned effect must be folded
    /// into a later publication barrier, which drains the VMEM-tracked DMA before LDS consumers run.
    pub fn global_load_lds_dword(
        &mut self,
        src: Buf<BF16>,
        src_offset: Idx,
        dst: Lds<BF16>,
        dst_offset: Idx,
        deps: &[Edge],
    ) -> Effect {
        Effect(self.ir.intern(Node::GlobalLoadLdsDword {
            src: src.id,
            src_offset: src_offset.0,
            dst: dst.id,
            dst_offset: dst_offset.0,
            deps: deps.iter().map(|e| e.raw()).collect(),
        }))
    }

    /// Extract scalar element `index` from a vector value ([`Node::VecExtract`] → `gep`).
    pub fn vec_extract<E: Elem>(&mut self, v: Val<E>, index: usize) -> Val<E> {
        Val::wrap(self.ir.intern(Node::VecExtract { vec: v.id, index, dtype: E::dtype() }))
    }

    /// Build a `<len×E>` vector from scalar `elements` ([`Node::VecBuild`] → `vectorize`) —
    /// the register-transpose store operand.
    pub fn vec_build<E: Elem>(&mut self, elements: &[Val<E>]) -> Val<E> {
        let elems = elements.iter().map(|e| e.id).collect();
        Val::wrap(self.ir.intern(Node::VecBuild { elements: elems, dtype: E::dtype() }))
    }

    /// Load an `E` value from an LDS buffer at flat `offset` **ordered after** the
    /// staging barrier (and any other `deps`): the cross-lane read of a staged tile.
    /// The `After` edge on the buffer makes the store→barrier→load order explicit —
    /// omitting it is the silent-miscompile class (§2.1). Prefer this over any bare
    /// LDS load (a lane may read another lane's write only past the barrier).
    pub fn load_lds_after<E: Elem>(&mut self, lds: Lds<E>, offset: Idx, deps: &[Edge]) -> Val<E> {
        let after = self.after_buf(lds.id, deps);
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: after, offset: offset.0, dtype: E::dtype() }))
    }

    /// A workgroup barrier fencing `body` (a store) plus every write in `deps`: all
    /// must complete before any consumer routed [`Self::load_lds_after`] (or otherwise
    /// `After` the returned effect) proceeds. The `store → barrier → load` fence the
    /// LDS stage needs (mirrors tk's `store.barrier(deps)`).
    pub fn barrier(&mut self, body: Effect, deps: &[Edge]) -> Effect {
        let deps = deps.iter().map(|e| e.raw()).collect();
        Effect(self.ir.intern(Node::Barrier { body: body.0, deps }))
    }

    /// A **bare workgroup barrier** (`s.barrier()` + a baked `sched.barrier(0)` wall, DESIGN §5c) —
    /// [`Self::barrier`] *without* the acq-rel fence, so the seal is a pure ping-pong rendezvous and
    /// does NOT force an `lgkmcnt(0)` LDS drain. The LDS ordering the fence dropped MUST be re-supplied
    /// by an explicit [`Self::swait_lgkmcnt`] at the RAW/WAR/pre-MFMA points (caller's obligation — a
    /// missing drain is a silent stale read). `body` + `deps` are pure happens-after anchors.
    pub fn bare_barrier(&mut self, body: Effect, deps: &[Edge]) -> Effect {
        let deps = deps.iter().map(|e| e.raw()).collect();
        Effect(self.ir.intern(Node::BareBarrier { body: body.0, deps }))
    }

    /// A **machine-scheduler fence** (`sched.barrier(mask)`, DESIGN §5c) positioned right after
    /// `anchors` — the load-pin that keeps the register-staged prefetch in flight: `anchors` are
    /// the prefetch load values, so the fence sits just past them and the AMDGPU scheduler may
    /// not sink them below it. Route the fence's [`Effect`] into a downstream consumer's deps to
    /// keep it live and force the rest of the body after it. `mask = 0` = a total fence.
    pub fn sched_fence(&mut self, mask: i64, anchors: &[Edge]) -> Effect {
        let deps = anchors.iter().map(|e| e.raw()).collect();
        Effect(self.ir.intern(Node::SchedFence { mask, deps }))
    }

    /// A **declarative interleave directive** (`sched.group.barrier(mask, size, group)`, FA-redesign
    /// §2.3) positioned after `anchors` — forms a `size`-instruction group of `mask` ops in scheduling
    /// group `group`. Emits NO instruction; drives the MFMA:VALU/exp interleave. Route its [`Effect`]
    /// into a downstream consumer to keep it live + positioned. Prefer the [`Self::interleave_valu`] /
    /// [`Self::interleave_exp`] ratio helpers; this is the raw primitive.
    pub fn sched_group(&mut self, mask: i64, size: i64, group: i64, anchors: &[Edge]) -> Effect {
        let deps = anchors.iter().map(|e| e.raw()).collect();
        Effect(self.ir.intern(Node::SchedGroupBarrier { mask, size, group, deps }))
    }

    /// `sched.group.barrier` mask bits (AMDGPU `SchedGroupBarrier`): matrix ops, vector ALU, and
    /// transcendental (`v_exp`) — HipKittens' `MFMA_MASK`/`VALU_MASK`/`EXP_MASK`.
    pub const SG_MFMA: i64 = 0x08;
    pub const SG_VALU: i64 = 0x02;
    pub const SG_EXP: i64 = 0x400;

    /// The declarative interleave ratio (HipKittens' `sched_barrier_pairs<Pairs,Cnt,Group>`): repeat
    /// `pairs`×{ 1 MFMA, then `valu` VALU } in scheduling group `group`, so the softmax reduction VALU
    /// runs *inside* the matrix pipeline. Emits `2·pairs` hints (zero instructions), chained so each is
    /// live + ordered after the last; `anchors` anchor the first. Returns the final hint [`Effect`] to
    /// thread onward. `pairs = 0` is a no-op (returns `None`).
    pub fn interleave_valu(&mut self, pairs: u32, valu: u32, group: i64, anchors: &[Edge]) -> Option<Effect> {
        self.interleave(Self::SG_VALU, pairs, valu, group, anchors)
    }

    /// HipKittens' `sched_barrier_exp_pairs`: repeat `pairs`×{ 1 MFMA, then `exp` transcendental } — the
    /// softmax `exp2` folded under the P·V MFMA. Same shape as [`Self::interleave_valu`], EXP mask.
    pub fn interleave_exp(&mut self, pairs: u32, exp: u32, group: i64, anchors: &[Edge]) -> Option<Effect> {
        self.interleave(Self::SG_EXP, pairs, exp, group, anchors)
    }

    /// Shared `pairs`×{ 1 MFMA, then `n` `mask`-ops } emitter for the ratio helpers.
    fn interleave(&mut self, mask: i64, pairs: u32, n: u32, group: i64, anchors: &[Edge]) -> Option<Effect> {
        let mut last: Option<Effect> = None;
        for _ in 0..pairs {
            let a = last.map_or_else(|| anchors.to_vec(), |e| vec![e.dep()]);
            let mfma = self.sched_group(Self::SG_MFMA, 1, group, &a);
            last = Some(self.sched_group(mask, n as i64, group, &[mfma.dep()]));
        }
        last
    }

    /// A **wave issue-priority** control (`s_setprio level`, DESIGN §5c), positioned after `after`.
    /// Bracket an MFMA cluster `set_prio(1, [entry]) … set_prio(0, [mma results])` so the compute
    /// wave wins SIMD issue over the co-resident loading wave. Route its [`Effect`] into a
    /// downstream consumer to keep it live and ordered.
    pub fn set_prio(&mut self, level: i64, after: &[Edge]) -> Effect {
        let deps = after.iter().map(|e| e.raw()).collect();
        Effect(self.ir.intern(Node::SetPrio { level, deps }))
    }

    /// A **value re-bound to observe `deps`** (an ordering edge on a `Val`, §5c) — the passthrough
    /// [`Node::After`] returns a value equal to `v` but happens-after `deps`. The way a scheduling hint
    /// (`interleave_valu`/`interleave_exp`) is kept LIVE + positioned inside a loop body: route the hint
    /// effect into a carried accumulator value, so it rides the carry to the sink instead of being DCE'd.
    pub fn val_after<E: Elem>(&mut self, v: Val<E>, deps: &[Edge]) -> Val<E> {
        Val::wrap(self.after_buf(v.id, deps))
    }

    /// An index value re-bound to observe `deps` (an ordering edge on a *value*, §5c): the way a
    /// schedule-steering custom (`wave_barrier`/`set_prio`) is ordered after a barrier without
    /// taking the barrier as a `Op::Custom` dep (which the renderer can't name) — the warp_row
    /// operand carries the ordering, mirroring tk's `a_smem.after([barrier])` (`gfx942.rs:156`).
    pub fn idx_after(&mut self, idx: Idx, deps: &[Edge]) -> Idx {
        Idx(self.after_buf(idx.0, deps))
    }

    /// Rebind an index leaf to a lexical scope. The value is unchanged; its identity is not.
    pub fn scope_idx(&mut self, idx: Idx, scope: Scope) -> Idx {
        match scope.id {
            Some(id) => Idx(self.after_buf(idx.0, &[Edge::anchor(id)])),
            None => idx,
        }
    }

    /// Rebind an LDS handle to a lexical scope. Derived LDS addresses/loads then remain local to that
    /// region through tile-IR and UOp hash-consing.
    pub fn scope_lds<E: Elem>(&mut self, lds: Lds<E>, scope: Scope) -> Lds<E> {
        match scope.id {
            Some(scope) => {
                let id = self.after_buf(lds.id, &[Edge::anchor(scope)]);
                Lds { id, len: lds.len, _e: PhantomData }
            }
            None => lds,
        }
    }

    /// The **HK barrier-wall opt-in** (DESIGN §5c): a void sentinel making codegen pair every
    /// `s_barrier` in this kernel with a positional `sched.barrier(0)`. Emit once; fold its
    /// [`Effect`] into the loop `End` so it stays live (it has no natural consumer).
    pub fn wall_marker(&mut self) -> Effect {
        Effect(self.ir.intern(Node::SchedWallMarker))
    }

    /// The **wave-phase asymmetric barrier** (`if warp_row == eq: s_barrier`, DESIGN §5c/3c) —
    /// `warp_row` is operand[0], `after` are ordering anchors. Route its [`Effect`] into a
    /// downstream consumer to keep it live and ordered (an un-executed barrier deadlocks, so it
    /// must never be DCE'd). Place OUTSIDE the loop (prologue `eq=1` / epilogue `eq=0`) — the asm
    /// skip-label is uniquified per construction, not per clang-unrolled copy.
    pub fn wave_barrier(&mut self, warp_row: Idx, eq: i64, after: &[Edge]) -> Effect {
        let mut deps: crate::ir::Edges = smallvec::smallvec![warp_row.0];
        deps.extend(after.iter().map(|e| e.raw()));
        Effect(self.ir.intern(Node::WaveBarrier { eq, deps }))
    }

    // ── elementwise binary ops (dtype-matched by construction) ───────────────

    pub fn add<E: Elem>(&mut self, a: Val<E>, b: Val<E>) -> Val<E> {
        self.binary(BinOp::Add, a, b)
    }
    pub fn sub<E: Elem>(&mut self, a: Val<E>, b: Val<E>) -> Val<E> {
        self.binary(BinOp::Sub, a, b)
    }
    pub fn mul<E: Elem>(&mut self, a: Val<E>, b: Val<E>) -> Val<E> {
        self.binary(BinOp::Mul, a, b)
    }
    pub fn max<E: Elem>(&mut self, a: Val<E>, b: Val<E>) -> Val<E> {
        self.binary(BinOp::Max, a, b)
    }

    fn binary<E: Elem>(&mut self, op: BinOp, a: Val<E>, b: Val<E>) -> Val<E> {
        Val::wrap(self.ir.intern(Node::EltwiseBinary { op, a: a.id, b: b.id }))
    }

    /// **Predicated select on an index comparison**: `lo < hi ? then : els`, per element. The
    /// ragged-tail mask primitive — FA folds an additive `-inf` where `global_kv ≥ n` so the masked
    /// keys `exp→0`. `lo`/`hi` are index-typed; `then`/`els` are the same-dtype value branches.
    pub fn select_lt<E: Elem>(&mut self, lo: Idx, hi: Idx, then: Val<E>, els: Val<E>) -> Val<E> {
        Val::wrap(self.ir.intern(Node::SelectLt { lo: lo.0, hi: hi.0, then: then.id, els: els.id }))
    }

    // ── elementwise unary math (the FA-forward additions — GEMM needed none) ──
    // The whole transcendental table already exists below the boundary (svod-ir's
    // `Op::Unary`); tk2 simply never surfaced it because matmul is transcendental-free.
    // FA's softmax is the first consumer: `exp2` for the online exp, `recip` for `1/norm`.

    /// Base-2 exponential (`2^x`) — the hardware `v_exp_f32` on gfx942. The softmax core.
    /// Lowers to `Op::Unary(UnaryOp::Exp2, x)` (see [`crate::lower`]).
    pub fn exp2(&mut self, x: Val<F32>) -> Val<F32> {
        Val::wrap(self.ir.intern(Node::Unary { op: crate::ir::UnOp::Exp2, x: x.id }))
    }

    /// Reciprocal (`1/x`) — the FA normalize `O / norm`. Lowers to `Op::Unary(UnaryOp::Reciprocal, x)`.
    pub fn recip(&mut self, x: Val<F32>) -> Val<F32> {
        Val::wrap(self.ir.intern(Node::Unary { op: crate::ir::UnOp::Recip, x: x.id }))
    }

    /// **Cross-lane lane-gather** (`llvm.amdgcn.ds.bpermute`, gfx942 — the ONLY way to
    /// exchange data between SIMD lanes without an LDS round-trip; svod-ir has NO
    /// cross-lane / DPP / shuffle op, so this is a hand-rolled inline-LLVM `Op::Custom`,
    /// mirroring tk1's `Group::shuffle_lane`). Returns `data` as computed by lane
    /// `src_lane`. Barrier-FREE (uses the LDS permute hardware, not a workgroup barrier),
    /// so a fragment reduction built on it keeps a [`crate::pipeline`] compute body
    /// edge-free. f32 is transported bitcast through i32 (bpermute is i32-only).
    pub fn shuffle_lane(&mut self, data: Val<F32>, src_lane: Idx) -> Val<F32> {
        let four = self.idx_const(4); // dword byte-stride: bpermute addr = lane·4
        let addr = self.idx_mul(src_lane, four);
        Val::wrap(self.ir.intern(Node::DsBpermute { addr: addr.0, data: data.id }))
    }

    /// **Cross-lane column reduction** of a Col-map fragment `val` (rows = the SPREAD/contraction
    /// axis, cols = the FLAT axis): fold every one of the `map.rows` spread-rows per flat-column,
    /// combine with the running `init`, and broadcast the per-column result to every `ept` slot (so a
    /// caller can subtract it row-wise). Barrier-FREE (`ds_bpermute`), so a softmax reduction built on
    /// it keeps a [`crate::pipeline`] compute body edge-free — the FA softmax row-max / row-sum.
    ///
    /// `add = false` ⇒ `max` (the online-softmax running max); `add = true` ⇒ `sum` (the norm). The
    /// fold is `(a) the `ept` slots (4 consecutive spread-rows in one lane-group) then (b) the lane
    /// tree {L+cols, L+2·cols, L+3·cols} mod WARP (the other lane-groups holding the same flat column).
    /// Promoted from the naive FA's hand-rolled helper (tk1's first-class `Group::col_reduce`).
    pub fn frag_col_reduce(&mut self, val: Val<F32>, lane: Idx, init: Val<F32>, add: bool) -> Val<F32> {
        const WARP: i64 = 64; // gfx942 wave64
        const EPT: usize = 4; // gfx942 16×16 fragment — a const, NOT meta-derived: `val` typically comes
        // through an `EltwiseBinary` (`mul`/`sub`) whose meta shape is bookkeeping-scalar, which would
        // collapse a meta-derived width to 1 and fold only one of the four spread-rows (a silent bug).
        let comb = |b: &mut Self, a: Val<F32>, c: Val<F32>| if add { b.add(a, c) } else { b.max(a, c) };
        let ept = EPT;
        let cols = 16i64; // Col-map flat-axis width (lanes per column-group)
        // (a) in-register fold of this lane's `ept` spread-rows.
        let mut partial = self.vec_extract(val, 0);
        for e in 1..ept {
            let x = self.vec_extract(val, e);
            partial = comb(self, partial, x);
        }
        // (b) the wave64 lane tree {L, L+cols, L+2·cols, L+3·cols} — every lane in a column ends equal.
        let mut acc = partial;
        let g = self.idx_const(WARP);
        for d in [cols, 2 * cols, 3 * cols] {
            let dc = self.idx_const(d);
            let sl = self.idx_add(lane, dc);
            let sl = self.idx_mod(sl, g);
            let sh = self.shuffle_lane(partial, sl);
            acc = comb(self, acc, sh);
        }
        // (c) fold the running accumulator, then broadcast to every `ept` slot.
        let init0 = self.vec_extract(init, 0);
        acc = comb(self, acc, init0);
        let copies: Vec<Val<F32>> = (0..ept).map(|_| acc).collect();
        self.vec_build(&copies)
    }

    /// **The 32×32×8 accumulator row-reduction** (the FA-32 softmax over kv — the `EPT_C = 16` analog of
    /// [`Self::frag_col_reduce`], per §Step 6). The QKᵀ accumulator holds `S[kv, q]` with **kv on the M
    /// (row) axis**: for a fixed q (`= lane % 32`), the 32 kv split into this lane's **16 in-register**
    /// [`AccDist`] elements PLUS the `lane ± 32` partner's other 16 (the `lane / 32` half — the reduce must
    /// use the AccDist geometry, NOT `ept = 4`). This folds (a) the 16 in-register elements, then (b) the
    /// single `L ↔ L+32` cross-lane partner (barrier-free `ds_bpermute`), folds the running `init`, and
    /// broadcasts the per-q result to all 16 slots so the caller subtracts it row-wise. `add = false` ⇒ max
    /// (online-softmax running max); `add = true` ⇒ sum (the norm).
    pub fn acc_row_reduce_32(&mut self, val: Val<F32>, lane: Idx, init: Val<F32>, add: bool) -> Val<F32> {
        const WARP: i64 = 64; // gfx942 wave64
        const EPT_C: usize = 16; // the 32×32×8 accumulator width — a const (the meta-shape gotcha, as above)
        const HALF: i64 = 32; // lane % 32 = the flat (q) axis; lane / 32 splits the M(kv) reduce in two
        let comb = |b: &mut Self, a: Val<F32>, c: Val<F32>| if add { b.add(a, c) } else { b.max(a, c) };
        // (a) in-register fold of this lane's 16 kv-elements (16 of the 32 kv for this q).
        let mut partial = self.vec_extract(val, 0);
        for e in 1..EPT_C {
            let x = self.vec_extract(val, e);
            partial = comb(self, partial, x);
        }
        // (b) the `L ↔ L+32` partner — the OTHER 16 kv of the same q (the `lane / 32` half).
        let g = self.idx_const(WARP);
        let dc = self.idx_const(HALF);
        let sl = self.idx_add(lane, dc);
        let sl = self.idx_mod(sl, g);
        let sh = self.shuffle_lane(partial, sl);
        let mut acc = comb(self, partial, sh);
        // (c) fold the running init, broadcast to all 16 slots (the caller subtracts it row-wise).
        let init0 = self.vec_extract(init, 0);
        acc = comb(self, acc, init0);
        let copies: Vec<Val<F32>> = (0..EPT_C).map(|_| acc).collect();
        self.vec_build(&copies)
    }

    // ── register fragments + MMA (the naive matmul vocabulary) ───────────────

    /// Allocate a per-lane register fragment carrying its [`FragMap`] MFMA lane-map.
    pub fn define_frag<E: Elem>(&mut self, map: FragMap) -> Frag<E> {
        let id = self.ir.fresh_reg_id();
        let id = self.ir.intern(Node::DefineFrag { id, dtype: E::dtype(), frag: map });
        Frag { id, map, _e: PhantomData }
    }

    /// Store an `E` value into a fragment cell at flat `offset` (a gather-scatter
    /// element write / the accumulator init).
    pub fn store_frag_elem<E: Elem>(&mut self, f: Frag<E>, offset: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreGlobal { buf: f.id, offset: offset.0, value: value.id }))
    }

    /// Load an `E` value from a fragment cell at flat `offset` (the post-loop scatter
    /// read; `f` is typically [`Self::frag_after`]-wrapped to observe the loop `End`).
    pub fn load_frag_elem<E: Elem>(&mut self, f: Frag<E>, offset: Idx) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: f.id, offset: offset.0, dtype: E::dtype() }))
    }

    /// Vector-read the whole `ept`-element fragment run as a WMMA operand.
    pub fn load_frag_vec<E: Elem>(&mut self, f: Frag<E>) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadRegVec { buf: f.id, ept: f.map.ept, dtype: E::dtype() }))
    }

    /// Vector-read the fragment run through completion of `deps` — the loop-carried
    /// accumulator read (`acc.after([init, range])`) / the post-gather operand read
    /// (`frag.after([gather stores])`).
    pub fn load_frag_vec_after<E: Elem>(&mut self, f: Frag<E>, deps: &[Edge]) -> Val<E> {
        let after = self.after_buf(f.id, deps);
        Val::wrap(self.ir.intern(Node::LoadRegVec { buf: after, ept: f.map.ept, dtype: E::dtype() }))
    }

    /// Vector-store a whole `ept`-element fragment run — the WMMA accumulator write-back
    /// (f32) or the vectorised gather's write into a bf16 operand fragment.
    pub fn store_frag_vec<E: Elem>(&mut self, f: Frag<E>, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreRegVec { buf: f.id, value: value.id }))
    }

    /// Zero-init an accumulator fragment with ONE **constant-index** `<ept×f32>` vector store
    /// (`store_frag_vec` of a `vec_build` of `ept` zeros) — the SROA-promotable init.
    ///
    /// The prior init (`store_frag_elem` inside a `range(ept)`) wrote the accumulator at a **runtime**
    /// index, and LLVM's SROA/mem2reg cannot promote an `alloca` accessed at a dynamic index — so the
    /// whole accumulator stayed in memory and its per-iteration load→mfma→store round-trip never became
    /// the loop-carried `phi` HipKittens relies on, which let LLVM fracture the 32-MFMA clusters
    /// (mfmautil 0.39 vs HK 0.65; proven by an `opt -sroa` de-risk on the dumped IR). A single vector
    /// store at offset 0 keeps every access constant-index + uniform-width ⇒ the fragment promotes to a
    /// `phi <ept×f32>` and the MFMA runs stay clustered. Returns the store `Effect` (the loop-carry seed
    /// threaded into the first `load_frag_vec_after`), replacing the old init `Range`/`End` entirely.
    pub fn zero_init_frag(&mut self, f: Frag<F32>) -> Effect {
        let zeros: Vec<Val<F32>> = (0..f.map.ept).map(|_| self.f32(0.0)).collect();
        let zvec = self.vec_build(&zeros);
        self.store_frag_vec(f, zvec)
    }

    /// Init an accumulator fragment to a **constant** with ONE constant-index `<ept×f32>` vector store
    /// (the [`Self::zero_init_frag`] generalisation) — the online-softmax `max = −∞` seed. Same
    /// SROA-promotable single-vector-store shape, so the fragment stays a loop-carried `phi`.
    pub fn const_init_frag(&mut self, f: Frag<F32>, v: f32) -> Effect {
        let c = self.f32(v);
        let cs: Vec<Val<F32>> = (0..f.map.ept).map(|_| c).collect();
        let cvec = self.vec_build(&cs);
        self.store_frag_vec(f, cvec)
    }

    /// ONE `<ept×E>` vector load of a contiguous LDS run at flat `base`, ordered after
    /// `deps` (the fill barrier) — the vectorised fragment gather ([`Node::LoadVecAt`] →
    /// `ds_read_b64`). Replaces `ept` scalar `load_lds_after` for a contiguous run.
    pub fn load_lds_vec_after<E: Elem>(&mut self, lds: Lds<E>, base: Idx, ept: usize, deps: &[Edge]) -> Val<E> {
        let after = self.after_buf(lds.id, deps);
        Val::wrap(self.ir.intern(Node::LoadVecAt { buf: after, base: base.0, ept, dtype: E::dtype() }))
    }

    /// The addr(3) **base pointer** of an LDS tile at flat element `base`, ordered after `raw`
    /// (the RAW barrier — the ordering rides the buffer handle, exactly as [`Self::load_lds_after`]):
    /// ONE `addrspacecast(index_off(lds, base))` VGPR the slice's [`Self::ds_read_b64`] gathers all
    /// read from (DESIGN §5c — HK's operand gather is ONE base + a per-fragment `offset:` immediate,
    /// so the `lane_rc` address is materialised once, not per fragment; the VGPR-spill-cliff cure).
    pub fn lds_ptr_as3<E: Elem>(&mut self, lds: Lds<E>, base: Idx, raw: &[Edge]) -> Idx {
        let buf = if raw.is_empty() { lds.id } else { self.after_buf(lds.id, raw) };
        Idx(self.ir.intern(Node::LdsPtrAs3 { buf, base: base.0 }))
    }

    /// ONE inline-asm `ds_read_b64 $d, $base offset:N` LDS gather (gfx942 §5c — HK's only asm):
    /// reads the `ept`-element bf16 run at `base_ptr + off_bytes` into a fresh `<ept×E>` value.
    /// `off_bytes` is a compile-time immediate (≤ 65535); `prev` is the prior fragment's store
    /// (an ordering-only operand chaining the `sideeffect` reads in program order so they can't
    /// hoist across the barriers — the silent-stale-read class). Store the value into the operand
    /// fragment with [`Self::store_frag_vec`]; the WMMA reads it via [`Self::load_frag_vec_after`].
    pub fn ds_read_b64<E: Elem>(&mut self, base_ptr: Idx, off_bytes: i64, ept: usize, prev: Option<TileId>) -> Val<E> {
        assert!((0..=65535).contains(&off_bytes), "ds_read_b64 offset {off_bytes}B exceeds the 16-bit immediate");
        assert_eq!(off_bytes % 8, 0, "ds_read_b64 offset must be b64-aligned");
        assert_eq!(ept * E::dtype().bytes(), 8, "ds_read_b64 payload must be exactly 64 bits");
        assert!(matches!(self.ir.node(base_ptr.0), Node::LdsPtrAs3 { .. }), "ds_read_b64 requires an LDS pointer");
        Val::wrap(self.ir.intern(Node::DsReadB64 {
            base_ptr: base_ptr.0,
            off_bytes,
            ept,
            dtype: E::dtype(),
            prev,
            hk_form: false,
        }))
    }

    /// HipKittens' **literal** `ds_read_b64` (the HK port, GAP/Tier-B): same gather as
    /// [`Self::ds_read_b64`] but rendered in HK's exact IR form — an `i32` raw-address operand
    /// (`base_i32`, from [`Self::ptr_to_i32`]), the offset as an `i` immediate operand, a `~{memory}`
    /// clobber, and an `i64` result bitcast to `<ept×E>` (matches `hk-micro_tk.ll`).
    pub fn ds_read_b64_hk<E: Elem>(
        &mut self,
        base_i32: Idx,
        off_bytes: i64,
        ept: usize,
        prev: Option<TileId>,
    ) -> Val<E> {
        assert!((0..=65535).contains(&off_bytes), "ds_read_b64 offset {off_bytes}B exceeds the 16-bit immediate");
        assert_eq!(off_bytes % 8, 0, "ds_read_b64 offset must be b64-aligned");
        assert_eq!(ept * E::dtype().bytes(), 8, "ds_read_b64 payload must be exactly 64 bits");
        assert!(matches!(self.ir.node(base_i32.0), Node::PtrToI32 { .. }), "HK ds_read_b64 requires a raw LDS address");
        Val::wrap(self.ir.intern(Node::DsReadB64 {
            base_ptr: base_i32.0,
            off_bytes,
            ept,
            dtype: E::dtype(),
            prev,
            hk_form: true,
        }))
    }

    /// **`ptrtoint ptr addrspace(3) → i32`** of an LDS base ([`Self::lds_ptr_as3`]) — the raw i32
    /// LDS address HK's `ds_read_b64`/`ds_write_b64` asm takes (HK port).
    pub fn ptr_to_i32(&mut self, ptr: Idx) -> Idx {
        assert!(matches!(self.ir.node(ptr.0), Node::LdsPtrAs3 { .. }), "ptr_to_i32 requires an LDS pointer");
        Idx(self.ir.intern(Node::PtrToI32 { ptr: ptr.0 }))
    }

    /// ONE inline-asm `ds_write_b64 $base, $val offset:N` LDS store (gfx942 §5c — HK's **commit**: the
    /// waitcnt-opaque write twin of [`Self::ds_read_b64`]): stores the `ept`-element `value` to
    /// `base_ptr + off_bytes`. Being `asm sideeffect` the `s_barrier` does NOT auto-drain it — pair it
    /// with an EXPOSED [`Self::swait_lgkmcnt`] to re-establish store→barrier→load order. `off_bytes` is
    /// a compile-time immediate (≤ 65535); `prev` chains the writes in program order (the prior write).
    pub fn ds_write_b64<E: Elem>(
        &mut self,
        base_ptr: Idx,
        off_bytes: i64,
        value: Val<E>,
        prev: Option<TileId>,
    ) -> Effect {
        assert!((0..=65535).contains(&off_bytes), "ds_write_b64 offset {off_bytes}B exceeds the 16-bit immediate");
        // `ept` = the operand vector width, read off the value's derived shape (`<ept×E>`).
        let ept = self.ir.meta(value.id).shape.iter().copied().product::<usize>().max(1);
        assert_eq!(off_bytes % 8, 0, "ds_write_b64 offset must be b64-aligned");
        assert_eq!(ept * E::dtype().bytes(), 8, "ds_write_b64 payload must be exactly 64 bits");
        assert!(matches!(self.ir.node(base_ptr.0), Node::LdsPtrAs3 { .. }), "ds_write_b64 requires an LDS pointer");
        Effect(self.ir.intern(Node::DsWriteB64 {
            base_ptr: base_ptr.0,
            off_bytes,
            value: value.id,
            ept,
            prev,
            hk_form: false,
        }))
    }

    /// HipKittens-form `ds_write_b64` commit: an `i32` raw-address operand, an `i64` value, and a
    /// `~{memory}` clobber. `off_bytes` uses the DS instruction's 16-bit immediate.
    pub fn ds_write_b64_hk<E: Elem>(
        &mut self,
        base_i32: Idx,
        off_bytes: i64,
        value: Val<E>,
        prev: Option<TileId>,
    ) -> Effect {
        assert!((0..=65535).contains(&off_bytes), "ds_write_b64 offset {off_bytes}B exceeds the 16-bit immediate");
        let ept = self.ir.meta(value.id).shape.iter().copied().product::<usize>().max(1);
        assert_eq!(off_bytes % 8, 0, "ds_write_b64 offset must be b64-aligned");
        assert_eq!(ept * E::dtype().bytes(), 8, "ds_write_b64 payload must be exactly 64 bits");
        assert!(
            matches!(self.ir.node(base_i32.0), Node::PtrToI32 { .. }),
            "HK ds_write_b64 requires a raw LDS address"
        );
        Effect(self.ir.intern(Node::DsWriteB64 {
            base_ptr: base_i32.0,
            off_bytes,
            value: value.id,
            ept,
            prev,
            hk_form: true,
        }))
    }

    /// Waitcnt-opaque scalar bf16 LDS store. `base_ptr + off_bytes` names the destination and `prev`
    /// chains writes (or the VMEM readiness wait). A later explicit `lgkmcnt(0)` is required.
    pub fn ds_write_b16(&mut self, base_ptr: Idx, off_bytes: i64, value: Val<BF16>, prev: Option<TileId>) -> Effect {
        assert!((0..=65535).contains(&off_bytes), "ds_write_b16 offset {off_bytes}B exceeds the 16-bit immediate");
        let ept = self.ir.meta(value.id).shape.iter().copied().product::<usize>().max(1);
        assert_eq!(off_bytes % 2, 0, "ds_write_b16 offset must be scalar-bf16 aligned");
        assert_eq!(ept, 1, "ds_write_b16 requires a scalar bf16 payload");
        assert!(matches!(self.ir.node(base_ptr.0), Node::LdsPtrAs3 { .. }), "ds_write_b16 requires an LDS pointer");
        Effect(self.ir.intern(Node::DsWriteB16 { base_ptr: base_ptr.0, off_bytes, value: value.id, prev }))
    }

    /// Queue-wide VMEM wait allowing `allowed_outstanding` younger operations to remain. The anchor
    /// positions the wait after a complete effect/batch; it does not identify one transfer's payload.
    pub fn swait_vmcnt_allowed(&mut self, anchor: impl Into<VmemWaitAnchor>, allowed_outstanding: u8) -> Effect {
        assert!(allowed_outstanding <= 63, "gfx942 vmcnt threshold must fit the 6-bit counter");
        Effect(self.ir.intern(Node::SWaitVmcnt { anchor: anchor.into().0, allowed_outstanding }))
    }

    /// Full VMEM drain (`s_waitcnt vmcnt(0)`).
    pub fn swait_vmcnt(&mut self, anchor: impl Into<VmemWaitAnchor>) -> Effect {
        self.swait_vmcnt_allowed(anchor, 0)
    }

    /// The **legacy `<4 x i32>` SRD** (HK's `make_srsrc`, config `0x110000`) of a global `buf` based
    /// at element `base_off` (HK port, GAP-1). Feeds [`Self::buffer_load_i128`]. `num_bytes`
    /// = the whole-buffer byte extent (the SRD range/bound).
    pub fn make_srsrc<E: Elem>(&mut self, buf: Buf<E>, base_off: Idx) -> Idx {
        let num_bytes = (buf.len * E::dtype().bytes()) as i64;
        Idx(self.ir.intern(Node::MakeSrsrc { buf: buf.id, base_off: base_off.0, num_bytes }))
    }

    /// ONE **`raw.buffer.load.i128`** MUBUF load over a legacy [`Self::make_srsrc`] SRD (HK's
    /// `load_global_to_register_buffer`, HK port, GAP-1): reads a 128-bit chunk (`ept`
    /// `E`-elements) at `rsrc[voffset]` bytes (`soffset = 0`). `order` pins the load into its
    /// authoring cluster (ordering-only). `ept · sizeof(E)` must be 128 bits.
    pub fn buffer_load_i128<E: Elem>(&mut self, rsrc: Idx, voffset: Idx, ept: usize, order: &[Edge]) -> Val<E> {
        assert_eq!(ept * E::dtype().bytes(), 16, "raw.buffer.load.i128 chunk must be 128 bits (16 bytes)");
        let order = order.iter().map(|e| e.raw()).collect();
        Val::wrap(self.ir.intern(Node::BufferLoadI128 {
            rsrc: rsrc.0,
            voffset: voffset.0,
            ept,
            dtype: E::dtype(),
            order,
        }))
    }

    /// **fp32 → bf16 truncation** (`(uint16_t)(bits(f) >> 16)`) — HK's `convertor<bf16,float>`, the
    /// truncating (not RNE) C store (HK port). Store the result to a bf16 global.
    pub fn bf16_trunc(&mut self, val: Val<F32>) -> Val<BF16> {
        Val::wrap(self.ir.intern(Node::Bf16Trunc { val: val.id }))
    }

    /// **Vector fp32 → bf16 truncation** of a gfx942 16×16 fragment vector (`ept = 4`): per-element
    /// [`Self::bf16_trunc`] then re-pack — the f32→bf16 relayout FA needs between the softmax weights
    /// `P` (f32) and the PV MMA operand. `ept` is the fixed fragment width, NOT meta-derived: `v`
    /// typically arrives through an `EltwiseBinary`/`Unary` (`exp2`) whose meta shape is bookkeeping-
    /// scalar, which would collapse a meta-derived width to 1 and cast only the first element.
    pub fn cast_vec_bf16(&mut self, v: Val<F32>) -> Val<BF16> {
        const EPT: usize = 4;
        let els: Vec<Val<BF16>> = (0..EPT)
            .map(|e| {
                let s = self.vec_extract(v, e);
                self.bf16_trunc(s)
            })
            .collect();
        self.vec_build(&els)
    }

    /// **Intra-lane byte permute** (`v_perm_b32 D, hi, lo, sel` / `llvm.amdgcn.perm`, gfx942 — the
    /// register-level 2×2 bf16 transpose aiter's 32×32×8 Flash-Attention uses). Over the 8-byte pool
    /// `{lo.bytes @ 0-3, hi.bytes @ 4-7}`, output byte `i = pool[sel.byte[i]]`; the result is a
    /// `<2×bf16>` dword. With `sel = S49` over two f32 operands this yields their **truncated bf16 pair**
    /// (bf16 = an f32's top 16 bits), fusing the f32→bf16 cast with the pack. Barrier-free (a pure ALU
    /// shuffle), mirroring [`Self::shuffle_lane`]'s `Op::Custom` shape.
    pub fn v_perm_b32(&mut self, hi: Val<F32>, lo: Val<F32>, selector: i64) -> Val<BF16> {
        Val::wrap(self.ir.intern(Node::VPerm { hi: hi.id, lo: lo.id, selector }))
    }

    /// Byte-permute two explicitly readied packed-bf16 dwords without exposing VMEM consumption to LLVM.
    pub fn v_perm_bf16x2_asm(&mut self, hi: Val<BF16>, lo: Val<BF16>, selector: i64) -> Val<BF16> {
        for value in [hi, lo] {
            let ept = self.ir.meta(value.id).shape.iter().copied().product::<usize>().max(1);
            assert_eq!(ept, 2, "v_perm_bf16x2_asm requires packed bf16 dwords");
        }
        Val::wrap(self.ir.intern(Node::VPermAsm { hi: hi.id, lo: lo.id, selector }))
    }

    /// Tie a packed b64 bf16 operand through side-effect asm after an opaque LDS readiness wait.
    pub fn opaque_ready_b64(&mut self, val: Val<BF16>, wait: TileId) -> Val<BF16> {
        let ept = self.ir.meta(val.id).shape.iter().copied().product::<usize>().max(1);
        assert_eq!(ept * BF16::dtype().bytes(), 8, "opaque_ready_b64 requires exactly four bf16 values");
        assert!(matches!(self.ir.node(wait), Node::SWaitLgkmcnt { .. }), "opaque_ready_b64 requires an lgkm wait");
        Val::wrap(self.ir.intern(Node::OpaqueReadyB64 { val: val.id, wait }))
    }

    /// aiter's `s49` selector — gather the HIGH bf16 of each dword pair (bytes {2,3,6,7}), i.e. the
    /// truncated-bf16 pair of two f32 operands. `s50` (LOW bf16, bytes {0,1,4,5}) is the V-deinterleave twin.
    pub const S49_HI_BF16: i64 = 0x07060302;
    /// aiter's `s50` selector — gather the LOW bf16 of each dword pair (bytes {0,1,4,5}); the V-transpose
    /// deinterleave half (see [`Self::v_perm_b32`]).
    pub const S50_LO_BF16: i64 = 0x05040100;

    /// **The P→PV relayout under 32×32×8** (aiter's `v_perm s49` pack): the 16-wide f32 QKᵀ accumulator
    /// (`Mfma32x32x8Bf16::EPT_C = 16`) truncated to bf16 and packed into the **4** PV B-operands (one per
    /// hardware `K = 8` slice, `EPT_B = 4` bf16 each). By the accumulator↔B-operand layout correspondence
    /// (proven in `pv_relayout_probe`), B-operand for slice `s` is exactly the truncated bf16 of accumulator
    /// elements `[4s, 4s+1, 4s+2, 4s+3]` — the intra-lane pack a single [`Self::v_perm_b32`] pair does per
    /// slice. This is the 32×32×8 replacement for the 16×16×16 zero-cost cast (which no longer holds).
    pub fn pv_relayout_s49(&mut self, acc: Val<F32>) -> Vec<Val<BF16>> {
        (0..4)
            .map(|s| {
                let e0 = self.vec_extract(acc, 4 * s);
                let e1 = self.vec_extract(acc, 4 * s + 1);
                let e2 = self.vec_extract(acc, 4 * s + 2);
                let e3 = self.vec_extract(acc, 4 * s + 3);
                // v_perm(hi, lo, s49) = <trunc(lo), trunc(hi)>: dword0 = <e0,e1>, dword1 = <e2,e3>.
                let d0 = self.v_perm_b32(e1, e0, Self::S49_HI_BF16);
                let d1 = self.v_perm_b32(e3, e2, Self::S49_HI_BF16);
                let b0 = self.vec_extract(d0, 0);
                let b1 = self.vec_extract(d0, 1);
                let b2 = self.vec_extract(d1, 0);
                let b3 = self.vec_extract(d1, 1);
                self.vec_build(&[b0, b1, b2, b3]) // <4×bf16> = bf16(acc[4s..4s+4])
            })
            .collect()
    }

    /// The **manual LDS drain** (`s_waitcnt lgkmcnt(0)`, §5c): a void `asm sideeffect` ordered after
    /// `prev` (the last commit write) that stalls until every outstanding LDS op completes — the
    /// exposed drain the asm [`Self::ds_write_b64`] commit needs (its writes are waitcnt-opaque).
    pub fn swait_lgkmcnt(&mut self, prev: Edge) -> Effect {
        Effect(self.ir.intern(Node::SWaitLgkmcnt { prev: prev.raw() }))
    }

    /// One K-fragment MFMA `D = A·B + C` (gfx942 16×16×16 bf16→f32) via the **intrinsic**.
    /// `a`/`b` are the bf16 fragment operands, `c` the f32 accumulator; returns the f32 result.
    pub fn mma(&mut self, a: Val<BF16>, b: Val<BF16>, c: Val<F32>, ept: usize) -> Val<F32> {
        Val::wrap(self.ir.intern(Node::Mma { a: a.id, b: b.id, c: c.id, ept, asm: false }))
    }

    /// The **shape-matched** intrinsic MFMA: `Tile<M,K> · Tile<K,N> → Tile<M,N>` — the
    /// *shared* `const K` type-checks operand composition, and the accumulator width comes from
    /// `S::EPT_C` as data (`4 → 16×16×16`, `16 → 32×32×8`). Erases via [`Self::mma`], so the interned
    /// `Node::Mma` is byte-identical (no `Node` field added). The `debug_assert` bridges the type-level
    /// `M/N/K` to the shape marker's data consts — the one seam where the type and the data must agree.
    ///
    /// A mismatched inner dim is a **compile error** (the shared `K` fails to unify):
    /// ```compile_fail
    /// use svod_tk2::build::{Builder, BF16, F32};
    /// use svod_tk2::shape::{Mfma16x16x16Bf16 as S, MfmaShape};
    /// let mut b = Builder::new("shape_fail");
    /// let fa = b.define_frag::<BF16>(S::a_map());
    /// let fb = b.define_frag::<BF16>(S::b_map());
    /// let fc = b.define_frag::<F32>(S::c_map());
    /// let (a, bb, c) = (b.load_frag_vec(fa), b.load_frag_vec(fb), b.load_frag_vec(fc));
    /// // A is 16×8, B is 16×16 → the inner dims (8 vs 16) disagree → no `mma_of` for these tiles.
    /// let _ = b.mma_of::<S, 16, 16, 16>(a.tile::<16, 8>(), bb.tile::<16, 16>(), c.tile::<16, 16>());
    /// ```
    pub fn mma_of<S: crate::shape::MfmaShape, const M: usize, const K: usize, const N: usize>(
        &mut self,
        a: Tile<BF16, M, K>,
        b: Tile<BF16, K, N>,
        c: Tile<F32, M, N>,
    ) -> Tile<F32, M, N> {
        debug_assert_eq!((M, N, K), (S::M, S::N, S::K), "Tile dims disagree with MfmaShape S");
        self.mma(a.erase(), b.erase(), c.erase(), S::EPT_C).tile::<M, N>()
    }

    /// The **asm** MFMA (`v_mfma_f32_16x16x16_bf16` as inline `asm sideeffect`, §5c): schedule-opaque
    /// so the cluster order survives `-O3` without the `sched.barrier(0)` walls that spill. Numerically
    /// identical to [`Self::mma`]. gfx942-only (the renderer falls back to the intrinsic on RDNA).
    pub fn mma_asm(&mut self, a: Val<BF16>, b: Val<BF16>, c: Val<F32>, ept: usize) -> Val<F32> {
        Val::wrap(self.ir.intern(Node::Mma { a: a.id, b: b.id, c: c.id, ept, asm: true }))
    }

    /// The **shape-matched** asm MFMA — the [`Self::mma_of`] twin over [`Self::mma_asm`] (the §5c
    /// schedule-opaque channel). Erases identically, so the clustered matmul's `Node::Mma` is unchanged.
    pub fn mma_asm_of<S: crate::shape::MfmaShape, const M: usize, const K: usize, const N: usize>(
        &mut self,
        a: Tile<BF16, M, K>,
        b: Tile<BF16, K, N>,
        c: Tile<F32, M, N>,
    ) -> Tile<F32, M, N> {
        debug_assert_eq!((M, N, K), (S::M, S::N, S::K), "Tile dims disagree with MfmaShape S");
        self.mma_asm(a.erase(), b.erase(), c.erase(), S::EPT_C).tile::<M, N>()
    }

    /// A fragment handle re-bound to observe `deps` (the post-loop carried read:
    /// `acc.after([end])`), symmetric with [`Self::reg_after`].
    pub fn frag_after<E: Elem>(&mut self, f: Frag<E>, deps: &[Edge]) -> Frag<E> {
        let id = self.after_buf(f.id, deps);
        Frag { id, map: f.map, _e: PhantomData }
    }

    // ── ordering edges (first-class, §2.1) ───────────────────────────────────

    /// Route the buffer handle `buf` through completion of `deps` — the ownership
    /// happens-before token: a later read of the returned handle *cannot* be
    /// observed before `deps`. Used for the loop-carried accumulator read/reinit.
    /// `deps` are raw handles so an edge may be an [`Effect`] (a store) OR a
    /// [`Range`] (keeps the read inside the loop body).
    fn after_buf(&mut self, buf: TileId, deps: &[Edge]) -> TileId {
        let deps = deps.iter().map(|e| e.raw()).collect();
        self.ir.intern(Node::After { val: buf, deps })
    }

    /// Close `ranges` loop(s) around effect `e` — exactly one `End` per `Range`
    /// (the linearizer's one-RANGE-one-END obligation).
    pub fn end(&mut self, e: Effect, ranges: &[Range]) -> Effect {
        let ranges = ranges.iter().map(|r| r.id).collect();
        Effect(self.ir.intern(Node::End { body: e.0, ranges }))
    }

    /// Combine effects: route store `e` through completion of `deps` (other stores),
    /// yielding ONE effect ordered after all of them. This is how a loop carrying
    /// MULTIPLE accumulators closes its single `End`: a RANGE admits one END, so the
    /// other accumulators' stores are folded in here as a shared input (they stay
    /// in-loop — each already depends on the range — and survive DCE), and the MMAs are
    /// NOT serialized against each other (only this combine waits for all). The tk
    /// `endrange_to` obligation, expressed as an ordering edge instead of acc-read
    /// chaining. Every accumulator then reads its final value post-loop via `.after([end])`.
    pub fn combine(&mut self, e: Effect, deps: &[Edge]) -> Effect {
        Effect(self.after_buf(e.0, deps))
    }

    /// A register handle re-bound to observe `deps` (the post-loop carried read:
    /// `reg.after([end])`). Returns a fresh [`Reg`] over the ordering-wrapped buffer.
    pub fn reg_after<E: Elem>(&mut self, reg: Reg<E>, deps: &[Edge]) -> Reg<E> {
        let id = self.after_buf(reg.id, deps);
        Reg { id, len: reg.len, _e: PhantomData }
    }

    /// Finish: the kernel sink over terminal `roots`, carrying the kernel name.
    pub fn finish(mut self, roots: &[Effect]) -> (TileIr, TileId) {
        let roots = roots.iter().map(|e| e.0).collect::<smallvec::SmallVec<[TileId; 4]>>();
        let sink = self.ir.intern(Node::Sink { roots });
        (self.ir, sink)
    }
}
