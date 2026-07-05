//! The typed builder — a thin, ergonomic front-end that emits interned [`TileId`]
//! handles, never eager UOp (DESIGN.md §A, §0).
//!
//! Typing is deliberately *gentle* (DESIGN.md §OPEN-2, ratified): the element
//! dtype rides in the types via a **sealed** [`Elem`] trait (so `add(a, b)`
//! type-checks that both operands share a dtype, and "no impl ⇒ no method"), while
//! shape/register-range/schedule validity stay in data + the verifier — never in
//! surface types (HK's single largest unreadability source). We do NOT push
//! typestate: handles are plain `Copy` wrappers over `TileId`.

use std::marker::PhantomData;

use svod_dtype::DType;

use crate::ir::{BinOp, FragMap, IndexOp, Node, Scalar, ScopeAxis, TileId, TileIr};

mod sealed {
    pub trait Sealed {}
}

/// A legal tile element dtype (sealed — the DSL's dtype whitelist). Adding a dtype
/// is one `impl`; there is no way for a caller to invent an unsupported element.
pub trait Elem: sealed::Sealed + Copy {
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

impl Range {
    /// This loop's range as an ordering edge (keeps a routed read in the loop body).
    pub fn dep(self) -> TileId {
        self.id
    }
}

/// An effect handle (a store, an ended store, an after-wrapped buffer).
#[derive(Copy, Clone, Debug)]
pub struct Effect(pub TileId);

impl Effect {
    /// This effect as an ordering edge (a happens-before token).
    pub fn dep(self) -> TileId {
        self.0
    }
}

impl<E: Elem> Val<E> {
    fn wrap(id: TileId) -> Self {
        Val { id, _e: PhantomData }
    }
}

/// The staged builder. Owns the [`TileIr`] arena; every method interns and returns
/// a `Copy` handle.
pub struct Builder {
    pub ir: TileIr,
    pub name: String,
}

impl Builder {
    pub fn new(name: impl Into<String>) -> Self {
        Builder { ir: TileIr::new(), name: name.into() }
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

    /// The block (thread) index with `bound` threads.
    pub fn block_axis(&mut self, bound: i64) -> Idx {
        Idx(self.ir.intern(Node::Axis { axis: ScopeAxis::Block, bound }))
    }

    /// Open a loop over `trips` iterations; nesting is emergent from the resulting
    /// [`Range`]/`End` edges — never authored (DESIGN.md §D boundary).
    pub fn range(&mut self, trips: i64) -> Range {
        let rid = self.ir.fresh_range_id();
        Range { id: self.ir.intern(Node::Range { id: rid, trips }) }
    }

    /// The loop counter as an index value.
    pub fn counter(&self, r: Range) -> Idx {
        Idx(r.id)
    }

    /// An index-typed integer constant.
    pub fn idx_const(&mut self, v: i64) -> Idx {
        Idx(self.ir.intern(Node::Const { scalar: Scalar::Int(v), dtype: DType::Index }))
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

    // ── loads / stores (movement, lowered to INDEX + LOAD/STORE) ─────────────

    /// Load an `E`-element from a global buffer at flat `offset`.
    pub fn load<E: Elem>(&mut self, buf: Buf<E>, offset: Idx) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: buf.id, offset: offset.0, dtype: E::dtype() }))
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
    pub fn load_reg_after<E: Elem>(&mut self, reg: Reg<E>, offset: Idx, deps: &[TileId]) -> Val<E> {
        let after = self.after_buf(reg.id, deps);
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: after, offset: offset.0, dtype: E::dtype() }))
    }

    /// Store an `E` value into a global buffer at flat `offset` (a terminal effect).
    pub fn store<E: Elem>(&mut self, buf: Buf<E>, offset: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreGlobal { buf: buf.id, offset: offset.0, value: value.id }))
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
    pub fn store_lds_after<E: Elem>(&mut self, lds: Lds<E>, offset: Idx, value: Val<E>, deps: &[TileId]) -> Effect {
        let after = self.after_buf(lds.id, deps);
        Effect(self.ir.intern(Node::StoreGlobal { buf: after, offset: offset.0, value: value.id }))
    }

    /// ONE `<ept×E>` vector load of a contiguous **global** run at flat `base` (the
    /// coalesced, vectorised fill read → `buffer_load_dwordx*`) — the global mirror of
    /// [`Self::load_lds_vec_after`], no ordering edge (a plain source read).
    pub fn load_vec<E: Elem>(&mut self, buf: Buf<E>, base: Idx, ept: usize) -> Val<E> {
        Val::wrap(self.ir.intern(Node::LoadVecAt { buf: buf.id, base: base.0, ept, dtype: E::dtype() }))
    }

    /// ONE `<ept×E>` vector store of a contiguous LDS run at flat `base` ([`Node::StoreVecAt`]
    /// → `ds_write_b64`/`b128`) — the store mirror of [`Self::load_lds_vec_after`], replacing
    /// `ept` scalar `store_lds` for a contiguous, aligned run (the vectorised fill).
    pub fn store_lds_vec<E: Elem>(&mut self, lds: Lds<E>, base: Idx, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreVecAt { buf: lds.id, base: base.0, value: value.id }))
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
    pub fn load_lds_after<E: Elem>(&mut self, lds: Lds<E>, offset: Idx, deps: &[TileId]) -> Val<E> {
        let after = self.after_buf(lds.id, deps);
        Val::wrap(self.ir.intern(Node::LoadGlobal { buf: after, offset: offset.0, dtype: E::dtype() }))
    }

    /// A workgroup barrier fencing `body` (a store) plus every write in `deps`: all
    /// must complete before any consumer routed [`Self::load_lds_after`] (or otherwise
    /// `After` the returned effect) proceeds. The `store → barrier → load` fence the
    /// LDS stage needs (mirrors tk's `store.barrier(deps)`).
    pub fn barrier(&mut self, body: Effect, deps: &[TileId]) -> Effect {
        let deps = deps.iter().copied().collect();
        Effect(self.ir.intern(Node::Barrier { body: body.0, deps }))
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
    pub fn load_frag_vec_after<E: Elem>(&mut self, f: Frag<E>, deps: &[TileId]) -> Val<E> {
        let after = self.after_buf(f.id, deps);
        Val::wrap(self.ir.intern(Node::LoadRegVec { buf: after, ept: f.map.ept, dtype: E::dtype() }))
    }

    /// Vector-store a whole `ept`-element fragment run — the WMMA accumulator write-back
    /// (f32) or the vectorised gather's write into a bf16 operand fragment.
    pub fn store_frag_vec<E: Elem>(&mut self, f: Frag<E>, value: Val<E>) -> Effect {
        Effect(self.ir.intern(Node::StoreRegVec { buf: f.id, value: value.id }))
    }

    /// ONE `<ept×E>` vector load of a contiguous LDS run at flat `base`, ordered after
    /// `deps` (the fill barrier) — the vectorised fragment gather ([`Node::LoadVecAt`] →
    /// `ds_read_b64`). Replaces `ept` scalar `load_lds_after` for a contiguous run.
    pub fn load_lds_vec_after<E: Elem>(&mut self, lds: Lds<E>, base: Idx, ept: usize, deps: &[TileId]) -> Val<E> {
        let after = self.after_buf(lds.id, deps);
        Val::wrap(self.ir.intern(Node::LoadVecAt { buf: after, base: base.0, ept, dtype: E::dtype() }))
    }

    /// One K-fragment MFMA `D = A·B + C` (gfx942 16×16×16 bf16→f32). `a`/`b` are the
    /// bf16 fragment operands, `c` the f32 accumulator; returns the f32 result vector.
    pub fn mma(&mut self, a: Val<BF16>, b: Val<BF16>, c: Val<F32>, ept: usize) -> Val<F32> {
        Val::wrap(self.ir.intern(Node::Mma { a: a.id, b: b.id, c: c.id, ept }))
    }

    /// A fragment handle re-bound to observe `deps` (the post-loop carried read:
    /// `acc.after([end])`), symmetric with [`Self::reg_after`].
    pub fn frag_after<E: Elem>(&mut self, f: Frag<E>, deps: &[TileId]) -> Frag<E> {
        let id = self.after_buf(f.id, deps);
        Frag { id, map: f.map, _e: PhantomData }
    }

    // ── ordering edges (first-class, §2.1) ───────────────────────────────────

    /// Route the buffer handle `buf` through completion of `deps` — the ownership
    /// happens-before token: a later read of the returned handle *cannot* be
    /// observed before `deps`. Used for the loop-carried accumulator read/reinit.
    /// `deps` are raw handles so an edge may be an [`Effect`] (a store) OR a
    /// [`Range`] (keeps the read inside the loop body).
    fn after_buf(&mut self, buf: TileId, deps: &[TileId]) -> TileId {
        let deps = deps.iter().copied().collect();
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
    pub fn combine(&mut self, e: Effect, deps: &[TileId]) -> Effect {
        Effect(self.after_buf(e.0, deps))
    }

    /// A register handle re-bound to observe `deps` (the post-loop carried read:
    /// `reg.after([end])`). Returns a fresh [`Reg`] over the ordering-wrapped buffer.
    pub fn reg_after<E: Elem>(&mut self, reg: Reg<E>, deps: &[TileId]) -> Reg<E> {
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
