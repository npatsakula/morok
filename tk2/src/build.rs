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

use crate::ir::{BinOp, IndexOp, Node, Scalar, ScopeAxis, TileId, TileIr};

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
