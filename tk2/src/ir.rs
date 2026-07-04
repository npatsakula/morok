//! The interned tile-IR — ONE hash-consed DAG carrying both the tile algorithm
//! and (eventually) the schedule as data (DESIGN.md §OPEN-1 → decided: one IR).
//!
//! Every node is interned into an arena keyed by its *structure*, and the builder
//! returns `Copy` [`TileId`] handles (§2.3). Structurally-identical nodes collapse
//! to one id — which is exactly why the schedule-carrying disambiguators
//! ([`Node::Range`]`.id`, [`Node::DefineReg`]`.id`, [`Node::Global`]`.slot`) are
//! part of the interning key: two distinct loops / registers / buffers must NOT
//! fold together, mirroring svod's `DefineReg.id` / buffer-`Unique` obligation, or
//! the lowered UOp collapses distinct registers into one (a miscompile).
//!
//! Ordering is a **first-class edge** ([`Node::After`] / [`Node::End`], §2.1): a
//! dependent value is routed *through* the edge, so a consumer cannot observe the
//! value without observing the ordering constraint. The linearizer below the
//! boundary orders purely by DAG edges, so a missing edge would be a silent wrong
//! answer — here it is a structural node instead.

use std::collections::HashMap;

use smallvec::SmallVec;
use svod_dtype::DType;

/// A `Copy` handle into the [`TileIr`] arena. Interning guarantees two handles are
/// equal iff their nodes are structurally identical (with disambiguators).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct TileId(pub u32);

/// A tile's logical shape (scalar tiles carry `[]` or `[1]`).
pub type Shape = SmallVec<[usize; 4]>;

/// A short list of ordering / range operands (edges).
pub type Edges = SmallVec<[TileId; 4]>;

// ── layout: a typed transform-graph ADT (DESIGN.md §2.4), stubbed to what the
//    trivial kernels need. The full CK transform graph (Unmerge/Merge/Xor/Pad/…)
//    plugs in as more `Transform` variants + a const-fold pass in Step 2. ──────

/// A single layout transform node. Composed by graph-append into a [`Layout`];
/// only `PassThrough` and a stub `Embed` exist today — swizzle (`Xor`), tiling
/// (`Unmerge`/`Merge`), padding, etc. are the extension points.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum Transform {
    /// Identity — the flat, contiguous, row-major addressing the skeleton uses.
    #[default]
    PassThrough,
    /// A strided embed of a logical axis (stubbed: carries the stride only).
    Embed { stride: i64 },
}

/// A tile's addressing layout: an ordered chain of [`Transform`]s (a graph, in a
/// vector for now). Contiguous == a single `PassThrough`.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct Layout {
    pub transforms: SmallVec<[Transform; 2]>,
}

impl Layout {
    /// The contiguous (identity) layout.
    pub fn contiguous() -> Self {
        Layout { transforms: SmallVec::new() }
    }
}

/// Where a tile value physically lives (DESIGN.md §2.5 — an enum field, not a
/// residency code-explosion). Only `Reg`/`Global` are exercised in Step 1.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum Residency {
    #[default]
    Reg,
    Lds,
    Global,
}

/// The register class a `Reg`-resident tile occupies (DESIGN.md §2.5/§6C —
/// present now, so the AGPR channel is a field flip in Step 3, not a 2⁴ rewrite).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum RegClass {
    #[default]
    Vgpr,
    Agpr,
}

/// Elementwise binary op vocabulary (a minimal slice of the ~30 TK/HK ops, §3).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Max,
}

/// Integer addressing arithmetic (the const-foldable index band, §2.4).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum IndexOp {
    Add,
    Mul,
}

/// A scope index source: a grid dimension or the block (thread) index. Nesting of
/// loops is *emergent* from the [`Node::Range`]/[`Node::End`] edges, never authored
/// as a boundary (DESIGN.md §D boundary constraint).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum ScopeAxis {
    Grid(u8),
    Block,
}

/// A hash-safe scalar constant (f32 stored as raw bits so [`Node`] stays `Eq`+`Hash`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Scalar {
    Int(i64),
    F32(u32),
}

impl Scalar {
    pub fn f32(v: f32) -> Self {
        Scalar::F32(v.to_bits())
    }
}

/// A tile-IR node. Value-producing and effectful nodes share one arena; the
/// interning key is the whole node (including disambiguators).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Node {
    /// A global buffer parameter. `slot` is the ABI position (outputs first) and a
    /// hash-cons disambiguator — two params at different slots never collapse.
    Global { slot: u32, dtype: DType, len: usize },
    /// A per-lane register accumulator cell (`DefineReg`). `id` is the per-kernel
    /// deterministic slot AND the disambiguator (svod's correctness obligation).
    DefineReg { id: u32, dtype: DType, len: usize },
    /// A grid/block index value (`Special`), carrying its bound.
    Axis { axis: ScopeAxis, bound: i64 },
    /// A loop counter (`Range`). `id` disambiguates identically-bounded loops.
    Range { id: u32, trips: i64 },
    /// A compile-time scalar constant.
    Const { scalar: Scalar, dtype: DType },
    /// Integer addressing arithmetic.
    IndexAlu { op: IndexOp, a: TileId, b: TileId },
    /// Load a scalar/tile from `buf` (a `Global`/`DefineReg`) at flat `offset`.
    LoadGlobal { buf: TileId, offset: TileId, dtype: DType },
    /// An elementwise binary op on two loaded values.
    EltwiseBinary { op: BinOp, a: TileId, b: TileId },
    /// Store `value` into `buf` at flat `offset` (an effect).
    StoreGlobal { buf: TileId, offset: TileId, value: TileId },
    /// Ordering edge: `val` is routed through completion of every dep in `deps`.
    After { val: TileId, deps: Edges },
    /// Close `ranges` loop(s) around effect `body` (one `End` per `Range`).
    End { body: TileId, ranges: Edges },
    /// The kernel sink over terminal effects.
    Sink { roots: Edges },
}

/// The derived per-value metadata every tile carries (DESIGN.md §2.5 — shape /
/// dtype / layout / residency / reg-class as fields). Computed at intern time from
/// the node + its operands; effect nodes carry a `Void`-ish placeholder.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TileMeta {
    pub shape: Shape,
    pub dtype: Option<DType>,
    pub layout: Layout,
    pub residency: Residency,
    pub reg_class: RegClass,
}

impl TileMeta {
    fn value(shape: Shape, dtype: DType, residency: Residency) -> Self {
        TileMeta { shape, dtype: Some(dtype), layout: Layout::contiguous(), residency, reg_class: RegClass::Vgpr }
    }
    fn effect() -> Self {
        TileMeta {
            shape: SmallVec::new(),
            dtype: None,
            layout: Layout::contiguous(),
            residency: Residency::Reg,
            reg_class: RegClass::Vgpr,
        }
    }
}

/// The interned tile-IR arena: hash-consed nodes + parallel derived metadata +
/// monotonic disambiguator counters. Children are always interned before parents
/// (the builder builds bottom-up), so a `TileId`'s operands have strictly smaller
/// ids — which the lowering and folder rely on for a single linear pass.
#[derive(Default)]
pub struct TileIr {
    nodes: Vec<Node>,
    meta: Vec<TileMeta>,
    dedup: HashMap<Node, TileId>,
    next_range: u32,
    next_reg: u32,
    next_slot: u32,
}

impl TileIr {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern `node`, returning its (possibly pre-existing) handle. Structurally
    /// identical nodes collapse to one id.
    pub fn intern(&mut self, node: Node) -> TileId {
        if let Some(&id) = self.dedup.get(&node) {
            return id;
        }
        let meta = self.derive_meta(&node);
        let id = TileId(self.nodes.len() as u32);
        self.dedup.insert(node.clone(), id);
        self.nodes.push(node);
        self.meta.push(meta);
        id
    }

    /// The node behind a handle.
    pub fn node(&self, id: TileId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// The derived metadata behind a handle.
    pub fn meta(&self, id: TileId) -> &TileMeta {
        &self.meta[id.0 as usize]
    }

    /// Total interned node count (reachable + dedup residue).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    // ── disambiguator mints (the hash-cons correctness obligation) ───────────

    /// A fresh global-buffer ABI slot.
    pub fn fresh_slot(&mut self) -> u32 {
        let s = self.next_slot;
        self.next_slot += 1;
        s
    }
    /// A fresh loop id (so two same-trip loops do not hash-cons together).
    pub fn fresh_range_id(&mut self) -> u32 {
        let r = self.next_range;
        self.next_range += 1;
        r
    }
    /// A fresh register slot / disambiguator.
    pub fn fresh_reg_id(&mut self) -> u32 {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    /// Apply `f` to each operand handle of `node`, returning a rebuilt node — the
    /// primitive the nanopass folder recurses through (children already rewritten).
    pub fn map_children(node: &Node, mut f: impl FnMut(TileId) -> TileId) -> Node {
        match node.clone() {
            Node::IndexAlu { op, a, b } => Node::IndexAlu { op, a: f(a), b: f(b) },
            Node::LoadGlobal { buf, offset, dtype } => Node::LoadGlobal { buf: f(buf), offset: f(offset), dtype },
            Node::EltwiseBinary { op, a, b } => Node::EltwiseBinary { op, a: f(a), b: f(b) },
            Node::StoreGlobal { buf, offset, value } => {
                Node::StoreGlobal { buf: f(buf), offset: f(offset), value: f(value) }
            }
            Node::After { val, deps } => Node::After { val: f(val), deps: deps.into_iter().map(&mut f).collect() },
            Node::End { body, ranges } => Node::End { body: f(body), ranges: ranges.into_iter().map(&mut f).collect() },
            Node::Sink { roots } => Node::Sink { roots: roots.into_iter().map(&mut f).collect() },
            // Leaves (no operands): returned unchanged.
            leaf => leaf,
        }
    }

    /// Every operand handle of `node` (for structural walks like `top_down`).
    pub fn children(node: &Node) -> Edges {
        let mut out = Edges::new();
        Self::map_children(node, |c| {
            out.push(c);
            c
        });
        out
    }

    fn derive_meta(&self, node: &Node) -> TileMeta {
        match node {
            Node::Global { dtype, len, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*len]), dtype.clone(), Residency::Global)
            }
            Node::DefineReg { dtype, len, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*len]), dtype.clone(), Residency::Reg)
            }
            Node::Axis { .. } | Node::Range { .. } => TileMeta::value(SmallVec::new(), DType::Index, Residency::Reg),
            Node::Const { dtype, .. } => TileMeta::value(SmallVec::new(), dtype.clone(), Residency::Reg),
            Node::IndexAlu { .. } => TileMeta::value(SmallVec::new(), DType::Index, Residency::Reg),
            Node::LoadGlobal { dtype, .. } => TileMeta::value(SmallVec::new(), dtype.clone(), Residency::Reg),
            Node::EltwiseBinary { a, .. } => {
                let dt = self.meta(*a).dtype.clone().unwrap_or(DType::Float32);
                TileMeta::value(SmallVec::new(), dt, Residency::Reg)
            }
            // `After` is a passthrough of its value (an ordering edge routed
            // through it), so it carries the value's residency/dtype/layout.
            Node::After { val, .. } => self.meta(*val).clone(),
            Node::StoreGlobal { .. } | Node::End { .. } | Node::Sink { .. } => TileMeta::effect(),
        }
    }
}
