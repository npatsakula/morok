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
    /// The HK/CK bank-conflict-avoiding XOR swizzle (§2.4, §5b): `col ^ delta(row)` over a
    /// `cols`-wide LDS tile. Set on an LDS tile's layout by `SwizzlePass` and materialised
    /// at each [`Node::LdsCol`] access — the first `.apply`-able layout refinement.
    Xor { cols: usize },
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

/// Integer addressing arithmetic (the const-foldable index band, §2.4). `Mod`/`Div`
/// carry the per-lane fragment `lane_rc` map (row = lane % rows, col = lane / rows …);
/// they are the div/mod the const-fold pass will later collapse for aligned shapes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum IndexOp {
    Add,
    Mul,
    Mod,
    Div,
    /// Bitwise XOR — the LDS bank swizzle (`col ^ delta`), a bijection applied
    /// identically on fill-store and gather-load so it never changes the result.
    Xor,
    /// Logical shift-right (the swizzle's `>> 7` bank-bit extraction).
    Shr,
    /// Logical shift-left (the swizzle's `<< 3` bank placement).
    Shl,
}

/// A register-tile **fragment** lane→(row,col) map — the CDNA 16×16×16 MFMA per-lane
/// layout, mirrored verbatim from tk's `RT_16X16` (`tk/src/tiles.rs`) + `lane_rc`
/// (`tk/src/group/mod.rs`). Each of the `threads` lanes holds `ept` elements of one
/// base fragment; element `inner` of lane `L` maps to:
/// - `transpose == false` (the A / Row operand): `(row = L % rows, col = (L / rows)·stride + inner)`
/// - `transpose == true`  (the B, C / Col operands): `(row = (L / cols)·stride + inner, col = L % cols)`
///
/// This is the load-bearing, silent-garbage-prone datum: a wrong map still lowers and
/// runs, computing plausible-looking garbage — the device allclose is the real proof.
/// It rides as **data on the tile** (`TileMeta::frag`), per DESIGN.md §B/§2.5.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct FragMap {
    /// Base-fragment rows (16 for the gfx942 MFMA edge).
    pub rows: usize,
    /// Base-fragment cols (16).
    pub cols: usize,
    /// Elements each lane holds for one base fragment (4 for gfx942 bf16→f32).
    pub ept: usize,
    /// The lane-group column step (`= ept` for gfx942; the K spread across lane-groups).
    pub stride: usize,
    /// Column-major-in-registers (the B/C operands); selects the transposed `lane_rc` branch.
    pub transpose: bool,
}

impl FragMap {
    /// The gfx942 16×16×16 base fragment (`RT_16X16`): `ept = stride = 4`. `transpose`
    /// selects Row (A) vs Col (B, C) — the only knob the naive matmul varies.
    pub const fn gfx942_16x16(transpose: bool) -> Self {
        FragMap { rows: 16, cols: 16, ept: 4, stride: 4, transpose }
    }
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
    /// A shared-memory (LDS) buffer allocation (`DefineLocal`) of `len` `dtype`
    /// elements. `id` is the per-kernel deterministic slot AND the hash-cons
    /// disambiguator — two LDS tiles must NOT collapse (the renderer names LDS
    /// `@local{id}`; a collision aliases distinct shared buffers = a miscompile). LDS
    /// load/store reuse [`Node::LoadGlobal`]/[`Node::StoreGlobal`] against this buffer
    /// (the ptr's `Local` address space carries the residency).
    DefineLocal { id: u32, dtype: DType, len: usize },
    /// A per-lane register **fragment** tile: an `ept`-element `DefineReg` carrying its
    /// [`FragMap`] lane→(row,col) layout as data (§B/§2.5). Lowers identically to a
    /// [`Node::DefineReg`] of `frag.ept` elements; the map drives the `lane_rc`
    /// addressing in the fragment gather/scatter and the `ept` of the MMA operand.
    DefineFrag { id: u32, dtype: DType, frag: FragMap },
    /// A grid/block index value (`Special`), carrying its bound.
    Axis { axis: ScopeAxis, bound: i64 },
    /// A loop counter (`Range`). `id` disambiguates identically-bounded loops.
    Range { id: u32, trips: i64 },
    /// A compile-time scalar constant.
    Const { scalar: Scalar, dtype: DType },
    /// Integer addressing arithmetic.
    IndexAlu { op: IndexOp, a: TileId, b: TileId },
    /// A **layout-application point** on an LDS tile access: the logical in-tile
    /// `(row, col)` of a `cols`-wide tile, to be materialised to a flat column offset by
    /// the tile's [`Layout`]. The base kernel emits it as the identity (`PassThrough`,
    /// lowering to just `col`); `SwizzlePass` rewrites it to the bank XOR
    /// `col ^ delta(row)` (§2.4 / §5b) — the `.apply`-able refinement, so the swizzle is a
    /// composable layout attribute, not hand-woven into the kernel's addressing.
    LdsCol { row: TileId, col: TileId, cols: usize },
    /// Load a scalar/tile from `buf` (a `Global`/`DefineReg`) at flat `offset`.
    LoadGlobal { buf: TileId, offset: TileId, dtype: DType },
    /// An elementwise binary op on two loaded values.
    EltwiseBinary { op: BinOp, a: TileId, b: TileId },
    /// Store `value` into `buf` at flat `offset` (an effect).
    StoreGlobal { buf: TileId, offset: TileId, value: TileId },
    /// Vector LOAD of a whole `ept`-element per-lane fragment run from register `buf`
    /// (offset 0) — the `<ept × dtype>` operand a WMMA consumes (mirrors tk's
    /// `load_vec_at`). `buf` is a `DefineFrag` (optionally `After`-wrapped for the
    /// loop-carry / post-gather read).
    LoadRegVec { buf: TileId, ept: usize, dtype: DType },
    /// Vector LOAD of `ept` **contiguous** elements from an LDS/global `buf` starting at
    /// flat `base` — one `LOAD(INDEX(buf,[base]), <ept×dtype>)`, which the AMD renderer
    /// lowers to a `ds_read_b64`/`b128` (the vectorised gather; the reused mechanism, not a
    /// hand-written intrinsic). Requires the `ept` run to be contiguous + aligned (the A /
    /// Row operand; B is strided until the `[N,K]` transpose). `buf` may be `After`-wrapped.
    LoadVecAt { buf: TileId, base: TileId, ept: usize, dtype: DType },
    /// Vector STORE of a `<ept × f32>` fragment `value` into register `buf` at offset 0
    /// — the WMMA accumulator write-back (an effect).
    StoreRegVec { buf: TileId, value: TileId },
    /// Vector STORE of a `<ept × dtype>` `value` into an LDS/global `buf` at flat `base`
    /// — one `STORE(INDEX(buf,[base]), value)`, the AMD renderer's `ds_write_b64`/`b128`
    /// (the vectorised fill; the store mirror of [`Node::LoadVecAt`]). Requires the `ept`
    /// run contiguous + aligned (the A / Row fill; B stays scalar under the transpose).
    StoreVecAt { buf: TileId, base: TileId, value: TileId },
    /// Extract scalar element `index` from vector value `vec` → `vec.gep([index])`
    /// ([`Op::Gep`](svod_ir::Op::Gep)). The register-transpose primitive (read a column
    /// out of a loaded row-vector); `dtype` is the scalar element type.
    VecExtract { vec: TileId, index: usize, dtype: DType },
    /// Build a `<len × dtype>` vector from scalar `elements` → `UOp::vectorize(elements)`
    /// ([`Op::Vectorize`](svod_ir::Op::Vectorize)). The register-transpose store operand
    /// (pack a transposed column of scalars into one b64 for a `ds_write_b64`).
    VecBuild { elements: SmallVec<[TileId; 4]>, dtype: DType },
    /// A single K-fragment matrix multiply-accumulate `D = A·B + C` → one
    /// [`Op::Wmma`](svod_ir::Op::Wmma) (gfx942 16×16×16 bf16→f32 MFMA intrinsic).
    /// `a`/`b` are bf16 `LoadRegVec` operands, `c` the f32 accumulator operand; the
    /// result is an f32 `<ept × f32>` vector stored back via [`Node::StoreRegVec`].
    /// One K-fragment MFMA. `asm = true` renders it as inline `asm sideeffect`
    /// (`v_mfma_f32_16x16x16_bf16`, `=v,v,v,0`) instead of the intrinsic — schedule-opaque, so
    /// the cluster program order survives `-O3` WITHOUT the `sched.barrier(0)` walls that extend
    /// live ranges past the VGPR spill cliff (§5c: the asm channel is how HK stays at 254 VGPR).
    /// The accumulator `"0"` tie keeps the K-reduction in one physical register either way.
    Mma { a: TileId, b: TileId, c: TileId, ept: usize, asm: bool },
    /// The addr(3) **base pointer** of an LDS tile at flat element `base` —
    /// `addrspacecast(index_off(buf, base))` → ONE base VGPR shared by a slice's
    /// [`Node::DsReadB64`] gathers (DESIGN §5c — HK's operand gather reads from ONE base +
    /// a per-fragment `offset:` immediate, so the `lane_rc` div/mod address is materialised
    /// once, not per fragment; that collapse is what breaks the VGPR-spill cliff). `buf` may
    /// be `After`-wrapped (the RAW ordering rides the buffer, exactly as the scalar
    /// [`Node::LoadGlobal`] LDS read). Lowers to a `Int32`-typed `Op::Custom`.
    LdsPtrAs3 { buf: TileId, base: TileId },
    /// ONE inline-asm `ds_read_b64 $d, $base offset:N` LDS gather (gfx942, DESIGN §5c — the
    /// ONLY asm HipKittens uses; its MFMA/set_prio/sched.barrier are all intrinsics we already
    /// emit): reads the `ept`-element bf16 run at `base_ptr + off_bytes` into a fresh
    /// `<ept×bf16>` value. `off_bytes` is a compile-time immediate (`≤ 65535`, the 16-bit
    /// `offset:` field); `prev` chains the `sideeffect` reads in program order (an ordering-only
    /// operand — the prior fragment's store — so the reads can't hoist across the barriers, the
    /// silent-stale-read class §2.1). Renders as `read.bitcast(<ept×bf16>)` of the `<ept×i16>`
    /// asm result.
    DsReadB64 { base_ptr: TileId, off_bytes: i64, ept: usize, dtype: DType, prev: Option<TileId> },
    /// ONE inline-asm `ds_write_b64 $base, $val offset:N` LDS store (gfx942, DESIGN §5c — the
    /// **commit** twin of [`Node::DsReadB64`]): writes the `ept`-element bf16 `value` to
    /// `base_ptr + off_bytes`. Being `asm sideeffect` it is OPAQUE to LLVM's waitcnt pass, so unlike
    /// [`Node::StoreVecAt`] an `s_barrier` does NOT auto-drain it — HK's escape from the barrier's
    /// implicit `lgkmcnt(0)` (the writes stay in flight until a manual [`Node::SWaitLgkmcnt`] drains
    /// them where the schedule wants). `prev` chains the `sideeffect` writes in program order (an
    /// ordering-only operand — the prior fragment's write — keeping them from hoisting across the
    /// barriers, the silent-stale class §2.1). An EFFECT (a side-effect store, like [`Node::Barrier`]).
    DsWriteB64 { base_ptr: TileId, off_bytes: i64, value: TileId, ept: usize, prev: Option<TileId> },
    /// The **manual LDS drain** (`s_waitcnt lgkmcnt(0)`, gfx942 §5c): a void `asm sideeffect` that
    /// stalls until every outstanding LDS op completes — the EXPOSED drain the [`Node::DsWriteB64`]
    /// commit needs (its writes are waitcnt-opaque, so the RAW `s_barrier` no longer fences them; this
    /// re-establishes the store→barrier→load order). `prev` is the last commit write (an ordering-only
    /// operand pinning the drain after the writes in program order). An EFFECT.
    SWaitLgkmcnt { prev: TileId },
    /// A workgroup synchronization barrier (`s.barrier`): `body` (a store) passes
    /// through as the effect, and every write in `deps` is fenced — all must complete
    /// before any consumer routed [`Node::After`] this barrier proceeds. This is the
    /// `store → barrier → load` order the cross-lane LDS stage needs, carried as a
    /// first-class node: a missing store→load edge would be a silent wrong answer (the
    /// exact class §2.1 targets), so the barrier is structural, not implicit. Mirrors
    /// tk's `store.barrier(deps)` idiom (`tk/src/kernel.rs`).
    Barrier { body: TileId, deps: Edges },
    /// A **bare workgroup barrier** (`@llvm.amdgcn.s.barrier()` + a positional `sched.barrier(0)`
    /// wall, DESIGN §5c) — the HK cluster-seal twin of [`Node::Barrier`] *without* the
    /// `fence acquire`/`fence release` acq-rel pair. The fence is what forces an implicit
    /// `s_waitcnt lgkmcnt(0)` at every seal AND acts as a machine-scheduler barrier that throttles
    /// MFMA overlap (`tk/src/arch/gfx9.rs:55`); a per-cluster schedule (9 seals/K-block) pays that
    /// 9× where HK drains only 3× and uses a bare `s_barrier` for the rest. Dropping the fence makes
    /// the seal a pure workgroup rendezvous (the ping-pong phase carrier), so the LDS ordering it no
    /// longer provides MUST be re-supplied explicitly by a [`Node::SWaitLgkmcnt`] at the RAW/WAR/
    /// pre-MFMA points (the caller's obligation — a missing drain is the silent-stale class §2.1).
    /// `body` + `deps` are pure happens-after anchors (no `{N}` refs). The baked-in `sched.barrier(0)`
    /// reproduces the [`Node::SchedWallMarker`] wall locally (this lowers to `Op::Custom`, which the
    /// codegen `wall_after_barriers` pass — keyed on `Op::Barrier` — would otherwise miss).
    BareBarrier { body: TileId, deps: Edges },
    /// A **machine-scheduler fence** (`@llvm.amdgcn.sched.barrier(mask)`, DESIGN §5c): a
    /// void side-effect the AMDGPU scheduler may not move any instruction across (mask 0 =
    /// total). Positioned in the instruction stream *right after* `deps` (its ordering
    /// anchors — typically the prefetch loads), so a later consumer routed after it cannot
    /// float above it and the anchored loads cannot sink below it — the **load-pin** that
    /// keeps the register-staged prefetch in flight across the MFMAs (the measured cure for
    /// LLVM sinking the loads to the loop tail). Unlike [`Node::Barrier`] this is NOT a
    /// workgroup sync — it is purely a compiler scheduling boundary. Kept live + ordered by
    /// a downstream consumer that lists it in *its* deps.
    SchedFence { mask: i64, deps: Edges },
    /// A **wave issue-priority** control (`s_setprio level`, DESIGN §5c): a void side-effect
    /// positioned after `deps`. Bracketing an MFMA cluster with `set_prio(1) … set_prio(0)`
    /// makes the compute wave win SIMD issue over the co-resident loading wave (the systolic
    /// array stays fed). Emitted as `asm sideeffect` (schedule-opaque by construction), so it
    /// pins its own position. Arch-gated (§2.8): a no-op where the arch has no priority model.
    SetPrio { level: i64, deps: Edges },
    /// The **wave-phase asymmetric barrier** (DESIGN §5c/3c): only lanes whose `warp_row == eq`
    /// run the `s_barrier` (a `readfirstlane`+`s_cmp`+`s_cbranch`+`s_barrier` asm block, mirroring
    /// tk's `wave_phase_barrier`). `deps[0]` IS the `warp_row` index operand; `deps[1..]` are
    /// ordering anchors. Placing an `eq=1` barrier in the prologue and an `eq=0` in the epilogue
    /// phase-offsets the two warp-groups one cluster apart (so one group's MFMA clusters overlap
    /// the other's memory clusters — the fill that makes the per-cluster barriers pay). The ONE
    /// asm/control-flow primitive (tk2 forbids authoring If/EndIf, so the predicate rides inside
    /// the asm). Balance-critical: an `eq=0`/`eq=1` count mismatch deadlocks the workgroup.
    WaveBarrier { eq: i64, deps: Edges },
    /// The **HK barrier-wall opt-in** (DESIGN §5c): a void sentinel that makes the codegen
    /// `wall_after_barriers` pass pair every `s_barrier` in this kernel with a positional
    /// `sched.barrier(0)` — the cluster wall lattice, placed by stream position (not value-anchored,
    /// so it cannot float into the prefetch). Emit once; kept live by folding into the `End`.
    SchedWallMarker,
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
    /// The per-lane fragment lane→(row,col) map, present only on register-fragment
    /// tiles ([`Node::DefineFrag`]) — the tile carries its own MFMA layout as data.
    pub frag: Option<FragMap>,
}

impl TileMeta {
    fn value(shape: Shape, dtype: DType, residency: Residency) -> Self {
        TileMeta {
            shape,
            dtype: Some(dtype),
            layout: Layout::contiguous(),
            residency,
            reg_class: RegClass::Vgpr,
            frag: None,
        }
    }
    fn effect() -> Self {
        TileMeta {
            shape: SmallVec::new(),
            dtype: None,
            layout: Layout::contiguous(),
            residency: Residency::Reg,
            reg_class: RegClass::Vgpr,
            frag: None,
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
    next_local: u32,
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
    /// A fresh LDS slot / disambiguator (a separate namespace from regs — the
    /// renderer names LDS `@local{id}`, so two shared tiles must not share an id).
    pub fn fresh_local_id(&mut self) -> u32 {
        let l = self.next_local;
        self.next_local += 1;
        l
    }

    /// Apply `f` to each operand handle of `node`, returning a rebuilt node — the
    /// primitive the nanopass folder recurses through (children already rewritten).
    pub fn map_children(node: &Node, mut f: impl FnMut(TileId) -> TileId) -> Node {
        match node.clone() {
            Node::IndexAlu { op, a, b } => Node::IndexAlu { op, a: f(a), b: f(b) },
            Node::LdsCol { row, col, cols } => Node::LdsCol { row: f(row), col: f(col), cols },
            Node::LoadGlobal { buf, offset, dtype } => Node::LoadGlobal { buf: f(buf), offset: f(offset), dtype },
            Node::EltwiseBinary { op, a, b } => Node::EltwiseBinary { op, a: f(a), b: f(b) },
            Node::StoreGlobal { buf, offset, value } => {
                Node::StoreGlobal { buf: f(buf), offset: f(offset), value: f(value) }
            }
            Node::LoadRegVec { buf, ept, dtype } => Node::LoadRegVec { buf: f(buf), ept, dtype },
            Node::LoadVecAt { buf, base, ept, dtype } => Node::LoadVecAt { buf: f(buf), base: f(base), ept, dtype },
            Node::StoreRegVec { buf, value } => Node::StoreRegVec { buf: f(buf), value: f(value) },
            Node::StoreVecAt { buf, base, value } => Node::StoreVecAt { buf: f(buf), base: f(base), value: f(value) },
            Node::VecExtract { vec, index, dtype } => Node::VecExtract { vec: f(vec), index, dtype },
            Node::VecBuild { elements, dtype } => {
                Node::VecBuild { elements: elements.into_iter().map(&mut f).collect(), dtype }
            }
            Node::Mma { a, b, c, ept, asm } => Node::Mma { a: f(a), b: f(b), c: f(c), ept, asm },
            Node::LdsPtrAs3 { buf, base } => Node::LdsPtrAs3 { buf: f(buf), base: f(base) },
            Node::DsReadB64 { base_ptr, off_bytes, ept, dtype, prev } => {
                Node::DsReadB64 { base_ptr: f(base_ptr), off_bytes, ept, dtype, prev: prev.map(&mut f) }
            }
            Node::DsWriteB64 { base_ptr, off_bytes, value, ept, prev } => {
                Node::DsWriteB64 { base_ptr: f(base_ptr), off_bytes, value: f(value), ept, prev: prev.map(&mut f) }
            }
            Node::SWaitLgkmcnt { prev } => Node::SWaitLgkmcnt { prev: f(prev) },
            Node::Barrier { body, deps } => {
                Node::Barrier { body: f(body), deps: deps.into_iter().map(&mut f).collect() }
            }
            Node::BareBarrier { body, deps } => {
                Node::BareBarrier { body: f(body), deps: deps.into_iter().map(&mut f).collect() }
            }
            Node::SchedFence { mask, deps } => Node::SchedFence { mask, deps: deps.into_iter().map(&mut f).collect() },
            Node::SetPrio { level, deps } => Node::SetPrio { level, deps: deps.into_iter().map(&mut f).collect() },
            Node::WaveBarrier { eq, deps } => Node::WaveBarrier { eq, deps: deps.into_iter().map(&mut f).collect() },
            Node::SchedWallMarker => Node::SchedWallMarker,
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
            Node::DefineLocal { dtype, len, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*len]), dtype.clone(), Residency::Lds)
            }
            Node::DefineFrag { dtype, frag, .. } => {
                let mut m = TileMeta::value(SmallVec::from_slice(&[frag.ept]), dtype.clone(), Residency::Reg);
                m.frag = Some(*frag);
                m
            }
            Node::Axis { .. } | Node::Range { .. } => TileMeta::value(SmallVec::new(), DType::Index, Residency::Reg),
            Node::Const { dtype, .. } => TileMeta::value(SmallVec::new(), dtype.clone(), Residency::Reg),
            Node::IndexAlu { .. } | Node::LdsCol { .. } => {
                TileMeta::value(SmallVec::new(), DType::Index, Residency::Reg)
            }
            Node::LoadGlobal { dtype, .. } => TileMeta::value(SmallVec::new(), dtype.clone(), Residency::Reg),
            Node::EltwiseBinary { a, .. } => {
                let dt = self.meta(*a).dtype.clone().unwrap_or(DType::Float32);
                TileMeta::value(SmallVec::new(), dt, Residency::Reg)
            }
            // A fragment vector value: an `ept`-lane register vector (bookkeeping only;
            // the lowered UOp carries the true `dtype.vec(ept)`).
            Node::LoadRegVec { dtype, ept, .. } | Node::LoadVecAt { dtype, ept, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*ept]), dtype.clone(), Residency::Reg)
            }
            Node::Mma { ept, .. } => TileMeta::value(SmallVec::from_slice(&[*ept]), DType::Float32, Residency::Reg),
            // The addr(3) base pointer VGPR (Int32-typed, mirroring tk's `base_as3` custom).
            Node::LdsPtrAs3 { .. } => TileMeta::value(SmallVec::new(), DType::Int32, Residency::Reg),
            // The asm gather's `<ept×bf16>` operand value.
            Node::DsReadB64 { dtype, ept, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*ept]), dtype.clone(), Residency::Reg)
            }
            // A scalar extracted from a vector; a `len`-vector built from scalars.
            Node::VecExtract { dtype, .. } => TileMeta::value(SmallVec::new(), dtype.clone(), Residency::Reg),
            Node::VecBuild { elements, dtype } => {
                TileMeta::value(SmallVec::from_slice(&[elements.len()]), dtype.clone(), Residency::Reg)
            }
            // `After` is a passthrough of its value (an ordering edge routed
            // through it), so it carries the value's residency/dtype/layout.
            Node::After { val, .. } => self.meta(*val).clone(),
            Node::StoreGlobal { .. }
            | Node::StoreRegVec { .. }
            | Node::StoreVecAt { .. }
            | Node::DsWriteB64 { .. }
            | Node::SWaitLgkmcnt { .. }
            | Node::Barrier { .. }
            | Node::BareBarrier { .. }
            | Node::SchedFence { .. }
            | Node::SetPrio { .. }
            | Node::WaveBarrier { .. }
            | Node::SchedWallMarker
            | Node::End { .. }
            | Node::Sink { .. } => TileMeta::effect(),
        }
    }
}
