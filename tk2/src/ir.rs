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

/// Elementwise **unary** op vocabulary (FA-forward addition — the GEMM path needed none).
/// The softmax core: `Exp2` is the hardware `v_exp_f32` (the AMD decomposition leaves f32
/// `Exp2` native), `Recip` the final `1/norm` normalize. Both lower to `Op::Unary` (svod-ir
/// has the whole transcendental table; tk2 just never surfaced it — see [`crate::build::Builder::exp2`]).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnOp {
    Exp2,
    Recip,
}

/// Integer addressing arithmetic (the const-foldable index band, §2.4). `Mod`/`Div`
/// carry the per-lane fragment `lane_rc` map (row = lane % rows, col = lane / rows …);
/// they are the div/mod the const-fold pass will later collapse for aligned shapes.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum IndexOp {
    Add,
    Sub,
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
    /// Integer minimum — the workgroup-specific clamp for causal/runtime iteration domains.
    Min,
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

/// The **MFMA accumulator lane→(row,col) distribution** — the datum a single [`FragMap`] cannot express
/// (§migration). The accumulator's per-lane elements decompose into a *two-level* M-block split that the
/// FragMap's one arithmetic-progression `lane_rc` run has no room for. Mirrors CK's `CWarpDstrEncoding`
/// (`ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`): the M axis is `sequence<kCM0PerLane, kCMLane,
/// kCM1PerLane>`, the N axis `sequence<kCNLane>`. For a per-lane accumulator element `i ∈ [0, ept_c)`
/// (`m_blk = i / m_inner`, `m_in = i % m_inner`):
/// - `row = m_blk·m_block_stride + (lane / n_lanes)·lane_m_stride + m_in`
/// - `col = lane % n_lanes`
///
/// 16×16×16 is the degenerate `m_blocks = 1` case (`row = (lane/16)·4 + i`, `col = lane%16`) — identical
/// to the `transpose` [`FragMap`] `lane_rc`; 32×32×8 is the `m_blocks = 4` case (four row-blocks 8 apart).
/// Rides as DATA on the shape (derived by the `MfmaShape` marker), never as a surface type (§OPEN-2).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct AccDist {
    /// Outer M row-blocks per lane — CK `kCM0PerLane` (16×16: 1, 32×32: 4).
    pub m_blocks: usize,
    /// Row stride between blocks — CK `kCMLane·kCM1PerLane` (16×16: 16, 32×32: 8).
    pub m_block_stride: usize,
    /// Rows per block (the fast per-lane element run) — CK `kCM1PerLane` (16×16: 4, 32×32: 4).
    pub m_inner: usize,
    /// Row stride between M-lane groups — CK `kCM1PerLane` (16×16: 4, 32×32: 4).
    pub lane_m_stride: usize,
    /// Lanes spanning the N (column) axis — CK `kCNLane` (16×16: 16, 32×32: 32).
    pub n_lanes: usize,
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
    /// A grid/block index whose launch bound is a runtime index expression.
    AxisDyn { axis: ScopeAxis, bound: TileId },
    /// A loop counter (`Range`). `id` disambiguates identically-bounded loops.
    Range { id: u32, trips: i64 },
    /// A statically-sized loop whose entry must follow `deps`. This is the peeled-prologue form: the
    /// trip count remains constant while the control-flow region has an explicit incoming edge.
    RangeAfter { id: u32, trips: i64, deps: Edges },
    /// A runtime-sized loop domain. `deps` optionally order a peeled prologue before loop entry.
    RangeDyn { id: u32, trips: TileId, deps: Edges },
    /// A zero-instruction lexical scope marker. `id` prevents address/value DAGs authored in distinct
    /// control-flow regions from hash-consing together; consumers bind through [`Node::After`].
    Scope { id: u32, deps: Edges },
    /// A bounded runtime integer kernel argument. The current device ABI requires i32-compatible bounds.
    ScalarParam { name: String, min: i64, max: i64 },
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
    /// A gated global load: `offset < bound ? buf[offset] : alt`.
    LoadGlobalBounded { buf: TileId, offset: TileId, bound: TileId, alt: TileId, dtype: DType },
    /// An elementwise binary op on two loaded values.
    EltwiseBinary { op: BinOp, a: TileId, b: TileId },
    /// A predicated **select on an index comparison**: `lo < hi ? then : els`, per element (FA's
    /// ragged-tail mask — `select(global_kv < n, score, -inf)`). `lo`/`hi` are index-typed;
    /// `then`/`els` are the value branches (same dtype). Lowers to `Op::Ternary(Where,
    /// Op::Binary(Lt, lo, hi), then, els)`. The result inherits `then`'s dtype (a value node).
    SelectLt { lo: TileId, hi: TileId, then: TileId, els: TileId },
    /// An elementwise **unary** math op (FA-forward addition — the softmax `exp2` / the
    /// normalize `recip`). Lowers to `Op::Unary(UnaryOp::{Exp2,Reciprocal}, x)` — svod-ir
    /// already carries the whole transcendental table; tk2's GEMM path just never used it.
    Unary { op: UnOp, x: TileId },
    /// A **cross-lane lane-gather** (`llvm.amdgcn.ds.bpermute`) — `data` as computed by
    /// lane `addr/4` (`addr` = the byte lane-address `src_lane·4`). The ONLY inter-lane
    /// exchange primitive on gfx942 that needs no LDS round-trip; svod-ir has no cross-lane
    /// op at all, so this lowers to a hand-written inline-LLVM `Op::Custom` (mirroring tk1's
    /// `shuffle_lane`). Barrier-free — the building block of FA's `exp2`-free-of-barrier
    /// online-softmax row reductions. Renders `bitcast f32→i32`, bpermute, `bitcast i32→f32`.
    DsBpermute { addr: TileId, data: TileId },
    /// An **intra-lane byte permute** (`v_perm_b32 D, hi, lo, sel` → `llvm.amdgcn.perm`, gfx942) — the
    /// register-level 2×2 bf16 transpose aiter's Flash-Attention uses for the 32×32×8 relayouts. Over an
    /// 8-byte pool `{lo.bytes[0..4] @ idx 0-3, hi.bytes[0..4] @ idx 4-7}`, output byte `i` = `pool[sel.byte[i]]`.
    /// The two aiter selectors: `s49 = 0x07060302` gathers the HIGH bf16 of each dword pair (bytes {2,3,6,7})
    /// — which, over two f32 operands, is exactly their **truncated (RTZ) bf16 pair** packed into one dword
    /// (bf16 = the top 16 bits of an f32); `s50 = 0x05040100` gathers the LOW bf16 (bytes {0,1,4,5}). `hi`/`lo`
    /// are the two source dwords (bitcast to i32); `selector` is the compile-time byte-select immediate. The
    /// result is a `<2 × bf16>` dword (the two gathered bf16 halves). Lowers to a `<2×bf16>`-bitcast of the
    /// `llvm.amdgcn.perm` i32 result — a hand-written `Op::Custom`, mirroring [`Node::DsBpermute`].
    VPerm { hi: TileId, lo: TileId, selector: i64 },
    /// Waitcnt-opaque packed-bf16 `v_perm_b32`. Used after an explicit partial VMEM wait in the d128
    /// V transpose so LLVM cannot strengthen `vmcnt(4)` to a full drain at the intrinsic use.
    VPermAsm { hi: TileId, lo: TileId, selector: i64 },
    /// A b64 register value tied through side-effect inline asm after an opaque LDS readiness wait.
    /// This prevents MachineScheduler from moving an MFMA consumer above `SWaitLgkmcnt`.
    OpaqueReadyB64 { val: TileId, wait: TileId },
    /// Store `value` into `buf` at flat `offset` (an effect).
    StoreGlobal { buf: TileId, offset: TileId, value: TileId },
    /// A gated global store: write only when `offset < bound`.
    StoreGlobalBounded { buf: TileId, offset: TileId, bound: TileId, value: TileId },
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
    /// Hardware direct GLOBAL→LDS DMA (`global_load_lds_dword` on gfx942). The source and destination
    /// offsets are in elements of their bf16 buffers; the intrinsic transfers one dword. `deps` are
    /// ordering-only anchors that pin the issue point without routing the payload through VGPRs.
    GlobalLoadLdsDword { src: TileId, src_offset: TileId, dst: TileId, dst_offset: TileId, deps: Edges },
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
    ///
    /// `hk_form` (default `false`) switches the rendered asm to HipKittens' **literal** IR form
    /// (`"ds_read_b64 $0, $1 offset:$2\0A", "=v,v,i,~{memory}"(i32 addr, i64 off)` → `i64` result,
    /// bitcast to `<ept×bf16>`) — an i32 raw-address operand, the offset as an `i` immediate operand,
    /// and a `~{memory}` clobber, matching `hk-micro_tk.ll`. The default form (existing clustered
    /// kernel) is byte-unchanged; the HK port uses the flagged form.
    DsReadB64 { base_ptr: TileId, off_bytes: i64, ept: usize, dtype: DType, prev: Option<TileId>, hk_form: bool },
    /// The **buffer-resource descriptor** (`ptr addrspace(8)`) of a global buffer — `make.buffer.rsrc.p0`
    /// of `&buf[base_off]`; `num_bytes` = the buffer byte-extent (the SRD bound), config `0x110000` (HK's
    /// `make_srsrc`, row_stride 0). Shipped with `base_off = 0` (fixed base, loop-invariant → hoisted);
    /// an advancing base (`origin·K + k_base`) is HK's perf scheme but flaky here. Feeds
    /// [`Node::BufferLoadRaw`]. Lowers to a pointer-typed `Op::Custom`.
    MakeBufferRsrc { buf: TileId, base_off: TileId, num_bytes: i64 },
    /// Runtime-bounded modern buffer resource descriptor. `num_bytes` is relative to `base_off`.
    MakeBufferRsrcDyn { buf: TileId, base_off: TileId, num_bytes: TileId },
    /// ONE `llvm.amdgcn.raw.buffer.load.v{dwords}i32` **MUBUF** load (gfx942 — HK's DRAM prefetch, the
    /// escape from FLAT `global_load`): reads the `ept`-element run at `rsrc[voffset]` bytes, `soffset = 0`.
    /// The address split rides in the DESCRIPTOR: `rsrc`'s base advances per K-tile in SCALAR (see
    /// [`Node::MakeBufferRsrc`]), and `voffset` is the per-lane, K-invariant within-tile byte offset
    /// (hoisted VGPR) — so the load issues WITHOUT a per-iteration VGPR-address `v_add` on its critical
    /// path, and without a non-zero `soffset` (mishandled by the config). `order` are ordering-only anchors
    /// (the authoring cluster A@C0 / B@C4). Renders as `load.bitcast(<ept×elem>)` of the `<dwords×i32>`.
    BufferLoadRaw { rsrc: TileId, voffset: TileId, ept: usize, dtype: DType, order: Edges },
    /// ONE inline-asm `ds_write_b64 $base, $val offset:N` LDS store (gfx942, DESIGN §5c — the
    /// **commit** twin of [`Node::DsReadB64`]): writes the `ept`-element bf16 `value` to
    /// `base_ptr + off_bytes`. Being `asm sideeffect` it is OPAQUE to LLVM's waitcnt pass, so unlike
    /// [`Node::StoreVecAt`] an `s_barrier` does NOT auto-drain it — HK's escape from the barrier's
    /// implicit `lgkmcnt(0)` (the writes stay in flight until a manual [`Node::SWaitLgkmcnt`] drains
    /// them where the schedule wants). `prev` chains the `sideeffect` writes in program order (an
    /// ordering-only operand — the prior fragment's write — keeping them from hoisting across the
    /// barriers, the silent-stale class §2.1). An EFFECT (a side-effect store, like [`Node::Barrier`]).
    ///
    /// `hk_form` (default `false`) switches the rendered asm to HipKittens' raw-address IR form: an
    /// i32 address operand, an i64 value, the DS immediate offset, and a `~{memory}` clobber.
    DsWriteB64 { base_ptr: TileId, off_bytes: i64, value: TileId, ept: usize, prev: Option<TileId>, hk_form: bool },
    /// Waitcnt-opaque scalar bf16 LDS store used by FA's write-transposed V commit. The raw i32 LDS
    /// address and memory-clobbering inline asm prevent LLVM from conservatively aliasing it with a
    /// younger direct-to-LDS K transfer and strengthening `vmcnt(4)` to `vmcnt(0)`.
    DsWriteB16 { base_ptr: TileId, off_bytes: i64, value: TileId, prev: Option<TileId> },
    /// The **manual LDS drain** (`s_waitcnt lgkmcnt(0)`, gfx942 §5c): a void `asm sideeffect` that
    /// stalls until every outstanding LDS op completes — the EXPOSED drain the [`Node::DsWriteB64`]
    /// commit needs (its writes are waitcnt-opaque, so the RAW `s_barrier` no longer fences them; this
    /// re-establishes the store→barrier→load order). `prev` is the last commit write (an ordering-only
    /// operand pinning the drain after the writes in program order). An EFFECT.
    SWaitLgkmcnt { prev: TileId },
    /// A queue-wide VMEM wait (`s_waitcnt vmcnt(allowed_outstanding)`, gfx942). Zero is a full drain;
    /// a positive threshold leaves that many younger queue entries outstanding. `anchor` is a complete
    /// effect/batch that positions the wait; it does not confer transfer-specific readiness.
    SWaitVmcnt { anchor: TileId, allowed_outstanding: u8 },
    /// **`ptrtoint ptr addrspace(3) → i32`** of an [`Node::LdsPtrAs3`] base — the raw i32 LDS byte
    /// address HipKittens' `ds_read_b64`/`ds_write_b64` asm takes as its `v` address operand (the
    /// oracle's `i32 %262`, not a typed pointer). Feeds the `hk_form` [`Node::DsReadB64`]/
    /// [`Node::DsWriteB64`]. Used only by the HK port.
    PtrToI32 { ptr: TileId },
    /// The **legacy `<4 x i32>` buffer-resource descriptor** (HipKittens' `make_srsrc`, `st.cuh`):
    /// `ptrtoint`→`bitcast i64→<2×i32>`→`shufflevector`→insert w3 = `1114112` (0x110000) + w2 =
    /// `num_bytes` (the range) — the SRD `{ptr, range, 0x110000}` the oracle's
    /// `raw.buffer.load.i128` consumes, distinct from the p0 [`Node::MakeBufferRsrc`]
    /// (`make.buffer.rsrc.p0`) the existing kernels use. Lowers to the multi-instruction SRD chain
    /// producing an `<4 x i32>` value. Used only by the HK port (GAP-1).
    MakeSrsrc { buf: TileId, base_off: TileId, num_bytes: i64 },
    /// ONE **`llvm.amdgcn.raw.buffer.load.i128`** MUBUF load over a legacy `<4 x i32>` SRD
    /// ([`Node::MakeSrsrc`]) — HipKittens' mainloop DRAM prefetch (`load_global_to_register_buffer`).
    /// Reads a 128-bit chunk (`ept` `dtype` elements) at `rsrc[voffset]` bytes (`soffset = 0`),
    /// bitcast `i128 → <ept×dtype>`. `order` are ordering-only anchors (the authoring cluster). The
    /// `.i128` legacy form the oracle emits, NOT the p0 `raw.ptr.buffer.load.v4i32`
    /// ([`Node::BufferLoadRaw`]) the existing kernels use. Used only by the HK port (GAP-1).
    BufferLoadI128 { rsrc: TileId, voffset: TileId, ept: usize, dtype: DType, order: Edges },
    /// **fp32 → bf16 truncation** (`bitcast float→i32`; `lshr 16`; `trunc i16`; `bitcast bfloat`) —
    /// HipKittens' `convertor<bf16,float>` = `(uint16_t)(bits(f) >> 16)`, the truncating (NOT
    /// round-to-nearest) C store the oracle emits. tk2's default f32→bf16 cast is RNE, so the
    /// HK `store` uses this explicit truncation to match HK's IR + numerics.
    Bf16Trunc { val: TileId },
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
    /// A **declarative interleave directive** (`@llvm.amdgcn.sched.group.barrier(mask, size, sync_id)`,
    /// FA-redesign §2.3): tells LLVM's post-RA scheduler to form a group of `size` instructions matching
    /// `mask`, keep the groups in program order, and interleave groups sharing a `group` (SyncID). Unlike
    /// [`Node::SchedFence`] (a *total* boundary LLVM may move nothing across) this is a *positive* ratio —
    /// "1 MFMA, then N VALU/exp, repeat" — the load-bearing primitive for softmax-under-MFMA (HipKittens'
    /// `sched_barrier_pairs`). Emits NO instruction: it survives to ASM only as a `; sched_group_barrier`
    /// comment; what it PRODUCES is the MFMA:VALU/exp interleave. `deps` are liveness/position anchors
    /// ONLY (never a correctness edge) — deleting every one leaves a still-correct kernel (the scheduling/
    /// correctness split). Lowered exactly like [`Node::SchedFence`] (a `Void` `Op::Custom`).
    SchedGroupBarrier { mask: i64, size: i64, group: i64, deps: Edges },
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
    next_scope: u32,
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
    /// A fresh lexical-scope disambiguator.
    pub fn fresh_scope_id(&mut self) -> u32 {
        let s = self.next_scope;
        self.next_scope += 1;
        s
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
            Node::AxisDyn { axis, bound } => Node::AxisDyn { axis, bound: f(bound) },
            Node::RangeAfter { id, trips, deps } => {
                Node::RangeAfter { id, trips, deps: deps.into_iter().map(&mut f).collect() }
            }
            Node::RangeDyn { id, trips, deps } => {
                Node::RangeDyn { id, trips: f(trips), deps: deps.into_iter().map(&mut f).collect() }
            }
            Node::Scope { id, deps } => Node::Scope { id, deps: deps.into_iter().map(&mut f).collect() },
            Node::LdsCol { row, col, cols } => Node::LdsCol { row: f(row), col: f(col), cols },
            Node::LoadGlobal { buf, offset, dtype } => Node::LoadGlobal { buf: f(buf), offset: f(offset), dtype },
            Node::LoadGlobalBounded { buf, offset, bound, alt, dtype } => {
                Node::LoadGlobalBounded { buf: f(buf), offset: f(offset), bound: f(bound), alt: f(alt), dtype }
            }
            Node::EltwiseBinary { op, a, b } => Node::EltwiseBinary { op, a: f(a), b: f(b) },
            Node::SelectLt { lo, hi, then, els } => Node::SelectLt { lo: f(lo), hi: f(hi), then: f(then), els: f(els) },
            Node::Unary { op, x } => Node::Unary { op, x: f(x) },
            Node::DsBpermute { addr, data } => Node::DsBpermute { addr: f(addr), data: f(data) },
            Node::VPerm { hi, lo, selector } => Node::VPerm { hi: f(hi), lo: f(lo), selector },
            Node::VPermAsm { hi, lo, selector } => Node::VPermAsm { hi: f(hi), lo: f(lo), selector },
            Node::OpaqueReadyB64 { val, wait } => Node::OpaqueReadyB64 { val: f(val), wait: f(wait) },
            Node::StoreGlobal { buf, offset, value } => {
                Node::StoreGlobal { buf: f(buf), offset: f(offset), value: f(value) }
            }
            Node::StoreGlobalBounded { buf, offset, bound, value } => {
                Node::StoreGlobalBounded { buf: f(buf), offset: f(offset), bound: f(bound), value: f(value) }
            }
            Node::LoadRegVec { buf, ept, dtype } => Node::LoadRegVec { buf: f(buf), ept, dtype },
            Node::LoadVecAt { buf, base, ept, dtype } => Node::LoadVecAt { buf: f(buf), base: f(base), ept, dtype },
            Node::StoreRegVec { buf, value } => Node::StoreRegVec { buf: f(buf), value: f(value) },
            Node::StoreVecAt { buf, base, value } => Node::StoreVecAt { buf: f(buf), base: f(base), value: f(value) },
            Node::GlobalLoadLdsDword { src, src_offset, dst, dst_offset, deps } => Node::GlobalLoadLdsDword {
                src: f(src),
                src_offset: f(src_offset),
                dst: f(dst),
                dst_offset: f(dst_offset),
                deps: deps.into_iter().map(&mut f).collect(),
            },
            Node::VecExtract { vec, index, dtype } => Node::VecExtract { vec: f(vec), index, dtype },
            Node::VecBuild { elements, dtype } => {
                Node::VecBuild { elements: elements.into_iter().map(&mut f).collect(), dtype }
            }
            Node::Mma { a, b, c, ept, asm } => Node::Mma { a: f(a), b: f(b), c: f(c), ept, asm },
            Node::LdsPtrAs3 { buf, base } => Node::LdsPtrAs3 { buf: f(buf), base: f(base) },
            Node::DsReadB64 { base_ptr, off_bytes, ept, dtype, prev, hk_form } => {
                Node::DsReadB64 { base_ptr: f(base_ptr), off_bytes, ept, dtype, prev: prev.map(&mut f), hk_form }
            }
            Node::MakeBufferRsrc { buf, base_off, num_bytes } => {
                Node::MakeBufferRsrc { buf: f(buf), base_off: f(base_off), num_bytes }
            }
            Node::MakeBufferRsrcDyn { buf, base_off, num_bytes } => {
                Node::MakeBufferRsrcDyn { buf: f(buf), base_off: f(base_off), num_bytes: f(num_bytes) }
            }
            Node::BufferLoadRaw { rsrc, voffset, ept, dtype, order } => Node::BufferLoadRaw {
                rsrc: f(rsrc),
                voffset: f(voffset),
                ept,
                dtype,
                order: order.into_iter().map(&mut f).collect(),
            },
            Node::DsWriteB64 { base_ptr, off_bytes, value, ept, prev, hk_form } => Node::DsWriteB64 {
                base_ptr: f(base_ptr),
                off_bytes,
                value: f(value),
                ept,
                prev: prev.map(&mut f),
                hk_form,
            },
            Node::DsWriteB16 { base_ptr, off_bytes, value, prev } => {
                Node::DsWriteB16 { base_ptr: f(base_ptr), off_bytes, value: f(value), prev: prev.map(&mut f) }
            }
            Node::SWaitLgkmcnt { prev } => Node::SWaitLgkmcnt { prev: f(prev) },
            Node::SWaitVmcnt { anchor, allowed_outstanding } => {
                Node::SWaitVmcnt { anchor: f(anchor), allowed_outstanding }
            }
            Node::PtrToI32 { ptr } => Node::PtrToI32 { ptr: f(ptr) },
            Node::MakeSrsrc { buf, base_off, num_bytes } => {
                Node::MakeSrsrc { buf: f(buf), base_off: f(base_off), num_bytes }
            }
            Node::BufferLoadI128 { rsrc, voffset, ept, dtype, order } => Node::BufferLoadI128 {
                rsrc: f(rsrc),
                voffset: f(voffset),
                ept,
                dtype,
                order: order.into_iter().map(&mut f).collect(),
            },
            Node::Bf16Trunc { val } => Node::Bf16Trunc { val: f(val) },
            Node::Barrier { body, deps } => {
                Node::Barrier { body: f(body), deps: deps.into_iter().map(&mut f).collect() }
            }
            Node::BareBarrier { body, deps } => {
                Node::BareBarrier { body: f(body), deps: deps.into_iter().map(&mut f).collect() }
            }
            Node::SchedFence { mask, deps } => Node::SchedFence { mask, deps: deps.into_iter().map(&mut f).collect() },
            Node::SchedGroupBarrier { mask, size, group, deps } => {
                Node::SchedGroupBarrier { mask, size, group, deps: deps.into_iter().map(&mut f).collect() }
            }
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
            Node::Axis { .. }
            | Node::AxisDyn { .. }
            | Node::Range { .. }
            | Node::RangeAfter { .. }
            | Node::RangeDyn { .. }
            | Node::ScalarParam { .. } => TileMeta::value(SmallVec::new(), DType::Index, Residency::Reg),
            Node::Scope { .. } => TileMeta::effect(),
            Node::Const { dtype, .. } => TileMeta::value(SmallVec::new(), dtype.clone(), Residency::Reg),
            Node::IndexAlu { .. } | Node::LdsCol { .. } => {
                TileMeta::value(SmallVec::new(), DType::Index, Residency::Reg)
            }
            Node::LoadGlobal { dtype, .. } | Node::LoadGlobalBounded { dtype, .. } => {
                TileMeta::value(SmallVec::new(), dtype.clone(), Residency::Reg)
            }
            Node::EltwiseBinary { a, .. } => {
                let dt = self.meta(*a).dtype.clone().unwrap_or(DType::Float32);
                TileMeta::value(SmallVec::new(), dt, Residency::Reg)
            }
            // A predicated select carries the `then` branch's value dtype+shape (the `els` matches it).
            Node::SelectLt { then, .. } => {
                let m = self.meta(*then);
                let dt = m.dtype.clone().unwrap_or(DType::Float32);
                TileMeta::value(m.shape.clone(), dt, Residency::Reg)
            }
            // Unary preserves its operand's dtype+shape (an `ept`-vec stays an `ept`-vec, so
            // a fused `exp2` over a fragment vector keeps its width for the downstream reduce).
            Node::Unary { x, .. } => {
                let m = self.meta(*x);
                let dt = m.dtype.clone().unwrap_or(DType::Float32);
                TileMeta::value(m.shape.clone(), dt, Residency::Reg)
            }
            // The bpermute'd lane value — a Float32 scalar (transported bitcast through i32).
            Node::DsBpermute { .. } => TileMeta::value(SmallVec::new(), DType::Float32, Residency::Reg),
            // The v_perm'd value — a `<2 × bf16>` dword (the two gathered/packed bf16 halves).
            Node::VPerm { .. } | Node::VPermAsm { .. } => {
                TileMeta::value(SmallVec::from_slice(&[2]), DType::BFloat16, Residency::Reg)
            }
            Node::OpaqueReadyB64 { val, .. } => self.meta(*val).clone(),
            // A fragment vector value: an `ept`-lane register vector (bookkeeping only;
            // the lowered UOp carries the true `dtype.vec(ept)`).
            Node::LoadRegVec { dtype, ept, .. } | Node::LoadVecAt { dtype, ept, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*ept]), dtype.clone(), Residency::Reg)
            }
            Node::Mma { ept, .. } => TileMeta::value(SmallVec::from_slice(&[*ept]), DType::Float32, Residency::Reg),
            // The addr(3) base pointer VGPR (Int32-typed, mirroring tk's `base_as3` custom).
            Node::LdsPtrAs3 { .. } => TileMeta::value(SmallVec::new(), DType::Int32, Residency::Reg),
            // The raw i32 LDS byte address (`ptrtoint`) HK's ds_read/ds_write asm takes.
            Node::PtrToI32 { .. } => TileMeta::value(SmallVec::new(), DType::Int32, Residency::Reg),
            // The legacy `<4 x i32>` SRD (bookkeeping shape; the lowered custom names its true type).
            Node::MakeSrsrc { .. } => TileMeta::value(SmallVec::from_slice(&[4]), DType::Int32, Residency::Reg),
            // The `raw.buffer.load.i128` prefetch's `<ept×dtype>` chunk value.
            Node::BufferLoadI128 { dtype, ept, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*ept]), dtype.clone(), Residency::Reg)
            }
            // The fp32→bf16 truncation scalar.
            Node::Bf16Trunc { .. } => TileMeta::value(SmallVec::new(), DType::BFloat16, Residency::Reg),
            // The asm gather's `<ept×bf16>` operand value.
            Node::DsReadB64 { dtype, ept, .. } => {
                TileMeta::value(SmallVec::from_slice(&[*ept]), dtype.clone(), Residency::Reg)
            }
            // The `ptr addrspace(8)` buffer descriptor (pointer-like; the lowered custom names its type).
            Node::MakeBufferRsrc { .. } => TileMeta::value(SmallVec::new(), DType::Int64, Residency::Reg),
            Node::MakeBufferRsrcDyn { .. } => TileMeta::value(SmallVec::new(), DType::Int64, Residency::Reg),
            // The MUBUF prefetch's `<ept×bf16>` value.
            Node::BufferLoadRaw { dtype, ept, .. } => {
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
            | Node::StoreGlobalBounded { .. }
            | Node::StoreRegVec { .. }
            | Node::StoreVecAt { .. }
            | Node::GlobalLoadLdsDword { .. }
            | Node::DsWriteB64 { .. }
            | Node::DsWriteB16 { .. }
            | Node::SWaitLgkmcnt { .. }
            | Node::SWaitVmcnt { .. }
            | Node::Barrier { .. }
            | Node::BareBarrier { .. }
            | Node::SchedFence { .. }
            | Node::SchedGroupBarrier { .. }
            | Node::SetPrio { .. }
            | Node::WaveBarrier { .. }
            | Node::SchedWallMarker
            | Node::End { .. }
            | Node::Sink { .. } => TileMeta::effect(),
        }
    }
}
