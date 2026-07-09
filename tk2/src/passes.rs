//! The `.apply`-able refinement passes over the tile-IR (DESIGN.md §2.4/§5b), as contract-checked
//! [`Pass`](crate::pass::Pass)es through the runner. Both kept kernels compose them:
//! `matmul_lds_kblock_mw_pipe2(cfg).apply(VectorizePass).apply(SwizzlePass)`.
//!
//! 1. [`VectorizePass`] (Band::Tiling) — fuse each fragment's `ept` contiguous scalar LDS gather
//!    loads into ONE `<ept×bf16>` vector load ([`Node::LoadVecAt`] → `ds_read_b64`).
//! 2. [`SwizzlePass`] (Band::MemoryPlacement) — materialise every [`Node::LdsCol`] layout hole to the
//!    HK/CK bank-conflict-avoiding XOR `col ^ delta(row)`.
//!
//! Both are **semantics-preserving by construction** (§2.1/§2.6) and commute: the fused load's base
//! *is* the `inner = 0` element's offset (whose `LdsCol` the swizzle relocates chunk-wise), and the
//! swizzle is a bijection — so all four combos (base/vec/sw/vec+sw) stay bit-exact.

use std::collections::{HashMap, HashSet};

use svod_dtype::DType;

use crate::ir::{Edges, IndexOp, Node, Scalar, TileId, TileIr};
use crate::pass::{Band, Fold, Pass, PassError, fold};

// ── shared analysis helpers ──────────────────────────────────────────────────

/// Every node id reachable from `root` (its dependency cone).
pub(crate) fn reachable(ir: &TileIr, root: TileId) -> Vec<TileId> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        order.push(id);
        for c in TileIr::children(ir.node(id)) {
            stack.push(c);
        }
    }
    order
}

/// True if `id` is an integer `Const`.
fn int_const(ir: &TileIr, id: TileId) -> Option<i64> {
    match ir.node(id) {
        Node::Const { scalar: Scalar::Int(v), .. } => Some(*v),
        _ => None,
    }
}

// ============================================================================
// SwizzlePass — the first `.apply`-able layout refinement (DESIGN §2.4 / §5b)
// ============================================================================

/// Materialise every [`Node::LdsCol`] to the HK/CK bank-conflict-avoiding XOR
/// `col ^ delta(row)`, turning the flat LDS layout into the swizzled one. The base
/// kernel emits `LdsCol` as the identity (a composable hole); `.apply(SwizzlePass)`
/// fills it — so the swizzle is a **layout refinement pass**, not hand-woven addressing.
/// Single-subtile bf16 formula (`cols ∈ {16,32,64}`), verified bit-exact + cross-checked
/// against HK `st.cuh` / CK `make_xor_transform` (§5b).
pub struct SwizzlePass;

impl Pass for SwizzlePass {
    fn name(&self) -> &str {
        "lds_bank_swizzle"
    }
    fn band(&self) -> Band {
        Band::MemoryPlacement
    }
    /// Postcondition: no `LdsCol` remains — every layout point is materialised.
    fn ensures(&self, ir: &TileIr, root: TileId) -> bool {
        reachable(ir, root).into_iter().all(|id| !matches!(ir.node(id), Node::LdsCol { .. }))
    }
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        Ok(fold(&mut Swizzle, ir, root))
    }
}

/// The nanopass folder: `LdsCol{row,col,cols}` → `col ^ (((row%16)·cols·2 >> 7 << 3) >> 1)`.
struct Swizzle;

impl Fold for Swizzle {
    fn fold_node(&mut self, ir: &mut TileIr, node: Node) -> TileId {
        let Node::LdsCol { row, col, cols } = node else {
            return ir.intern(node);
        };
        let konst = |ir: &mut TileIr, v: i64| ir.intern(Node::Const { scalar: Scalar::Int(v), dtype: DType::Index });
        let alu = |ir: &mut TileIr, op, a, b| ir.intern(Node::IndexAlu { op, a, b });
        let c16 = konst(ir, 16);
        let r16 = alu(ir, IndexOp::Mod, row, c16);
        let sb2 = konst(ir, (cols * 2) as i64); // swizzle_bytes = cols · itemsize (bf16)
        let t = alu(ir, IndexOp::Mul, r16, sb2);
        let c7 = konst(ir, 7);
        let t = alu(ir, IndexOp::Shr, t, c7);
        let c3 = konst(ir, 3);
        let t = alu(ir, IndexOp::Shl, t, c3);
        let c1 = konst(ir, 1);
        let delta = alu(ir, IndexOp::Shr, t, c1); // >> log2(itemsize)
        alu(ir, IndexOp::Xor, col, delta)
    }
}

// ============================================================================
// VectorizePass — fuse each scalar gather run into one ds_read_b64 (DESIGN §5b)
// ============================================================================

/// Fuse each fragment's `ept` **contiguous scalar** LDS gather loads into ONE `<ept×bf16>`
/// vector load ([`Node::LoadVecAt`] → `ds_read_b64`) + a [`Node::StoreRegVec`]. The base
/// kernel emits the scalar run (`LdsView::gather`, the movement layer); this is the
/// composable gather refinement (`.apply(VectorizePass)`), the counterpart to
/// [`SwizzlePass`] — the two commute (the fused load's base *is* the `inner = 0` element's
/// offset, whose `LdsCol` the swizzle relocates chunk-wise; §5b). Fills stay builder-
/// structural (the B transpose isn't a fusible run — a scalar B-fill needs no transpose).
pub struct VectorizePass;

/// A fusible gather: a bf16 `DefineFrag` written by exactly `ept` `StoreGlobal`s at const
/// offsets `0..ept` whose values are scalar `LoadGlobal`s. Returns each such frag → `ept`.
fn gather_frags(ir: &TileIr, root: TileId) -> HashMap<TileId, usize> {
    let mut writers: HashMap<TileId, (usize, Vec<i64>)> = HashMap::new();
    for id in reachable(ir, root) {
        let Node::StoreGlobal { buf, offset, value } = ir.node(id) else { continue };
        let Node::DefineFrag { dtype, frag, .. } = ir.node(*buf) else { continue };
        if *dtype != DType::BFloat16 || !matches!(ir.node(*value), Node::LoadGlobal { .. }) {
            continue;
        }
        let Some(e) = int_const(ir, *offset) else { continue };
        let w = writers.entry(*buf).or_insert((frag.ept, Vec::new()));
        w.1.push(e);
    }
    writers
        .into_iter()
        .filter_map(|(f, (ept, mut es))| {
            es.sort_unstable();
            (es.len() == ept && es.iter().enumerate().all(|(i, &e)| e == i as i64)).then_some((f, ept))
        })
        .collect()
}

impl Pass for VectorizePass {
    fn name(&self) -> &str {
        "vectorize_gathers"
    }
    fn band(&self) -> Band {
        Band::Tiling
    }
    /// Postcondition: no fusible scalar gather run survives — every one is now a vector load.
    fn ensures(&self, ir: &TileIr, root: TileId) -> bool {
        gather_frags(ir, root).is_empty()
    }
    fn apply(&self, ir: &mut TileIr, root: TileId) -> Result<TileId, PassError> {
        let valid = gather_frags(ir, root);
        Ok(fold(&mut VecGather { valid, fused: HashMap::new() }, ir, root))
    }
}

/// The folder: replace each gather group's `ept` scalar stores with one vector store, and
/// dedup the edge lists that referenced them (they now all point at the fused store).
struct VecGather {
    valid: HashMap<TileId, usize>,
    fused: HashMap<TileId, TileId>,
}

/// Order-preserving de-duplication of an edge list (the `ept`→1 collapse leaves duplicates).
fn dedup(edges: &Edges) -> Edges {
    let mut seen = HashSet::new();
    edges.iter().copied().filter(|e| seen.insert(*e)).collect()
}

impl Fold for VecGather {
    fn fold_node(&mut self, ir: &mut TileIr, node: Node) -> TileId {
        match node {
            // A gather store: the `inner = 0` leader synthesises the fused vector load/store
            // (its offset is the run start); later elements collapse onto it. Leaders fold
            // first (emitted first ⇒ smaller id ⇒ visited first by the ascending-id driver).
            Node::StoreGlobal { buf, offset, value } if self.valid.contains_key(&buf) => {
                let ept = self.valid[&buf];
                if int_const(ir, offset) == Some(0) {
                    let Node::LoadGlobal { buf: lds, offset: base, dtype } = ir.node(value).clone() else {
                        unreachable!("gather store value is a scalar LoadGlobal")
                    };
                    let vec = ir.intern(Node::LoadVecAt { buf: lds, base, ept, dtype });
                    let fused = ir.intern(Node::StoreRegVec { buf, value: vec });
                    self.fused.insert(buf, fused);
                    fused
                } else {
                    self.fused[&buf]
                }
            }
            // Edge lists that routed through the collapsed scalar stores: dedup the now-
            // coincident fused ids (the barrier drops its body from its own fence set).
            Node::After { val, deps } => ir.intern(Node::After { val, deps: dedup(&deps) }),
            Node::Barrier { body, deps } => {
                let deps = dedup(&deps).into_iter().filter(|&d| d != body).collect();
                ir.intern(Node::Barrier { body, deps })
            }
            Node::BareBarrier { body, deps } => {
                let deps = dedup(&deps).into_iter().filter(|&d| d != body).collect();
                ir.intern(Node::BareBarrier { body, deps })
            }
            Node::Sink { roots } => ir.intern(Node::Sink { roots: dedup(&roots) }),
            // Schedule controls anchored on the collapsed scalar gather stores: dedup the now-
            // coincident fused ids (§5c cluster fences ride the same edges the gather stores did).
            Node::SchedFence { mask, deps } => ir.intern(Node::SchedFence { mask, deps: dedup(&deps) }),
            Node::SetPrio { level, deps } => ir.intern(Node::SetPrio { level, deps: dedup(&deps) }),
            Node::WaveBarrier { eq, deps } => ir.intern(Node::WaveBarrier { eq, deps: dedup(&deps) }),
            Node::SchedWallMarker => ir.intern(Node::SchedWallMarker),
            other => ir.intern(other),
        }
    }
}
