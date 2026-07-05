//! Instruction-scheduling subsystem (the "instruction manager").
//!
//! A *declarative* layer the tile-DSL author opts into per pipeline loop, plus a
//! post-linearization pass that lowers the intent into gfx9 (CDNA) machine
//! scheduling controls. The author marks a loop with a single [`pipeline`] node
//! (anchored to the loop counter, as the prior `iglp_opt` was); the pass
//! [`apply_pipeline_scheduling`] finds it in the linearized stream and splices the
//! controls in *by instruction class* — no data-dependency threading, so the
//! brittle "control mis-ordered past its consumer" failure mode cannot occur.
//!
//! The intent is GPU-agnostic; only this lowering table is gfx9-specific and it
//! no-ops on every other target.
//!
//! ## Lowering recipes
//!
//! - [`SchedKind::Gemm`] → `@llvm.amdgcn.iglp.opt(0)`: delegate the load/MFMA
//!   interleave to the backend. In svod's *dataflow* model — where global loads
//!   are scheduled by LLVM, not hand-placed — this beats the hand-built
//!   `s_setprio`/`sched.barrier` scaffold HipKittens uses. Measured on gfx942
//!   (`build_matmul_db`): per-MFMA `sched.barrier(0)` fences *regress* to 0.6–0.9×
//!   of the plain kernel (they pin the very load/MFMA overlap the double buffer
//!   exists to create), and `s_setprio` brackets (with or without iglp) do not
//!   reach iglp-alone — HipKittens' fences only pay off because they protect a
//!   *manual* load schedule, which a dataflow DSL does not emit.
//! - [`SchedKind::Attention`] → the MFMA/softmax interleave comb (Stage 2): the one
//!   place manual scheduling has structural reason to beat iglp, because iglp's
//!   generic mode is blind to the online-softmax work threaded between the QKᵀ and
//!   A·V matmuls. Until that lands, attention also delegates to iglp.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{Op, UOp};

use crate::llvm::common::LlvmTarget;

/// Sentinel prefix identifying a [`pipeline`] marker in an `Op::Custom` body. It is
/// a valid LLVM line comment, so an un-lowered marker (non-CDNA target, or the pass
/// disabled) is an inert no-op rather than malformed IR.
const PIPELINE_PREFIX: &str = "; svod.sched.pipeline";

/// The compute pattern of a marked loop — selects the lowering recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedKind {
    /// Pure MFMA accumulation (matmul).
    Gemm,
    /// MFMA + online softmax (flash attention).
    Attention,
}

impl SchedKind {
    fn tag(self) -> &'static str {
        match self {
            SchedKind::Gemm => "gemm",
            SchedKind::Attention => "attention",
        }
    }

    fn parse(rest: &str) -> Self {
        if rest.contains("attention") { SchedKind::Attention } else { SchedKind::Gemm }
    }
}

/// Mark the loop whose counter is `dep` as a scheduled compute pipeline of `kind`.
/// Place once at the loop top, threaded through the in-loop buffers the body reads
/// (as the prior `iglp_opt` was) so it stays loop-scoped and live. The
/// [`apply_pipeline_scheduling`] pass lowers it on gfx9 (CDNA) targets and ignores
/// it everywhere else.
pub fn pipeline(kind: SchedKind, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(smallvec![dep], format!("{PIPELINE_PREFIX} kind={}", kind.tag()), DType::Void)
}

/// `@llvm.amdgcn.iglp.opt(mode)`: delegate the MFMA/memory interleave to the AMDGPU
/// machine scheduler's canned pipeline (`mode = 0` = GEMM). The dataflow model's
/// natural lever — it beats hand-placed fences when there is no manual load
/// schedule to protect.
fn iglp_opt(mode: i64) -> Arc<UOp> {
    UOp::custom(
        smallvec![],
        format!(
            "declare void @llvm.amdgcn.iglp.opt(i32)\n\
             call void @llvm.amdgcn.iglp.opt(i32 {mode})"
        ),
        DType::Void,
    )
}

/// `s_setprio N`: raise (`1`) / lower (`0`) this wave's issue priority. Reserved for
/// the Stage 2 attention comb (HipKittens brackets its FA compute clusters).
#[allow(dead_code)]
fn setprio(prio: i64) -> Arc<UOp> {
    UOp::custom(smallvec![], format!("call void asm sideeffect \"s_setprio {prio}\", \"\"()"), DType::Void)
}

/// `@llvm.amdgcn.sched.barrier(mask)`: a hard fence the machine scheduler may not
/// move any instruction across. Harmful for GEMM only when *value-anchored* (it floats
/// into the prefetch and pins the load/MFMA overlap); placed **positionally** right after
/// each `s_barrier` (see [`wall_after_barriers`]) it reproduces HK's cluster wall lattice.
fn sched_barrier(mask: i64) -> Arc<UOp> {
    UOp::custom(
        smallvec![],
        format!(
            "declare void @llvm.amdgcn.sched.barrier(i32)\n\
             call void @llvm.amdgcn.sched.barrier(i32 {mask})"
        ),
        DType::Void,
    )
}

/// Recognize a [`pipeline`] marker node and recover its kind.
fn marker_kind(node: &Arc<UOp>) -> Option<SchedKind> {
    match node.op() {
        Op::Custom { code, .. } => code.strip_prefix(PIPELINE_PREFIX).map(SchedKind::parse),
        _ => None,
    }
}

/// Lower [`pipeline`] markers in a linearized instruction stream into gfx9 machine
/// scheduling controls. A no-op unless the target is a CDNA (MFMA-class) AMD GPU
/// *and* the stream carries a marker, so it is safe to run over every kernel.
pub fn apply_pipeline_scheduling(nodes: Vec<Arc<UOp>>, target: LlvmTarget) -> Vec<Arc<UOp>> {
    if !target.amd_arch().is_some_and(|a| a.is_cdna()) {
        return nodes;
    }
    if !nodes.iter().any(|n| marker_kind(n).is_some()) {
        return nodes;
    }

    // Splice `iglp_opt(0)` right after EVERY marker — a kernel may pipeline more
    // than one loop, and each marked loop needs its own scheduling control (an
    // un-lowered marker is an inert comment). Both kinds currently delegate to
    // `iglp_opt(0)` (Stage 2 swaps the attention path for the softmax/MFMA comb).
    let mut out: Vec<Arc<UOp>> = Vec::with_capacity(nodes.len() + 1);
    for node in nodes {
        let is_marker = marker_kind(&node).is_some();
        out.push(node);
        if is_marker {
            out.push(iglp_opt(0));
        }
    }
    out
}

/// Sentinel a kernel carries to opt into HK-style barrier walling ([`wall_after_barriers`]).
/// A valid LLVM line comment, so an un-lowered marker is inert.
const WALL_PREFIX: &str = "; svod.sched.wall_barriers";

/// The opt-in marker: emit once (kept live so it survives DCE) to request that
/// [`wall_after_barriers`] pair every `s_barrier` with a `sched.barrier(0)`.
pub fn wall_marker() -> Arc<UOp> {
    UOp::custom(smallvec![], WALL_PREFIX.to_string(), DType::Void)
}

fn is_wall_marker(node: &Arc<UOp>) -> bool {
    matches!(node.op(), Op::Custom { code, .. } if code.starts_with(WALL_PREFIX))
}

/// Splice `@llvm.amdgcn.sched.barrier(0)` immediately after EVERY `s_barrier` in the
/// **linearized** stream — HK's 1:1 `s_barrier`↔`sched.barrier(0)` wall lattice. This is
/// *positional* (a fixed stream index), so unlike a value-anchored `sched.barrier` it cannot
/// float into the prefetch and force early `vmcnt` drains — a wall only forbids instruction
/// *motion* across each cluster boundary while the load latency still overlaps naturally.
/// Opt-in (only a stream carrying a [`wall_marker`]) and CDNA-only, so safe over every kernel.
/// tk2's `WaveBarrier` is inline asm (not `Op::Barrier`), so the prologue/epilogue wave-phase
/// barriers are correctly left unwalled — only the hot-loop cluster/RAW/WAR barriers get walls.
pub fn wall_after_barriers(nodes: Vec<Arc<UOp>>, target: LlvmTarget) -> Vec<Arc<UOp>> {
    if !target.amd_arch().is_some_and(|a| a.is_cdna()) {
        return nodes;
    }
    if !nodes.iter().any(is_wall_marker) {
        return nodes;
    }
    let mut out: Vec<Arc<UOp>> = Vec::with_capacity(nodes.len() + 16);
    for node in nodes {
        if is_wall_marker(&node) {
            continue; // drop the sentinel so it does not render as a stray comment
        }
        let is_barrier = matches!(node.op(), Op::Barrier { .. });
        out.push(node);
        if is_barrier {
            out.push(sched_barrier(0));
        }
    }
    out
}
