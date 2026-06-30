//! Raw gfx942 scheduling/synchronization primitives, injected as `Op::Custom`
//! Void side-effects (K-loop pipeline). These carry no data — each takes a
//! `dep` purely so the linearizer's toposort sequences it *after* the prior
//! cluster and a consumer can `.after([..])` it to sequence the next cluster
//! *after* it (and keep it live through DCE).
//!
//! `s_setprio`/`s_waitcnt` ride `call … asm sideeffect` (a scheduling boundary
//! the AMDGPU backend cannot reorder across); `sched_barrier` rides the
//! `@llvm.amdgcn.sched.barrier` intrinsic (its `declare` is auto-hoisted to the
//! module prefix by the CUSTOM renderer). The `dep` is referenced only for
//! ordering — the emitted text has no `{N}` placeholder, which the strict
//! template validator allows (it only bounds-checks present placeholders).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::UOp;

/// Monotonic counter for unique skip-labels in [`wave_phase_barrier`] (the LLVM `%=`
/// asm-uniquifier does not survive svod's render path — it reaches the assembler
/// verbatim and fails to parse — so we mint the label in Rust).
static WAVE_PHASE_LABEL: AtomicU64 = AtomicU64::new(0);

/// `s_setprio N` (N ∈ 0..=3): raise/lower this wave's issue priority around an
/// MFMA burst so the scheduler keeps the systolic array fed (`GEMM` cluster
/// `s_setprio(1)` before the MFMAs, `s_setprio(0)` after).
pub fn s_setprio(prio: i64, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(smallvec![dep], format!("call void asm sideeffect \"s_setprio {prio}\", \"\"()"), DType::Void)
}

/// `s_waitcnt lgkmcnt(n)`: drain outstanding LDS (`ds_read`/`ds_write`) traffic
/// down to `n` before proceeding — the deferred-wait the register-staged
/// prefetch relies on (issue the next tile's loads, then wait only at the
/// consuming cluster).
pub fn s_waitcnt_lgkmcnt(n: i64, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(smallvec![dep], format!("call void asm sideeffect \"s_waitcnt lgkmcnt({n})\", \"\"()"), DType::Void)
}

/// `@llvm.amdgcn.sched.barrier(mask)`: a hard instruction-scheduling fence.
/// `mask = 0` forbids *any* instruction from moving across it, pinning each
/// cluster's loads/MFMAs/`ds_write` into their program-order region so the
/// pipeline structure survives the AMDGPU machine scheduler.
pub fn sched_barrier(mask: i64, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![dep],
        format!(
            "declare void @llvm.amdgcn.sched.barrier(i32)\n\
             call void @llvm.amdgcn.sched.barrier(i32 {mask})"
        ),
        DType::Void,
    )
}

/// `@llvm.amdgcn.iglp.opt(mode)`: invoke the AMDGPU machine scheduler's canned
/// "initiate-and-group-level-parallelism" pipeline over the loop region
/// *following* this intrinsic. `mode = 0` is the MFMA/memory interleave (overlaps
/// the next tile's global load under the current tile's MFMAs); `mode = 1/2` are
/// attention variants. Place once at the K-loop top. It must NOT share a loop
/// region with [`sched_barrier`] (the two scheduling mutators fight). `dep` orders
/// it after the loop counter so it lands at the top of the loop body.
pub fn iglp_opt(mode: i64, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![dep],
        format!(
            "declare void @llvm.amdgcn.iglp.opt(i32)\n\
             call void @llvm.amdgcn.iglp.opt(i32 {mode})"
        ),
        DType::Void,
    )
}

/// HK's wave-phase ping-pong: a **wave-uniform conditional `s_barrier`** — only the
/// waves whose `warp_row == eq` execute the barrier (`if(warp_row==N) s_barrier`). The
/// wave-uniform `warp_row` (a VGPR-derived value) is `v_readfirstlane`'d into a scalar,
/// compared (`s_cmp_eq_u32`), and the `s_barrier` is skipped (`s_cbranch_scc0`) on
/// mismatch. Placed once in the prologue (`eq = 1`) and once in the epilogue (`eq = 0`),
/// it offsets warp-row 1 by one barrier going into the loop and rebalances it after — so
/// during the loop the two warp-rows are one cluster-barrier out of phase, and one row's
/// MFMAs overlap the other row's memory clusters (the ping-pong HK relies on).
///
/// A typed (`i32`) custom — the readfirstlane scratch SGPR is its (unused) output; the
/// effect is the conditional barrier. `dep` orders it (referenced only for sequencing).
///
/// # Safety / placement
/// The per-wave `s_barrier` execution count MUST stay balanced across the two warp-rows
/// (warp-row 1's extra in the prologue is matched by warp-row 0's in the epilogue) — an
/// unbalanced count **deadlocks the workgroup**. Place OUTSIDE the (clang-`-O3`-unrolled)
/// loop body: the skip label is uniquified per *construction*, not per unrolled copy.
pub fn wave_phase_barrier(warp_row: Arc<UOp>, eq: i64, dep: Arc<UOp>) -> Arc<UOp> {
    let uid = WAVE_PHASE_LABEL.fetch_add(1, Ordering::Relaxed);
    let label = format!(".Lwpb{uid}");
    // `warp_row` is an index (i64); the `v`/`s` 32-bit asm operand needs i32 (the value is
    // 0/1, so the truncation is exact).
    let warp_row = warp_row.cast(DType::Int32);
    UOp::custom(
        smallvec![warp_row, dep],
        // `{{{{scc}}}}` → `format!` → `{{scc}}` (svod template) → rendered `~{scc}` (the
        // LLVM SCC clobber); `{{0}}` → `{0}` → the `warp_row` operand (dep 0).
        format!(
            "call i32 asm sideeffect \"v_readfirstlane_b32 $0, $1\\0A\\09\
             s_cmp_eq_u32 $0, {eq}\\0A\\09s_cbranch_scc0 {label}\\0A\\09s_barrier\\0A\\09\
             {label}:\", \"=s,v,~{{{{scc}}}}\"(i32 {{0}})"
        ),
        DType::Int32,
    )
}
