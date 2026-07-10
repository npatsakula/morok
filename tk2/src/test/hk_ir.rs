//! Headless LLVM-IR verification of the [`crate::hk`] leaf helpers against HipKittens' `micro_tk`
//! oracle (`hk-micro_tk.ll`, gfx942). Each test builds a minimal driver `Program`, renders the AMD
//! IR via [`render_amd_ir`] (no device / clang / KFD), and asserts the presence + count of the
//! oracle's exact intrinsic / asm form AND the absence of the wrong form. Tests T1–T9 mirror the
//! per-helper build+verify plan.
//!
//! GAP-3 note (T9): the headless renderer is alloca-based (it defers register promotion to clang
//! `-O3`), so the accumulator's loop-carried `<4 x float>` **phi** materialises only after clang's
//! SROA/mem2reg — which the oracle `.ll` (post-`-O3`) shows and the device bit-exact test proves. T9
//! therefore asserts the SROA-promotable *precondition* headlessly (one real K-loop; a constant-index
//! `<4 x float>` accumulator round-trip; a single-store `zero`-init, NOT an init range-loop).

use svod_dtype::AmdArch;

use crate::build::{BF16, Builder, Effect, F32};
use crate::hk::memory::{G_load, load, load_global_to_register_buffer, store, store_register_buffer_to_shared};
use crate::hk::mma::{mma_ABt, zero};
use crate::hk::swizzle;
use crate::hk::sync::{s_barrier, s_setprio, s_waitcnt_lgkmcnt, s_waitcnt_vmcnt, sched_barrier};
use crate::hk::types::{rt_bf, rt_fl, st_bf};
use crate::kernels::Program;
use crate::launch::render_amd_ir;
use crate::passes::SwizzlePass;

// ── shared harness ───────────────────────────────────────────────────────────

/// Finish `b` over `roots`, (optionally) apply [`SwizzlePass`], and render the gfx942 AMD IR.
fn render(b: Builder, roots: &[Effect], swizzle: bool) -> String {
    let (ir, sink) = b.finish(roots);
    let mut p = Program { ir, sink, name: "hk_ir_test".into() };
    if swizzle {
        p = p.apply(SwizzlePass);
    }
    render_amd_ir(&p, AmdArch::Gfx942).expect("hk driver renders to gfx942 IR")
}

/// Count the CALL sites of `pat` (excluding the hoisted `declare` line).
fn count_calls(ir: &str, pat: &str) -> usize {
    ir.lines().filter(|l| l.contains(pat) && !l.trim_start().starts_with("declare")).count()
}

/// True iff some rendered line is a `ds_read_b64`/`ds_write_b64` asm carrying `needle`.
fn ds_line_has(ir: &str, mnemonic: &str, needle: &str) -> bool {
    ir.lines().any(|l| l.contains(mnemonic) && l.contains(needle))
}

// ── T1: mma_ABt / mfma161616 — the intrinsic `.1k` MFMA (GAP-2) ──────────────

#[test]
fn t1_mma_abt_emits_16_mfma_1k() {
    let mut b = Builder::new("t1");
    // rt_bf<64,16> = 4 A-fragments + 4 B-fragments; rt_fl<64,64> = 16 accumulators.
    let a = rt_bf::new(&mut b, 64, 16);
    let bt = rt_bf::new(&mut b, 64, 16);
    let c = rt_fl::new(&mut b, 64, 64);
    let a_ops: Vec<_> = a.frags.iter().map(|&f| b.load_frag_vec(f)).collect();
    let bt_ops: Vec<_> = bt.frags.iter().map(|&f| b.load_frag_vec(f)).collect();
    let c_ops: Vec<_> = c.frags.iter().map(|&f| b.load_frag_vec(f)).collect();
    let outs = mma_ABt(&mut b, &a_ops, &bt_ops, &c_ops, 4, 4);
    let roots: Vec<Effect> = outs.iter().zip(c.frags.iter()).map(|(&v, &f)| b.store_frag_vec(f, v)).collect();
    let ir = render(b, &roots, false);

    // One mma_ABt over the 4×4 output grid = exactly 16 MFMA calls, all the `.1k` intrinsic.
    assert_eq!(count_calls(&ir, "@llvm.amdgcn.mfma.f32.16x16x16bf16.1k("), 16, "mma_ABt must emit 16 `.1k` MFMAs");
    // Every `mfma` occurrence is the `.1k` variant (no non-`.1k` form) and none is the asm form.
    assert_eq!(ir.matches("mfma").count(), ir.matches("mfma.f32.16x16x16bf16.1k").count(), "only the `.1k` MFMA");
    assert!(!ir.contains("v_mfma"), "mma_ABt uses the intrinsic, not the inline-asm MFMA");
    // Operand packing + immediates match the oracle: `<4 x i16>,<4 x i16>,<4 x float>, i32 0,i32 0,i32 0`.
    let call = ir.lines().find(|l| l.contains("mfma.f32.16x16x16bf16.1k(") && l.contains("call")).expect("a call");
    assert!(call.contains("<4 x i16>") && call.contains("<4 x float>"), "operands <4 x i16>/<4 x float>");
    assert!(call.contains("i32 0, i32 0, i32 0)"), "cbsz/abid/blgp = 0");
}

// ── T2: swizzle idx() — pure numeric golden vs SwizzlePass (no IR) ────────────

proptest::proptest! {
    #[test]
    fn t2_swizzle_matches_tk2_swizzlepass(r in 0u32..256, c in 0u32..64) {
        // HK's `st::idx` returns a swizzled BYTE offset; tk2's `SwizzlePass` a swizzled ELEMENT
        // offset. They are the same map in element space (`hk_byte / sizeof(bf16) == tk2_offset`).
        let hk_byte = swizzle::idx(0, r, c, 256, 64);
        proptest::prop_assert_eq!(hk_byte % 2, 0);
        proptest::prop_assert_eq!(hk_byte / 2, swizzle::tk2_offset(r, c, 64));
    }
}

#[test]
fn t2_swizzle_delta_is_row_times_four() {
    // For st_bf<256,64>: delta(row) = (row % 16) · 4 (the closed form the equivalence rests on).
    for row in 0u32..64 {
        assert_eq!(swizzle::tk2_delta(row, 64), (row % 16) * 4);
    }
}

// ── T3: load (shared→register ds_read gather), HK form ───────────────────────

#[test]
fn t3_load_emits_hk_ds_read() {
    let mut b = Builder::new("t3");
    let st = st_bf::new(&mut b, 256, 64);
    let sub = st.subtile_inplace(64, 16, 0, 0);
    let tid = b.block_axis(512);
    let mut rt = rt_bf::new(&mut b, 64, 16);
    let ops = load(&mut b, &mut rt, sub, tid, &[]);
    let scratch = b.define_local::<BF16>(64);
    let roots: Vec<Effect> = ops
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let off = b.idx_const((i * 4) as i64);
            b.store_lds_vec(scratch, off, v)
        })
        .collect();
    let ir = render(b, &roots, true);

    // rt_bf<64,16> = 4 fragments → 4 `ds_read_b64` in HK's literal form.
    assert_eq!(count_calls(&ir, "ds_read_b64 $0, $1 offset:$2"), 4, "4 HK-form ds_read gathers");
    assert!(ir.contains("=v,v,i,~{memory}"), "HK ds_read constraints (i32 addr + `i` offset + memory)");
    // ONE base VGPR + per-fragment offset immediates 0/2048/4096/6144 (= i·EDGE·64·2).
    for off in ["i64 0)", "i64 2048)", "i64 4096)", "i64 6144)"] {
        assert!(ds_line_has(&ir, "ds_read_b64", off), "ds_read offset {off}");
    }
    // NOT the default (clustered-kernel) `ptr addrspace(3)` + literal-offset form.
    assert!(!ir.contains("ds_read_b64 $0, $1 offset:0\", \"=v,v\""), "no default (non-HK) ds_read form");
}

// ── T4: store_register_buffer_to_shared (ds_write commit), HK form ───────────

#[test]
fn t4_commit_emits_hk_ds_write() {
    let mut b = Builder::new("t4");
    let dst = st_bf::new(&mut b, 256, 64);
    let tid = b.block_axis(512);
    let src = b.global::<BF16>(8192 * 256);
    let base = b.idx_const(0);
    let (v0, v1) = (b.idx_const(0), b.idx_const(128));
    let chunks = load_global_to_register_buffer(&mut b, src, base, &[v0, v1], &[]);
    let stores = store_register_buffer_to_shared(&mut b, dst, &chunks, tid, None);
    let ir = render(b, &stores, true);

    // 2 float4 chunks × 2 halves = 4 HK-form `ds_write_b64`.
    assert_eq!(count_calls(&ir, "ds_write_b64 $0, $1"), 4, "4 HK-form ds_write commits");
    assert!(ir.contains("\"v,v,~{memory}\""), "HK ds_write constraints (i32 addr + i64 value + memory)");
    // HK folds the offset into the address (NO `offset:` immediate) and writes an i64 value.
    assert!(!ir.contains("ds_write_b64 $0, $1 offset:"), "HK ds_write has no `offset:` immediate");
    assert!(ds_line_has(&ir, "ds_write_b64", "i64"), "HK ds_write value is i64");
}

// ── T5: make_srsrc + load_global_to_register_buffer (raw.buffer.load.i128) — GAP-1

#[test]
fn t5_buffer_load_emits_legacy_i128_srd() {
    let mut b = Builder::new("t5");
    let a = b.global::<BF16>(8192 * 256);
    let base = b.idx_const(0);
    let (v0, v1) = (b.idx_const(0), b.idx_const(128));
    let chunks = load_global_to_register_buffer(&mut b, a, base, &[v0, v1], &[]);
    let scratch = b.define_local::<BF16>(16);
    let roots: Vec<Effect> = chunks
        .iter()
        .enumerate()
        .map(|(i, &ch)| {
            let off = b.idx_const((i * 8) as i64);
            b.store_lds_vec(scratch, off, ch)
        })
        .collect();
    let ir = render(b, &roots, false);

    // Two chunks → two `raw.buffer.load.i128` over the legacy `<4 x i32>` SRD.
    assert_eq!(count_calls(&ir, "@llvm.amdgcn.raw.buffer.load.i128("), 2, "2 legacy i128 buffer loads");
    assert!(ir.contains("<4 x i32>"), "legacy `<4 x i32>` SRD operand");
    assert!(ir.contains("i32 1114112, i64 3"), "SRD config word 0x110000 at w3");
    assert!(ir.contains("shufflevector <2 x i32>"), "SRD base-ptr shuffle to <4 x i32>");
    assert!(ir.contains("bitcast i128"), "i128 chunk bitcast to <8 x bfloat>");
    // NOT the p0 form the existing pipe2/clustered kernels emit.
    assert!(!ir.contains("make.buffer.rsrc.p0"), "hk buffer load must NOT emit the p0 SRD");
    assert!(!ir.contains("raw.ptr.buffer.load"), "hk buffer load must NOT emit the p0 buffer load");
}

// ── T6: sync leaves ──────────────────────────────────────────────────────────

#[test]
fn t6_sync_leaves_render_each_intrinsic() {
    let mut b = Builder::new("t6");
    // Seed the chain with a real store effect (an LDS commit).
    let a = b.global::<BF16>(8192 * 256);
    let base = b.idx_const(0);
    let v0 = b.idx_const(0);
    let chunk = load_global_to_register_buffer(&mut b, a, base, &[v0], &[]);
    let scratch = b.define_local::<BF16>(8);
    let seed = b.store_lds_vec(scratch, base, chunk[0]);

    let bar = s_barrier(&mut b, seed, &[]);
    let p1 = s_setprio(&mut b, 1, &[bar.dep()]);
    let p0 = s_setprio(&mut b, 0, &[p1.dep()]);
    let sb = sched_barrier(&mut b, &[p0.dep()]);
    let wl = s_waitcnt_lgkmcnt(&mut b, sb.dep());
    let wv = s_waitcnt_vmcnt(&mut b, wl.dep());
    let ir = render(b, &[wv], false);

    assert!(ir.contains("@llvm.amdgcn.s.barrier()"), "s_barrier → s.barrier()");
    assert!(ir.contains("@llvm.amdgcn.s.setprio(i16 1)"), "s_setprio(1)");
    assert!(ir.contains("@llvm.amdgcn.s.setprio(i16 0)"), "s_setprio(0)");
    assert!(ir.contains("@llvm.amdgcn.sched.barrier(i32 0)"), "sched_barrier(0)");
    assert!(ir.contains("s_waitcnt lgkmcnt(0)"), "s_waitcnt lgkmcnt(0)");
    assert!(ir.contains("s_waitcnt vmcnt(0)"), "s_waitcnt vmcnt(0)");
}

// ── T7: G::load (cooperative global→shared) ──────────────────────────────────

#[test]
fn t7_group_load_renders_the_cooperative_chain() {
    let mut b = Builder::new("t7");
    let dst = st_bf::new(&mut b, 256, 64);
    let src = b.global::<BF16>(8192 * 256);
    let tid = b.block_axis(512);
    let base = b.idx_const(0);
    let v0 = b.idx_const(0);
    let drained = G_load(&mut b, dst, src, base, &[v0], tid, &[]);
    let ir = render(b, &[drained], true);

    // buffer_load (global→VGPR) → vmcnt(0) → ds_write (VGPR→LDS) → lgkmcnt(0).
    assert_eq!(count_calls(&ir, "@llvm.amdgcn.raw.buffer.load.i128("), 1, "one global load chunk");
    assert!(ir.contains("s_waitcnt vmcnt(0)"), "VMEM drain before the LDS commit");
    assert_eq!(count_calls(&ir, "ds_write_b64 $0, $1"), 2, "one float4 chunk → 2 ds_write halves");
    assert!(ir.contains("s_waitcnt lgkmcnt(0)"), "LDS drain after the commit");
    // Program order: the vmcnt drain precedes the first ds_write which precedes the lgkmcnt drain.
    let pos = |p: &str| ir.find(p).unwrap_or(usize::MAX);
    assert!(pos("s_waitcnt vmcnt(0)") < pos("ds_write_b64"), "vmcnt before ds_write");
    assert!(pos("ds_write_b64") < pos("s_waitcnt lgkmcnt(0)"), "ds_write before lgkmcnt");
}

// ── T8: store (register→global C, fp32→bf16 truncation) ──────────────────────

#[test]
fn t8_store_truncates_fp32_to_bf16() {
    let mut b = Builder::new("t8");
    let c = b.global::<BF16>(256 * 256);
    let acc = rt_fl::new(&mut b, 16, 16); // one 16×16 fragment → 4 scalar stores
    let tid = b.block_axis(64);
    let base = b.idx_const(0);
    let roots = store(&mut b, c, &acc, base, 256, tid);
    let ir = render(b, &roots, false);

    // HK's `convertor<bf16,float>` = `(uint16_t)(bits(f) >> 16)`: bitcast→lshr 16→trunc i16→bitcast.
    assert_eq!(count_calls(&ir, "lshr i32"), 4, "4 truncating shifts (one per stored element)");
    assert!(ir.contains("bitcast float"), "f32 bits reinterpret");
    assert!(ir.contains("trunc i32") && ir.contains("to i16"), "truncate to the bf16 high half");
    assert!(ir.contains("bitcast i16") && ir.contains("to bfloat"), "reinterpret the i16 as bf16");
    assert!(ir.contains("store bfloat"), "scalar bf16 global store");
    // Truncation, NOT round-to-nearest (svod's default cast): no RNE `+ 0x7fff` rounding term.
    assert!(!ir.contains("32767"), "HK store truncates; no RNE rounding");
}

// ── T9: loop-phi carry — the SROA-promotable accumulator round-trip (GAP-3) ──

#[test]
fn t9_accumulator_is_sroa_promotable_across_the_kloop() {
    let mut b = Builder::new("t9");
    let acc = rt_fl::new(&mut b, 16, 16); // one fp32 accumulator fragment
    let a = rt_bf::new(&mut b, 16, 16);
    let bt = rt_bf::new(&mut b, 16, 16);
    let seed = zero(&mut b, &acc); // ONE constant-index <4×f32> vector store (the loop-carry seed)

    let kr = b.range(2);
    let a_op = b.load_frag_vec(a.frags[0]);
    let b_op = b.load_frag_vec(bt.frags[0]);
    let acc_in = b.load_frag_vec_after(acc.frags[0], &[seed[0].dep(), kr.dep()]);
    let d = mma_ABt(&mut b, &[a_op], &[b_op], &[acc_in], 1, 1);
    let st = b.store_frag_vec(acc.frags[0], d[0]);
    let ended = b.end(st, &[kr]);

    // Post-loop: read the carried accumulator + store one element to global (keeps the loop live).
    let c = b.global::<BF16>(256);
    let acc_final = b.frag_after(acc.frags[0], &[ended.dep()]);
    let off = b.idx_const(0);
    let v = b.load_frag_elem::<F32>(acc_final, off);
    let bf = b.bf16_trunc(v);
    let root = b.store(c, off, bf);
    let ir = render(b, &[root], false);

    // Exactly ONE real hardware loop (the K-loop). `zero` is a single store, NOT an init range-loop
    // — that is what keeps every accumulator access constant-index (SROA-promotable). Counting the
    // `loop_body_N:` label definitions (not the substring, which also appears in `br` targets) and the
    // single conditional latch `br i1` both pin "one loop".
    let loop_defs = ir.lines().filter(|l| l.trim().starts_with("loop_body_") && l.trim().ends_with(':')).count();
    assert_eq!(loop_defs, 1, "exactly one K-loop; zero-init added no loop");
    assert_eq!(ir.matches("br i1").count(), 1, "one loop-latch conditional branch (not an unrolled body)");
    // The accumulator round-trip is a constant-index `<4 x float>` load+store into a fixed alloca —
    // the SROA-promotable form clang `-O3` turns into a loop-carried `<4 x float>` phi (the oracle's
    // form; headless render is pre-SROA so we assert the precondition, not the phi itself).
    assert!(ir.contains("alloca [4 x float]"), "accumulator is a [4 x float] register cell");
    assert!(ir.contains("store <4 x float>"), "constant-index vector write-back of the MFMA result");
    assert!(ir.contains("load <4 x float>") || ir.contains("= load"), "constant-index vector read-back");
    assert_eq!(count_calls(&ir, "@llvm.amdgcn.mfma.f32.16x16x16bf16.1k("), 1, "the K-loop body MFMA");
}

// ── T10: the assembled `micro_tk` kernel — the rolled K-loop CFG + per-body static op counts ──
//
// The full-kernel gate (spec §1/§2): render `micro_tk` (HK's exact 8192³ tiling) and assert the
// rolled K-loop matches HK's oracle `hk-micro_tk-mainloop.ll` — exactly ONE self-loop body block, and
// the per-body static counts. NB the spec §2 prose says "8 setprio(1)+8 setprio(0)"; the ORACLE `.ll`
// (the ground truth) has **4+4** (one bracket per compute cluster C1/C3/C5/C7) and **0** vmcnt in the
// mainloop — this test asserts the oracle's counts, verified by `grep` on `hk-micro_tk-mainloop.ll`.

/// The rendered lines of the single rolled K-loop body — from the `loop_body_N:` label to the next
/// block label. `micro_tk` has no control flow inside the loop, so the body is one basic block.
fn loop_body(ir: &str) -> Vec<&str> {
    let lines: Vec<&str> = ir.lines().collect();
    let is_label = |l: &str| {
        let t = l.trim();
        t.ends_with(':') && t[..t.len() - 1].chars().all(|c| c.is_alphanumeric() || c == '_' || c == '.')
    };
    let start = lines.iter().position(|l| l.trim().starts_with("loop_body_") && is_label(l)).expect("a loop body");
    let end = start + 1 + lines[start + 1..].iter().position(|l| is_label(l)).expect("a block after the loop body");
    lines[start..end].to_vec()
}

#[test]
fn t10_micro_tk_matches_the_oracle_kloop_cfg() {
    let prog = crate::hk::micro_tk(8192, 8192, 8192).apply(SwizzlePass);
    let ir = render_amd_ir(&prog, AmdArch::Gfx942).expect("micro_tk renders to gfx942 IR");

    // ── Exactly ONE rolled K-loop: one `loop_body_N:` label + one conditional latch branch. ──
    let loop_defs = ir.lines().filter(|l| l.trim().starts_with("loop_body_") && l.trim().ends_with(':')).count();
    assert_eq!(loop_defs, 1, "exactly one rolled K-loop body block (NOT unrolled at the K level)");
    assert_eq!(ir.matches("br i1").count(), 1, "exactly one loop-latch conditional branch");

    // ── Per-loop-body static op counts (spec §2, verified against `hk-micro_tk-mainloop.ll`). ──
    let body = loop_body(&ir);
    let n = |pat: &str| body.iter().filter(|l| l.contains(pat)).count();
    assert_eq!(n("raw.buffer.load.i128("), 8, "8 legacy i128 DRAM prefetches (A@C0 + B@C4, 4 chunks each)");
    assert_eq!(n("ds_read_b64 $0, $1 offset:$2"), 48, "48 HK-form ds_read gathers (12 tiles × 4 frags)");
    assert_eq!(n("ds_write_b64 $0, $1"), 16, "16 HK-form ds_write commits (2 tiles × 4 chunks × 2 halves)");
    assert_eq!(n("mfma.f32.16x16x16bf16.1k("), 128, "128 MFMAs (4 compute clusters × 32)");
    assert_eq!(n("@llvm.amdgcn.s.barrier()"), 8, "8 per-cluster s_barrier seals");
    assert_eq!(n("@llvm.amdgcn.sched.barrier(i32 0)"), 8, "8 sched.barrier(0) walls (one per bare seal)");
    assert_eq!(n("@llvm.amdgcn.s.setprio(i16 1)"), 4, "4 setprio(1) — one per compute cluster (oracle: 4)");
    assert_eq!(n("@llvm.amdgcn.s.setprio(i16 0)"), 4, "4 setprio(0) — one per compute cluster (oracle: 4)");
    assert_eq!(n("s_waitcnt lgkmcnt(0)"), 3, "3 manual LDS drains (C1, C3, C6)");
    assert_eq!(n("s_waitcnt vmcnt(0)"), 0, "no VMEM drain in the mainloop (only the prologue G::load)");

    // ── HK's legacy SRD form + absence of the p0 buffer-load the existing kernels emit. ──
    assert!(ir.contains("i32 1114112, i64 3"), "legacy SRD config word 0x110000 at w3");
    assert!(ir.contains("<4 x i32>"), "legacy `<4 x i32>` buffer-resource descriptor");
    assert!(!ir.contains("make.buffer.rsrc.p0"), "micro_tk must NOT emit the p0 SRD");
    assert!(!ir.contains("raw.ptr.buffer.load"), "micro_tk must NOT emit the p0 buffer load");
    // ── HK's truncating fp32→bf16 C store (`bits >> 16`), not svod's default RNE. ──
    assert!(ir.contains("lshr i32") && ir.contains("store bfloat"), "truncating bf16 C store");
    assert!(!ir.contains("32767"), "HK store truncates; no RNE rounding term");
}
