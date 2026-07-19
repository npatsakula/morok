//! Host tests: the ADT interns/disambiguates correctly, and the verified lowering
//! produces spec-valid device-UOp for BOTH proof kernels.

use svod_dtype::DType;

use crate::ir::{Node, RegClass, Residency, TileIr};
use crate::lower;

// ── the ADT: interning, disambiguators, residency/reg-class fields ───────────

#[test]
fn structurally_identical_nodes_hash_cons() {
    let mut ir = TileIr::new();
    let a = ir.intern(Node::Global { slot: 0, dtype: DType::Float32, len: 4 });
    let b = ir.intern(Node::Global { slot: 0, dtype: DType::Float32, len: 4 });
    assert_eq!(a, b, "identical Global nodes must collapse to one id");
}

#[test]
fn slot_disambiguator_keeps_distinct_globals_apart() {
    let mut ir = TileIr::new();
    let s0 = ir.fresh_slot();
    let s1 = ir.fresh_slot();
    let a = ir.intern(Node::Global { slot: s0, dtype: DType::Float32, len: 4 });
    let b = ir.intern(Node::Global { slot: s1, dtype: DType::Float32, len: 4 });
    assert_ne!(a, b, "different ABI slots must NOT hash-cons together (miscompile guard)");
}

#[test]
fn range_id_disambiguator_keeps_distinct_loops_apart() {
    let mut ir = TileIr::new();
    let r0 = ir.fresh_range_id();
    let r1 = ir.fresh_range_id();
    // Same trip count, different loop — must stay distinct or two loops collapse.
    let a = ir.intern(Node::Range { id: r0, trips: 16 });
    let b = ir.intern(Node::Range { id: r1, trips: 16 });
    assert_ne!(a, b, "same-trip distinct loops must not collapse");
}

#[test]
fn lexical_scopes_keep_region_local_expressions_apart() {
    use crate::build::Builder;

    let mut b = Builder::new("scope_identity");
    let base = b.idx_const(7);
    let one = b.idx_const(1);
    let s0 = b.scope(&[]);
    let s1 = b.scope(&[]);
    let a = b.scope_idx(base, s0);
    let c = b.scope_idx(base, s1);
    assert_ne!(a.0, c.0, "the same value rebound in distinct scopes must remain distinct");
    let aa = b.idx_add(a, one);
    let cc = b.idx_add(c, one);
    assert_ne!(aa.0, cc.0, "scope identity must propagate into derived address DAGs");
}

#[test]
#[should_panic(expected = "reused with different bounds")]
fn runtime_scalar_rejects_conflicting_bounds() {
    let mut b = crate::Builder::new("runtime_scalar_bounds");
    let _ = b.scalar_param("n", 1, 16);
    let _ = b.scalar_param("n", 1, 32);
}

#[test]
fn residency_and_reg_class_fields_present() {
    let mut ir = TileIr::new();
    let g = ir.intern(Node::Global { slot: 0, dtype: DType::Float32, len: 8 });
    let r = ir.intern(Node::DefineReg { id: 0, dtype: DType::Float32, len: 1 });
    assert_eq!(ir.meta(g).residency, Residency::Global);
    assert_eq!(ir.meta(r).residency, Residency::Reg);
    // Reg-class channel exists now (the AGPR pass flips this field in Step 3).
    assert_eq!(ir.meta(r).reg_class, RegClass::Vgpr);
}

#[test]
fn runtime_domains_and_bounded_views_lower_spec_valid() {
    use crate::build::{Builder, F32};

    // Dynamic grid plus bounded load/store.
    let mut b = Builder::new("tk2_dynamic_grid_bounded");
    let out = b.global::<F32>(16);
    let input = b.global::<F32>(16);
    let n = b.scalar_param("n", 1, 16);
    assert_eq!(n.0, b.scalar_param("n", 1, 16).0, "same runtime scalar declaration must deduplicate");
    let gid = b.grid_axis_dyn(0, n);
    let input = b.bounded(input, n);
    let out = b.bounded(out, n);
    let zero = b.f32(0.0);
    let row_stride = b.scalar_param("row_stride", 1, 2);
    let zero_idx = b.idx_const(0);
    let one = b.idx_const(1);
    let value = b.load_strided_bounded(input, [zero_idx, gid], [one, row_stride], zero);
    let root = b.store_strided_bounded(out, [zero_idx, gid], [one, row_stride], value);
    let (ir, sink) = b.finish(&[root]);
    let p = crate::Program { ir, sink, name: "tk2_dynamic_grid_bounded".into() };
    lower::verify(&p).expect("dynamic grid and bounded accesses must lower spec-valid");
    let llvm = crate::launch::render_amd_ir(&p, svod_dtype::AmdArch::Gfx942).expect("render dynamic grid");
    assert!(llvm.contains("i32 %n"), "bounded runtime scalar must render as an i32 kernel argument");
    assert!(llvm.contains("i32 %row_stride"), "runtime strides must remain body scalar arguments");

    // Dynamic Range over the same authoring primitive. The ended store is the sink root.
    let mut b = Builder::new("tk2_dynamic_range_bounded");
    let out = b.global::<F32>(16);
    let input = b.global::<F32>(16);
    let n = b.scalar_param("n", 0, 16);
    let out = b.bounded(out, n);
    let input = b.bounded(input, n);
    let limit = b.scalar_param("limit", 0, 16);
    let trips = b.idx_min(n, limit);
    let domain = b.iter_domain(trips);
    let r = b.range_dyn(domain);
    let i = b.counter(r);
    let zero = b.f32(0.0);
    let value = b.load_bounded(input, i, zero);
    let store = b.store_bounded(out, i, value);
    let ended = b.end(store, &[r]);
    let (ir, sink) = b.finish(&[ended]);
    let p = crate::Program { ir, sink, name: "tk2_dynamic_range_bounded".into() };
    lower::verify(&p).expect("dynamic Range must lower spec-valid");
    let llvm = crate::launch::render_amd_ir(&p, svod_dtype::AmdArch::Gfx942).expect("render dynamic Range");
    assert!(llvm.contains("i32 %n"), "dynamic Range end must use the runtime scalar argument");
    assert!(llvm.contains("i32 %limit"), "workgroup/domain clamp must use its runtime scalar argument");
    assert!(llvm.contains("icmp ult i32"), "dynamic Range must emit a runtime loop comparison");

    // Dynamic raw-buffer descriptor over the same bounded-view contract.
    let mut b = Builder::new("tk2_dynamic_buffer_resource");
    let out = b.global::<F32>(16);
    let input = b.global::<F32>(16);
    let n = b.scalar_param("n", 1, 16);
    let gid = b.grid_axis_dyn(0, n);
    let input = b.bounded(input, n);
    let out = b.bounded(out, n);
    let rsrc = b.make_buffer_rsrc_bounded(input);
    let four = b.idx_const(4);
    let byte_offset = b.idx_mul(gid, four);
    let value = b.buffer_load_raw::<F32>(rsrc, byte_offset, 1, &[]);
    let root = b.store_bounded(out, gid, value);
    let (ir, sink) = b.finish(&[root]);
    let p = crate::Program { ir, sink, name: "tk2_dynamic_buffer_resource".into() };
    lower::verify(&p).expect("runtime-bounded raw resource must lower spec-valid");
    let llvm = crate::launch::render_amd_ir(&p, svod_dtype::AmdArch::Gfx942).expect("render dynamic resource");
    assert!(llvm.contains("make.buffer.rsrc"), "bounded raw view must render a dynamic resource descriptor");
}

#[test]
fn direct_global_to_lds_dword_lowers_and_renders() {
    use crate::build::{BF16, Builder};

    let mut b = Builder::new("tk2_direct_lds_probe");
    let src = b.global::<BF16>(1024);
    let dst = b.define_local::<BF16>(1024);
    let tid = b.block_axis(512);
    let two = b.idx_const(2);
    let off = b.idx_mul(tid, two);
    let dma = b.global_load_lds_dword(src, off, dst, off, &[]);
    let partial = b.swait_vmcnt_allowed(dma, 4);
    let root = b.barrier(partial, &[dma.dep()]);
    let (ir, sink) = b.finish(&[root]);
    let p = crate::Program { ir, sink, name: "tk2_direct_lds_probe".into() };
    lower::verify(&p).expect("direct GLOBAL→LDS DMA must lower spec-valid");
    let llvm = crate::launch::render_amd_ir(&p, svod_dtype::AmdArch::Gfx942).expect("render direct LDS");
    assert!(llvm.contains("llvm.amdgcn.global.load.lds"), "must render the gfx942 direct-to-LDS intrinsic");
    assert!(
        llvm.contains("call void @llvm.amdgcn.s.waitcnt(i32 3956)"),
        "must render LLVM's tracked vmcnt(4) readiness wait"
    );
}

#[test]
#[should_panic(expected = "payload must be exactly 64 bits")]
fn ds_read_b64_rejects_non_b64_payloads() {
    use crate::build::{BF16, Builder};

    let mut b = Builder::new("bad_b64_payload");
    let lds = b.define_local::<BF16>(16);
    let zero = b.idx_const(0);
    let ptr = b.lds_ptr_as3(lds, zero, &[]);
    let _ = b.ds_read_b64::<BF16>(ptr, 0, 2, None);
}

#[test]
#[should_panic(expected = "requires an lgkm wait")]
fn opaque_ready_b64_rejects_non_wait_anchors() {
    use crate::build::{BF16, Builder};

    let mut b = Builder::new("missing_opaque_readiness");
    let input = b.global::<BF16>(4);
    let zero = b.idx_const(0);
    let scalar = b.load(input, zero);
    let packed = b.vec_build(&[scalar, scalar, scalar, scalar]);
    let not_a_wait = b.sched_fence(0, &[packed.id]);
    let _ = b.opaque_ready_b64(packed, not_a_wait.dep());
}

#[test]
#[should_panic(expected = "rows must tile")]
fn gather_run_rejects_truncated_geometry() {
    use crate::build::{BF16, Builder};
    use crate::shape::Mfma32x32x8Bf16;
    use crate::tile::Plain;

    let mut b = Builder::new("bad_gather_geometry");
    let lds = b.define_local::<BF16>(4096);
    let zero = b.idx_const(0);
    let _ = crate::tile_move::gather_run::<BF16, Plain, Mfma32x32x8Bf16>(
        &mut b,
        lds,
        32,
        31,
        32,
        zero,
        zero,
        zero,
        false,
        &[],
    );
}

#[test]
fn fa32_final_asm_v_gather_is_tied_through_opaque_readiness() {
    use crate::ir::TileId;

    // The d128 production (fast) path gathers V through waitcnt-opaque asm reads, so the post-loop DRAIN
    // must tie every final P·V V operand through the drain's lgkmcnt readiness (`opaque_ready_b64`).
    let p = crate::kernels::fa::flash_attention_fwd_32(1, 512, 128).apply(crate::SwizzlePass);
    let live = crate::passes::reachable(&p.ir, p.sink);
    let drain_wait = live
        .iter()
        .copied()
        .filter(|&id| matches!(p.ir.node(id), Node::SWaitLgkmcnt { .. }))
        .max()
        .expect("final gather has an lgkm wait");
    let tied_final_mmas = live
        .iter()
        .filter(|&&id| match p.ir.node(id) {
            Node::Mma { a, .. } => matches!(
                p.ir.node(*a),
                Node::OpaqueReadyB64 {
                    wait,
                    ..
                } if *wait == drain_wait
            ),
            _ => false,
        })
        .count();
    assert_eq!(tied_final_mmas, 128 / 8, "every final P·V V operand must be tied through the drain wait");
    assert!(matches!(p.ir.node(drain_wait), Node::SWaitLgkmcnt { prev: TileId(_) }));
}

#[test]
fn fa32_public_d128_routes_to_two_crew_pingpong() {
    // The public d128 default now routes to the merged two-crew ping-pong (device-fastest at every size):
    // asm-opaque register-staged K (NOT direct-to-LDS, which is stagger-incompatible) + the wave-phase
    // stagger (one eq=0 rebalance + one eq=1 seed).
    for n in [512usize, 1024, 2048] {
        let p = crate::kernels::fa::flash_attention_fwd_32(1, n, 128).apply(crate::SwizzlePass);
        let live = crate::passes::reachable(&p.ir, p.sink);
        let has_direct = live.iter().any(|&id| matches!(p.ir.node(id), Node::GlobalLoadLdsDword { .. }));
        let phases = live.iter().filter(|&&id| matches!(p.ir.node(id), Node::WaveBarrier { .. })).count();
        assert!(!has_direct, "d128 ping-pong uses asm-opaque register-staged K, not direct-to-LDS, at S={n}");
        assert_eq!(phases, 2, "d128 ping-pong must carry the eq=0/eq=1 wave-phase stagger at S={n}");
    }
    // d64 stays single-crew register-staged (the ping-pong is d128-only).
    let p = crate::kernels::fa::flash_attention_fwd_32(1, 2048, 64).apply(crate::SwizzlePass);
    let live = crate::passes::reachable(&p.ir, p.sink);
    assert!(
        live.iter().all(|&id| !matches!(p.ir.node(id), Node::GlobalLoadLdsDword { .. })),
        "d64 must retain register-staged K"
    );
    assert!(
        live.iter().all(|&id| !matches!(p.ir.node(id), Node::WaveBarrier { .. })),
        "d64 stays single-crew (no wave-phase stagger)"
    );
}

#[test]
fn fa32_pingpong_constructs_balanced_and_lowers() {
    // The d128 production kernel (the ping-pong) must CONSTRUCT (both `pipeline::verify`'s wave-phase balance
    // check and the `verify_v2` scheduling gate run at build time), carry EXACTLY one eq=1 stagger seed + one
    // eq=0 rebalance (an imbalance panics in `verify` as a would-be workgroup deadlock), and lower to
    // spec-valid gfx942 LLVM IR. n=512,d128 ⇒ 16 KV blocks (≥3, so a steady body carries the eq=1 barrier).
    let p = crate::kernels::fa::flash_attention_fwd_32(1, 512, 128).apply(crate::SwizzlePass);
    let live = crate::passes::reachable(&p.ir, p.sink);
    let count_eq = |want: i64| {
        live.iter().filter(|&&id| matches!(p.ir.node(id), Node::WaveBarrier { eq, .. } if *eq == want)).count()
    };
    assert_eq!(
        (count_eq(0), count_eq(1)),
        (1, 1),
        "ping-pong FA must carry exactly one eq=0 rebalance and one eq=1 stagger barrier"
    );
    let llvm = crate::launch::render_amd_ir(&p, svod_dtype::AmdArch::Gfx942).expect("render ping-pong FA-32");
    assert!(llvm.contains("barrier"), "ping-pong FA must emit workgroup barriers");
}

#[test]
fn fa32_public_domain_is_tile_exact_and_head_dim_restricted() {
    assert!(
        std::panic::catch_unwind(|| crate::kernels::fa::flash_attention_fwd_32(1, 0, 128)).is_err(),
        "public FA-32 must reject an empty sequence"
    );
    for d in [64usize, 128] {
        crate::kernels::fa::flash_attention_fwd_32(1, 1024, d);
        assert!(
            std::panic::catch_unwind(|| crate::kernels::fa::flash_attention_fwd_32(1, 1023, d)).is_err(),
            "public FA-32 must reject a Q256-ragged n at d={d}"
        );
    }
    for d in [32usize, 96, 160, 192] {
        assert!(
            std::panic::catch_unwind(|| crate::kernels::fa::flash_attention_fwd_32(1, 1024, d)).is_err(),
            "public FA-32 must reject unsupported d={d}"
        );
    }
    for n in [64usize, 80, 96, 128] {
        assert!(
            std::panic::catch_unwind(|| crate::kernels::fa::flash_attention_fwd_32(1, n, 128)).is_err(),
            "public FA-32 must reject qualification-only n={n}"
        );
        crate::kernels::fa::flash_attention_fwd_32_register_k(1, n, 128);
    }
}

// ── the verified lowering ────────────────────────────────────────────────────

#[test]
fn matmul_lds_kblock_clustered_lowers_and_balances_the_wave_phase() {
    // The §5c clustered HK replica (2 warp-rows → 128², 4 K-blocks, ksteps=4): the interpreter walks
    // the 8-cluster schedule placing per-cluster barriers + set_prio brackets + the warp-phase
    // ping-pong. It must lower spec-valid, carry the SetPrio brackets, and the wave barriers must be
    // balanced (eq=0 count == eq=1 count == 1) — else `matmul_..._clustered` would have panicked in
    // `verify_warp_phase_balance` at construction. The clustered kernel now gathers via the asm
    // `ds_read_b64 offset:N` primitive (LdsPtrAs3 + DsReadB64) and re-enables the positional
    // `wall_after_barriers` lattice (SchedWallMarker) — both must lower spec-valid.
    let count = |p: &crate::Program, pred: &dyn Fn(&Node) -> bool| {
        (0..p.ir.len()).filter(|&i| pred(p.ir.node(crate::ir::TileId(i as u32)))).count()
    };
    let p = crate::kernels::matmul::matmul_lds_kblock_mw_clustered(
        128,
        128,
        256,
        crate::kernels::matmul::Tiling { bm: 64, bn: 64, wm: 2, wn: 2, k_step: 64 },
    );
    lower::verify(&p).expect("clustered HK replica must lower to spec-valid UOp");
    assert!(count(&p, &|n| matches!(n, Node::SetPrio { .. })) > 0, "compute clusters ⇒ SetPrio nodes");
    assert_eq!(count(&p, &|n| matches!(n, Node::WaveBarrier { eq: 1, .. })), 1, "one eq=1 prologue wave barrier");
    assert_eq!(count(&p, &|n| matches!(n, Node::WaveBarrier { eq: 0, .. })), 1, "one eq=0 epilogue wave barrier");
    // Composes with the refinement passes: VectorizePass is a no-op on the asm gather (no fusible
    // scalar run), and SwizzlePass folds the (fragment-invariant) XOR delta into the asm base offset's
    // `lds_col` — so the swizzled clustered kernel still lowers spec-valid.
    let sw = crate::kernels::matmul::matmul_lds_kblock_mw_clustered(
        128,
        128,
        256,
        crate::kernels::matmul::Tiling { bm: 64, bn: 64, wm: 2, wn: 2, k_step: 64 },
    )
    .apply(crate::passes::VectorizePass)
    .apply(crate::passes::SwizzlePass);
    lower::verify(&sw).expect("clustered.apply(Vectorize).apply(Swizzle) must lower spec-valid");
}

// ── the FA-forward experiment: the ClusterCx pipeline generalised to a second kernel shape ─────

// ── the SchedGroupBarrier interleave primitive (FA-redesign step 2) ──────────────────────────────

/// The declarative interleave directive ([`crate::build::Builder::interleave_valu`]) must (a) intern
/// `SchedGroupBarrier` nodes, (b) lower spec-valid, and (c) RENDER to the `@llvm.amdgcn.sched.group.
/// barrier` builtin the AMDGPU backend emits as the `; sched_group_barrier` interleave comment. A tiny
/// 2-slice 32×32×8 MFMA burst (intrinsic accumulator) + a VALU scale carries an `interleave_valu<2,5>`
/// hint threaded live into the store — the minimal proof the primitive emits before FA depends on it.
#[test]
fn sched_group_barrier_lowers_and_renders_the_builtin() {
    use crate::build::{BF16, Builder, F32};
    use crate::shape::{Mfma32x32x8Bf16 as S, MfmaShape};
    let mut b = Builder::new("tk2_sched_group_probe");
    let c = b.global::<F32>(S::M * S::N);
    let a = b.global::<BF16>(S::M * 2 * S::K);
    let bmat = b.global::<BF16>(S::N * 2 * S::K);
    let _wg = b.grid_axis(0, 1);
    let lane = b.block_axis(64);
    let (a_map, b_map, dist) = (S::a_map(), S::b_map(), S::acc_dist());
    // 2-slice K-loop into one intrinsic-MFMA accumulator.
    let mut acc = {
        let zs: Vec<_> = (0..S::EPT_C).map(|_| b.f32(0.0)).collect();
        b.vec_build(&zs)
    };
    for ki in 0..2 {
        let af = crate::test::probes::load_op_frag(&mut b, a, a_map, 0, ki * S::K, 2 * S::K, lane);
        let bf = crate::test::probes::load_op_frag(&mut b, bmat, b_map, 0, ki * S::K, 2 * S::K, lane);
        acc = b.mma_of::<S>(af, bf, acc);
    }
    // A VALU op the interleave can pull under the MFMAs (the softmax-rescale analog).
    let two = b.f32(2.0);
    let mut scaled = Vec::with_capacity(S::EPT_C);
    for i in 0..S::EPT_C {
        let e = b.vec_extract(acc, i);
        scaled.push(b.mul(e, two));
    }
    let acc = b.vec_build(&scaled);
    // Scatter, then thread an interleave_valu<pairs=2, valu=5> hint anchored on the last store — live
    // via the roots, so it survives DCE and reaches the renderer.
    let n_c = b.idx_const(S::N as i64);
    let mut roots = Vec::new();
    for i in 0..S::EPT_C {
        let (row, col) = b.acc_rc(dist, lane, i);
        let rn = b.idx_mul(row, n_c);
        let off = b.idx_add(rn, col);
        let v = b.vec_extract(acc, i);
        roots.push(b.store(c, off, v));
    }
    let anchor = roots.last().expect("stores").dep();
    let hint = b.interleave_valu(2, 5, 1, &[anchor]).expect("pairs>0");
    roots.push(hint);
    let (ir, sink) = b.finish(&roots);
    let p = crate::Program { ir, sink, name: "tk2_sched_group_probe".into() };

    // (a) the nodes are interned — 2 pairs × 2 hints = 4 SchedGroupBarrier.
    let n_sgb = (0..p.ir.len())
        .filter(|&i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::SchedGroupBarrier { .. }))
        .count();
    assert_eq!(n_sgb, 4, "interleave_valu<2,_> ⇒ 2×(MFMA+VALU) = 4 SchedGroupBarrier nodes");
    // (b) spec-valid lowering.
    lower::verify(&p).expect("sched_group probe must lower spec-valid");
    // (c) renders the builtin (→ the `; sched_group_barrier` ASM comment).
    let llvm = crate::launch::render_amd_ir(&p, svod_dtype::AmdArch::Gfx942).expect("render");
    assert!(llvm.contains("llvm.amdgcn.sched.group.barrier"), "must render the sched.group.barrier builtin");
    assert!(llvm.contains("i32 8, i32 1"), "MFMA-mask(0x8) size-1 group present");
    assert!(llvm.contains("i32 2, i32 5"), "VALU-mask(0x2) size-5 group present");
}

/// The 32×32×8 MFMA isolation probe ([`crate::test::probes::mfma_32x32x8_probe`]) must lower to spec-valid
/// device-UOp — proving the `Node::Mma` accumulator-width dispatch (`ept 16 → 32×32×8`), the wide
/// `v_mfma_f32_32x32x8_bf16` intrinsic selection, and the 16-VGPR `acc_rc` scatter survive lowering +
/// `type_verify` BEFORE the device gate. Covers one MFMA (32×32×8), a K-loop (32×32×16), and a tiled
/// output (64×64×8) so the accumulation chain + the M/N tiling are all exercised in the linearizer.
#[test]
fn mfma_32x32x8_probe_lowers_spec_valid() {
    for (m, n, k) in [(32usize, 32usize, 8usize), (32, 32, 16), (64, 64, 8)] {
        let p = crate::test::probes::mfma_32x32x8_probe(m, n, k);
        // Exactly the tiled MFMA count: (m/32)·(n/32)·(k/8).
        let n_mma =
            (0..p.ir.len()).filter(|&i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::Mma { .. })).count();
        assert_eq!(n_mma, (m / 32) * (n / 32) * (k / 8), "probe {m}×{n}×{k} MFMA count");
        lower::verify(&p).expect("32×32×8 probe must lower to spec-valid UOp");
    }
}

/// A **ragged-`n` FA-32** ([`crate::kernels::fa::flash_attention_fwd_32`]) must lower spec-valid: `n=80`
/// is not a KV-block (32) multiple, so the last KV block is partial and the online softmax carries the
/// per-element ragged-tail mask (`global_kv < n ? score : −∞`). Proves the new `Node::SelectLt` →
/// `WHERE(LT,…)` lowering + `type_verify` accept the mask BEFORE the device gate, at `d=64` (2 KV
/// fragments) and `d=128`, both base and `SwizzlePass` forms. (Constructing it also runs the build-time
/// `verify_v2` scheduling-coherence + pipeline completeness checks over the masked schedule.)
#[test]
fn fa32_ragged_tail_lowers_spec_valid() {
    for d in [64usize, 128] {
        let p = crate::kernels::fa::flash_attention_fwd_32_register_k(1, 80, d);
        let n_sel = (0..p.ir.len())
            .filter(|&i| matches!(p.ir.node(crate::ir::TileId(i as u32)), Node::SelectLt { .. }))
            .count();
        assert!(n_sel > 0, "ragged FA-32 (n=80, d={d}) must emit the SelectLt ragged-tail mask");
        lower::verify(&p).expect("ragged FA-32 must lower spec-valid (base)");
        let ps = crate::kernels::fa::flash_attention_fwd_32_register_k(1, 80, d).apply(crate::SwizzlePass);
        lower::verify(&ps).expect("ragged FA-32 must lower spec-valid (swizzled)");
    }
}

/// Tile-exact FA-32 uses a straight-line QK(0) warmup, so the rolled steady loop processes only blocks
/// `1..nblocks-2`. Two-block inputs take the explicit warmup-to-epilogue transition and emit no loop;
/// scoped movement keeps ragged warmup/loop/epilogue address DAGs distinct.
#[test]
fn fa32_warmup_covers_scoped_and_no_steady_paths() {
    let trips = |p: &crate::Program| {
        (0..p.ir.len())
            .filter_map(|i| match p.ir.node(crate::ir::TileId(i as u32)) {
                Node::Range { trips, .. } | Node::RangeAfter { trips, .. } => Some(*trips),
                _ => None,
            })
            .collect::<Vec<_>>()
    };

    let full = crate::kernels::fa::flash_attention_fwd_32(32, 2048, 64);
    assert_eq!(trips(&full), vec![62], "64 KV blocks minus warmup and epilogue leaves 62 steady iterations");

    for d in [64usize, 128] {
        let ragged = crate::kernels::fa::flash_attention_fwd_32_register_k(1, 80, d);
        assert_eq!(trips(&ragged), vec![1], "three KV blocks minus warmup and epilogue leaves one steady iteration");
        assert!(
            (0..ragged.ir.len()).any(|i| matches!(ragged.ir.node(crate::ir::TileId(i as u32)), Node::Scope { .. })),
            "ragged warmup must carry explicit lexical scope identity"
        );
        let ragged_llvm =
            crate::launch::render_amd_ir(&ragged, svod_dtype::AmdArch::Gfx942).expect("render ragged warmup");
        assert!(ragged_llvm.contains("loop_entry_"), "three-block ragged warmup has one real steady loop trip");

        let two_blocks = crate::kernels::fa::flash_attention_fwd_32_register_k(1, 64, d);
        assert!(trips(&two_blocks).is_empty(), "two KV blocks use warmup plus epilogue with no synthetic loop");
        let two_llvm =
            crate::launch::render_amd_ir(&two_blocks, svod_dtype::AmdArch::Gfx942).expect("render two-block warmup");
        assert!(!two_llvm.contains("loop_entry_"), "two-block warmup must not lower a dead/zero-trip loop");
    }
}
