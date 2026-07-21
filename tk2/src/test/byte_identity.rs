//! **Byte-identity gate (migration Step 1).** Re-deriving the 16×16×16 shape constants (`EDGE`, the
//! operand/accumulator `FragMap`s, `ept`, the `Node::Mma` intrinsic dims) FROM the `Mfma16x16x16Bf16`
//! marker must leave the EMITTED tile-IR unchanged — the type now *computes* what was hardcoded, and
//! nothing else moves. This gate hashes the reachable node DAG of both proof kernels (matmul + FA +
//! the `atb_probe`), base AND `VectorizePass`+`SwizzlePass`, and asserts it equals the golden signature
//! captured at HEAD `00aea47f` (pre-marker). A mismatch means a re-derived site is NOT byte-identical —
//! STOP and report (the abstraction is subtly wrong), do not paper over it.

use crate::ir::{TileId, TileIr};
use crate::kernels::Program;
use crate::kernels::fa::{flash_attention_fwd_32, flash_attention_fwd_32_register_k};
use crate::kernels::matmul::{Tiling, matmul_lds_kblock_mw_clustered};
use crate::test::probes::atb_probe;
use crate::{SwizzlePass, VectorizePass};

/// FNV-1a over the reachable node DAG from `sink` — each node's id-tagged `Debug`, in id order. The
/// signature captures the exact emitted program: DAG structure AND every data field (`FragMap`, `ept`,
/// `off_bytes`, dtypes, …), ignoring only unreachable dedup residue. No `std` hasher, so it is stable
/// across runs/toolchains (an FNV-1a with the standard 64-bit offset/prime).
fn structural_sig(ir: &TileIr, sink: TileId) -> u64 {
    let mut seen = vec![false; ir.len()];
    let mut stack = vec![sink];
    let mut order = Vec::new();
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id.0 as usize], true) {
            continue;
        }
        order.push(id);
        for c in TileIr::children(ir.node(id)) {
            stack.push(c);
        }
    }
    order.sort();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in order {
        for byte in format!("{}={:?};", id.0, ir.node(id)).bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn sig(p: &Program) -> u64 {
    structural_sig(&p.ir, p.sink)
}

/// The five representative programs whose emitted IR must stay byte-identical across the refactor:
/// the clustered matmul (base + vec+sw) at the bit-exact test's tiling, FA (base + vec+sw), and the
/// `atb_probe`. Same shapes the device gates use, so a host diff localises what a device gate would.
fn signatures() -> Vec<(&'static str, u64)> {
    let mm = || matmul_lds_kblock_mw_clustered(256, 256, 256, Tiling::default());
    // FA-32 (the 32×32×8 wide-core FA): the `tile_move::{gather_run, commit_run}` derivation of
    // `Fa32Hooks`'s fragment addressing must leave the emitted IR unchanged. `fa32` is the double-buffered
    // (d=128) historical short-shape case; it uses the crate-private register-staged oracle because the
    // public production domain is Q256-exact. `fa32.d64` is the single-buffered (d=64) path (parity offset
    // folds to 0). `fa32.sw` runs SwizzlePass (the K-tile bank swizzle used in production).
    let fa32 = || flash_attention_fwd_32_register_k(2, 128, 128);
    vec![
        ("matmul", sig(&mm())),
        ("matmul.vec.sw", sig(&mm().apply(VectorizePass).apply(SwizzlePass))),
        ("atb_probe", sig(&atb_probe(16, 64, 64))),
        ("fa32", sig(&fa32())),
        ("fa32.sw", sig(&fa32().apply(SwizzlePass))),
        ("fa32.d64", sig(&flash_attention_fwd_32_register_k(3, 128, 64))),
        ("fa32.d64.xcd", sig(&flash_attention_fwd_32(8, 512, 64).apply(SwizzlePass))),
        ("fa32.d128.prod", sig(&flash_attention_fwd_32(2, 2048, 128))),
        ("fa32.d128.prod.sw", sig(&flash_attention_fwd_32(2, 2048, 128).apply(SwizzlePass))),
    ]
}

/// GOLDEN signatures captured at HEAD `00aea47f` (pre-`MfmaShape`-marker). Step 1's whole point is that
/// these do NOT change: the marker re-derives the 16×16×16 constants, the emitted IR is identical.
const GOLDEN: &[(&str, u64)] = &[
    ("matmul", 0xc592_10e1_e156_59aa),
    ("matmul.vec.sw", 0xb6c6_8775_50c9_547c),
    ("atb_probe", 0xa506_f161_b28e_11fc),
    // FA-32 golden re-baselined for the compute-software-pipeline ROTATION (fused QKᵀ∥softmax cluster,
    // carried double-buffered `s`, post-loop drain) + the pipeline WAR guard that orders a read-then-
    // INDEPENDENTLY-written slot's store after the cluster's read-carrying stores (fixes the epilogue
    // reading the wrong carried scores — the deleted-cross-lane-reduce bug). A real STRUCTURAL change, not a
    // refactor — device-correct via `flash_attention32_matches_reference_on_gfx942` (uniform ~1.8e-3 at all
    // shapes incl. ragged n=80). matmul/fa/atb goldens are UNCHANGED (the WAR guard is a no-op there).
    // FA-32 rebaselined after the tile-exact QK(0) warmup removed the empty seed softmax/PV. Device
    // correctness, resource, scheduler-cadence, and repeated performance gates all passed; non-FA
    // signatures remain unchanged.
    ("fa32", 0x1d8b_e75f_649b_e62b),
    ("fa32.sw", 0x336c_a678_a3d3_aa41),
    ("fa32.d64", 0x84d2_b447_0013_36a5),
    // d64-only 8-XCD remap, with two Q tiles per slice so the permutation is non-identity.
    ("fa32.d64.xcd", 0xc36a_7cd6_ac9f_85cd),
    // The d128 PRODUCTION fast/ping-pong path (asm gather+commit, packed V, two-crew ping-pong) — frozen
    // so the flag-matrix collapse (asm_gather/asm_commit/packed_v/ping_pong → one `fast` flag) is proven
    // IR-preserving on the FAST path, not only the register-staged oracle path the other FA-32 goldens hit.
    ("fa32.d128.prod", 0x06bd_40b2_ec99_cf3c),
    ("fa32.d128.prod.sw", 0x9440_d1c8_4dde_1343),
];

/// Print the live signatures — run at HEAD to capture the golden values, and any time to eyeball a diff.
#[test]
fn print_signatures() {
    for (name, s) in signatures() {
        println!("BYTE_IDENTITY {name} = 0x{s:016x}");
    }
}

/// The accumulator C-site abstraction is correct: for 16×16×16, `acc_rc(acc_dist())` produces the
/// IDENTICAL index nodes as the former `lane_rc(c_map())` — interning collapses them to ONE `TileId`
/// per (row, col), so re-deriving the accumulator distribution from the marker is provably byte-exact
/// (the `m_blocks == 1` block term folds away; the `lane_m_stride`/`n_lanes` consts match `stride`/`cols`).
#[test]
fn acc_rc_matches_lane_rc_for_16x16x16() {
    use crate::build::Builder;
    use crate::shape::{Mfma16x16x16Bf16, MfmaShape};
    let mut b = Builder::new("acc_rc_probe");
    let lane = b.block_axis(64);
    let dist = Mfma16x16x16Bf16::acc_dist();
    let cmap = Mfma16x16x16Bf16::c_map();
    for i in 0..Mfma16x16x16Bf16::EPT_C {
        let inner = b.idx_const(i as i64);
        let (r_lane, c_lane) = b.lane_rc(cmap, lane, inner);
        let (r_acc, c_acc) = b.acc_rc(dist, lane, i);
        // `Idx` wraps a `TileId`; interning makes equal structure ⇒ equal id.
        assert_eq!(r_lane.0, r_acc.0, "acc_rc row must intern-equal lane_rc row for accumulator elem {i}");
        assert_eq!(c_lane.0, c_acc.0, "acc_rc col must intern-equal lane_rc col for accumulator elem {i}");
    }
}

/// The shape-matched MMA entry points (`mma_of`/`mma_asm_of`) erase to the plain
/// [`Builder::mma`]/[`Builder::mma_asm`]: the `Tile<M,K>·Tile<K,N>→Tile<M,N>` composition check lives
/// only in the types, so the interned `Node::Mma` is identical and hash-consing collapses the shaped and
/// unshaped calls to ONE `TileId`. This proves the type layer adds zero IR — the matmul clustered kernel
/// stays byte-identical after the `mma_asm` → `mma_asm_of` migration (its GOLDEN below is unchanged).
#[test]
fn shaped_mma_interns_equal_to_plain_mma_16x16x16() {
    use crate::build::Builder;
    use crate::shape::{Mfma16x16x16Bf16 as S, MfmaShape};
    let mut b = Builder::new("mma_shape_probe");
    let fa = b.define_frag::<crate::build::BF16>(S::a_map());
    let fb = b.define_frag::<crate::build::BF16>(S::b_map());
    let fc = b.define_frag::<crate::build::F32>(S::c_map());
    let (a, bb, c) = (b.load_frag_vec(fa), b.load_frag_vec(fb), b.load_frag_vec(fc));
    // Intrinsic path: shaped `mma_of` == plain `mma`.
    let plain = b.mma(a, bb, c, S::EPT_C);
    let shaped = b.mma_of::<S, 16, 16, 16>(a.tile::<16, 16>(), bb.tile::<16, 16>(), c.tile::<16, 16>());
    assert_eq!(plain.id, shaped.erase().id, "shaped mma_of must intern-equal plain mma");
    // Asm path (the matmul kernel's site): shaped `mma_asm_of` == plain `mma_asm`.
    let plain_asm = b.mma_asm(a, bb, c, S::EPT_C);
    let shaped_asm = b.mma_asm_of::<S, 16, 16, 16>(a.tile::<16, 16>(), bb.tile::<16, 16>(), c.tile::<16, 16>());
    assert_eq!(plain_asm.id, shaped_asm.erase().id, "shaped mma_asm_of must intern-equal plain mma_asm");
}

/// The gate: the emitted IR of every proof kernel must equal its golden signature.
#[test]
fn emitted_ir_is_byte_identical_after_marker_refactor() {
    for ((name, got), (gname, want)) in signatures().into_iter().zip(GOLDEN.iter()) {
        assert_eq!(&name, gname, "signature list drifted from GOLDEN order");
        assert_eq!(
            got, *want,
            "{name}: emitted IR changed (0x{got:016x} != golden 0x{want:016x}) — a re-derived 16×16×16 \
             site is NOT byte-identical; STOP and report the discrepancy (do not update GOLDEN to hide it)"
        );
    }
}
