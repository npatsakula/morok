//! **Byte-identity gate (migration Step 1).** Re-deriving the 16×16×16 shape constants (`EDGE`, the
//! operand/accumulator `FragMap`s, `ept`, the `Node::Mma` intrinsic dims) FROM the `Mfma16x16x16Bf16`
//! marker must leave the EMITTED tile-IR unchanged — the type now *computes* what was hardcoded, and
//! nothing else moves. This gate hashes the reachable node DAG of both proof kernels (matmul + FA +
//! the `atb_probe`), base AND `VectorizePass`+`SwizzlePass`, and asserts it equals the golden signature
//! captured at HEAD `00aea47f` (pre-marker). A mismatch means a re-derived site is NOT byte-identical —
//! STOP and report (the abstraction is subtly wrong), do not paper over it.

use crate::ir::{TileId, TileIr};
use crate::kernels::Program;
use crate::kernels::fa::{flash_attention_fwd, flash_attention_fwd_32};
use crate::kernels::matmul::matmul_lds_kblock_mw_clustered;
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
    let mm = || matmul_lds_kblock_mw_clustered(256, 256, 256, 128, 64, 2, 4, 64);
    let fa = || flash_attention_fwd(2, 128, 128);
    // FA-32 (the 32×32×8 wide-core FA): the `tile_move::{gather_run, commit_run}` derivation of
    // `Fa32Hooks`'s fragment addressing must leave the emitted IR unchanged. `fa32` is the double-buffered
    // (d=128) tile-exact case (the device gate's `(2,128,128)`); `fa32.d64` is the single-buffered (d=64)
    // path (parity offset folds to 0). `fa32.sw` runs SwizzlePass (the K-tile bank swizzle — the as-used
    // path the device gate applies).
    let fa32 = || flash_attention_fwd_32(2, 128, 128);
    vec![
        ("matmul", sig(&mm())),
        ("matmul.vec.sw", sig(&mm().apply(VectorizePass).apply(SwizzlePass))),
        ("fa", sig(&fa())),
        ("fa.vec.sw", sig(&fa().apply(VectorizePass).apply(SwizzlePass))),
        ("atb_probe", sig(&atb_probe(16, 64, 64))),
        ("fa32", sig(&fa32())),
        ("fa32.sw", sig(&fa32().apply(SwizzlePass))),
        ("fa32.d64", sig(&flash_attention_fwd_32(3, 128, 64))),
    ]
}

/// GOLDEN signatures captured at HEAD `00aea47f` (pre-`MfmaShape`-marker). Step 1's whole point is that
/// these do NOT change: the marker re-derives the 16×16×16 constants, the emitted IR is identical.
const GOLDEN: &[(&str, u64)] = &[
    ("matmul", 0xc592_10e1_e156_59aa),
    ("matmul.vec.sw", 0xb6c6_8775_50c9_547c),
    ("fa", 0x21cb_5221_fdc1_2b88),
    ("fa.vec.sw", 0x50d3_dc07_32b9_ac3e),
    ("atb_probe", 0xa506_f161_b28e_11fc),
    // FA-32 golden re-baselined for Phase 2: the `RowPartition` grid decode interns the same node set
    // in a different ORDER (front-loaded vs. interspersed with tid/warp), shifting ids — equivalent IR,
    // device-bit-exact (the `flash_attention32_matches_reference_on_gfx942` gate), NOT a regression.
    ("fa32", 0x663b_b07a_90be_3bc5),
    ("fa32.sw", 0x1070_a446_f24e_c88b),
    ("fa32.d64", 0x979d_3059_9c11_cf0e),
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
