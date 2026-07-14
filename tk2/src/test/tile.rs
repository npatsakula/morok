//! **Tile-layer step-1 gate** (`crate::tile`). The deriving device must compute *exactly* the data the
//! kernels thread by hand today — the `MfmaShape` operand/accumulator maps, the `ept` triple, the
//! `n_frags` arithmetic (`movement.rs`), and the swizzle policy — for both shapes and all operand
//! roles. A mismatch means a `RegTile<…>`-authored tile would address differently than the hand-filled
//! `gather_view`, so the later migration (steps 3–5) could NOT be byte-identical. Pure host asserts
//! against the seed (`shape.rs`); no device, no IR interning yet.

use crate::ir::{Layout, Residency, Transform};
use crate::shape::{Mfma16x16x16Bf16, Mfma32x32x8Bf16, MfmaShape};
use crate::tile::{ARow, Acc, BCol, GlobalTile, LdsTile, Plain, RegTile, TileType, Xor};

use crate::build::{BF16, F32};

// ── register-tile operand roles derive the shape's FragMap / AccDist verbatim ──

#[test]
fn regtile_row_operand_derives_a_map() {
    // 16×16×16
    let d = <RegTile<BF16, 16, 16, ARow, Mfma16x16x16Bf16>>::desc();
    assert_eq!(d.residency, Residency::Reg);
    assert_eq!(d.frag, Some(Mfma16x16x16Bf16::a_map()));
    assert_eq!(d.acc, None);
    assert_eq!(d.ept, Mfma16x16x16Bf16::EPT_A);
    // 32×32×8
    let d = <RegTile<BF16, 32, 8, ARow, Mfma32x32x8Bf16>>::desc();
    assert_eq!(d.frag, Some(Mfma32x32x8Bf16::a_map()));
    assert_eq!(d.ept, Mfma32x32x8Bf16::EPT_A);
}

#[test]
fn regtile_col_operand_derives_b_map() {
    let d = <RegTile<BF16, 16, 16, BCol, Mfma16x16x16Bf16>>::desc();
    assert_eq!(d.frag, Some(Mfma16x16x16Bf16::b_map()));
    assert_eq!(d.acc, None);
    let d = <RegTile<BF16, 8, 32, BCol, Mfma32x32x8Bf16>>::desc();
    assert_eq!(d.frag, Some(Mfma32x32x8Bf16::b_map()));
    assert_eq!(d.ept, Mfma32x32x8Bf16::EPT_B);
}

#[test]
fn regtile_accumulator_derives_acc_dist_not_a_map() {
    // The accumulator addresses via AccDist, NOT a FragMap (which cannot express the M-block split).
    let d = <RegTile<F32, 16, 16, Acc, Mfma16x16x16Bf16>>::desc();
    assert_eq!(d.frag, None, "acc tile carries no operand FragMap");
    assert_eq!(d.acc, Some(Mfma16x16x16Bf16::acc_dist()));
    assert_eq!(d.ept, Mfma16x16x16Bf16::EPT_C);
    let d = <RegTile<F32, 32, 32, Acc, Mfma32x32x8Bf16>>::desc();
    assert_eq!(d.acc, Some(Mfma32x32x8Bf16::acc_dist()));
    assert_eq!(d.ept, 16, "32×32×8 accumulator is EPT_C = 16");
}

// ── n_frags matches the movement-layer arithmetic ((R/M)·(C/K) for A, etc.) ──

#[test]
fn regtile_n_frags_tracks_tile_over_mfma_shape() {
    // A: (R/M)·(C/K). A 32-row × 128-col K-operand tile at 32×32×8: (32/32)·(128/8) = 16 frags.
    assert_eq!(<RegTile<BF16, 32, 128, ARow, Mfma32x32x8Bf16>>::desc().n_frags, 16);
    // B: (R/K)·(C/N). A 8×32 tile at 32×32×8: (8/8)·(32/32) = 1.
    assert_eq!(<RegTile<BF16, 8, 32, BCol, Mfma32x32x8Bf16>>::desc().n_frags, 1);
    // Acc: (R/M)·(C/N). A 32×64 accumulator at 32×32×8: (32/32)·(64/32) = 2.
    assert_eq!(<RegTile<F32, 32, 64, Acc, Mfma32x32x8Bf16>>::desc().n_frags, 2);
    // 16×16×16 A: 32×32 → (32/16)·(32/16) = 4.
    assert_eq!(<RegTile<BF16, 32, 32, ARow, Mfma16x16x16Bf16>>::desc().n_frags, 4);
}

// ── the VGPR ledger: n_frags · ept · sizeof(E) / 4 (dwords) ──

#[test]
fn regtile_vgprs_ledger_counts_dwords() {
    // 32×128 bf16 K-operand @ 32×32×8: 16 frags · 4 bf16 · 2 B / 4 = 32 VGPR.
    assert_eq!(<RegTile<BF16, 32, 128, ARow, Mfma32x32x8Bf16>>::desc().vgprs, 32);
    assert_eq!(<RegTile<BF16, 32, 128, ARow, Mfma32x32x8Bf16>>::vgprs(), 32);
    // 32×32 f32 accumulator @ 32×32×8: 1 frag · 16 f32 · 4 B / 4 = 16 VGPR (the O d-tile cost).
    assert_eq!(<RegTile<F32, 32, 32, Acc, Mfma32x32x8Bf16>>::vgprs(), 16);
    // 16×16 f32 accumulator @ 16×16×16: 1 · 4 · 4 / 4 = 4 VGPR.
    assert_eq!(<RegTile<F32, 16, 16, Acc, Mfma16x16x16Bf16>>::vgprs(), 4);
    // LDS/global tiles cost 0 VGPRs — the register-diet point (K/V direct-to-LDS contribute nothing).
    assert_eq!(<LdsTile<BF16, 32, 128, Plain>>::vgprs(), 0);
    assert_eq!(<GlobalTile<BF16, 4096, 128>>::vgprs(), 0);
}

// ── LDS swizzle is a property of the type, not a threaded parameter ──

#[test]
fn ldstile_swizzle_comes_from_the_type() {
    let plain = <LdsTile<BF16, 32, 128, Plain>>::desc();
    assert_eq!(plain.residency, Residency::Lds);
    assert_eq!(plain.swizzle, Layout::contiguous(), "Plain ⇒ contiguous (direct-to-LDS compatible)");
    assert_eq!(plain.inner, 128);

    let xor = <LdsTile<BF16, 32, 128, Xor>>::desc();
    assert_eq!(
        xor.swizzle,
        Layout { transforms: [Transform::Xor { cols: 128 }].into_iter().collect() },
        "Xor ⇒ the bank XOR over the tile width"
    );
}

// ── residency is a const on the type ──

#[test]
fn residency_is_a_type_constant() {
    assert_eq!(<RegTile<BF16, 16, 16, ARow, Mfma16x16x16Bf16>>::RES, Residency::Reg);
    assert_eq!(<LdsTile<BF16, 32, 128, Plain>>::RES, Residency::Lds);
    assert_eq!(<GlobalTile<F32, 4096, 128>>::RES, Residency::Global);
}
