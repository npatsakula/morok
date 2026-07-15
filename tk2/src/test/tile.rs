//! **Marker-derivation gate** (`crate::tile`). The operand-role + swizzle markers must derive *exactly*
//! the data the kernels thread by hand today — the `MfmaShape` operand/accumulator maps, the `ept`
//! triple, the `n_frags` arithmetic (`movement.rs`), and the swizzle policy — for both shapes and all
//! operand roles. A mismatch means a marker-authored gather would address differently than the
//! hand-filled `gather_view`. Pure host asserts against the seed (`shape.rs`); no device, no IR.

use crate::ir::{Layout, Transform};
use crate::shape::{Mfma16x16x16Bf16, Mfma32x32x8Bf16, MfmaShape};
use crate::tile::{ARow, Acc, BCol, Plain, RegLayout, Swizzle, Xor};

// ── register-tile operand roles derive the shape's FragMap / AccDist verbatim ──

#[test]
fn row_operand_derives_a_map() {
    assert_eq!(ARow::frag::<Mfma16x16x16Bf16>(), Some(Mfma16x16x16Bf16::a_map()));
    assert_eq!(ARow::acc::<Mfma16x16x16Bf16>(), None);
    assert_eq!(ARow::ept::<Mfma16x16x16Bf16>(), Mfma16x16x16Bf16::EPT_A);
    assert_eq!(ARow::frag::<Mfma32x32x8Bf16>(), Some(Mfma32x32x8Bf16::a_map()));
    assert_eq!(ARow::ept::<Mfma32x32x8Bf16>(), Mfma32x32x8Bf16::EPT_A);
}

#[test]
fn col_operand_derives_b_map() {
    assert_eq!(BCol::frag::<Mfma16x16x16Bf16>(), Some(Mfma16x16x16Bf16::b_map()));
    assert_eq!(BCol::acc::<Mfma16x16x16Bf16>(), None);
    assert_eq!(BCol::frag::<Mfma32x32x8Bf16>(), Some(Mfma32x32x8Bf16::b_map()));
    assert_eq!(BCol::ept::<Mfma32x32x8Bf16>(), Mfma32x32x8Bf16::EPT_B);
}

#[test]
fn accumulator_derives_acc_dist_not_a_map() {
    // The accumulator addresses via AccDist, NOT a FragMap (which cannot express the M-block split).
    assert_eq!(Acc::frag::<Mfma16x16x16Bf16>(), None, "acc role carries no operand FragMap");
    assert_eq!(Acc::acc::<Mfma16x16x16Bf16>(), Some(Mfma16x16x16Bf16::acc_dist()));
    assert_eq!(Acc::ept::<Mfma16x16x16Bf16>(), Mfma16x16x16Bf16::EPT_C);
    assert_eq!(Acc::acc::<Mfma32x32x8Bf16>(), Some(Mfma32x32x8Bf16::acc_dist()));
    assert_eq!(Acc::ept::<Mfma32x32x8Bf16>(), 16, "32×32×8 accumulator is EPT_C = 16");
}

// ── n_frags matches the movement-layer arithmetic ((R/M)·(C/K) for A, etc.) ──

#[test]
fn n_frags_tracks_tile_over_mfma_shape() {
    // A: (R/M)·(C/K). A 32-row × 128-col K-operand tile at 32×32×8: (32/32)·(128/8) = 16 frags.
    assert_eq!(ARow::n_frags::<Mfma32x32x8Bf16>(32, 128), 16);
    // B: (R/K)·(C/N). A 8×32 tile at 32×32×8: (8/8)·(32/32) = 1.
    assert_eq!(BCol::n_frags::<Mfma32x32x8Bf16>(8, 32), 1);
    // Acc: (R/M)·(C/N). A 32×64 accumulator at 32×32×8: (32/32)·(64/32) = 2.
    assert_eq!(Acc::n_frags::<Mfma32x32x8Bf16>(32, 64), 2);
    // 16×16×16 A: 32×32 → (32/16)·(32/16) = 4.
    assert_eq!(ARow::n_frags::<Mfma16x16x16Bf16>(32, 32), 4);
}

// ── LDS swizzle is a property of the marker, not a threaded parameter ──

#[test]
fn swizzle_comes_from_the_marker() {
    assert_eq!(Plain::layout(128), Layout::contiguous(), "Plain ⇒ contiguous");
    assert_eq!(
        Xor::layout(128),
        Layout { transforms: [Transform::Xor { cols: 128 }].into_iter().collect() },
        "Xor ⇒ the bank XOR over the tile width"
    );
    // The swizzle carries the runtime width through unchanged.
    assert_eq!(Xor::layout(64), Layout { transforms: [Transform::Xor { cols: 64 }].into_iter().collect() },);
}
