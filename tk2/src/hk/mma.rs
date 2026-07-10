//! HipKittens MFMA (`ops/warp/register/tile/mma.cuh`) as thin tk2 wrappers over the intrinsic
//! [`Builder::mma`] — which already renders the oracle's exact
//! `@llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32 0, i32 0, i32 0)`
//! (GAP-2: verified, no new code needed). `zero` maps to the SROA-promotable `zero_init_frag`.

#![allow(non_snake_case)]

use crate::build::{BF16, Builder, Effect, F32, Val};
use crate::hk::types::rt_fl;

/// Elements per lane per fragment (4 bf16 / 4 f32 for gfx942 16×16×16).
const EPT: usize = 4;

/// HK's `mfma161616` (bf16 overload, `mma.cuh:28`) — ONE
/// `v_mfma_f32_16x16x16_bf16` (`d = a·bᵀ + c`, cbsz/abid/blgp = 0). Wraps the intrinsic
/// [`Builder::mma`], which renders `mfma.f32.16x16x16bf16.1k` verbatim.
pub fn mfma161616(b: &mut Builder, a: Val<BF16>, bt: Val<BF16>, c: Val<F32>) -> Val<F32> {
    b.mma(a, bt, c, EPT)
}

/// HK's `mma_ABt_base` (bf16 overload, `mma.cuh:103`) — one base-fragment MFMA, delegating to
/// [`mfma161616`].
pub fn mma_ABt_base(b: &mut Builder, a: Val<BF16>, bt: Val<BF16>, c: Val<F32>) -> Val<F32> {
    mfma161616(b, a, bt, c)
}

/// HK's `mma_ABt` (`mma.cuh:242`) — `D = A·Bᵀ + C` over the `ri × cj` output subtile grid
/// (`ri = D.height = A.rows/16`, `cj = D.width = B.rows/16`; `A.width == 1`, so only k = 0). One
/// `mma_ABt_base` per `(n, m)`: `D[n][m] = A[n] · B[m]ᵀ + C[n][m]` — `ri·cj` MFMAs (16 for `<64,64>`).
///
/// `a` = the `ri` A-operand fragments (Row), `bt` = the `cj` B-operand fragments (Row, indexed by the
/// output column as its row — the transpose), `c` = the `ri·cj` accumulator inputs (row-major).
pub fn mma_ABt(
    b: &mut Builder,
    a: &[Val<BF16>],
    bt: &[Val<BF16>],
    c: &[Val<F32>],
    ri: usize,
    cj: usize,
) -> Vec<Val<F32>> {
    assert_eq!(a.len(), ri, "mma_ABt: A operand count = ri");
    assert_eq!(bt.len(), cj, "mma_ABt: B operand count = cj");
    assert_eq!(c.len(), ri * cj, "mma_ABt: accumulator count = ri·cj");
    let mut out = Vec::with_capacity(ri * cj);
    for n in 0..ri {
        for m in 0..cj {
            out.push(mma_ABt_base(b, a[n], bt[m], c[n * cj + m]));
        }
    }
    out
}

/// HK's `zero(C_accum)` (`maps.cuh:377` → `base_ops::zero`) — VGPR-zero every fragment of the
/// accumulator. Uses [`Builder::zero_init_frag`] (ONE constant-index `<4×f32>` vector store per
/// fragment) so the accumulator SROA-promotes to a loop-carried `<4 x float>` phi (the mechanism that
/// keeps the 32-MFMA clusters from fracturing — DESIGN §1/GAP-3). Returns the per-fragment seed
/// effects (each threaded into the first loop-carried accumulator read).
pub fn zero(b: &mut Builder, acc: &rt_fl) -> Vec<Effect> {
    acc.frags.iter().map(|&f| b.zero_init_frag(f)).collect()
}
