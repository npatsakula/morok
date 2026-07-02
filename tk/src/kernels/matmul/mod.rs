//! The bf16→f32 tile matmul, split into a per-GPU module tree. The graph-native
//! entry ([`matmul`]), the per-arch [`MatmulCfg`] + its selector ([`cfg_for_arch`]),
//! and the shared [`K_STEP`] default live here in `mod.rs`; the arch-generic kernel
//! machinery lives in [`common`] and the per-arch specializations under [`amd`].
//!
//! Computes `C = A·Bᵀ` with **B in `[N,K]` layout** (the B[N,K] contract): both operands
//! then have K contiguous, so both gather from LDS as one `ds_read_b64` per fragment
//! (`mma_abt`). One arch-generic, single-buffered builder ([`build_matmul_cfg`], in
//! [`common`]) is driven by a
//! per-arch [`MatmulCfg`]: one `cfg.block × cfg.block` C tile per workgroup,
//! `cfg.n_accum` `reg × reg` accumulators/wave reduced over a tracked K-loop in
//! `cfg.k_step()`-wide strips out of XOR-swizzled LDS. The tile shortcuts
//! ([`Kernel::acc`] / [`Kernel::operand`] / [`Kernel::shared_sw`]) resolve the right
//! WMMA fragment per arch (gfx942 CDNA MFMA vs gfx1151 RDNA WMMA), so one builder
//! serves both.
//!
//! Per-arch tuning lives under [`amd`]: gfx942 runs the software-pipelined inline-`asm`
//! MFMA microkernel (256²/128² by N via
//! [`cfg_for_gfx942`](amd::gfx942::cfg_for_gfx942)) in [`amd::gfx942`]; the
//! occupancy-tuned RDNA3.5 [`GFX1151_CFG`] drives the generic [`gemm_core`] in
//! [`amd::gfx1151`]. [`matmul`] dispatches by arch. The strongest gfx1151 lever is `k_step`: a smaller strip
//! shrinks the live WMMA-input fragment VGPR/lane (each input replicates all
//! `k_step`/16 K-sub-steps), raising occupancy. A port of tinygrad
//! `test_tk.py::test_simple_matmul` lifted to a reusable kernel builder.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::Kernel;

pub mod amd;
mod common;

pub use amd::gfx942::{BLOCK128_CFG, BLOCK256_CFG, SMALL_CFG};
pub(crate) use amd::gfx942::{build_matmul_asm_cfg, cfg_for_gfx942};
pub use amd::gfx1151::GFX1151_CFG;
pub use common::{build_matmul_cfg, build_matmul_splitk, gemm_core};

/// K-reduction step (the LDS strip depth, shared by every config).
pub const K_STEP: usize = 64;

/// Block / wave geometry of a multi-wave matmul: a
/// `wave_rows × wave_cols`-wave workgroup computes a `block × block` C tile,
/// each wave owning `n_accum` col-major `reg × reg` f32 accumulators
/// (`reg = block / wave_cols`) reduced over K in [`K_STEP`]-wide steps. The
/// `wave_cols = wave_rows * n_accum` invariant keeps `reg` square: the M side is
/// split into `wave_rows * n_accum` row-blocks, the N side into `wave_cols`.
#[derive(Clone, Copy)]
pub struct MatmulCfg {
    /// The square C-tile edge (in elements) one workgroup computes.
    pub block: usize,
    /// Wave grid rows — the M side splits into `wave_rows * n_accum` row-blocks.
    pub wave_rows: usize,
    /// Wave grid columns — the N side splits into `wave_cols` col-blocks.
    pub wave_cols: usize,
    /// `reg × reg` f32 accumulators per wave.
    pub n_accum: usize,
    /// Drive `(pid_m, pid_n)` from a flattened 1-D grid via the chiplet/L2
    /// [`l2_swizzle`](crate::grid::l2_swizzle) instead of the plain 2-D
    /// `block_idx`. Grid becomes `[grid² , 1, 1]`.
    pub l2_swizzle: bool,
    /// Fill the GLOBAL→LDS strips with 128-bit (`vec8` bf16) coalesced loads
    /// instead of the scalar/`vec4`-folded path.
    pub vec_load: bool,
    /// K-reduction step (LDS strip depth) for the single-buffered K-loop. Must be a
    /// multiple of 16 (the WMMA K-edge) and divide N. Lowering it cuts the live
    /// operand VGPR/lane (each WMMA input replicates all `k_step`/16 K-sub-steps),
    /// raising occupancy — the dominant occupancy lever on RDNA3.5/wave32
    /// ([`GFX1151_CFG`] uses 32). gfx942 keeps [`K_STEP`]
    /// (64). `0` means "use [`K_STEP`]" so older literal/`..BLOCK256_CFG` builders that
    /// predate the field still get the default — see [`MatmulCfg::k_step`].
    pub k_step: usize,
}

impl MatmulCfg {
    /// The per-accumulator square edge (`block / wave_cols`).
    pub const fn reg(&self) -> usize {
        self.block / self.wave_cols
    }
    /// The K-reduction step, resolving the `0` sentinel (older literal builders) to
    /// the default [`K_STEP`]. The resolved value must be a multiple of 16 (the WMMA
    /// K-edge) and divide N; a violation panics in [`gemm_core`].
    pub const fn k_step(&self) -> usize {
        if self.k_step == 0 { K_STEP } else { self.k_step }
    }
    /// `reg`-blocks per C-tile side (= `wave_cols` = `wave_rows * n_accum`); the
    /// grid→C-block coordinate multiplier.
    pub const fn blocks_per_side(&self) -> usize {
        self.block / self.reg()
    }
    /// Launch block size (threads) = `wave_rows * wave_cols * wave_size`.
    pub const fn threads(&self, wave_size: usize) -> i64 {
        (self.wave_rows * self.wave_cols * wave_size) as i64
    }
    /// Grid edge (`n / block`).
    pub const fn grid(&self, n: usize) -> i64 {
        (n / self.block) as i64
    }
    /// Launch grid for a general `m × n` C: a flattened 1-D `[gm·gn, 1, 1]` when
    /// the chiplet swizzle ([`l2_swizzle`](Self::l2_swizzle)) is on (it re-derives `(pid_m, pid_n)`), else
    /// the plain 2-D `[gn, gm, 1]` (x = n-blocks → `block_idx[0]` = pid_n, y = m-blocks
    /// → `block_idx[1]` = pid_m — matching [`block_coords`](common::block_coords)).
    pub const fn grid_dims_mn(&self, m: usize, n: usize) -> [i64; 3] {
        let (gm, gn) = ((m / self.block) as i64, (n / self.block) as i64);
        if self.l2_swizzle { [gm * gn, 1, 1] } else { [gn, gm, 1] }
    }
    /// Square convenience: [`grid_dims_mn`](Self::grid_dims_mn) with `m = n` (the `[grid², 1, 1]` /
    /// `[grid, grid, 1]` the square matmul launches with).
    pub const fn grid_dims(&self, n: usize) -> [i64; 3] {
        self.grid_dims_mn(n, n)
    }
}

/// Unified per-arch tile config: gfx942 (CDNA wave64) → the asm-microkernel tile
/// (256²/128² by N via [`cfg_for_gfx942`](amd::gfx942::cfg_for_gfx942)); gfx1151
/// (RDNA3.5 wave32) / any future arch → the generic-`gemm_core` [`GFX1151_CFG`]. Paired
/// with [`build_matmul_dispatch`], which builds the matching kernel — one per-arch kernel
/// per card (gfx942 / gfx950 / gfx1250 …).
pub fn cfg_for_arch(arch: svod_dtype::AmdArch, n: usize) -> MatmulCfg {
    match arch {
        svod_dtype::AmdArch::Gfx942 => cfg_for_gfx942(n),
        _ => GFX1151_CFG,
    }
}

/// Build the arch-appropriate **single-kernel** matmul into `ker`: gfx942 → the inline-
/// `asm` MFMA microkernel ([`build_matmul_asm_cfg`]); gfx1151 / other → the generic
/// [`gemm_core`] ([`build_matmul_cfg`]). `cfg` must be [`cfg_for_arch`]`(arch, n)`. This
/// is the ONE per-arch selection point [`matmul`] and the direct-launch tests share, so a
/// test's direct launch matches production on whatever card it runs on (split-K is a
/// separate graph the gfx1151 path layers on top — see [`matmul`]).
pub(crate) fn build_matmul_dispatch(ker: &Kernel, arch: svod_dtype::AmdArch, n: usize, cfg: MatmulCfg) {
    match arch {
        svod_dtype::AmdArch::Gfx942 => build_matmul_asm_cfg(ker, n, cfg, true),
        _ => build_matmul_cfg(ker, n, cfg),
    }
}

/// The GPU arch(es) the tile matmul is built for: gfx942 (CDNA MFMA, wave64) and
/// gfx1151 (RDNA3.5 WMMA, wave32 — the `_W32_*` fragment shapes). The launcher
/// gates against this; see [`crate::target::check_target`].
/// Validated on gfx942 (CDNA3) and gfx1151 (RDNA3.5).
pub const MATMUL_SUPPORTED_ARCHS: &[svod_dtype::AmdArch] = &[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151];

/// **Graph-native** `n×n` matrix multiply — returns a lazy output [`Tensor`] (a
/// `custom_kernel` / `Op::Call` node), the matmul peer of [`crate::flash_attention`].
/// Composes into a model graph and realizes / benchmarks through the normal
/// `prepare()` → `execute_profiled` path like any other tensor op.
///
/// Computes `C = A·Bᵀ`: **`b` is the `[N, K]` operand** (the B[N,K] contract — B stored
/// N-major so both operands gather K-contiguous; see the module docs). `a`/`b` are
/// square `[n, n]` of **any float dtype**: they are cast to bf16 internally (the
/// kernel is a bf16-input matrix-engine GEMM), and the result is the f32 WMMA/MFMA
/// accumulator. So a caller needs no kernel knowledge — pass plain tensors, get a
/// tensor back. The per-arch occupancy config is picked by [`cfg_for_arch`].
///
/// Like [`crate::flash_attention_with`], the outcome is three-way (via
/// [`crate::launch_custom`]): `Ok(None)` when the device can't run the kernel,
/// `Err` when the request is malformed (an operand that isn't a statically-shaped
/// rank-2 tensor, non-square operands, or a size that isn't a multiple of the arch's
/// block), `Ok(Some)` when it ran.
///
/// ```no_run
/// use svod_tensor::Tensor;
/// let a = Tensor::randn(&[256, 256]).unwrap();
/// let b = Tensor::randn(&[256, 256]).unwrap(); // B is [N, K]: result is A·Bᵀ
/// if let Some(mut c) = svod_tk::matmul(&a, &b).unwrap() { // lazy bf16→f32 GEMM node
///     c.prepare().unwrap();                                // realize through the scheduler
/// }
/// ```
pub fn matmul(a: &Tensor, b: &Tensor) -> crate::LaunchResult<Option<Tensor>> {
    use snafu::{ResultExt, ensure};

    let ad = crate::launch::concrete_dims(a, "matmul", "a", 2)?;
    let bd = crate::launch::concrete_dims(b, "matmul", "b", 2)?;
    let (am, an) = (ad[0], ad[1]);
    let (bm, bn) = (bd[0], bd[1]);
    let n = am;

    crate::launch_custom(
        &a.device(),
        MATMUL_SUPPORTED_ARCHS,
        // Operands must be square + equal-sized; `n % block` (arch-dependent) is checked
        // in `build`. Both are structural request errors (`Err`), not fallback triggers.
        move |_arch| {
            ensure!(
                an == am && bm == am && bn == am,
                crate::launch::NotSquareSnafu { kernel: "matmul", a: [am, an], b: [bm, bn] }
            );
            Ok(())
        },
        true, // no runtime-applicability fallback — a bad size is an error, not `None`.
        move |profile| {
            let caps = profile.caps;
            let cfg = cfg_for_arch(caps.arch, n);
            ensure!(
                n % cfg.block == 0,
                crate::launch::DimMultipleSnafu { kernel: "matmul", dim: "n", value: n, multiple: cfg.block }
            );
            // Operands → bf16 (the matrix-engine operand dtype). ELIDE the cast when the
            // operand is already bf16 (`Tensor::cast` does not) so bf16-native callers bind
            // the ABI's bf16 globals directly with no cast node. Output stays f32.
            let to_bf16 = |t: &Tensor| -> crate::LaunchResult<Tensor> {
                if t.uop().dtype() == DType::BFloat16 {
                    Ok(t.clone())
                } else {
                    t.cast(DType::BFloat16).context(crate::launch::OperandSnafu)
                }
            };
            let a_bf = to_bf16(a)?;
            let b_bf = to_bf16(b)?;
            let threads = cfg.threads(caps.wave_size);
            // Split-K only on the generic path (gfx1151): the gfx942 asm microkernel's
            // per-workgroup MFMA duty beats the split-K generic kernel at every size (asm
            // TF 61/218/415/490 vs 57/171/339/401 @1024/2048/4096/8192), so it absorbs the
            // small-N grid-starvation via its smaller 128² tile instead of splitting K.
            let k_splits = if caps.arch == svod_dtype::AmdArch::Gfx942 {
                1
            } else {
                common::split_k_for(n, cfg, profile.cu_count)
            };
            if k_splits > 1 {
                // Split-K: a `[(n/block)², k_splits]` grid keeps a small problem's grid
                // wide enough to fill the machine. Each workgroup writes a partial tile
                // to `scratch[k_slice]`; the graph reduce sums them into the result.
                let g = (n / cfg.block) as i64;
                let scratch = Tensor::empty(&[k_splits, n, n], DType::Float32);
                let partial = crate::graph_launch(
                    "matmul_splitk",
                    [g * g, k_splits as i64, 1],
                    threads,
                    scratch,
                    &[&a_bf, &b_bf],
                    caps,
                    move |ker| {
                        build_matmul_splitk(ker, n, cfg, k_splits);
                        ker.finish(cfg.n_accum)
                    },
                )?;
                Ok(partial.sum(0).expect("split-K: reduce scratch over k_slice"))
            } else {
                // Single kernel — the arch-appropriate build (gfx942 asm / gfx1151 generic).
                let out = Tensor::empty(&[n, n], DType::Float32);
                crate::graph_launch("matmul", cfg.grid_dims(n), threads, out, &[&a_bf, &b_bf], caps, move |ker| {
                    build_matmul_dispatch(ker, caps.arch, n, cfg);
                    ker.finish(cfg.n_accum)
                })
            }
        },
    )
}
