//! WMMA/MFMA matrix-multiply tile ops — the four `mma_{ab,abt,atb,atbt}`
//! variants and their shared looped/unrolled bodies. Each `mma` reduces over the
//! K-edge with one [`Op::Wmma`](svod_ir::Op::Wmma) per K-iteration.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{AmdArch, DType};
use svod_ir::{AxisType, RendererDevice, UOp, WmmaMetadata, WmmaUpcastAxes};
use svod_schedule::optimizer::{Renderer, TensorCore};

use super::Group;
use crate::index::{Idx, flat_index, load_vec_at};
use crate::tile::RT;

/// Bridge a scheduler [`TensorCore`] (the per-arch×dtype matrix-op table, the
/// single source of truth — `schedule::optimizer::renderer`) into the IR
/// [`WmmaMetadata`] that a hand-built [`Op::Wmma`](svod_ir::Op::Wmma) consumes.
///
/// `dims`/`dtype_in`/`dtype_out`/`threads`/`tile_grid` copy straight across. The
/// `upcast_axes` are `log2(elements_per_thread)` size-2 entries per operand
/// (mirrors the optimizer's `tc.rs` construction, where every upcast/reduce split
/// is by 2); the axis-id *values* are cosmetic on tk's expander-free direct path —
/// codegen/devectorize read only the sizes — so we descend from 4, which
/// reproduces the prior hand layout `[(4,2),(3,2)]` for the gfx942 16×16×16 case.
/// `reduce_axes` is empty: tk's `mma` carries the K reduce as its own `inner`
/// range, not inside the WMMA metadata.
fn wmma_from_tc(tc: &TensorCore, device: RendererDevice) -> WmmaMetadata {
    let axes = |ept: usize| -> Vec<(usize, usize)> { (0..(ept as f64).log2() as usize).map(|i| (4 - i, 2)).collect() };
    WmmaMetadata {
        name: format!("WMMA_{}_{}_{}_{:?}_{:?}", tc.dims.0, tc.dims.1, tc.dims.2, tc.dtype_in, tc.dtype_out),
        dims: tc.dims,
        dtype_in: tc.dtype_in.clone(),
        dtype_out: tc.dtype_out.clone(),
        device,
        threads: tc.threads,
        upcast_axes: WmmaUpcastAxes {
            a: axes(tc.elements_per_thread.0),
            b: axes(tc.elements_per_thread.1),
            c: axes(tc.elements_per_thread.2),
        },
        reduce_axes: vec![],
        tile_grid: tc.tile_grid,
        asm: false,
    }
}

/// The K=16 WMMA descriptor for input dtype `dtype_in` on `arch`, looked up from
/// the shared per-arch tensor-core table (`Renderer::for_amd_arch`) rather than
/// re-encoded here — so bf16/f16 on CDNA's MFMA cores and the RDNA wave32 cores
/// come from one source. Both accumulate in f32. `arch` is threaded from
/// [`crate::ArchCaps::arch`] (gfx942 in practice; the table already carries the
/// RDNA cores for when a wave32 arch is enabled).
fn wmma_desc(arch: AmdArch, dtype_in: &DType) -> WmmaMetadata {
    let ren = Renderer::for_amd_arch(arch);
    let tc =
        ren.tensor_cores.iter().find(|tc| &tc.dtype_in == dtype_in && tc.dims == (16, 16, 16)).unwrap_or_else(|| {
            // Precondition violation by the kernel author, not end-user input: the
            // matrix-core operand dtype must be bf16/f16 (the only 16×16×16 WMMA
            // inputs). The USE-face kernels pre-cast; an AUTHOR calling `mma_*`
            // with an unsupported RT dtype lands here.
            unimplemented!(
                "mma: operand dtype {dtype_in:?} has no 16×16×16 WMMA on {arch:?} — operands must be bf16 or f16"
            )
        });
    wmma_from_tc(tc, ren.device)
}

/// Per-lane element count for a WMMA operand = product of its upcast-axis sizes
/// (`wmma_from_tc` builds these as `log2(elements_per_thread)` size-2 entries, so
/// the product is the elements-per-thread). gfx942 16×16×16 → A/B/C = 4/4/4; RDNA
/// → 16/16/8 (replicated 16-wide inputs, 8-wide accumulator). Empty axes ⇒ 1.
fn upcast_count(axes: &[(usize, usize)]) -> i64 {
    axes.iter().map(|(_, sz)| *sz as i64).product()
}

impl<'k> Group<'k> {
    /// `C += A·B` over a tile (tinygrad `mma_AB`): for every output fragment
    /// `(height, width)` accumulate `WMMA(A[height,inner], B[inner,width])`
    /// across the reduce axis `inner`. One [`Op::Wmma`](svod_ir::Op::Wmma) per
    /// K-iteration → one `mfma.f32.16x16x16bf16.1k`.
    ///
    /// # Panics
    /// The operand tiles `a`/`b` must be **bf16 or f16** — the only 16×16×16
    /// matrix-core input dtypes. An operand of any other dtype panics (a kernel-
    /// authoring error). Accumulation is always f32; this precondition holds for
    /// all four `mma_{ab,abt,atb,atbt}` variants. Also panics unless the operand
    /// tile's base is the 16-column WMMA base, and on an operand-rank mismatch
    /// (the index permutation reads the trailing fragment-grid axes).
    pub fn mma_ab(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, false, false, false)
    }

    /// `C += A·Bᵀ` (tinygrad `mma_ABt`): B fragment is read transposed
    /// (`b[width, inner]`); reduce axis stays `a.shape[-2]`.
    ///
    /// # Panics
    /// Panics on an unsupported operand dtype, unless the operand base is the
    /// 16-column WMMA base, and on an operand-rank mismatch (see [`Self::mma_ab`]).
    pub fn mma_abt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, false, true, false)
    }

    /// [`Self::mma_abt`] emitting the matrix op as an inline-`asm sideeffect` MFMA
    /// (opaque to the AMDGPU machine scheduler, so the inner-loop program order
    /// survives `-O3`) instead of the `@llvm.amdgcn.mfma.*` intrinsic. The explicit
    /// per-call counterpart of the old kernel-global `asm_mfma` mode — only the
    /// gfx942 asm microkernel ([`crate::kernels::matmul`] `gemm_core_asm`) calls it.
    /// Valid only for the f32-accumulating bf16 K=16 MFMA (the shape it is used for).
    pub fn mma_abt_asm(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, false, true, true)
    }

    /// `C += Aᵀ·B` (tinygrad `mma_AtB`): A fragment is read transposed
    /// (`a[inner, height]`) and the reduce axis is `a.shape[-3]`.
    ///
    /// # Panics
    /// Panics on an unsupported operand dtype, unless the operand base is the
    /// 16-column WMMA base, and on an operand-rank mismatch (see [`Self::mma_ab`]).
    pub fn mma_atb(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, true, false, false)
    }

    /// [`Self::mma_atb`] emitting the matrix op as an inline-`asm sideeffect` MFMA
    /// (opaque to the AMDGPU machine scheduler, so the inner-loop program order
    /// survives `-O3`) instead of the `@llvm.amdgcn.mfma.*` intrinsic — the [`Self::mma_atb`]
    /// counterpart of [`Self::mma_abt_asm`]. Used by the gfx942 flash-attention microkernel
    /// (both the QKᵀ and A·V matmuls are `AtB`). Valid only for the f32-accumulating
    /// bf16 K=16 MFMA; on non-CDNA targets the asm flag is inert (the intrinsic renders).
    pub fn mma_atb_asm(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, true, false, true)
    }

    /// `C += Aᵀ·Bᵀ` (tinygrad `mma_AtBt`): both fragments read transposed.
    ///
    /// # Panics
    /// Panics on an unsupported operand dtype, unless the operand base is the
    /// 16-column WMMA base, and on an operand-rank mismatch (see [`Self::mma_ab`]).
    pub fn mma_atbt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, true, true, false)
    }

    /// The shared WMMA body. The four `mma_{AB,ABt,AtB,AtBt}` variants differ
    /// only in the operand index permutation and the reduce-axis selection:
    /// - `a_t` (Aᵀ): A is read `a[inner, height]` and the reduce axis is
    ///   `a.shape[-3]`; otherwise `a[height, inner]`, reduce axis `a.shape[-2]`.
    /// - `b_t` (Bᵀ): B is read `b[width, inner]`; otherwise `b[inner, width]`.
    fn mma(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>, a_t: bool, b_t: bool, asm: bool) -> RT<'k> {
        // Flat (cross-tile-pipeline) FA opts into the fully-unrolled body so the
        // QKᵀ / A·V MFMAs render loop-free for the attention scheduling comb.
        if self.ker.unrolled() {
            return self.mma_u(c, a, b, a_t, b_t, asm);
        }
        // Wave-agnostic: each wave runs the WMMA on its own per-lane RT operands
        // (the wave sub-tile selection happens in the LDS→REG load, not here). The
        // per-lane operand widths come from the descriptor (gfx942 4/4/4; RDNA
        // 16/16/8), not a hardcoded 4.
        assert_eq!(a.base.base.cols, 16, "mma: only the 16-col WMMA base is supported");
        let mut meta = wmma_desc(self.ker.caps.arch, a.elem());
        meta.asm = asm;
        let (a_w, b_w, c_w) =
            (upcast_count(&meta.upcast_axes.a), upcast_count(&meta.upcast_axes.b), upcast_count(&meta.upcast_axes.c));

        let h_end = c.shape()[c.shape().len() - 3] as i64;
        let w_end = c.shape()[c.shape().len() - 2] as i64;
        let k_end = if a_t { a.shape()[a.shape().len() - 3] } else { a.shape()[a.shape().len() - 2] } as i64;
        let height = self.ker.raw_range(h_end, AxisType::Loop);
        let width = self.ker.raw_range(w_end, AxisType::Loop);
        let inner = self.ker.raw_range(k_end, AxisType::Reduce);

        // ONE vector load per operand (the `ept` run is unit-stride) instead of `ept`
        // scalar loads + insertelement — the base index carries the loop ranges, the
        // vector width is the static `ept`.
        let a_idx0 = if a_t {
            [Idx::from(&inner), Idx::from(&height), Idx::Const(0)]
        } else {
            [Idx::from(&height), Idx::from(&inner), Idx::Const(0)]
        };
        let a_in = load_vec_at(a.uop(), a.shape(), &a_idx0, a_w as usize);
        let b_idx0 = if b_t {
            [Idx::from(&width), Idx::from(&inner), Idx::Const(0)]
        } else {
            [Idx::from(&inner), Idx::from(&width), Idx::Const(0)]
        };
        let b_in = load_vec_at(b.uop(), b.shape(), &b_idx0, b_w as usize);
        // The accumulator read must depend on the reduce range `inner`, or it is
        // loop-invariant w.r.t. the K loop and gets hoisted *out* of it — every
        // K-iteration would then re-read the pre-loop C and the WMMA's
        // accumulation chain breaks. Mirrors svod's `reduce_to_acc`
        // (`acc.after([..reduce_range]).index(..)`): the `After([inner])` keeps
        // the read inside the K loop so it observes the prior iteration's store.
        let c_acc = c.uop().after(smallvec![inner.clone()]);
        let d_in =
            load_vec_at(&c_acc, c.shape(), &[Idx::from(&height), Idx::from(&width), Idx::Const(0)], c_w as usize);

        let out = UOp::wmma(a_in, b_in, d_in, meta);
        // ONE vector store of the whole `<c_w x f32>` fragment result.
        let c_store = flat_index(c.uop(), c.shape(), &[Idx::from(&height), Idx::from(&width), Idx::Const(0)])
            .store(out)
            .end(smallvec![height, width, inner]);
        self.finalize_reg(c, c_store)
    }

    /// Fully **unrolled** [`Self::mma`]: emit one [`Op::Wmma`](svod_ir::Op::Wmma)
    /// per `(height, width, k)` fragment via Rust `for` loops — **no inner
    /// `RANGE`** — so the MFMAs render as a *flat* schedulable LLVM region the
    /// attention scheduling comb can weave the online softmax through. tk's
    /// direct-launch path skips the optimizer's `pre_expand`, so the looped
    /// [`Self::mma`] stays rolled (three `loop_body_*` around the mfma); explicit
    /// unroll is the only way to flatten it (the cheap axis-flip is dead
    /// on the direct path).
    ///
    /// Each fragment's K-accumulation chains (`c[h,w]`'s k-step read observes the
    /// k−1 store); fragments chain into one terminal store so the enclosing rolled
    /// KV loop's `END` scopes them all (cf. the matmul accumulator chain,
    /// `kernels/matmul.rs:201`). Bit-identical accumulation order to [`Self::mma`].
    fn mma_u(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>, a_t: bool, b_t: bool, asm: bool) -> RT<'k> {
        assert_eq!(a.base.base.cols, 16, "mma_u: only the 16-col WMMA base is supported");
        let mut meta = wmma_desc(self.ker.caps.arch, a.elem());
        meta.asm = asm;
        let (a_w, b_w, c_w) =
            (upcast_count(&meta.upcast_axes.a), upcast_count(&meta.upcast_axes.b), upcast_count(&meta.upcast_axes.c));

        let h_end = c.shape()[c.shape().len() - 3] as i64;
        let w_end = c.shape()[c.shape().len() - 2] as i64;
        let k_end = if a_t { a.shape()[a.shape().len() - 3] } else { a.shape()[a.shape().len() - 2] } as i64;

        // Fragment-scoping chain: each fragment's first (k=0) accumulator read
        // orders after the previous fragment's terminal store, so the LAST
        // fragment's store transitively scopes them all under one loop `END`.
        let mut prev_frag: Option<Arc<UOp>> = None;
        for h in 0..h_end {
            for w in 0..w_end {
                // Per-fragment K accumulation: the k-step read observes the k−1
                // store to this same fragment (the unrolled analog of the looped
                // `c.after([inner])` loop-carry).
                let mut frag_prev: Option<Arc<UOp>> = None;
                for k in 0..k_end {
                    // ONE vector load of the fragment's `ept` run (unit-stride trailing
                    // axis) instead of `ept` scalar loads + insertelement — collapses the
                    // per-element load/shuffle bloat that inflated VGPR pressure.
                    let a_idx0 = if a_t {
                        [Idx::Const(k), Idx::Const(h), Idx::Const(0)]
                    } else {
                        [Idx::Const(h), Idx::Const(k), Idx::Const(0)]
                    };
                    let a_in = load_vec_at(a.uop(), a.shape(), &a_idx0, a_w as usize);
                    let b_idx0 = if b_t {
                        [Idx::Const(w), Idx::Const(k), Idx::Const(0)]
                    } else {
                        [Idx::Const(k), Idx::Const(w), Idx::Const(0)]
                    };
                    let b_in = load_vec_at(b.uop(), b.shape(), &b_idx0, b_w as usize);
                    // Accumulator source: the prior k-step's store for this
                    // fragment; on k==0 the incoming `c` carrying the
                    // fragment-scoping dep on the previous fragment's store.
                    let mut deps: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();
                    match &frag_prev {
                        Some(fp) => deps.push(fp.clone()),
                        None => {
                            if let Some(pf) = &prev_frag {
                                deps.push(pf.clone());
                            }
                        }
                    }
                    // Anchor the incoming accumulator read (no chain dep yet) to the
                    // enclosing rolled loop so a carried accumulator (`o_reg`) is not
                    // hoisted out (see `Group::anchor`); subsequent k/fragment reads
                    // chain through their stores, which are already loop-scoped.
                    let c_src = if deps.is_empty() { self.anchor(c.uop()) } else { c.uop().after(deps) };
                    let d_in =
                        load_vec_at(&c_src, c.shape(), &[Idx::Const(h), Idx::Const(w), Idx::Const(0)], c_w as usize);
                    let out = UOp::wmma(a_in, b_in, d_in, meta.clone());
                    // ONE vector store of the whole `<c_w x f32>` fragment result.
                    frag_prev =
                        Some(flat_index(c.uop(), c.shape(), &[Idx::Const(h), Idx::Const(w), Idx::Const(0)]).store(out));
                }
                prev_frag = frag_prev;
            }
        }
        let terminal = prev_frag.expect("mma_u: at least one (height, width) fragment");
        self.finalize_reg(c, terminal)
    }
}
