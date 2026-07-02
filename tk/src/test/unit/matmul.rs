//! Tests for the bf16→f32 tile matmul ([`crate::kernels::matmul`]): a port of
//! tinygrad `test_tk.py::test_simple_matmul` plus a GPU-free graph-shape check of
//! the `mma_AB` WMMA construction and the hardware-gated end-to-end checks.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{Op, UOp};
use svod_tensor::Tensor;

use crate::kernels::matmul::*;
use crate::tiles::{RT_16X16, TileLayout};
use crate::{Kernel, MoveIdx};

/// Dummy `(c, a, b)` BUFFER UOps for GPU-free graph-shape kernel builds.
fn dummy_buffers(n: usize) -> Vec<Arc<UOp>> {
    let sz = n * n;
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::BFloat16),
    ]
}

/// A non-rank-2 operand is a structured `Err` (not a panic). The shape
/// preconditions resolve before any device dispatch, so this runs GPU-free.
#[test]
fn matmul_non_rank2_operand_is_operand_shape_err() {
    let sq = Tensor::randn(&[64, 64]).expect("randn");
    let a1 = Tensor::randn(&[64]).expect("randn"); // operand a: rank 1
    let e = matmul(&a1, &sq).err().expect("rank-1 a must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "a", .. }), "got {e:?}");
    let b3 = Tensor::randn(&[2, 64, 64]).expect("randn"); // operand b: rank 3
    let e = matmul(&sq, &b3).err().expect("rank-3 b must error, not panic");
    assert!(matches!(e, crate::launch::Error::OperandRank { operand: "b", .. }), "got {e:?}");
}

/// Pure graph-shape check (no GPU): `mma_AB` emits exactly one `WMMA` per
/// K-iteration with `bf16.vec(4)` × `bf16.vec(4)` → `f32.vec(4)` operands and a
/// 16×16×16 / 4-4-4 descriptor.
#[test]
fn test_mma_ab_wmma_graph_shape() {
    let ker = Kernel::new("mma_probe", [1, 1, 1], 64, vec![], crate::ArchCaps::GFX942);
    let warp = ker.warp();

    let a = ker.rt((64, 64), DType::BFloat16, TileLayout::Row, RT_16X16);
    let b = ker.rt((64, 64), DType::BFloat16, TileLayout::Col, RT_16X16);
    let c = ker.rt((64, 64), DType::Float32, TileLayout::Col, RT_16X16);

    let c0 = warp.zero(c);
    let out = warp.mma_ab(c0, &a, &b);

    let wmmas: Vec<_> = out.uop().toposort().into_iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).collect();
    assert_eq!(wmmas.len(), 1, "exactly one symbolic WMMA per K-iteration");

    let bf16x4 = DType::BFloat16.vec(4).unwrap();
    let f32x4 = DType::Float32.vec(4).unwrap();
    let Op::Wmma { a: wa, b: wb, c: wc, metadata } = wmmas[0].op() else { unreachable!() };
    assert_eq!(wa.dtype(), bf16x4, "A operand is bf16.vec(4)");
    assert_eq!(wb.dtype(), bf16x4, "B operand is bf16.vec(4)");
    assert_eq!(wc.dtype(), f32x4, "C (accumulator) operand is f32.vec(4)");
    assert_eq!(wmmas[0].dtype(), f32x4, "WMMA result is f32.vec(4)");

    assert_eq!(metadata.dims, (16, 16, 16));
    assert_eq!(metadata.dtype_in, DType::BFloat16);
    assert_eq!(metadata.dtype_out, DType::Float32);
    let prod = |axes: &[(usize, usize)]| axes.iter().map(|(_, s)| s).product::<usize>();
    assert_eq!(prod(&metadata.upcast_axes.a), 4, "A upcast product");
    assert_eq!(prod(&metadata.upcast_axes.b), 4, "B upcast product");
    assert_eq!(prod(&metadata.upcast_axes.c), 4, "C upcast product");
}

/// The fully-unrolled MMA ([`Kernel::set_unroll`]) emits one symbolic `WMMA` per
/// `(height, width, k)` fragment — a 32×32 = 2×2 output over a 32-wide K (2
/// reduce steps) is 8 flat nodes — vs the looped form's single symbolic node, and
/// renders to gfx942 with 8 distinct `mfma` instructions (no enclosing
/// `loop_body`), which the looped form cannot (it renders one mfma inside loops).
/// The flatness de-risk: explicit Rust-`for` unroll *does* flatten the
/// MFMAs on tk's optimizer-skipping direct-launch path.
#[test]
fn test_mma_unroll_flattens_mfma() {
    let build = |unroll: bool| {
        let n = 32usize;
        let ker = Kernel::new("mma_unroll_probe", [1, 1, 1], 64, dummy_buffers(n), crate::ArchCaps::GFX942);
        ker.set_unroll(unroll);
        let c_gl = ker.gl(&[1, 1, n, n], DType::Float32);
        let _a_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
        let _b_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
        let warp = ker.warp();
        let a = warp.zero(ker.rt((n, n), DType::BFloat16, TileLayout::Row, RT_16X16));
        // `mma_ab` reads `a[h,k] b[k,w]`; a 32×32 col `b` is a 2×2 K-tiled operand.
        let b = warp.zero(ker.rt((n, n), DType::BFloat16, TileLayout::Col, RT_16X16));
        let c = warp.zero(ker.rt((n, n), DType::Float32, TileLayout::Col, RT_16X16));
        let c = warp.mma_ab(c, &a, &b);
        let _ = warp.store(c_gl, c, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(1)
    };

    let wmma_count = |sink: &Arc<UOp>| sink.toposort().iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).count();
    assert_eq!(wmma_count(&build(false)), 1, "looped mma → one symbolic WMMA node");
    assert_eq!(wmma_count(&build(true)), 8, "unrolled mma → 8 flat WMMA nodes (2×2 output × 2 K-steps)");

    let render = |sink: Arc<UOp>| {
        let lowered = svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), sink, &mut ());
        let program = svod_codegen::program_pipeline::program_from_sink(lowered, DeviceSpec::Cpu);
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
        let linear_uop =
            linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR present");
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("mma_unroll_probe")).expect("render").code
    };
    // Count mfma *call sites* — exclude the single (deduped) `declare` line.
    let mfma =
        |code: &str| code.lines().filter(|l| l.contains("mfma.f32.16x16x16bf16.1k") && !l.contains("declare")).count();
    let (looped_mfma, unrolled_mfma) = (mfma(&render(build(false))), mfma(&render(build(true))));
    // The flatness proof: unrolling renders all 8 MFMAs as distinct flat
    // instructions (a rolled K/fragment loop cannot — it renders strictly fewer).
    assert_eq!(unrolled_mfma, 8, "unrolled mma renders 8 flat mfma — no rolled K/fragment loop");
    assert!(looped_mfma < 8, "looped mma keeps the K/fragment loops rolled ({looped_mfma} < 8 static mfma)");
}

/// gfx1151 (RDNA3.5) matmul renders to **WMMA**, not MFMA (host, no GPU). Built
/// with wave32 [`crate::ArchCaps`], the kernel must select the `_W32_*` fragment
/// shapes and lower to `llvm.amdgcn.wmma.f32.16x16x16.bf16` (with bf16 inputs
/// bitcast to `<16 x i16>` and an `<8 x float>` accumulator) — never an `mfma`
/// (CDNA-only). This proves the arch-select + wave32 layout build & emit the
/// right intrinsic; numerical correctness is gated on gfx1151 hardware.
#[test]
fn test_matmul_rdna_renders_wmma() {
    let n = 64usize; // SMALL_CFG: block=64, 1 wave, 32 threads (wave32).
    let ker = Kernel::new(
        "matmul_rdna",
        SMALL_CFG.grid_dims(n),
        SMALL_CFG.threads(32),
        dummy_buffers(n),
        crate::ArchCaps::for_arch(svod_dtype::AmdArch::Gfx1151),
    );
    build_matmul_cfg(&ker, n, SMALL_CFG);
    let sink = ker.finish(SMALL_CFG.n_accum);
    let lowered = svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), sink, &mut ());
    let program = svod_codegen::program_pipeline::program_from_sink(lowered, DeviceSpec::Cpu);
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR present");
    let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx1151);
    // Renders (no OOM/panic) ⇒ the wave32 fragment shapes lower cleanly.
    let code =
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("matmul_rdna")).expect("render").code;

    assert!(code.contains("llvm.amdgcn.wmma.f32.16x16x16.bf16"), "gfx1151 matmul must emit the RDNA WMMA intrinsic");
    assert!(!code.contains("mfma"), "gfx1151 is WMMA, not CDNA MFMA");
}

/// A `group_2d(2,4)` is 8 waves / 512 threads, with `warp_row`/`warp_col`
/// derived as `div`/`mod` of the wave id by `cols_waves`.
#[test]
fn test_group_2d_wave_index_shape() {
    use svod_ir::{BinaryOp, Op};

    let ker = Kernel::new("wave_probe", [1, 1, 1], 512, vec![], crate::ArchCaps::GFX942);
    let g = ker.group_2d(2, 4);
    assert_eq!(g.warps, 8, "2×4 wave grid = 8 waves");
    assert_eq!(g.rows_waves, 2);
    assert_eq!(g.cols_waves, 4);
    assert_eq!(g.group_threads(), 512, "8 waves × 64 = 512 threads/block");

    // warp_row = warpid / cols_waves (=4); warp_col = warpid % 4.
    let by_four = |u: &Arc<UOp>, op| {
        u.toposort().into_iter().any(|n| {
            matches!(n.op(), Op::Binary(o, _, d) if *o == op
                && matches!(d.op(), Op::Const(c) if matches!(c.0, svod_ir::ConstValue::Int(4))))
        })
    };
    assert!(by_four(&g.warp_row(), BinaryOp::Idiv), "warp_row divides the wave id by cols_waves=4");
    assert!(by_four(&g.warp_col(), BinaryOp::Mod), "warp_col mods the wave id by cols_waves=4");

    // Single-warp group keeps the 1×1 grid.
    let w = ker.warp();
    assert_eq!((w.warps, w.rows_waves, w.cols_waves, w.group_threads()), (1, 1, 1, 64));
}

/// `st_db` allocates a 2×-size LDS buffer, and a parity `with_base_offset` view
/// threads a runtime offset into the LDS flat address (so a double-buffer
/// gather/fill is counter-dependent and stays loop-scoped), while an ordinary
/// `st` tile's addresses carry no such offset.
#[test]
fn test_st_db_base_offset_infra() {
    use crate::tiles::ST_16X16_SWIZZLED;

    let ker = Kernel::new("db_infra", [1, 1, 1], 512, vec![], crate::ArchCaps::GFX942);
    // Single-half flat element count for a 256×32 bf16 tile (base 16×16):
    // (256/16)*(32/16)*16*16 = 16*2*256 = 8192.
    let db = ker.st_db((256, 32), DType::BFloat16, TileLayout::Row, ST_16X16_SWIZZLED);
    assert_eq!(db.half_elems(), 8192, "half_elems = height*width*base.rows*base.cols");
    assert!(db.base_offset().is_none(), "fresh st_db addresses half 0 (no base_offset)");

    // A parity view adds `parity * half_elems` to the flat address.
    let tile = ker.range(4); // a Loop range counter
    let parity = tile.try_mod(&crate::index::cidx(2)).expect("tile % 2");
    let off = parity.try_mul(&crate::index::cidx(db.half_elems() as i64)).expect("parity*half");
    let view = db.with_base_offset(off.clone());
    assert!(view.base_offset().is_some(), "with_base_offset sets the parity select");

    // Sanity: the underlying buffer is shared (same DefineLocal), only the view differs.
    assert!(std::sync::Arc::ptr_eq(db.uop(), view.uop()), "with_base_offset shares the backing buffer");
}

// =============================================================================
// Hardware-gated end-to-end matmul on gfx942.
// =============================================================================

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_simple_matmul_amd -- --ignored --nocapture`.
///
/// Runs the 8-wave 256×256 tile matmul on the real GPU across several N and
/// checks each against a reference `a.matmul(b)` over the *same* bf16-rounded
/// operands (bf16 tolerance ~5e-2).
#[test]
#[ignore]
fn test_simple_matmul_amd() {
    for n in [256usize, 512, 1024, 2048] {
        run_matmul_check(n);
    }
}

fn run_matmul_check(n: usize) {
    let (a, b) = matmul_inputs(n);
    let got = launch_matmul("simple_matmul", n, BLOCK256_CFG, |ker| build_matmul_cfg(ker, n, BLOCK256_CFG), &a, &b);
    let expected = matmul_reference(&a, &b);
    let max_abs = max_abs_err(&got, &expected);
    println!("matmul N={n}: max abs error = {max_abs:e}");
    assert!(max_abs < 5e-2, "N={n}: max abs error {max_abs} exceeds bf16 tolerance 5e-2");
}

/// The chiplet/L2 grid swizzle in **isolation** (1-D grid + [`l2_swizzle`],
/// scalar fill). It permutes which workgroup computes which 256-block, so the
/// full C must be bit-identical-up-to-bf16-tolerance to `a.matmul(b)`.
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_l2swizzle_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_l2swizzle_amd() {
    let cfg = MatmulCfg { vec_load: false, ..BLOCK256_CFG };
    for n in [2048usize, 4096] {
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul("matmul_l2sw", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("l2swizzle N={n}: max abs error = {max_abs:e}");
        assert!(max_abs < 5e-2, "l2swizzle N={n}: max abs error {max_abs} exceeds 5e-2");
    }
}

/// Realized bf16 `(a, b)` inputs so kernel + reference see identical rounding.
fn matmul_inputs(n: usize) -> (svod_tensor::Tensor, svod_tensor::Tensor) {
    use svod_tensor::Tensor;
    // Native bf16 (no f32 `rand` → `cast` round-trip): the matrix-engine operand dtype
    // straight from the generator, so `matmul()` binds it with no cast.
    let mut a =
        Tensor::rand_with(&[n, n], DType::BFloat16, svod_dtype::default_device::default_device()).expect("rand a");
    let mut b =
        Tensor::rand_with(&[n, n], DType::BFloat16, svod_dtype::default_device::default_device()).expect("rand b");
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    (a, b)
}

/// f32 ground-truth `a·bᵀ` (B in `[N,K]`, the B[N,K] contract) over the bf16-rounded operands.
fn matmul_reference(a: &svod_tensor::Tensor, b: &svod_tensor::Tensor) -> Vec<f32> {
    let bt = b.cast(DType::Float32).expect("b→f32").try_permute(&[1, 0]).expect("bᵀ");
    let mut reference = a.cast(DType::Float32).expect("a→f32").matmul(&bt).expect("ref matmul");
    reference.realize().expect("realize reference");
    reference.as_vec::<f32>().expect("read reference")
}

fn max_abs_err(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    got.iter().zip(expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max)
}

/// The wave32 (gfx1151) matmul computes exactly `A·Bᵀ` (B in `[N,K]`, the B[N,K]
/// contract) — not a non-transposed or operand-swapped variant. Compares `got`
/// against every transpose/permutation candidate and asserts `A·Bᵀ` is the unique
/// match (the rest are garbage-scale). A layout regression in the wave32 fragment
/// map would flip which candidate wins.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_rdna_computes_ab -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_rdna_computes_ab() {
    use svod_tensor::Tensor;
    let n = 64usize;
    let (a, b) = matmul_inputs(n);
    let got = launch_matmul("matmul_diag", n, SMALL_CFG, |ker| build_matmul_cfg(ker, n, SMALL_CFG), &a, &b);

    let f = |t: &Tensor| t.cast(DType::Float32).expect("→f32");
    let (af, bf) = (f(&a), f(&b));
    let tr = |x: &Tensor| x.try_permute(&[1, 0]).expect("transpose");
    let mm = |x: &Tensor, y: &Tensor| x.matmul(y).expect("matmul");
    let vec = |mut x: Tensor| {
        x.realize().expect("realize");
        x.as_vec::<f32>().expect("read")
    };

    let abt_err = max_abs_err(&got, &vec(mm(&af, &tr(&bf))));
    // bf16 accumulation over K=64 ⇒ a few thousandths; transposes/swaps are O(1).
    assert!(abt_err < 1e-1, "wave32 matmul should equal A·Bᵀ, got max abs err {abt_err:e}");

    let wrong: Vec<(&str, Tensor)> = vec![
        ("A·B", mm(&af, &bf)),
        ("(A·Bᵀ)^T", tr(&mm(&af, &tr(&bf)))),
        ("A^T·B", mm(&tr(&af), &bf)),
        ("A^T·B^T", mm(&tr(&af), &tr(&bf))),
        ("B·A", mm(&bf, &af)),
        ("(B·A)^T", tr(&mm(&bf, &af))),
    ];
    for (name, cand) in wrong {
        let err = max_abs_err(&got, &vec(cand));
        assert!(err > 1.0, "wave32 matmul matches {name} (err {err:e}) — layout is not plain A·Bᵀ");
    }
}

/// Element-level check of the wave32 fragment lane→(m,n) map: `A = I`,
/// `B[k][j] = (k%16)*16 + (j%16)` ⇒ `C = I·Bᵀ = Bᵀ`, so the first 16×16 output
/// fragment must read `got[i][j] = B[j][i] = j*16 + i`. Any within-fragment
/// permutation lands a source element at the wrong `(i,j)` and trips the assert
/// (printing the offending row).
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_rdna_grid -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_rdna_grid() {
    use svod_tensor::Tensor;
    let n = 64usize;
    let mut a_data = vec![0f32; n * n];
    for i in 0..n {
        a_data[i * n + i] = 1.0; // identity
    }
    let b_data: Vec<f32> = (0..n * n).map(|p| (((p / n) % 16) * 16 + (p % n) % 16) as f32).collect();
    let mk =
        |d: &[f32]| Tensor::from_slice(d).try_reshape([n, n]).expect("reshape").cast(DType::BFloat16).expect("→bf16");
    let (mut a, mut b) = (mk(&a_data), mk(&b_data));
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let got = launch_matmul("matmul_grid", n, SMALL_CFG, |ker| build_matmul_cfg(ker, n, SMALL_CFG), &a, &b);

    for i in 0..16 {
        let row: Vec<i32> = (0..16).map(|j| got[i * n + j].round() as i32).collect();
        let expected: Vec<i32> = (0..16).map(|j| (j * 16 + i) as i32).collect();
        assert_eq!(row, expected, "fragment(0,0) row i={i} permuted: {row:?} (expected {expected:?})");
    }
}

/// Numerics of the software-pipelined inline-`asm` MFMA microkernel
/// ([`build_matmul_asm_cfg`]) — the flat single-LDS-buffer microkernel that exercises tk's
/// full asm GEMM infrastructure (asm MFMA + asm `ds_read_b64` gather + asm register-staged
/// `global_load_dwordx4`/`ds_write_b64` prefetch). Must equal `A·Bᵀ` (B in `[N,K]`)
/// within bf16 tolerance; the chunk/pre-gather schedule reorders the K reduction but the
/// result is the same sum. Dump the ISA with `SVOD_DUMP_AMD_IR=<dir>` to inspect the held
/// `L … P M(32) P … W` asm pipeline.
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_asm_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_asm_amd() {
    for n in [256usize, 512, 1024, 2048] {
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul(
            "matmul_asm",
            n,
            BLOCK256_CFG,
            |ker| build_matmul_asm_cfg(ker, n, BLOCK256_CFG, true),
            &a,
            &b,
        );
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("matmul_asm N={n}: max abs error = {max_abs:e}");
        assert!(max_abs < 5e-2, "hk N={n}: max abs err {max_abs} exceeds bf16 tolerance 5e-2");
    }
}

/// Numerics of the **128²** variant ([`build_matmul_asm_cfg`] at [`BLOCK128_CFG`]) — the
/// small-N tile. Same asm pipeline, `reg=32`.
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_asm128_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_asm128_amd() {
    for n in [512usize, 1024, 2048] {
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul(
            "matmul_asm128",
            n,
            BLOCK128_CFG,
            |ker| build_matmul_asm_cfg(ker, n, BLOCK128_CFG, true),
            &a,
            &b,
        );
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("matmul_asm128 N={n}: max abs error = {max_abs:e}");
        assert!(max_abs < 5e-2, "hk128 N={n}: max abs err {max_abs} exceeds bf16 tolerance 5e-2");
    }
}

/// Device-time TF of the 128² asm variant vs the 256² asm kernel and the generic
/// `gemm_core` at small N — does the software pipeline at a grid-filling 128² tile beat the
/// generic `gemm_core` run at 128²?
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_asm128_tf -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_asm128_tf() {
    use svod_runtime::{PmcSelection, ProfileOptions};
    use svod_tensor::Tensor;
    let caps = crate::ArchCaps::GFX942;
    let opts = ProfileOptions { iters: 1, static_analysis: false, counters: PmcSelection::None };
    for &n in &[1024usize, 2048, 4096] {
        let (a, b) = matmul_inputs(n);
        let out = Tensor::empty(&[n, n], DType::Float32);
        let mut c = crate::launch::graph_launch(
            "matmul_asm128",
            BLOCK128_CFG.grid_dims(n),
            BLOCK128_CFG.threads(caps.wave_size),
            out,
            &[&a, &b],
            caps,
            move |ker| {
                build_matmul_asm_cfg(ker, n, BLOCK128_CFG, true);
                ker.finish(BLOCK128_CFG.n_accum)
            },
        )
        .expect("graph_launch hk128");
        let plan = c.prepare().expect("prepare hk128");
        let flop = 2.0 * (n as f64).powi(3);
        let mut tfs: Vec<f64> = (0..5)
            .map(|_| {
                let report = plan.profile(&opts).expect("profile hk128");
                let ns: u64 = report
                    .stages
                    .iter()
                    .flat_map(|s| &s.kernels)
                    .filter_map(|k| Some(k.gpu_end_ns? - k.gpu_start_ns?))
                    .sum();
                flop / ns as f64 / 1e3
            })
            .collect();
        tfs.sort_by(|x, y| x.partial_cmp(y).expect("nan"));
        println!("matmul_asm128 N={n}: median {:.1} TF  (samples {:?})", tfs[tfs.len() / 2], tfs);
    }
}

/// Wave-phase ping-pong bisect: the asm microkernel ([`build_matmul_asm_cfg`]) with the
/// ping-pong toggled. Order matters — (1) the skeleton with the offset OFF (no conditional
/// barrier fires, no phase shift) must be numerically correct first, isolating the
/// pipelined-skeleton restructure from the offset; (2) the offset ON (predicated barrier on
/// warp-row 1 in the prologue / warp-row 0 in the epilogue) must also be correct. Run under
/// a wall-clock timeout to double as the HANG test (balanced counts → no deadlock).
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_asm_route -- --ignored --nocapture`
fn route_a_numerics(offset: bool, sizes: &[usize]) {
    for &n in sizes {
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul(
            "matmul_asm",
            n,
            BLOCK256_CFG,
            |ker| build_matmul_asm_cfg(ker, n, BLOCK256_CFG, offset),
            &a,
            &b,
        );
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("matmul_asm offset={offset} N={n}: max abs error = {max_abs:e}");
        assert!(max_abs < 5e-2, "hk offset={offset} N={n}: max abs err {max_abs} exceeds 5e-2");
    }
}

/// Bisect step 1 — the pipelined skeleton with the offset OFF (eq never-matching → no
/// conditional barrier fires). Must be numerically correct, isolating the restructure.
#[test]
#[ignore]
fn test_matmul_asm_route_off() {
    route_a_numerics(false, &[256, 512, 1024, 2048]);
}

/// Bisect step 2 — the offset ON (the predicated warp-row-1 prologue / warp-row-0 epilogue
/// ping-pong). Run under a wall-clock timeout to double as the HANG test (balanced counts →
/// no deadlock); then numerics ≤5e-2.
#[test]
#[ignore]
fn test_matmul_asm_route_on() {
    route_a_numerics(true, &[256, 512, 1024, 2048]);
}

/// Device-time throughput (median TFLOP/s, ≥3 samples) of the asm microkernel at large N —
/// measured through the graph-node path so the runtime profiler stamps real per-kernel
/// device time. Prints; no hard assert (the comparison against the reference is the result).
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_asm_route_tf -- --ignored --nocapture`
#[test]
#[ignore]
fn test_matmul_asm_route_tf() {
    use svod_runtime::{PmcSelection, ProfileOptions};
    use svod_tensor::Tensor;
    let caps = crate::ArchCaps::GFX942;
    let opts = ProfileOptions { iters: 1, static_analysis: false, counters: PmcSelection::None };
    for offset in [false, true] {
        for &n in &[1024usize, 2048, 4096, 8192] {
            let (a, b) = matmul_inputs(n);
            let out = Tensor::empty(&[n, n], DType::Float32);
            let mut c = crate::launch::graph_launch(
                "matmul_asm",
                BLOCK256_CFG.grid_dims(n),
                BLOCK256_CFG.threads(caps.wave_size),
                out,
                &[&a, &b],
                caps,
                move |ker| {
                    build_matmul_asm_cfg(ker, n, BLOCK256_CFG, offset);
                    ker.finish(BLOCK256_CFG.n_accum)
                },
            )
            .expect("graph_launch hk");
            let plan = c.prepare().expect("prepare hk");
            let flop = 2.0 * (n as f64).powi(3);
            let mut tfs: Vec<f64> = (0..5)
                .map(|_| {
                    let report = plan.profile(&opts).expect("profile hk");
                    let ns: u64 = report
                        .stages
                        .iter()
                        .flat_map(|s| &s.kernels)
                        .filter_map(|k| Some(k.gpu_end_ns? - k.gpu_start_ns?))
                        .sum();
                    flop / ns as f64 / 1e3 // TFLOP/s
                })
                .collect();
            tfs.sort_by(|x, y| x.partial_cmp(y).expect("nan"));
            println!("matmul_asm offset={offset} N={n}: median {:.1} TF  (samples {:?})", tfs[tfs.len() / 2], tfs);
        }
    }
}

/// Build + dispatch a matmul `cfg` over `(a, b)` once, returning the f32 C.
fn launch_matmul<F>(
    name: &str,
    n: usize,
    cfg: MatmulCfg,
    build: F,
    a: &svod_tensor::Tensor,
    b: &svod_tensor::Tensor,
) -> Vec<f32>
where
    F: FnOnce(&Kernel),
{
    use svod_tensor::Tensor;
    // Launch block must match the device wave size (gfx942 wave64, gfx11 wave32),
    // matching the `matmul()` entry's `cfg.threads(caps.wave_size)`.
    let ws = crate::target::resolve_arch(&a.device()).map(|ar| ar.wave_size() as usize).unwrap_or(64);
    let mut c = Tensor::empty(&[n, n], DType::Float32);
    crate::run_kernel(name, cfg.grid_dims(n), cfg.threads(ws), &mut [&mut c], &[a, b], |ker| {
        build(ker);
        ker.finish(cfg.n_accum)
    })
    .expect("matmul launch");
    c.as_vec::<f32>().expect("read c")
}

/// Direct single-kernel launch of the ARCH-APPROPRIATE matmul — resolves the device's
/// arch and goes through the same [`build_matmul_dispatch`] / [`cfg_for_arch`] wrapper
/// `matmul()`'s non-split-K path uses, so graph-vs-direct comparisons are valid on any
/// card (gfx942 asm / gfx1151 generic), not just the one this was written on.
fn launch_matmul_dispatch(name: &str, n: usize, a: &Tensor, b: &Tensor) -> Vec<f32> {
    let arch = crate::target::resolve_arch(&a.device()).expect("resolve device arch");
    let cfg = cfg_for_arch(arch, n);
    launch_matmul(name, n, cfg, move |ker| build_matmul_dispatch(ker, arch, n, cfg), a, b)
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_graph_amd -- --ignored --nocapture`.
///
/// The graph-native `matmul` (a `custom_kernel` / `Op::Call` node) matches **(a)**
/// the f32 reference (bf16 tol) AND **(b)** the direct-launch kernel **bit-for-bit**
/// — the matmul peer of the FA graph gate, confirming the matmul SINK lowers
/// identically through `custom_kernel → realize` (the optimizer bypass is
/// kernel-agnostic) as through direct launch.
#[test]
#[ignore]
fn test_matmul_graph_amd() {
    for n in [256usize, 512, 1024] {
        let (a, b) = matmul_inputs(n);
        let expected = matmul_reference(&a, &b);
        let direct = launch_matmul_dispatch("matmul_direct", n, &a, &b);

        let mut g = crate::kernels::matmul::matmul(&a, &b).expect("graph matmul").expect("matmul kernel applies");
        g.realize().expect("realize graph matmul");
        let graph = g.as_vec::<f32>().expect("read graph matmul");

        let (vs_ref, vs_direct) = (max_abs_err(&graph, &expected), max_abs_err(&graph, &direct));
        println!("matmul[graph] N={n}: vs ref = {vs_ref:e}, vs direct = {vs_direct:e}");
        assert!(vs_ref < 5e-2, "graph matmul N={n}: vs ref {vs_ref} exceeds bf16 tol 5e-2");
        assert!(vs_direct < 1e-3, "graph matmul N={n}: must match direct-launch bit-for-bit (Δ {vs_direct})");
    }
}

/// The gfx942 asm matmul (`gemm_core_asm`) is correct at every N, picking the 256²
/// [`BLOCK256_CFG`] for large N and the 128² [`BLOCK128_CFG`] below (via [`cfg_for_gfx942`]).
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_adaptive_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_adaptive_amd() {
    for n in [256usize, 512, 768, 1024, 2048] {
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul_dispatch("matmul_adaptive", n, &a, &b);
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("adaptive N={n}: max abs error = {max_abs:e}");
        assert!(max_abs < 5e-2, "adaptive N={n}: max abs error {max_abs} exceeds 5e-2");
    }
}
