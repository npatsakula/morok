//! Hardware direct global→LDS DMA ([`crate::Group::fill_local_direct`], gfx942
//! `global_load_lds_dword`): round-trip a bf16 tile GLOBAL→LDS→GLOBAL and verify
//! it matches — proving the DMA fill and its lane-contiguous (row-major) LDS
//! layout. `#[ignore]` (device); CDNA-only (the intrinsic is gfx942).

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::group::MoveIdx;
use crate::index::Idx;
use crate::tiles::TileLayout;

#[test]
#[ignore]
fn test_direct_lds_fill_amd() {
    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let Some(arch) = crate::target::resolve_arch(&dev) else {
        eprintln!("skip test_direct_lds_fill_amd: no AMD device");
        return;
    };
    if !arch.is_cdna() {
        eprintln!("skip test_direct_lds_fill_amd: global_load_lds is CDNA (gfx942)-only");
        return;
    }
    let (rows, cols) = (32usize, 64usize); // FA K/V shape: kv_blk=32 (2 row-fragments) × d=64
    let data: Vec<f32> = (0..rows * cols).map(|i| (i % 128 + 1) as f32).collect();
    let mut a = Tensor::from_slice(&data)
        .try_reshape([1usize, 1, rows, cols])
        .expect("reshape a")
        .cast(DType::BFloat16)
        .expect("→bf16");
    a.realize().expect("realize a");
    let mut out = Tensor::empty(&[1, 1, rows, cols], DType::BFloat16);

    let block = 8 * arch.wave_size() as i64; // 8 warps collaboratively DMA the tile
    crate::run_kernel("direct_lds_fill", [1, 1, 1], block, &mut [&mut out], &[&a], |ker| {
        let g = ker.group(8);
        let warp = ker.warp();
        let o = ker.gl(&[1, 1, rows, cols], DType::BFloat16);
        let ain = ker.gl(&[1, 1, rows, cols], DType::BFloat16);
        let lds = ker.shared_sw((rows, cols), DType::BFloat16, TileLayout::Row);
        // Direct global→LDS DMA of the whole tile, then read it back and store out.
        let lds = g.fill_local_direct(lds, &ain, &[Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::Const(0)], 2);
        let tile = warp.load(ker.operand((rows, cols), DType::BFloat16, TileLayout::Row), lds, MoveIdx::default());
        let _ = warp.store(o, tile, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(1)
    })
    .expect("direct_lds_fill launch");

    let mut of = out.cast(DType::Float32).expect("out→f32");
    of.realize().expect("realize out→f32");
    let got = of.as_vec::<f32>().expect("read out");
    let mut bad = 0usize;
    for (i, &g) in got.iter().enumerate().take(rows * cols) {
        let want = (i % 128 + 1) as f32;
        if (g - want).abs() > 1e-3 {
            if bad < 8 {
                eprintln!("out[{i}] = {g} want {want}");
            }
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "{bad}/{} elements wrong on {arch:?}", rows * cols);
    println!("direct global→LDS DMA round-trip: {}/{} correct on {arch:?}", rows * cols, rows * cols);
}
