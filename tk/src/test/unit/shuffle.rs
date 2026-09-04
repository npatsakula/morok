//! Cross-lane primitives ([`crate::Group::shuffle`]/`shuffle_xor`/`compare_exchange`)
//! — the `ds_bpermute`-based foundation for sorting networks, arg-reduce, and scan.
//! Graph-shape checks run GPU-free; the butterfly all-reduce is `#[ignore]`
//! (gfx942 wave64 + gfx1151 wave32 — the f32 fragment is arch-selected by role).

use svod_dtype::{AmdArch, DType};
use svod_ir::Op;

use crate::arch::FragRole;
use crate::tiles::{RT_16X16, TileLayout};
use crate::{ArchCaps, Kernel, MoveIdx, SwapDir};
use svod_ir::ops;

const ROW: TileLayout = TileLayout::Row;

/// `shuffle_xor` lowers to a `ds_bpermute` cross-lane gather (an `Op::Custom`) with
/// no LDS scratch and no barrier — and it does so on BOTH wave64 (gfx942) and wave32
/// (gfx1151), i.e. the lane math is arch-blind (`ArchCaps::wave_size`).
#[test]
fn test_shuffle_xor_graph_shape() {
    let build = |caps: ArchCaps, block: i64| {
        let ker = Kernel::new("shuf", [1, 1, 1], block, vec![], caps);
        let warp = ker.warp();
        let src = warp.zero(ker.rt((16, 16), DType::Float32, ROW, RT_16X16));
        let dst = ker.rt((16, 16), DType::Float32, ROW, RT_16X16);
        warp.shuffle_xor(dst, &src, 16).uop().toposort()
    };
    for (caps, block) in [(ArchCaps::GFX942, 64), (ArchCaps::for_arch(AmdArch::Gfx1151), 32)] {
        let topo = build(caps, block);
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Custom(..))),
            "{:?}: shuffle_xor emits a ds_bpermute Op::Custom",
            caps.arch
        );
        assert!(
            !topo
                .iter()
                .any(|u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))),
            "{:?}: no LDS scratch",
            caps.arch
        );
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))), "{:?}: no barrier", caps.arch);
    }
}

/// `compare_exchange` lowers to a `ds_bpermute` gather plus an ALU min/max select
/// (a `Ternary` `where`), with no LDS and no barrier.
#[test]
fn test_compare_exchange_graph_shape() {
    let ker = Kernel::new("ce", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    let warp = ker.warp();
    let src = warp.zero(ker.rt((16, 16), DType::Float32, ROW, RT_16X16));
    let dst = ker.rt((16, 16), DType::Float32, ROW, RT_16X16);
    let topo = warp.compare_exchange(dst, &src, 1, SwapDir::ByLaneBit(2)).uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Custom(..))), "ds_bpermute gather present");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(..))), "min/max select (where) present");
    assert!(
        !topo
            .iter()
            .any(|u| matches!(u.op(), Op::Buffer(ops::Buffer { arg, .. }) if arg.addrspace == Some(svod_ir::AddrSpace::Local))),
        "no LDS scratch"
    );
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier(..))), "no barrier");
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib shuffle::test_shuffle_xor_allreduce_amd -- --ignored`.
///
/// End-to-end (gfx942): a butterfly all-reduce — `x += shuffle_xor(x, mask)` for
/// `mask ∈ {1,2,…,32}` — sums each tile element across all 64 lanes. Seeded with
/// `1.0` everywhere, every element must become `wave_size = 64`. This validates the
/// `shuffle_xor` transport + lane math end-to-end and is **layout-independent** (the
/// sum of 1s over the wave is `wave_size` regardless of the lane↔element map).
#[test]
#[ignore]
fn test_shuffle_xor_allreduce_amd() {
    use svod_tensor::Tensor;

    // Arch-aware: derive the wave width (64 on CDNA, 32 on RDNA) so the butterfly
    // mask sequence, launch block, and expected sum all match the device.
    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let Some(arch) = crate::target::resolve_arch(&dev) else {
        eprintln!("skip test_shuffle_xor_allreduce_amd: no AMD device");
        return;
    };
    let w = arch.wave_size() as i64;
    let masks: Vec<i64> = (0..).map(|i| 1i64 << i).take_while(|&m| m < w).collect();

    let mut out = Tensor::empty(&[1, 1, 16, 16], DType::Float32);
    crate::run_kernel("allreduce", [1, 1, 1], w, &mut [&mut out], &[], |ker| {
        let warp = ker.warp();
        let o = ker.gl(&[1, 1, 16, 16], DType::Float32);
        // The arch-correct f32 16×16 fragment: RT_16X16 (ept=4) on CDNA wave64,
        // RT_16X16_W32_ACC (ept=8, even/odd interleave) on RDNA wave32 — so the store
        // covers all 256 elements, not just the wave64 half.
        let frag = ker.caps.frag(FragRole::Accumulator);
        let mut x = warp.ones(ker.rt((16, 16), DType::Float32, ROW, frag));
        for &mask in &masks {
            let tmp = warp.shuffle_xor(ker.rt((16, 16), DType::Float32, ROW, frag), &x, mask);
            x = warp.add(x, &tmp);
        }
        let _ = warp.store(o, x, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(1)
    })
    .expect("allreduce launch");

    let got = out.as_vec::<f32>().expect("read out");
    assert_eq!(got.len(), 256, "16x16 tile");
    let expected = w as f32;
    let bad = got.iter().filter(|&&v| (v - expected).abs() > 1e-3).count();
    assert_eq!(bad, 0, "every element must be the {w}-lane sum of 1.0 = {expected}; got e.g. {:?}", &got[..8]);
}
