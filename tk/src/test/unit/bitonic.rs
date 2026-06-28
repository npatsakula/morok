//! Tests for the counted bitonic sorting-network primitives
//! ([`crate::Group::arg_compare_exchange`] / `bitonic_argsort` /
//! `bitonic_merge_topk`): GPU-free graph-shape checks (both archs) of the
//! `ds_bpermute` butterfly structure, and hardware-gated end-to-end sorts vs a
//! Rust reference (arch-portable: gfx942 wave64 / gfx1151 wave32).
//!
//! The sort axis is the 16-wide lane axis (`laneid % 16`). The exec tests load a
//! `[query, corpus]` matrix into a `Col` accumulator (so `corpus = laneid % 16`,
//! matching the KNN integration's score tile) and sort the corpus per query row.

use svod_dtype::{AmdArch, DType};
use svod_ir::Op;

use crate::arch::FragRole;
use crate::index::Idx;
use crate::tiles::TileLayout;
use crate::{ArchCaps, Kernel, MoveIdx};

const COL: TileLayout = TileLayout::Col;

/// Reference sort: per query row, the corpus values ordered by the `(value, index)`
/// total order (ties → smaller corpus index, matching `arg_compare_exchange`).
/// Returns `(sorted_values, argsort_indices)` flattened `[query*corpus]`.
fn ref_argsort(m: &[f32], query: usize, corpus: usize) -> (Vec<f32>, Vec<i32>) {
    let (mut vs, mut is) = (Vec::with_capacity(query * corpus), Vec::with_capacity(query * corpus));
    for q in 0..query {
        let mut pairs: Vec<(f32, i32)> = (0..corpus).map(|c| (m[q * corpus + c], c as i32)).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
        for (v, i) in pairs {
            vs.push(v);
            is.push(i);
        }
    }
    (vs, is)
}

// =============================================================================
// GPU-free graph-shape checks.
// =============================================================================

/// `bitonic_argsort` lays down the Batcher network: 10 `arg_compare_exchange`
/// stages, each gathering its partner's value AND index via a `ds_bpermute`
/// `Op::Custom` (2 per stage per lane-register), selecting with `where`/`Ternary`,
/// and using `Lt`/`Eq` compares for the `(value, index)` total order — with no LDS,
/// no barrier, no WMMA. Holds on both wave64 (gfx942) and wave32 (gfx1151).
#[test]
fn test_bitonic_argsort_graph_shape() {
    for caps in [ArchCaps::GFX942, ArchCaps::for_arch(AmdArch::Gfx1151)] {
        let arch = caps.arch;
        let ker = Kernel::new("argsort", [1, 1, 1], caps.wave_size as i64, vec![], caps);
        let warp = ker.warp();
        let frag = ker.caps.frag(FragRole::Accumulator);
        let val = warp.zero(ker.rt((16, 16), DType::Float32, COL, frag));
        let idx = warp.zero(ker.rt((16, 16), DType::Int32, COL, frag));
        let (_, idx) = warp.bitonic_argsort(val, idx);
        let topo = idx.uop().toposort();

        let customs = topo.iter().filter(|u| matches!(u.op(), Op::Custom { .. })).count();
        assert!(customs >= 10, "{arch:?}: argsort emits ds_bpermute Op::Customs (got {customs})");
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(..))), "{arch:?}: keep-select where");
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Lt, ..))),
            "{arch:?}: total-order Lt compare"
        );
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "{arch:?}: no LDS scratch");
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "{arch:?}: no barrier");
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "{arch:?}: no WMMA");
    }
}

/// `bitonic_merge_topk` carries the reverse-gather + arg-min + 4 ascending
/// compare-exchange stages: `ds_bpermute` gathers and `where`-selects, no LDS /
/// barrier / WMMA. Both archs.
#[test]
fn test_bitonic_merge_topk_graph_shape() {
    for caps in [ArchCaps::GFX942, ArchCaps::for_arch(AmdArch::Gfx1151)] {
        let arch = caps.arch;
        let ker = Kernel::new("merge", [1, 1, 1], caps.wave_size as i64, vec![], caps);
        let warp = ker.warp();
        let frag = ker.caps.frag(FragRole::Accumulator);
        let av = warp.zero(ker.rt((16, 16), DType::Float32, COL, frag));
        let ai = warp.zero(ker.rt((16, 16), DType::Int32, COL, frag));
        let bv = warp.zero(ker.rt((16, 16), DType::Float32, COL, frag));
        let bi = warp.zero(ker.rt((16, 16), DType::Int32, COL, frag));
        let (_, idx) = warp.bitonic_merge_topk(&av, &ai, &bv, &bi);
        let topo = idx.uop().toposort();

        let customs = topo.iter().filter(|u| matches!(u.op(), Op::Custom { .. })).count();
        assert!(customs >= 4, "{arch:?}: merge emits ds_bpermute Op::Customs (got {customs})");
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(..))), "{arch:?}: keep-select where");
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "{arch:?}: no LDS scratch");
        assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "{arch:?}: no barrier");
    }
}

// =============================================================================
// Hardware-gated end-to-end sorts on gfx942 / gfx1151.
// =============================================================================

fn device_arch() -> Option<AmdArch> {
    let dev = svod_tensor::Tensor::empty(&[1], DType::Float32).device();
    crate::target::resolve_arch(&dev)
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib bitonic::test_bitonic_argsort_amd -- --ignored --nocapture`.
///
/// Load a `[16, 16]` matrix as `[query, corpus]` into a `Col` accumulator (so
/// `corpus = laneid % 16`), argsort the corpus per query, and compare against the
/// Rust `(value, index)` reference. Includes a forced-tie row (repeated values →
/// the smaller corpus index must rank first).
#[test]
#[ignore]
fn test_bitonic_argsort_amd() {
    use svod_tensor::Tensor;

    let Some(arch) = device_arch() else {
        eprintln!("skip test_bitonic_argsort_amd: no AMD device");
        return;
    };
    let w = arch.wave_size() as i64;
    let (query, corpus) = (16usize, 16usize);

    // Random matrix with a forced-tie row 3 (a few repeated values).
    svod_tensor::rand::manual_seed(0xB170_0001);
    let mut base_t = Tensor::randn(&[query, corpus]).expect("randn");
    base_t.realize().expect("realize base");
    let mut m = base_t.as_vec::<f32>().expect("vec");
    for c in 0..corpus {
        m[3 * corpus + c] = ((c % 3) as f32) - 1.0; // values {-1,0,1} repeated → ties
    }

    let a = Tensor::from_slice(&m).try_reshape([1usize, 1, query, corpus]).expect("reshape");
    let mut a = a.cast(DType::Float32).expect("f32");
    a.realize().expect("realize a");
    let mut vout = Tensor::empty(&[1, 1, query, corpus], DType::Float32);
    let mut iout = Tensor::empty(&[1, 1, query, corpus], DType::Int32);

    crate::run_kernel("argsort", [1, 1, 1], w, &mut [&mut vout, &mut iout], &[&a], |ker| {
        let warp = ker.warp();
        let frag = ker.caps.frag(FragRole::Accumulator);
        let vo = ker.gl(&[1, 1, query, corpus], DType::Float32);
        let io = ker.gl(&[1, 1, query, corpus], DType::Int32);
        let ain = ker.gl(&[1, 1, query, corpus], DType::Float32);

        // Load [query, corpus] into a Col accumulator: corpus = laneid % 16.
        let val = warp.load(ker.acc((query, corpus), COL), ain, MoveIdx::block((0, 0, 0, 0), 2));
        // idx[query, corpus] = the corpus column (global position, col_blk = 0).
        let idx = warp.map_position(
            ker.rt((query, corpus), DType::Int32, COL, frag),
            Idx::Const(0),
            Idx::Const(0),
            |_, _, _, col| col.cast(DType::Int32),
        );
        let (sv, si) = warp.bitonic_argsort(val, idx);
        let _ = warp.store(vo, sv, MoveIdx::block((0, 0, 0, 0), 2));
        let _ = warp.store(io, si, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(2)
    })
    .expect("argsort launch");

    let gv = vout.as_vec::<f32>().expect("read vout");
    let gi = iout.as_vec::<i32>().expect("read iout");
    let (ev, ei) = ref_argsort(&m, query, corpus);

    let mut bad = 0usize;
    for q in 0..query {
        for p in 0..corpus {
            let o = q * corpus + p;
            if (gv[o] - ev[o]).abs() > 1e-4 || gi[o] != ei[o] {
                bad += 1;
                if bad <= 8 {
                    eprintln!("q={q} p={p}: got (v={}, i={}) want (v={}, i={})", gv[o], gi[o], ev[o], ei[o]);
                }
            }
        }
    }
    assert_eq!(bad, 0, "bitonic_argsort mismatches on {arch:?}");
    println!("bitonic_argsort: {query}x{corpus} sorted OK on {arch:?} (tie row included)");
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib bitonic::test_bitonic_merge_topk_amd -- --ignored --nocapture`.
///
/// Two ASCENDING runs A, B per query; merge keeps the smallest 16 ascending. The
/// reference is the sorted lower half of A∪B by `(value, index)`. Includes a row
/// where A and B share an equal value (tie → smaller index kept).
#[test]
#[ignore]
fn test_bitonic_merge_topk_amd() {
    use svod_tensor::Tensor;

    let Some(arch) = device_arch() else {
        eprintln!("skip test_bitonic_merge_topk_amd: no AMD device");
        return;
    };
    let w = arch.wave_size() as i64;
    let (query, n) = (16usize, 16usize);

    // Build A, B ascending per query, with distinct indices: A indices 0..16,
    // B indices 16..32. Row 5 forces an A/B value tie.
    svod_tensor::rand::manual_seed(0xB170_0002);
    let mut araw_t = Tensor::randn(&[query, n]).expect("randn a");
    let mut braw_t = Tensor::randn(&[query, n]).expect("randn b");
    araw_t.realize().expect("realize a");
    braw_t.realize().expect("realize b");
    let araw = araw_t.as_vec::<f32>().expect("vec a");
    let braw = braw_t.as_vec::<f32>().expect("vec b");
    let mut amat = vec![0f32; query * n];
    let mut bmat = vec![0f32; query * n];
    for q in 0..query {
        let mut av: Vec<f32> = araw[q * n..(q + 1) * n].to_vec();
        let mut bv: Vec<f32> = braw[q * n..(q + 1) * n].to_vec();
        av.sort_by(|x, y| x.partial_cmp(y).unwrap());
        bv.sort_by(|x, y| x.partial_cmp(y).unwrap());
        if q == 5 {
            bv[0] = av[0]; // tie: A[5,0] == B[5,0]; keep the smaller index (A's 0..16)
            bv.sort_by(|x, y| x.partial_cmp(y).unwrap());
        }
        for c in 0..n {
            amat[q * n + c] = av[c];
            bmat[q * n + c] = bv[c];
        }
    }

    let mk = |v: &[f32]| {
        let mut t = Tensor::from_slice(v)
            .try_reshape([1usize, 1, query, n])
            .expect("reshape")
            .cast(DType::Float32)
            .expect("f32");
        t.realize().expect("realize");
        t
    };
    let a = mk(&amat);
    let b = mk(&bmat);
    let mut vout = Tensor::empty(&[1, 1, query, n], DType::Float32);
    let mut iout = Tensor::empty(&[1, 1, query, n], DType::Int32);

    crate::run_kernel("merge", [1, 1, 1], w, &mut [&mut vout, &mut iout], &[&a, &b], |ker| {
        let warp = ker.warp();
        let frag = ker.caps.frag(FragRole::Accumulator);
        let vo = ker.gl(&[1, 1, query, n], DType::Float32);
        let io = ker.gl(&[1, 1, query, n], DType::Int32);
        let ain = ker.gl(&[1, 1, query, n], DType::Float32);
        let bin = ker.gl(&[1, 1, query, n], DType::Float32);

        let av = warp.load(ker.acc((query, n), COL), ain, MoveIdx::block((0, 0, 0, 0), 2));
        let bv = warp.load(ker.acc((query, n), COL), bin, MoveIdx::block((0, 0, 0, 0), 2));
        // A indices = corpus col (0..16); B indices = col + 16 (16..32).
        let ai = warp.map_position(
            ker.rt((query, n), DType::Int32, COL, frag),
            Idx::Const(0),
            Idx::Const(0),
            |_, _, _, c| c.cast(DType::Int32),
        );
        let bi = warp.map_position(
            ker.rt((query, n), DType::Int32, COL, frag),
            Idx::Const(0),
            Idx::Const(0),
            |_, _, _, c| c.add(&crate::index::cidx(16)).cast(DType::Int32),
        );
        let (mv, mi) = warp.bitonic_merge_topk(&av, &ai, &bv, &bi);
        let _ = warp.store(vo, mv, MoveIdx::block((0, 0, 0, 0), 2));
        let _ = warp.store(io, mi, MoveIdx::block((0, 0, 0, 0), 2));
        ker.finish(2)
    })
    .expect("merge launch");

    let gv = vout.as_vec::<f32>().expect("read vout");
    let gi = iout.as_vec::<i32>().expect("read iout");

    let mut bad = 0usize;
    for q in 0..query {
        // Reference: lower half of A∪B by (value, index).
        let mut pairs: Vec<(f32, i32)> = Vec::with_capacity(2 * n);
        for c in 0..n {
            pairs.push((amat[q * n + c], c as i32));
            pairs.push((bmat[q * n + c], (c + n) as i32));
        }
        pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap().then(x.1.cmp(&y.1)));
        for (p, &(want_v, want_i)) in pairs.iter().take(n).enumerate() {
            let o = q * n + p;
            if (gv[o] - want_v).abs() > 1e-4 || gi[o] != want_i {
                bad += 1;
                if bad <= 8 {
                    eprintln!("q={q} p={p}: got (v={}, i={}) want (v={want_v}, i={want_i})", gv[o], gi[o]);
                }
            }
        }
    }
    assert_eq!(bad, 0, "bitonic_merge_topk mismatches on {arch:?}");
    println!("bitonic_merge_topk: {query} queries merged OK on {arch:?} (tie row included)");
}
