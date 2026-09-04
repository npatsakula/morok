//! Storage-dtype decomposition: FP8/BF16 widening (`pm_float_decomp`) and the
//! 64-bit word split (`pm_long_decomp`), plus the target table that picks them.

use std::sync::Arc;

use svod_dtype::{AmdArch, DType, ScalarDType};
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use super::helpers::{create_bool_const, create_buffer_typed};
use crate::devectorize::{Fp8DecompCtx, pm_float_decomp};
use crate::optimizer::{Renderer, apply_dtype_decomps, get_dtype_decomps};
use svod_ir::ops;

fn decompose(from: ScalarDType, root: Arc<UOp>) -> Arc<UOp> {
    let mut ctx = Fp8DecompCtx { from, to: ScalarDType::Float16 };
    svod_ir::rewrite::graph_rewrite_bottom_up(&pm_float_decomp(), root, &mut ctx)
}

fn store_value_dtypes(root: &Arc<UOp>) -> Vec<DType> {
    root.toposort()
        .into_iter()
        .filter_map(|node| match node.op() {
            Op::Store(ops::Store { value, .. }) => Some(value.dtype()),
            _ => None,
        })
        .collect()
}

/// Widening an FP8 load must not drop the gate/alt pair the late gater installed.
#[test]
fn fp8_decomp_preserves_alt_on_gated_load() {
    let index = UOp::index()
        .buffer(create_buffer_typed(64, ScalarDType::FP8E5M2))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
        .call()
        .unwrap();
    let load = UOp::load()
        .index(index)
        .alt(UOp::const_(DType::Scalar(ScalarDType::FP8E5M2), ConstValue::Float(0.0)))
        .gate(create_bool_const(false))
        .call();

    let decomposed = decompose(ScalarDType::FP8E5M2, load);

    let gated: Vec<_> = decomposed
        .toposort()
        .into_iter()
        .filter(|node| matches!(node.op(), Op::Load(ops::Load { gate: Some(_), .. })))
        .collect();
    assert!(!gated.is_empty(), "the gated load must survive decomposition");
    assert!(
        gated.iter().all(|node| matches!(node.op(), Op::Load(ops::Load { alt: Some(_), .. }))),
        "{}",
        decomposed.tree()
    );
}

#[test]
fn vector_fp8_load_decomposes_to_scalar_loads_and_stack() {
    let indices = UOp::stack((0..4).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect());
    let index = UOp::index()
        .buffer(create_buffer_typed(4, ScalarDType::FP8E4M3))
        .indices(vec![indices])
        .call()
        .unwrap()
        .with_dtype(DType::FP8E4M3.vec(4).unwrap());
    let load = UOp::load().index(index).dtype(DType::FP8E4M3.vec(4).unwrap()).call();

    let decomposed = decompose(ScalarDType::FP8E4M3, load);

    assert!(matches!(decomposed.op(), Op::Stack(..)), "{}", decomposed.tree());
    assert_eq!(decomposed.dtype(), DType::Float16);
    assert_eq!(decomposed.toposort().iter().filter(|u| matches!(u.op(), Op::Load(..))).count(), 4);
}

/// Both directions are rewritten: the STORE narrows to the uint8 storage form and
/// the LOAD reads it back, leaving no FNUZ node behind.
#[test]
fn fnuz_store_and_load_are_both_decomposed() {
    let index = UOp::index()
        .buffer(create_buffer_typed(4, ScalarDType::FP8E4M3FNUZ))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
        .call()
        .unwrap();
    let root = UOp::sink(vec![
        index.clone().store(UOp::const_(DType::Scalar(ScalarDType::FP8E4M3FNUZ), ConstValue::Float(1.0))),
        UOp::load().index(index).call(),
    ]);

    let decomposed = decompose(ScalarDType::FP8E4M3FNUZ, root);

    assert!(!decomposed.toposort().iter().any(|u| u.dtype().base() == ScalarDType::FP8E4M3FNUZ));
    assert!(store_value_dtypes(&decomposed).contains(&DType::UInt8), "{}", decomposed.tree());
    assert!(decomposed.toposort().iter().any(|u| matches!(u.op(), Op::Load(..)) && u.dtype() == DType::UInt8));
}

/// Which storage dtypes need decomposing is a property of the target, not of the AST.
#[test]
fn dtype_decomposition_mapping_is_target_sensitive() {
    let values = [
        (ScalarDType::FP8E4M3, ConstValue::Float(1.0)),
        (ScalarDType::FP8E4M3FNUZ, ConstValue::Float(1.0)),
        (ScalarDType::FP8E5M2, ConstValue::Float(1.0)),
        (ScalarDType::FP8E5M2FNUZ, ConstValue::Float(1.0)),
        (ScalarDType::Float16, ConstValue::Float(1.0)),
        (ScalarDType::BFloat16, ConstValue::Float(1.0)),
        (ScalarDType::Int64, ConstValue::Int(1)),
        (ScalarDType::UInt64, ConstValue::UInt(1)),
    ];
    let sink = UOp::sink(values.into_iter().map(|(dt, value)| UOp::const_(DType::Scalar(dt), value)).collect());
    let all_fp8_to_half = vec![
        (ScalarDType::FP8E4M3, ScalarDType::Float16),
        (ScalarDType::FP8E5M2, ScalarDType::Float16),
        (ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16),
        (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16),
    ];
    let fnuz_only =
        vec![(ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16), (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16)];

    assert_eq!(get_dtype_decomps(&sink, &Renderer::cpu()), all_fp8_to_half);
    // CDNA renders OCP FP8 natively and only lacks the FNUZ encodings.
    assert_eq!(get_dtype_decomps(&sink, &Renderer::for_amd_arch(AmdArch::Gfx942)), fnuz_only);
    assert_eq!(get_dtype_decomps(&sink, &Renderer::for_amd_arch(AmdArch::Gfx950)), fnuz_only);
    assert_eq!(get_dtype_decomps(&sink, &Renderer::for_amd_arch(AmdArch::Gfx1151)), all_fp8_to_half);
    // WebGPU has neither 64-bit integers nor any sub-f32 float.
    assert_eq!(
        get_dtype_decomps(&sink, &Renderer::webgpu()),
        vec![
            (ScalarDType::Int64, ScalarDType::Int32),
            (ScalarDType::FP8E4M3, ScalarDType::Float32),
            (ScalarDType::FP8E5M2, ScalarDType::Float32),
            (ScalarDType::Float16, ScalarDType::Float32),
            (ScalarDType::BFloat16, ScalarDType::Float32),
            (ScalarDType::FP8E4M3FNUZ, ScalarDType::Float32),
            (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float32),
        ]
    );
}

/// The combined pass must commit weak dtypes before decomposing, or the stored value
/// keeps a weak type the narrowing rules never match.
#[test]
fn combined_dtype_pass_commits_weak_stores_before_decomposition() {
    let index = |scalar| {
        UOp::index()
            .buffer(create_buffer_typed(4, scalar))
            .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
            .call()
            .unwrap()
    };
    let root = UOp::sink(vec![
        index(ScalarDType::FP8E4M3).store(UOp::const_(DType::WeakFloat, ConstValue::Float(1.5))),
        index(ScalarDType::BFloat16).store(UOp::const_(DType::WeakFloat, ConstValue::Float(-2.0))),
    ]);

    let decomposed = apply_dtype_decomps(root, Renderer::webgpu());

    assert!(
        !decomposed.toposort().iter().any(|u| matches!(u.dtype().base(), ScalarDType::FP8E4M3 | ScalarDType::BFloat16)),
        "{}",
        decomposed.tree()
    );
    let dtypes = store_value_dtypes(&decomposed);
    assert!(dtypes.contains(&DType::UInt8) && dtypes.contains(&DType::UInt16), "{}", decomposed.tree());
}

/// Same for the word split: a weak 64-bit value must be committed to `Int64` before
/// it can be cut into two `Int32` words.
#[test]
fn combined_dtype_pass_commits_long_weak_store_before_word_split() {
    let index = UOp::index()
        .buffer(create_buffer_typed(4, ScalarDType::Int64))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
        .call()
        .unwrap();
    let value = UOp::new(
        Op::Binary(
            svod_ir::BinaryOp::Shl,
            UOp::const_(DType::WeakInt, ConstValue::Int(0x1_0000_0000)),
            UOp::const_(DType::WeakInt, ConstValue::Int(0x7654_3210)),
        ),
        DType::WeakInt,
    );

    let decomposed = apply_dtype_decomps(UOp::sink(vec![index.store(value)]), Renderer::webgpu());

    let dtypes = store_value_dtypes(&decomposed);
    assert_eq!(dtypes, vec![DType::Int32; 2], "{}", decomposed.tree());
}

/// The two words of a split 64-bit STORE address *adjacent* elements of the doubled
/// 32-bit buffer, at `2*i` and `2*i+1`. The word values themselves are covered by
/// `test/property/long_shift.rs`.
#[test_case::test_case(0)]
#[test_case::test_case(3)]
fn long_store_words_address_adjacent_elements(at: i64) {
    use crate::test::property::long_shift::{long_const, split_store};

    for from in [ScalarDType::Int64, ScalarDType::UInt64] {
        let split = split_store(from, at, long_const(from, 0xdead_beef_feed_face));
        assert_eq!(split.map(|(_, address)| address), [2 * at, 2 * at + 1], "{from:?} at {at}");
        assert_eq!(split.map(|(word, _)| word), [0xfeed_face, 0xdead_beef], "{from:?} at {at}");
    }
}

/// `any::<u64>()` never samples these, so the property test cannot reach the carry
/// and all-ones boundaries of the multiply word split.
#[test_case::test_case(u64::MAX, u64::MAX; "all ones")]
#[test_case::test_case(0x0000_0000_ffff_ffff, 0x0000_0000_0000_0002; "low word carry")]
fn long_arithmetic_word_split_matches_native_at_boundaries(a: u64, b: u64) {
    for from in [ScalarDType::Int64, ScalarDType::UInt64] {
        crate::test::property::long_shift::assert_long_arithmetic(a, b, from);
    }
}
