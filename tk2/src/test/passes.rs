//! Host tests for the two real addressing passes (unroll + const-fold): the
//! pass-runner applies them through the banded pipeline, the contracts
//! (`requires`/`ensures`, band monotonicity) are exercised, and the fold is proven
//! structurally (address-op count delta + flat body).

use svod_ir::Op;

use crate::ir::{IndexOp, Node};
use crate::kernels::{Program, matmul};
use crate::lower;
use crate::pass::{Pass, PassError, Pipeline};
use crate::passes::{ConstFoldPass, UnrollPass, count_reachable, optimize_addressing};

fn is_range_or_end(n: &Node) -> bool {
    matches!(n, Node::Range { .. } | Node::End { .. })
}

/// `IndexAlu` mul/div/mod — the address-cone recompute the const-fold targets (Add
/// is left out: additive offsets survive, only the multiplicative/div/mod steps fold).
fn is_addr_muldivmod(n: &Node) -> bool {
    matches!(n, Node::IndexAlu { op, .. } if !matches!(op, IndexOp::Add))
}

// ── unroll ───────────────────────────────────────────────────────────────────

#[test]
fn unroll_flattens_the_matmul_body() {
    let p = matmul(64, 64, 64);
    let mut ir = p.ir;
    // Rolled: the init range + the K-loop range, each with its END (2 ranges, 2 ends).
    assert_eq!(count_reachable(&ir, p.sink, is_range_or_end), 4, "rolled matmul has 2 RANGE + 2 END");
    let root = UnrollPass.apply(&mut ir, p.sink).expect("unroll applies");
    // Flat: the ensures postcondition — no RANGE / END reachable.
    assert_eq!(count_reachable(&ir, root, is_range_or_end), 0, "unrolled matmul is flat");
    assert!(UnrollPass.ensures(&ir, root), "unroll ensures (flat) must hold");
}

// ── const-fold + the §2.4 ordering contract ──────────────────────────────────

#[test]
fn const_fold_requires_unroll_first() {
    // DESIGN.md §2.4: the fold only fires on compile-time-constant steps, so
    // const-fold's `requires` rejects a still-rolled graph — the runner surfaces a
    // wrong pass order as a Requires error (the de-risk of the contract model).
    let p = matmul(64, 64, 64);
    let mut ir = p.ir;
    assert!(!ConstFoldPass.requires(&ir, p.sink), "const-fold must reject a rolled graph");
    let err = Pipeline::new().then(ConstFoldPass).run(&mut ir, p.sink).expect_err("const-fold before unroll must fail");
    assert!(matches!(err, PassError::Requires { .. }), "expected a Requires error, got {err:?}");
}

#[test]
fn const_fold_collapses_constant_address_arith() {
    let p = matmul(64, 64, 64);
    let mut ir = p.ir;
    let unrolled = UnrollPass.apply(&mut ir, p.sink).expect("unroll");
    let before = count_reachable(&ir, unrolled, is_addr_muldivmod);
    let folded = ConstFoldPass.apply(&mut ir, unrolled).expect("const-fold");
    let after = count_reachable(&ir, folded, is_addr_muldivmod);
    // The counter-derived `tk·16` (now const·const) muls fold to immediates; the
    // lane-dependent `(lane/16)·4`, `lane/16`, `lane%16` stay runtime.
    assert!(after < before, "const-fold must reduce mul/div/mod address ops ({before} → {after})");
    assert!(ConstFoldPass.ensures(&ir, folded), "const-fold ensures (no const·const IndexAlu) must hold");
}

// ── the pipeline end-to-end (banded, contract-checked) ────────────────────────

#[test]
fn addressing_pipeline_runs_and_lowers_spec_valid() {
    let mut p = matmul(64, 64, 64);
    let root = optimize_addressing(&mut p.ir, p.sink).expect("banded unroll→const-fold pipeline runs");
    let opt = Program { ir: p.ir, sink: root, name: p.name };
    // The optimized program still lowers to spec-valid device-UOp.
    lower::verify(&opt).expect("optimized matmul must lower spec-valid");
}

/// Prints the structural before/after deltas for the report (proof (c)). Run with
/// `cargo test -p svod-tk2 --lib passes::report_ -- --nocapture`.
#[test]
fn report_address_op_deltas() {
    let is_indexalu = |n: &Node| matches!(n, Node::IndexAlu { .. });
    for k in [64usize, 128] {
        let p = matmul(k, k, k);
        let (mut ir, rolled) = (p.ir, p.sink);
        let rng0 = count_reachable(&ir, rolled, is_range_or_end);
        let alu0 = count_reachable(&ir, rolled, is_indexalu);
        let mul0 = count_reachable(&ir, rolled, is_addr_muldivmod);
        let mma0 = count_reachable(&ir, rolled, |n| matches!(n, Node::Mma { .. }));

        let unrolled = UnrollPass.apply(&mut ir, rolled).expect("unroll");
        let alu_u = count_reachable(&ir, unrolled, is_indexalu);
        let mul_u = count_reachable(&ir, unrolled, is_addr_muldivmod);

        let folded = ConstFoldPass.apply(&mut ir, unrolled).expect("fold");
        let rng1 = count_reachable(&ir, folded, is_range_or_end);
        let alu1 = count_reachable(&ir, folded, is_indexalu);
        let mul1 = count_reachable(&ir, folded, is_addr_muldivmod);
        let mma1 = count_reachable(&ir, folded, |n| matches!(n, Node::Mma { .. }));

        println!(
            "K={k}: RANGE+END {rng0}→{rng1} | Mma {mma0}→{mma1} (unrolled) | tile-IR IndexAlu total {alu0}→(unroll {alu_u})→(fold {alu1}) | mul/div/mod {mul0}→(unroll {mul_u})→(fold {mul1})"
        );
    }
}

#[test]
fn optimized_matmul_lowers_flat_with_one_wmma_per_kfrag() {
    let mut p = matmul(64, 64, 64); // K/16 = 4 K-fragments
    let root = optimize_addressing(&mut p.ir, p.sink).expect("pipeline");
    let opt = Program { ir: p.ir, sink: root, name: p.name.clone() };
    let sink = lower::lower(&opt.ir, opt.sink, &opt.name);
    let topo = sink.toposort();
    // Flat: the lowered sink cone has no RANGE / END.
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Range { .. } | Op::End { .. })), "lowered body must be flat");
    // Fully unrolled: one WMMA per K-fragment.
    let wmmas = topo.iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).count();
    assert_eq!(wmmas, 4, "K=64 → 4 unrolled WMMA fragments");
}
