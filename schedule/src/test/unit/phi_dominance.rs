//! Phi-dominance regression: the kmeans generic baseline (`matmul → min over K`)
//! produced invalid LLVM IR at K≥1024 on gfx1151, because a value derived from an
//! inner-loop counter was used after that loop exited. These are the minimal
//! graphs that reproduce it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, Op, ReduceOp, UOp};
use test_case::test_case;

use crate::linearize::linearize_with_cfg;
use crate::optimizer::config::OptStrategy;
use crate::optimizer::tc;
use crate::optimizer::{
    OptimizerConfig, Renderer, Scheduler, apply_post_optimization_with_renderer, optimize_kernel_with_config,
};
use svod_ir::ops;

/// `C[n,k] = Σ_d A[n,d] · B[d,k]`, optionally followed by `MIN_k` — the kmeans
/// baseline `x @ cᵀ → min(1)`. BFloat16 inputs keep RDNA4 WMMA (bf16→f32)
/// selectable.
fn build_matmul(n: i64, k: i64, d: i64, min_over_k: bool) -> Arc<UOp> {
    let n_r = UOp::range_axis(UOp::index_const(n), AxisId::Renumbered(0), AxisType::Global);
    let k_r = UOp::range_axis(UOp::index_const(k), AxisId::Renumbered(1), AxisType::Global);
    let d_r = UOp::range_axis(UOp::index_const(d), AxisId::Renumbered(2), AxisType::Reduce);

    let nf = n_r.clone().cast(DType::BFloat16);
    let kf = k_r.clone().cast(DType::BFloat16);
    let df = d_r.clone().cast(DType::BFloat16);

    let a = nf.try_add(&df).unwrap();
    let b = df.try_add(&kf).unwrap();
    let matmul = a.try_mul(&b).unwrap().reduce(smallvec![d_r], ReduceOp::Add);

    if min_over_k {
        UOp::sink(vec![matmul.reduce(smallvec![k_r], ReduceOp::Min), n_r])
    } else {
        UOp::sink(vec![matmul, n_r, k_r])
    }
}

/// Validate that no instruction in the linearized list references a value from a
/// closed (ended) loop scope without going through AFTER.
///
/// Scans left-to-right maintaining `open_ranges` (RANGEs whose END has not been
/// seen) and `range_deps` (transitive RANGE dependencies per UOp). AFTER merges
/// source scopes and removes only ranges ended by its dependency chain, matching
/// Tinygrad's `ended_ranges` semantics.
fn check_phi_dominance(linear: &[Arc<UOp>]) -> Result<(), String> {
    let mut range_deps: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut open_ranges: HashSet<u64> = HashSet::new();

    for (idx, uop) in linear.iter().enumerate() {
        match uop.op() {
            Op::Range(..) => {
                open_ranges.insert(uop.id);
                let mut deps = HashSet::from([uop.id]);
                for src in uop.op().sources() {
                    deps.extend(range_deps.get(&src.id).cloned().unwrap_or_default());
                }
                range_deps.insert(uop.id, deps);
            }

            Op::End(ops::End { ranges, .. }) => {
                let mut deps = HashSet::new();
                for src in uop.op().sources() {
                    deps.extend(range_deps.get(&src.id).cloned().unwrap_or_default());
                }
                for rid in &deps {
                    if !open_ranges.contains(rid) {
                        return Err(format!("END at [{idx}] depends on closed range {rid}"));
                    }
                }
                range_deps.insert(uop.id, deps);
                for r in ranges {
                    open_ranges.remove(&r.id);
                }
            }

            Op::After(..) => {
                let mut deps = HashSet::new();
                for src in uop.op().sources() {
                    deps.extend(range_deps.get(&src.id).cloned().unwrap_or_default());
                }
                for ended in uop.op().ended_ranges() {
                    match ended.op() {
                        Op::Range(..) => {
                            deps.remove(&ended.id);
                        }
                        _ => {
                            for rid in range_deps.get(&ended.id).cloned().unwrap_or_default() {
                                deps.remove(&rid);
                            }
                        }
                    }
                }
                for rid in &deps {
                    if !open_ranges.contains(rid) {
                        return Err(format!("AFTER at [{idx}] depends on closed range {rid}"));
                    }
                }
                range_deps.insert(uop.id, deps);
            }

            _ => {
                let mut deps = HashSet::new();
                for src in uop.op().sources() {
                    deps.extend(range_deps.get(&src.id).cloned().unwrap_or_default());
                }
                for rid in &deps {
                    if !open_ranges.contains(rid) {
                        return Err(format!(
                            "phi-dominance violation at [{idx}]: {:?} depends on closed range {rid}",
                            uop.op()
                        ));
                    }
                }
                range_deps.insert(uop.id, deps);
            }
        }
    }
    Ok(())
}

/// Check the pre-linearization DAG for cross-scope dependencies: node `u` with
/// RANGE `r` in its `InScopeRanges` consumed by `v` that neither has `r` in scope
/// nor ends it. Such a tree is malformed — no linearizer can produce valid code.
fn check_tree_scope(root: &Arc<UOp>) -> Result<(), String> {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::InScopeRangesProperty;

    let topo = root.toposort();
    for u in &topo {
        let u_scope = InScopeRangesProperty::get(u);
        if u_scope.is_empty() {
            continue;
        }
        for v in &topo {
            if !v.op().sources().iter().any(|s| s.id == u.id) || matches!(v.op(), Op::After(..)) {
                continue;
            }
            let v_scope = InScopeRangesProperty::get(v);
            let v_ended: HashSet<u64> = v.op().ended_ranges().iter().map(|r| r.id).collect();
            for r in u_scope.iter() {
                if !v_scope.contains(r) && !v_ended.contains(r) {
                    return Err(format!(
                        "tree-scope violation: {:?} (scope={{{:?}}}) → consumed by {:?} (scope={{{:?}}}) which doesn't end range {}",
                        u.op(),
                        u_scope.iter().copied().collect::<Vec<_>>(),
                        v.op(),
                        v_scope.iter().copied().collect::<Vec<_>>(),
                        r
                    ));
                }
            }
        }
    }
    Ok(())
}

fn all_capabilities(renderer: Renderer) -> Renderer {
    renderer.with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None)
}

#[test_case(Renderer::amd_rdna4(), 1024, false; "rdna4 matmul only")]
#[test_case(Renderer::amd_cdna3(), 1024, false; "cdna3 matmul only")]
#[test_case(Renderer::amd_rdna4(), 256, true; "rdna4 matmul+min small k")]
#[test_case(Renderer::amd_rdna4(), 1024, true; "rdna4 matmul+min large k")]
#[test_case(Renderer::amd_cdna3(), 1024, true; "cdna3 matmul+min large k")]
fn heuristic_optimizer_keeps_phi_dominance(renderer: Renderer, k: i64, min_over_k: bool) {
    let renderer = all_capabilities(renderer);
    let config = OptimizerConfig { strategy: OptStrategy::Heuristic, ..Default::default() };
    let optimized = optimize_kernel_with_config(build_matmul(64, k, 64, min_over_k), &renderer, &config)
        .expect("optimizer should succeed");

    check_phi_dominance(&linearize_with_cfg(optimized)).unwrap();
}

/// The heuristic optimizer does not always pick TC for these hand-built graphs,
/// so apply it explicitly before the post-optimization + linearize pipeline.
#[test_case(false; "matmul only")]
#[test_case(true; "matmul+min")]
fn tensor_cores_keep_phi_dominance_on_rdna4(min_over_k: bool) {
    let renderer = all_capabilities(Renderer::amd_rdna4());
    let mut scheduler = Scheduler::new(build_matmul(64, 1024, 64, min_over_k), renderer.clone());
    tc::apply(&mut scheduler, -1, 0, 1).expect("TC apply");

    let ast = scheduler.get_optimized_ast(None);
    assert!(ast.toposort().iter().any(|u| matches!(u.op(), Op::Wmma(..))), "TC apply did not produce WMMA");

    let post = apply_post_optimization_with_renderer(ast, &renderer).expect("post optimization");
    check_tree_scope(&post).unwrap();
    check_phi_dominance(&linearize_with_cfg(post)).unwrap();
}
