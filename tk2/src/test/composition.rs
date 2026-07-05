//! The **composition test** (DESIGN §5b "Step 0"): a loop microkernel that exercises the
//! `kloop stages=2` prerequisite — an LDS read carried across the back-edge through a barrier
//! (iteration `t` reads iteration `t-1`'s commit) plus a same-iteration WAR — so a carry bug
//! surfaces in isolation, not as silent garbage in the 385-TF matmul. The `broken` variant
//! (loop-carry `range` edge dropped) is the negative control that proves the test isn't blind.

use svod_dtype::DType;
use svod_tensor::Tensor;
use svod_tensor::testing::allclose_f32;

use crate::graph_kernel;
use crate::kernels::{Program, lds_carry_loop};
use crate::lower;

#[test]
fn lds_carry_loop_lowers_spec_valid() {
    // Both the carried loop AND the broken variant must lower to spec-valid UOp (the broken
    // one is a legal-but-wrong schedule — the device test is what distinguishes them).
    lower::verify(&lds_carry_loop(8, 4, false)).expect("LDS-carry loop must lower spec-valid");
    lower::verify(&lds_carry_loop(8, 4, true)).expect("broken variant must also lower spec-valid");
}

/// `out[lane] = Σ_{t=0}^{T-1} in[(lane+1+t) % n]`.
fn reference(inp: &[f32], n: usize, t: usize) -> Vec<f32> {
    (0..n).map(|lane| (0..t).map(|s| inp[(lane + 1 + s) % n]).sum()).collect()
}

fn run(prog: Program, inp: &Tensor, n: usize) -> Vec<f32> {
    let out = Tensor::empty(&[n], DType::Float32);
    let mut y = graph_kernel(prog, out, &[inp]).expect("wrap");
    let plan = y.prepare().expect("prepare");
    plan.execute().expect("execute");
    y.as_vec::<f32>().expect("read output")
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk2 --lib -- --ignored composition:: --nocapture`
#[test]
#[ignore]
fn lds_carry_through_barrier_is_correct_and_the_test_catches_a_stale_carry() {
    for &n in &[8usize, 16, 64] {
        for &t in &[1usize, 3, n / 2, n] {
            let dev = svod_dtype::default_device::default_device();
            let mut inp = Tensor::rand_with(&[n], DType::Float32, dev).expect("rand");
            inp.realize().expect("realize");
            let iv = inp.as_vec::<f32>().expect("in");
            let expected = reference(&iv, n, t);

            let got = run(lds_carry_loop(n, t, false), &inp, n);
            assert!(allclose_f32(&got, &expected, 1e-3, 1e-4).ok, "n={n} t={t}: carried loop must match reference");

            // Negative control: dropping the range edge re-reads the seed every iteration.
            let broken = run(lds_carry_loop(n, t, true), &inp, n);
            let stale: Vec<f32> = (0..n).map(|l| t as f32 * iv[(l + 1) % n]).collect();
            assert!(allclose_f32(&broken, &stale, 1e-3, 1e-4).ok, "n={n} t={t}: broken must equal the stale-read");
            if t > 1 {
                assert!(
                    !allclose_f32(&broken, &expected, 1e-3, 1e-4).ok,
                    "n={n} t={t}: broken must differ (test not blind)"
                );
            }
        }
    }
}
