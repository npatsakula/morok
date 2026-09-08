//! Looped vs. host-unrolled GRU: kernel inventory, rendered step source and timings.
//!
//! `Tensor::gru` compiles one step kernel and re-launches it per time slot
//! through the schedule-level scan loop. The unrolled baseline here is the
//! shape the builder had before that: the same cell, driven by a host `for`
//! loop over constant time offsets, so every step carries its own AST.
//!
//! Run with `cargo run --release -p svod-tensor --example rnn_step_bench`;
//! `SVOD_DEVICE` selects the backend.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use svod_dtype::DType;
use svod_runtime::ExecutionPlan;
use svod_tensor::Tensor;
use svod_tensor::nn::{GruCell, RecurrentCell};

const I: usize = 256;
const H: usize = 256;

fn values(n: usize, seed: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32 + 1.0) * seed).sin() * 0.5).collect()
}

struct Weights {
    x: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
}

fn weights(t: usize, b: usize) -> Weights {
    let x = Tensor::from_slice(values(t * b * I, 0.31)).try_reshape([t as isize, b as isize, I as isize]).unwrap();
    let w_ih = Tensor::from_slice(values(3 * H * I, 0.17)).try_reshape([(3 * H) as isize, I as isize]).unwrap();
    let w_hh = Tensor::from_slice(values(3 * H * H, 0.23)).try_reshape([(3 * H) as isize, H as isize]).unwrap();
    Weights { x, w_ih, w_hh }
}

/// The looped builder: one step graph indexed by a scan variable.
fn looped(w: &Weights) -> Tensor {
    w.x.gru().weight_ih(&w.w_ih).weight_hh(&w.w_hh).call().unwrap().output
}

/// The host-unrolled baseline: `T` structurally identical step graphs whose
/// input slices differ only by a constant offset.
fn unrolled(w: &Weights, t_len: usize, batch: usize) -> Tensor {
    let cell = GruCell::new(w.w_ih.clone(), w.w_hh.clone(), None, None).unwrap();
    let gx = cell.project_input(&w.x).unwrap().contiguous();
    let mut state = Tensor::zeros(&[batch, H], DType::Float32);
    let mut outs = Vec::with_capacity(t_len);
    for t in 0..t_len {
        let gx_t = gx.narrow(0, t, 1usize).unwrap().try_squeeze(Some(0)).unwrap();
        let (y, next) = cell.step_projected(&gx_t, &state).unwrap();
        state = next;
        outs.push(y);
    }
    Tensor::stack(&outs.iter().collect::<Vec<_>>(), 0).unwrap()
}

/// Rendered sources by entry point, with the launch count of each.
fn inventory(plan: &ExecutionPlan) -> BTreeMap<String, (usize, String)> {
    let mut by_source: BTreeMap<String, (usize, String)> = BTreeMap::new();
    for kernel in plan.prepared_kernels() {
        let entry =
            by_source.entry(kernel.kernel.entry_point.clone()).or_insert_with(|| (0, kernel.kernel.code.clone()));
        entry.0 += 1;
    }
    by_source
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let value = f();
    (value, start.elapsed())
}

/// The longest kernel body — for a GRU step that is always the fused matmul
/// plus its elementwise tail, the kernel G1 compares.
fn widest(inventory: &BTreeMap<String, (usize, String)>) -> (String, String) {
    inventory
        .iter()
        .max_by_key(|(_, (_, code))| code.len())
        .map(|(name, (_, code))| (name.clone(), code.clone()))
        .expect("plan has kernels")
}

/// Rewrite SSA names and integer literals to placeholders.
///
/// Two renderings of one kernel differ in every SSA number, and the looped
/// step's buffer offsets are `t * stride` where the unrolled step's are a
/// literal — normalising both away leaves exactly the structural difference
/// G1 asks about.
fn normalize(code: &str) -> Vec<String> {
    code.lines()
        .map(|line| {
            let mut out = String::with_capacity(line.len());
            let mut digits = false;
            for ch in line.chars() {
                if ch.is_ascii_digit() {
                    if !digits {
                        out.push('#');
                    }
                    digits = true;
                } else {
                    digits = false;
                    out.push(ch);
                }
            }
            out
        })
        .collect()
}

/// Lines present in exactly one of the two normalized sources.
fn diff(a: &str, b: &str) -> (Vec<String>, Vec<String>) {
    let (left, right) = (normalize(a), normalize(b));
    let only_left = left.iter().filter(|l| !right.contains(l)).cloned().collect();
    let only_right = right.iter().filter(|l| !left.contains(l)).cloned().collect();
    (only_left, only_right)
}

fn main() {
    println!("I = {I}, H = {H}, f32\n");
    println!(
        "{:<9} {:>4} {:>7} {:>9} {:>9} {:>11} {:>11}",
        "variant", "T", "B", "programs", "launches", "prepare ms", "exec ms"
    );

    let mut sources: BTreeMap<(usize, usize), (String, String)> = BTreeMap::new();
    for batch in [1usize, 8] {
        for t_len in [8usize, 64, 256] {
            for (name, build) in [
                ("looped", &looped as &dyn Fn(&Weights) -> Tensor),
                ("unrolled", &|w: &Weights| unrolled(w, t_len, batch)),
            ] {
                let w = weights(t_len, batch);
                let out = build(&w);
                let (plan, prepare) = timed(|| out.prepare().unwrap());
                plan.execute().unwrap();
                // The readback is inside the timer on purpose: a device submit
                // returns before the work does, so without it the GPU numbers
                // would measure dispatch, not execution. It costs both variants
                // the same T*B*H copyout.
                let (_, exec) = timed(|| {
                    for _ in 0..3 {
                        plan.execute().unwrap();
                    }
                    out.to_vec::<f32>().unwrap()
                });
                let inv = inventory(&plan);
                let launches: usize = inv.values().map(|(n, _)| n).sum();
                println!(
                    "{name:<9} {t_len:>4} {batch:>7} {:>9} {launches:>9} {:>11.1} {:>11.2}",
                    inv.len(),
                    prepare.as_secs_f64() * 1e3,
                    exec.as_secs_f64() * 1e3 / 3.0
                );
                if t_len == 8 && batch == 8 {
                    let (entry, code) = widest(&inv);
                    sources.insert((0, usize::from(name == "unrolled")), (entry, code));
                }
            }
        }
    }

    let (looped_entry, looped_src) = &sources[&(0, 0)];
    let (unrolled_entry, unrolled_src) = &sources[&(0, 1)];
    let (only_looped, only_unrolled) = diff(looped_src, unrolled_src);
    println!("\nwidest step kernel, T=8 B=8 (entry point encodes the applied opts):");
    println!("  looped   {looped_entry}: {} lines", looped_src.lines().count());
    println!("  unrolled {unrolled_entry}: {} lines", unrolled_src.lines().count());
    println!("  normalized lines only in looped ({}):", only_looped.len());
    for line in &only_looped {
        println!("    + {}", line.trim());
    }
    println!("  normalized lines only in unrolled ({}):", only_unrolled.len());
    for line in &only_unrolled {
        println!("    - {}", line.trim());
    }
}
