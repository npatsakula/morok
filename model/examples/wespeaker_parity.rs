//! Parity / smoke driver for [`WeSpeakerResNet34`].
//!
//! Loads a reference NPZ produced by the upstream Python pipeline (e.g.
//! pyannote / DiariZen) and the pyannote `pytorch_model.bin` checkpoint, runs
//! the svod JIT forward, and prints how far the svod output is from the
//! reference embeddings.
//!
//! ## NPZ contract
//!
//! Three arrays, all `float32`:
//! - `fbank`   shape `[B, 1598, 80]` — Kaldi-fbank features
//! - `weights` shape `[B, 799]`      — per-frame attention weights
//! - `expected` shape `[B, 256]`     — reference embeddings to compare against
//!
//! Aliases accepted as a convenience (the upstream dump script may use them):
//! `feats`/`features` for `fbank`, `mask`/`weight` for `weights`,
//! `embeddings`/`embedding` for `expected`.
//!
//! ## Usage
//!
//! ```text
//! cargo run -p svod-model --release --example wespeaker_parity -- \
//!     --bin /path/to/pytorch_model.bin --data /path/to/reference.npz
//!
//! # or pull the bin from HF Hub:
//! cargo run -p svod-model --release --example wespeaker_parity -- \
//!     --hub --data /path/to/reference.npz
//! ```

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use ndarray::{Array2, Array3};
use ndarray_npy::NpzReader;

use svod_model::jit::InputSpec;
use svod_model::wespeaker::{WeSpeakerConfig, WeSpeakerResNet34, WeSpeakerResNet34Jit};

const DEFAULT_HUB_ID: &str = "pyannote/wespeaker-voxceleb-resnet34-LM";

#[derive(Parser, Debug)]
#[command(about = "WeSpeaker ResNet34 parity smoke test", long_about = None)]
struct Args {
    /// Path to `pytorch_model.bin` (pyannote / PyTorch-Lightning format).
    #[arg(long, conflicts_with = "hub", required_unless_present = "hub")]
    bin: Option<PathBuf>,

    /// Pull the checkpoint from HuggingFace Hub instead of a local path.
    #[arg(long)]
    hub: bool,

    /// Override the HF Hub model id when using `--hub`.
    #[arg(long, default_value = DEFAULT_HUB_ID)]
    hf_id: String,

    /// Reference NPZ file with `fbank` + `weights` + `expected` arrays.
    #[arg(long)]
    data: PathBuf,

    /// Maximum batch baked into the JIT plan. Must be ≥ the batch dim of the
    /// NPZ inputs.
    #[arg(long, default_value_t = 16)]
    max_batch: usize,
}

fn pick_3d(npz: &mut NpzReader<std::fs::File>, candidates: &[&str], present: &[String]) -> Result<Array3<f32>, String> {
    for name in candidates {
        if present.iter().any(|p| p == name || p == &format!("{name}.npy")) {
            return npz.by_name(name).map_err(|e| format!("read `{name}`: {e}"));
        }
    }
    Err(format!("none of {candidates:?} found in NPZ (present: {present:?})"))
}

fn pick_2d(npz: &mut NpzReader<std::fs::File>, candidates: &[&str], present: &[String]) -> Result<Array2<f32>, String> {
    for name in candidates {
        if present.iter().any(|p| p == name || p == &format!("{name}.npy")) {
            return npz.by_name(name).map_err(|e| format!("read `{name}`: {e}"));
        }
    }
    Err(format!("none of {candidates:?} found in NPZ (present: {present:?})"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // -----------------------------------------------------------------------
    // Load reference data
    // -----------------------------------------------------------------------
    println!("Loading reference NPZ: {}", args.data.display());
    let mut npz = NpzReader::new(std::fs::File::open(&args.data)?)?;
    let names = npz.names()?;
    let fbank = pick_3d(&mut npz, &["feats", "fbank", "features"], &names)?;
    let weights = pick_2d(&mut npz, &["weights", "weight", "mask"], &names)?;
    let expected = pick_2d(&mut npz, &["embeddings", "expected", "embedding"], &names)?;

    let (b, t, f) = fbank.dim();
    let (b_w, t_w) = weights.dim();
    let (b_e, d) = expected.dim();

    println!("  fbank    : [{b}, {t}, {f}] f32");
    println!("  weights  : [{b_w}, {t_w}] f32");
    println!("  expected : [{b_e}, {d}] f32");

    if b != b_w || b != b_e {
        return Err(format!("batch mismatch: fbank={b} weights={b_w} expected={b_e}").into());
    }
    if t != 1598 || f != 80 {
        eprintln!("warning: fbank is [{b}, {t}, {f}] but the LM checkpoint expects [_, 1598, 80]");
    }
    if t_w != 799 {
        eprintln!("warning: weights are [{b}, {t_w}] but the LM checkpoint expects [_, 799]");
    }
    if d != 256 {
        eprintln!("warning: expected dim is {d}, not 256");
    }
    if b > args.max_batch {
        return Err(format!("NPZ batch {b} exceeds --max-batch {}", args.max_batch).into());
    }

    // -----------------------------------------------------------------------
    // Load model
    // -----------------------------------------------------------------------
    let cfg = WeSpeakerConfig::new().with_max_batch_size(args.max_batch);
    let t_load = Instant::now();
    let model = if args.hub {
        println!("Pulling weights from HF Hub: {}", args.hf_id);
        WeSpeakerResNet34::from_hub(&args.hf_id, cfg)?
    } else {
        let p = args.bin.as_ref().expect("bin required when --hub is not set");
        println!("Loading weights from {}", p.display());
        WeSpeakerResNet34::from_pytorch_bin(p, cfg)?
    };
    println!("  loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // -----------------------------------------------------------------------
    // Prepare JIT plan
    // -----------------------------------------------------------------------
    let mut jit = WeSpeakerResNet34Jit::new(model);
    println!("Preparing JIT plan [max_b={}, T={t}, F={f}] / [max_b, T_w={t_w}]...", args.max_batch);
    let t_prepare = Instant::now();
    jit.prepare(InputSpec::f32(&[args.max_batch, t, f]), InputSpec::f32(&[args.max_batch, t_w]))?;
    println!("  prepared in {:.2}s", t_prepare.elapsed().as_secs_f64());

    // -----------------------------------------------------------------------
    // Copy inputs (pad up to max_batch with zeros)
    // -----------------------------------------------------------------------
    let mut fbank_pad = vec![0f32; args.max_batch * t * f];
    for (i, row) in fbank.outer_iter().enumerate() {
        let off = i * t * f;
        let flat = row.as_standard_layout();
        fbank_pad[off..off + t * f].copy_from_slice(flat.as_slice().expect("standard layout"));
    }
    let mut weights_pad = vec![0f32; args.max_batch * t_w];
    for (i, row) in weights.outer_iter().enumerate() {
        let off = i * t_w;
        let flat = row.as_standard_layout();
        weights_pad[off..off + t_w].copy_from_slice(flat.as_slice().expect("standard layout"));
    }
    jit.feats_mut()?.copyin(bytemuck::cast_slice(&fbank_pad))?;
    jit.weights_mut()?.copyin(bytemuck::cast_slice(&weights_pad))?;

    // -----------------------------------------------------------------------
    // Execute and read out
    // -----------------------------------------------------------------------
    let t_exec = Instant::now();
    jit.execute_bound(b as i64)?;
    println!("Execute (b={b}): {:.2}s", t_exec.elapsed().as_secs_f64());

    // The buffer is sized for `max_batch`; the declared output's live shape is
    // `[b, 256]`, so the read-back is exactly the rows we bound.
    let out_flat = jit.embeddings_to_vec::<f32>()?;

    // -----------------------------------------------------------------------
    // Compare
    // -----------------------------------------------------------------------
    let mut max_abs = 0f32;
    let mut sum_sq = 0f64;
    let mut sum_abs = 0f64;
    let mut count = 0usize;
    let mut per_row_max = vec![0f32; b];
    for r in 0..b {
        let mut row_max = 0f32;
        for c in 0..d.min(256) {
            let got = out_flat[r * 256 + c];
            let want = expected[[r, c]];
            let diff = (got - want).abs();
            row_max = row_max.max(diff);
            max_abs = max_abs.max(diff);
            sum_sq += (diff as f64) * (diff as f64);
            sum_abs += diff as f64;
            count += 1;
        }
        per_row_max[r] = row_max;
    }
    let mean_abs = sum_abs / count as f64;
    let rms = (sum_sq / count as f64).sqrt();

    println!();
    println!("Difference vs reference:");
    println!("  max |delta|   {max_abs:.3e}");
    println!("  mean |delta|  {mean_abs:.3e}");
    println!("  rms          {rms:.3e}");
    println!("  per-row max:");
    for (r, m) in per_row_max.iter().enumerate() {
        println!("    row {r:>3}: {m:.3e}");
    }

    // Same tolerance as the request doc: max-abs ≤ 2e-5 is what the existing
    // ORT-vs-Python-ORT baseline reaches.
    let pass = max_abs <= 2e-5;
    println!();
    println!("{} (tolerance 2e-5)", if pass { "PASS" } else { "FAIL" });
    if !pass {
        std::process::exit(1);
    }

    Ok(())
}
