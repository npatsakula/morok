//! ResNet inference demo.
//!
//! Builds a [`ResNet`] backbone, prepares the [`ResNetJit`] wrapper, feeds it a
//! raw float32 NCHW image batch, and prints the top-k logit indices (when in
//! classification mode) or the feature-map size (in features mode).
//!
//! The example accepts a raw `.bin` file containing `f32` values in NCHW order
//! (`B × C × H × W × 4` bytes). For a quick smoke run without an input file,
//! omit `--image`; the demo synthesises a deterministic sine pattern.
//!
//! ## Usage
//!
//! ```text
//! cargo run -p svod-model --release --example resnet_classify
//! cargo run -p svod-model --release --example resnet_classify -- --image /tmp/cat.bin --hub
//! ```
//!
//! ## Performance note
//!
//! Realising the full ResNet graph through the CPU backend is currently slow
//! (see `perf-pm-apply-rangeify` follow-up). Keep `--side` at 32 unless you've
//! got time on your hands.

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_dtype::DType;
use svod_model::jit::InputSpec;
use svod_model::resnet::{OutputMode, ResNet, ResNetConfig, ResNetDepth, ResNetJit};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DepthArg {
    R18,
    R34,
    R50,
    R101,
    R152,
}

impl From<DepthArg> for ResNetDepth {
    fn from(value: DepthArg) -> Self {
        match value {
            DepthArg::R18 => ResNetDepth::R18,
            DepthArg::R34 => ResNetDepth::R34,
            DepthArg::R50 => ResNetDepth::R50,
            DepthArg::R101 => ResNetDepth::R101,
            DepthArg::R152 => ResNetDepth::R152,
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "ResNet inference smoke test", long_about = None)]
struct Args {
    /// Path to a raw `f32` NCHW image file (`3 * side * side * 4` bytes).
    /// When omitted, a deterministic sine pattern is used.
    #[arg(long)]
    image: Option<PathBuf>,

    /// ResNet variant to instantiate.
    #[arg(long, value_enum, default_value_t = DepthArg::R34)]
    depth: DepthArg,

    /// Square image side length (32 keeps the smoke run fast; ImageNet expects 224).
    #[arg(long, default_value_t = 32)]
    side: usize,

    /// Skip the classification head; print the feature-map size instead.
    #[arg(long)]
    features: bool,

    /// FC head output dimension.
    #[arg(long, default_value_t = 1000)]
    classes: usize,

    /// Print top-K logit indices in classification mode.
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Pull pretrained weights from HuggingFace Hub.
    #[arg(long)]
    hub: bool,

    /// Override the HF Hub model id (defaults to `timm/resnet<depth>.a1_in1k`).
    #[arg(long)]
    hf_id: Option<String>,
}

fn load_input(args: &Args) -> Result<Vec<f32>, String> {
    let elem_count = 3 * args.side * args.side;
    let Some(path) = args.image.as_ref() else {
        // Deterministic pseudo-random fp32 pattern so reruns produce the same logits.
        return Ok((0..elem_count).map(|i| ((i as f32) * 0.137).sin() * 0.5).collect());
    };
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let expected = elem_count * std::mem::size_of::<f32>();
    if bytes.len() != expected {
        return Err(format!(
            "input file is {} bytes; expected {expected} for shape [1, 3, {side}, {side}] f32",
            bytes.len(),
            side = args.side,
        ));
    }
    Ok(bytemuck::cast_slice::<u8, f32>(&bytes).to_vec())
}

fn default_hf_id(depth: ResNetDepth) -> &'static str {
    match depth {
        ResNetDepth::R18 => "timm/resnet18.a1_in1k",
        ResNetDepth::R34 => "timm/resnet34.a1_in1k",
        ResNetDepth::R50 => "timm/resnet50.a1_in1k",
        ResNetDepth::R101 => "timm/resnet101.a1_in1k",
        ResNetDepth::R152 => "timm/resnet152.a1_in1k",
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let depth: ResNetDepth = args.depth.into();
    let output =
        if args.features { OutputMode::Features } else { OutputMode::Classification { num_classes: args.classes } };

    println!("Loading ResNet-{depth:?} (output: {output:?})...");
    let t_load = Instant::now();
    let model = if args.hub {
        let id = args.hf_id.as_deref().unwrap_or_else(|| default_hf_id(depth));
        println!("  pulling weights from HF Hub: {id}");
        ResNet::from_hub(id, depth, output)?
    } else {
        ResNet::with_zero_weights(ResNetConfig::new(depth, output))
    };
    println!("  loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    println!("Preparing JIT plan for [1, 3, {side}, {side}]...", side = args.side);
    let mut jit = ResNetJit::new(model);
    let t_prepare = Instant::now();
    jit.prepare(InputSpec::new(&[1, 3, args.side, args.side], DType::Float32))?;
    println!("  prepared in {:.2}s", t_prepare.elapsed().as_secs_f64());

    let input = load_input(&args)?;
    jit.images_mut()?.copyin(bytemuck::cast_slice(&input))?;

    let t_exec = Instant::now();
    jit.execute_bound(1)?;
    println!("Execute: {:.2}s", t_exec.elapsed().as_secs_f64());

    let result = jit.logits_to_vec::<f32>()?;

    if args.features {
        println!("Feature map: {:?}", jit.logits_shape()?);
        println!("  first 8 values: {:?}", &result[..result.len().min(8)]);
    } else {
        let mut indexed: Vec<(usize, f32)> = result.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        println!("Top-{} logits:", args.top_k);
        for (rank, (idx, score)) in indexed.iter().take(args.top_k).enumerate() {
            println!("  {:>2}. class {:>4}  score {:>8.4}", rank + 1, idx, score);
        }
    }

    Ok(())
}
