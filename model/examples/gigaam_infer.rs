//! GigaAM inference demo (CTC + RN-T).
//!
//! Loads a WAV and runs it through an [`Asr`]: a [`FireRedVadSplitter`]-built
//! `VadSplitter` feeds a [`GigaAmTranscriber`]. The head is dispatched from the
//! loaded weights — a CTC revision drives the fused encoder+head JIT, an RN-T
//! revision the encoder + per-step predictor/joint backend (SentencePiece
//! `▁ → space` post-processing inside the transcriber).
//!
//! Substitute a `FixedLengthSplitter` (in `svod_arch::pipelines::audio`,
//! sized from `bounds.max_samples()` / `bounds.align_to_samples()`) for the
//! VAD splitter to skip the FireRedVAD hub download — useful for tests, short
//! utterances, or pipelines that already segmented the input.
//!
//! Usage:
//!   cargo run -p svod-model --release --example gigaam_infer -- audio.wav
//!   cargo run -p svod-model --release --example gigaam_infer -- audio.wav --encoder-dtype int8
//!   cargo run -p svod-model --release --example gigaam_infer -- audio.wav --encoder-dtype fp8
//!   cargo run -p svod-model --release --example gigaam_infer -- audio.wav --rnnt --profile
//!   SVOD_ORIGIN=1 cargo run -p svod-model --release --example gigaam_infer -- \
//!     audio.wav --profile --origin-depth 3 --profile-json profile.json
//!
//! Env knobs (all optional):
//!   SVOD_VAD_THRESHOLD=f       FireRedVAD speech threshold (default 0.4).

use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};

use svod_arch::pipelines::audio::{Asr, RunOptions};
use svod_dtype::DType;
use svod_model::audio::EncoderBounds;
use svod_model::firered_vad::FireRedVadSplitter;
use svod_model::gigaam::{GigaAm, GigaAmTranscriber, TranscribeOpts};

#[derive(Parser, Debug)]
#[command(about = "GigaAM transcription demo (CTC + RN-T)", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono; ints or floats).
    wav: PathBuf,

    /// HF Hub repo or local model directory with the model weights.
    #[arg(long, default_value = "vpermilp/GigaAM-v3")]
    repo: String,

    /// HF Hub revision; the head (CTC vs RN-T) follows the weights.
    /// Defaults to `ctc`, or `e2e_rnnt` under `--rnnt`.
    #[arg(long)]
    revision: Option<String>,

    /// Shorthand for the default RN-T revision.
    #[arg(long)]
    rnnt: bool,

    /// Emit per-word `[start - end] word` lines.
    #[arg(long)]
    timestamps: bool,

    /// Promote greedy CTC to beam search (no-op for RN-T).
    #[arg(long)]
    beam_decode: bool,

    /// Collect and print the typed per-stage GPU profile.
    #[arg(long)]
    profile: bool,

    /// Write the full profile (stages, kernel rows, origin rollups and the
    /// origin arena) as JSON. Implies --profile collection; set SVOD_ORIGIN=1
    /// for the rollups to carry scopes.
    #[arg(long, value_name = "PATH")]
    profile_json: Option<PathBuf>,

    /// Roll origin paths up to this many outermost frames (default: the full
    /// module path). Call frames are never a rollup level.
    #[arg(long, value_name = "N")]
    origin_depth: Option<usize>,

    /// SDPA scores buffer budget (MiB).
    #[arg(long, default_value_t = 256)]
    max_scores_mib: usize,

    /// Encoder compute/storage format (default: FP16).
    /// FP8/INT8 select their named checkpoints and use FP16 activations.
    #[arg(long, value_enum)]
    encoder_dtype: Option<EncoderDtype>,

    /// Override the safetensors file selected by --encoder-dtype.
    #[arg(long)]
    weights: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EncoderDtype {
    F32,
    F16,
    Bf16,
    Fp8,
    Int8,
}

impl EncoderDtype {
    fn compute_dtype(self) -> DType {
        match self {
            EncoderDtype::F32 => DType::Float32,
            EncoderDtype::F16 | EncoderDtype::Fp8 | EncoderDtype::Int8 => DType::Float16,
            EncoderDtype::Bf16 => DType::BFloat16,
        }
    }

    fn weights(self) -> &'static str {
        match self {
            EncoderDtype::Fp8 => "model_fp8.safetensors",
            EncoderDtype::Int8 => "model_int8.safetensors",
            EncoderDtype::F32 | EncoderDtype::F16 | EncoderDtype::Bf16 => "model.safetensors",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let t_total = Instant::now();
    let args = Args::parse();
    let revision = args.revision.clone().unwrap_or_else(|| if args.rnnt { "e2e_rnnt" } else { "ctc" }.to_string());
    let encoder_dtype = args.encoder_dtype.map_or(DType::Float16, EncoderDtype::compute_dtype);
    let weights = args
        .weights
        .as_deref()
        .unwrap_or_else(|| args.encoder_dtype.map_or("model.safetensors", EncoderDtype::weights));
    let opts = TranscribeOpts::builder().beam_decode(args.beam_decode).max_scores_mib(args.max_scores_mib).build();

    println!("Loading audio: {}", args.wav.display());
    let (waveform, sample_rate) = load_wav(&args.wav)?;
    let duration_s = waveform.len() as f32 / sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, sample_rate);

    let local_repo = PathBuf::from(&args.repo);
    println!(
        "\nLoading GigaAM from {}{}...",
        args.repo,
        if local_repo.is_dir() { String::new() } else { format!(" ({revision})") }
    );
    let model = if local_repo.is_dir() {
        GigaAm::from_dir_with_weights_and_encoder_dtype(&local_repo, weights, encoder_dtype)?
    } else {
        GigaAm::from_hub_with_revision_and_weights_and_encoder_dtype(&args.repo, &revision, weights, encoder_dtype)?
    };
    if args.rnnt && model.head.as_rnnt().is_none() {
        return Err(format!("{}@{revision} has a CTC head, not RN-T.", args.repo).into());
    }
    if sample_rate as usize != model.config.sample_rate {
        return Err(format!("WAV is {sample_rate} Hz; model expects {} Hz", model.config.sample_rate).into());
    }
    // Encoder bounds (capacity / frame stride) drive both the splitter's chunk
    // sizing and the transcriber's eager JIT prepare.
    let bounds = EncoderBounds {
        sample_rate: model.config.sample_rate as u32,
        hop_length: model.config.hop_length,
        subsampling_factor: model.config.subsampling_factor,
        max_mel_frames: model.config.max_mel_frames,
        recommended_target_secs: model.recommended_chunk_secs(),
    };
    // `assemble` sizes the (eagerly JIT-prepared) transcriber from the
    // splitter's chunk ceiling — no hand-threaded buffer size.
    let splitter = FireRedVadSplitter::from_hub(&bounds)?;
    let mut asr = Asr::assemble(splitter, |max_chunk| GigaAmTranscriber::new(model, opts.clone(), max_chunk))?;

    println!("Transcribing...");
    let t_transcribe = Instant::now();
    // VAD split → arch pipeline machinery (decode windows → crop → stitch),
    // with the VAD stage folded into the profile.
    let profile = args.profile || args.profile_json.is_some();
    let result = asr.transcribe(&waveform, RunOptions { words: args.timestamps, profile, ..Default::default() })?;
    let dt_transcribe = t_transcribe.elapsed();

    if args.timestamps {
        for chunk in &result.chunks {
            let off = chunk.start_sec;
            for w in chunk.words.iter().flatten() {
                println!("  [{:>6.2} - {:>6.2}] {}", w.start + off, w.end + off, w.text);
            }
        }
    } else {
        for chunk in &result.chunks {
            if !chunk.text.is_empty() {
                println!("  [{:>6.1}s] {}", chunk.start_sec, chunk.text);
            }
        }
    }
    if let Some(profile) = &result.profile {
        if args.profile {
            println!("\n--- Profile ---\n{}", profile.render_report(args.origin_depth));
        }
        if let Some(path) = &args.profile_json {
            std::fs::write(path, profile.to_json(args.origin_depth))?;
            println!("Profile JSON: {}", path.display());
        }
    }

    println!("\n--- Transcript ---\n{}", result.text);
    println!(
        "\nTotal: {:.2}s; transcribe: {:.2}s; loop RTF: {:.4}x",
        t_total.elapsed().as_secs_f32(),
        dt_transcribe.as_secs_f32(),
        if duration_s > 0.0 { dt_transcribe.as_secs_f32() / duration_s } else { 0.0 },
    );
    Ok(())
}

fn load_wav(path: &PathBuf) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?
        }
    };
    Ok((samples, spec.sample_rate))
}
