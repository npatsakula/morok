//! Golden parity test — downloads real Whisper weights from HuggingFace,
//! runs encoder forward, and compares against a PyTorch reference output.
//!
//! ```text
//! SVOD_WHISPER=$PWD/data/whisper cargo test -p svod-model --lib whisper::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::state::{self, StateDict};
use crate::whisper::{ModelDimensions, Whisper, WhisperSize};

const HUB_REPO: &str = "openai/whisper-tiny";

fn resolve_file(name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_WHISPER") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/whisper").join(name);
    if p.exists() {
        return p;
    }
    let repo = crate::hub::HubRepo::open(HUB_REPO, "main").expect("HF Hub API");
    repo.get(name).unwrap_or_else(|_| panic!("download {name} from {HUB_REPO}"))
}

fn load_golden_vec<T: Clone + Default + svod_dtype::ext::HasDType>(sd: &StateDict, key: &str) -> Vec<T> {
    let t = sd.get(key).unwrap_or_else(|| panic!("missing golden key: {key}")).clone();
    t.realize().unwrap();
    t.as_vec::<T>().unwrap()
}

fn max_abs_delta(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

/// Encoder parity: our encoder (f32) vs the PyTorch reference.
/// Input: mel spectrogram `[1, 80, 3000]` → output `[1, 1500, D]`.
#[test]
#[ignore = "heavy: real whisper-tiny weights + PyTorch golden (local or HF Hub download)"]
fn encoder_output_matches_pytorch() {
    let weights = resolve_file("model.safetensors");
    let golden_path = resolve_file("golden.safetensors");

    let mut dims = ModelDimensions::for_size(WhisperSize::Tiny);
    dims.dtype = DType::Float32;
    let sd = state::load_safetensors(&weights).expect("load weights");
    // Strip "model." prefix if present (HF safetensors)
    let sd: StateDict = sd
        .iter()
        .map(|(k, v)| {
            let k2 = k.strip_prefix("model.").unwrap_or(k);
            (k2.to_string(), v.clone())
        })
        .collect();
    let model = Whisper::from_state_dict(&sd, dims).expect("load model");

    // Load golden input (mel) + output (encoder features)
    let golden = state::load_safetensors(&golden_path).expect("load golden");
    let mel_vec = load_golden_vec::<f32>(&golden, "mel");
    let mel_shape = load_golden_vec::<i64>(&golden, "mel_shape");
    let mel =
        Tensor::from_slice(&mel_vec).try_reshape(mel_shape.iter().map(|&d| d as isize).collect::<Vec<_>>()).unwrap();

    let want: Vec<f32> = load_golden_vec(&golden, "encoder_output");

    let out = model.encode(&mel).expect("encoder forward");
    out.realize().expect("realize output");
    let got = out.as_vec::<f32>().expect("output readout");

    assert_eq!(got.len(), want.len(), "encoder output length mismatch");
    let delta = max_abs_delta(&got, &want);
    eprintln!("encoder max |delta| = {delta:.3e}");
    assert!(delta < 1e-3, "encoder output drifted from PyTorch golden: max |delta| = {delta}");
}

/// Decoder logits parity: our decoder (f32) vs the PyTorch reference.
/// Input: encoder features + token ids → logits `[1, L, n_vocab]`.
#[test]
#[ignore = "heavy: real whisper-tiny weights + PyTorch golden (local or HF Hub download)"]
fn decoder_logits_match_pytorch() {
    let weights = resolve_file("model.safetensors");
    let golden_path = resolve_file("golden.safetensors");

    let mut dims = ModelDimensions::for_size(WhisperSize::Tiny);
    dims.dtype = DType::Float32;
    let sd = state::load_safetensors(&weights).expect("load weights");
    let sd: StateDict = sd
        .iter()
        .map(|(k, v)| {
            let k2 = k.strip_prefix("model.").unwrap_or(k);
            (k2.to_string(), v.clone())
        })
        .collect();
    let model = Whisper::from_state_dict(&sd, dims).expect("load model");

    let golden = state::load_safetensors(&golden_path).expect("load golden");

    // Encoder features from golden (or run encoder ourselves)
    let enc_vec = load_golden_vec::<f32>(&golden, "encoder_output");
    let enc_shape = load_golden_vec::<i64>(&golden, "encoder_output_shape");
    let audio_features =
        Tensor::from_slice(&enc_vec).try_reshape(enc_shape.iter().map(|&d| d as isize).collect::<Vec<_>>()).unwrap();

    // Token ids from golden
    let tokens_vec = load_golden_vec::<i64>(&golden, "tokens");
    let tokens = Tensor::from_slice(tokens_vec.iter().map(|&t| t as i32).collect::<Vec<_>>())
        .try_reshape([1isize, tokens_vec.len() as isize])
        .unwrap()
        .cast(DType::Int32);

    let want: Vec<f32> = load_golden_vec(&golden, "logits");

    let out = model.decode(&tokens, &audio_features, 0).expect("decoder forward");
    out.realize().expect("realize logits");
    let got = out.as_vec::<f32>().expect("logits readout");

    assert_eq!(got.len(), want.len(), "logits length mismatch");
    let delta = max_abs_delta(&got, &want);
    eprintln!("decoder logits max |delta| = {delta:.3e}");
    assert!(delta < 1e-2, "decoder logits drifted from PyTorch golden: max |delta| = {delta}");
}
