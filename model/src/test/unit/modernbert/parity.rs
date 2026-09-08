//! Parity against the PyTorch reference (`answerdotai/ModernBERT-base`).
//! Heavy: loads the real checkpoint + a golden `last_hidden_state` produced by
//! HuggingFace `transformers` (`uv run scripts/convert_modernbert.py`).
//!
//! Runs in **f32** (config dtype overridden) so it works on CPU backends
//! without GPU bf16 transcendentals. bf16 numerical parity is implied by the
//! framework's f32-accumulator guarantees in layernorm/matmul/attention.

use std::path::{Path, PathBuf};

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::modernbert::{ModernBert, ModernBertConfig, ModernBertForMaskedLm};
use crate::state::StateDict;

const HUB_REPO: &str = "answerdotai/ModernBERT-base";

/// Resolve `model.safetensors` / `golden.safetensors` for the real-checkpoint
/// tests: `SVOD_MODERNBERT` dir override → local `data/modernbert/` (output of
/// `scripts/convert_modernbert.py`) → HF Hub download.
fn real_file(name: &str) -> PathBuf {
    let dir = std::env::var_os("SVOD_MODERNBERT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/modernbert"));
    let local = dir.join(name);
    if local.exists() {
        local
    } else {
        let repo = crate::hub::HubRepo::open(HUB_REPO, "main").expect("HF Hub API");
        repo.get(name).unwrap_or_else(|_| panic!("download {name} from HF Hub"))
    }
}

fn load_golden_vec<T: svod_dtype::ext::HasDType + Default + Clone>(sd: &StateDict, key: &str) -> Vec<T> {
    let mut t = sd.get(key).unwrap_or_else(|| panic!("golden key {key}")).clone();
    t.realize().expect("realize golden");
    t.as_vec::<T>().expect("golden readout")
}

/// Load the model + golden fixture once, returning (model, ids, mask, want).
/// `want` is `(T, D)` row-major (batch squeezed by the generator); `mask` is
/// the per-token attention mask (1 = real, 0 = pad).
fn load_fixture() -> (ModernBert, Tensor, Vec<i64>, Vec<f32>) {
    let weights = real_file("model.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");

    let cfg_path = real_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;

    let model = ModernBert::from_safetensors(&weights, cfg).expect("load weights");

    let input_ids: Vec<i64> = load_golden_vec(&golden, "input_ids");
    let want: Vec<f32> = load_golden_vec(&golden, "last_hidden_state");
    let mask: Vec<i64> = golden
        .get("attention_mask")
        .map(|_| load_golden_vec(&golden, "attention_mask"))
        .unwrap_or_else(|| vec![1; input_ids.len()]);
    let (b, l) = match golden.get("input_ids_shape") {
        Some(t) => {
            let mut t = t.clone();
            t.realize().unwrap();
            let s = t.as_vec::<i64>().unwrap();
            (s[0] as usize, s[1] as usize)
        }
        None => (1, input_ids.len()),
    };
    let ids = Tensor::from_slice(input_ids).try_reshape([b as isize, l as isize]).unwrap();

    (model, ids, mask, want)
}

/// Run the model with `mask` (bool, true=real) and return the flat (B, L, D) f32 output.
fn run_forward(model: &ModernBert, ids: &Tensor, mask: Option<&Tensor>) -> Vec<f32> {
    let mut out = model.forward(ids, mask).expect("forward");
    out.realize().expect("realize output");
    out.as_vec::<f32>().expect("output readout")
}

/// Max |delta| over the REAL-token positions only (where `mask == 1`), folded
/// across the hidden dim. Pad positions are excluded: transformers' pad outputs
/// are an artifact (the mask zeroes their attention) and a divergence there is
/// not a model-correctness signal.
fn real_token_max_delta(got: &[f32], want: &[f32], mask: &[i64], d: usize) -> f32 {
    got.chunks_exact(d)
        .zip(want.chunks_exact(d))
        .zip(mask.iter())
        .filter(|&(_, m)| *m == 1)
        .flat_map(|((g, w), _)| g.iter().zip(w.iter()).map(|(a, e)| (a - e).abs()))
        .fold(0.0f32, f32::max)
}

/// `last_hidden_state` parity: our backbone (f32) vs the PyTorch reference.
/// Compares **real-token positions only** — the mask is load-bearing.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn last_hidden_state_matches_pytorch() {
    let (model, ids, mask, want) = load_fixture();
    let d = want.len() / mask.len();

    let mask_t =
        Tensor::from_slice(mask.clone()).cast(DType::Bool).unwrap().try_reshape([1isize, mask.len() as isize]).unwrap();
    let got = run_forward(&model, &ids, Some(&mask_t));

    let real_max = real_token_max_delta(&got, &want, &mask, d);
    eprintln!("real-token max |delta| = {real_max:.3e}");
    assert!(real_max < 1e-3, "real-token last_hidden_state drifted from PyTorch golden: max |delta| = {real_max}");
}

/// Control: the mask must be load-bearing. Running with the correct mask matches
/// the golden on real tokens (above); running with an all-ones mask (ignoring
/// padding) must DIVERGE on real tokens — the 20 pad tokens would otherwise
/// contaminate every real token's attention. If this test ever passes, the
/// golden is no longer exercising the mask.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn ignoring_padding_diverges_from_golden() {
    let (model, ids, mask, want) = load_fixture();
    let d = want.len() / mask.len();

    // All-ones mask: attend to every position including padding.
    let all_ones = Tensor::from_slice(vec![1i64; mask.len()])
        .cast(DType::Bool)
        .unwrap()
        .try_reshape([1isize, mask.len() as isize])
        .unwrap();
    let got_unmasked = run_forward(&model, &ids, Some(&all_ones));
    let real_max_unmasked = real_token_max_delta(&got_unmasked, &want, &mask, d);

    // The golden was produced WITH the mask; ignoring padding must diverge on
    // real tokens by orders of magnitude more than the masked run (1e-3 bound).
    assert!(
        real_max_unmasked > 1e-2,
        "ignoring padding did NOT diverge from the golden on real tokens \
         (max |delta| = {real_max_unmasked:.3e}); the mask is not load-bearing in the golden"
    );
    eprintln!("unmasked real-token max |delta| = {real_max_unmasked:.3e} (diverges as expected)");
}

/// Load the MLM model (backbone + head) from the same weights + config as the
/// backbone fixture, plus the golden `mlm_logits`.
fn load_mlm_fixture() -> (ModernBertForMaskedLm, Tensor, Vec<i64>, Vec<f32>) {
    let weights = real_file("model.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");

    let cfg_path = real_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;

    let model = ModernBertForMaskedLm::from_safetensors(&weights, cfg).expect("load MLM weights");

    let input_ids: Vec<i64> = load_golden_vec(&golden, "input_ids");
    let want: Vec<f32> = load_golden_vec(&golden, "mlm_logits");
    let mask: Vec<i64> = golden
        .get("attention_mask")
        .map(|_| load_golden_vec(&golden, "attention_mask"))
        .unwrap_or_else(|| vec![1; input_ids.len()]);
    let (b, l) = match golden.get("input_ids_shape") {
        Some(t) => {
            let mut t = t.clone();
            t.realize().unwrap();
            let s = t.as_vec::<i64>().unwrap();
            (s[0] as usize, s[1] as usize)
        }
        None => (1, input_ids.len()),
    };
    let ids = Tensor::from_slice(input_ids).try_reshape([b as isize, l as isize]).unwrap();

    (model, ids, mask, want)
}

/// MLM-logits parity: our backbone + MLM head (f32) vs `AutoModelForMaskedLM`.
/// Compares **real-token positions only** (the mask is load-bearing), folding
/// across the vocab axis. The head reuses the f32-accumulator guarantees of
/// matmul/layernorm/GELU, so the same sub-1e-2 regime as the backbone applies.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn mlm_logits_match_pytorch() {
    let (model, ids, mask, want) = load_mlm_fixture();
    let v = want.len() / mask.len();

    let mask_t =
        Tensor::from_slice(mask.clone()).cast(DType::Bool).unwrap().try_reshape([1isize, mask.len() as isize]).unwrap();
    let mut got = model.forward(&ids, Some(&mask_t)).expect("MLM forward");
    got.realize().expect("realize logits");
    let got = got.as_vec::<f32>().expect("logits readout");

    let real_max = real_token_max_delta(&got, &want, &mask, v);
    eprintln!("MLM real-token max |delta| = {real_max:.3e}");
    assert!(real_max < 1e-2, "MLM logits drifted from PyTorch golden: real-token max |delta| = {real_max}");
}
