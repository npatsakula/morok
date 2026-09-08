//! Golden parity tests against real `Qwen/Qwen3-Embedding-0.6B` weights.
//!
//! Run with:
//! ```text
//! SVOD_QWEN3=$PWD/data/qwen3 cargo test -p svod-model --lib qwen3::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::qwen3::{Qwen3Config, Qwen3Embedding, Qwen3Model, Qwen3Reranker};
use crate::state;

const HUB_REPO: &str = "Qwen/Qwen3-Embedding-0.6B";
const RERANKER_HUB_REPO: &str = "Qwen/Qwen3-Reranker-0.6B";

fn resolve_file(name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_QWEN3") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/qwen3").join(name);
    if p.exists() {
        return p;
    }
    let repo = crate::hub::HubRepo::open(HUB_REPO, "main").expect("HF Hub API");
    repo.get(name).unwrap_or_else(|_| panic!("download {name} from {HUB_REPO}"))
}

fn load_cfg() -> Qwen3Config {
    let cfg_path = resolve_file("config.json");
    let mut cfg = Qwen3Config::from_json(&cfg_path).expect("config");
    cfg.dtype = DType::Float32;
    cfg
}

fn load_model() -> Qwen3Model {
    let cfg = load_cfg();
    let weights_path = resolve_file("model.safetensors");
    Qwen3Model::from_safetensors(&weights_path, cfg).expect("load model")
}

fn load_golden_vec(key: &str) -> Vec<f32> {
    let golden_path = resolve_file("golden.safetensors");
    let sd = state::load_safetensors(&golden_path).expect("load golden");
    let t = sd.get(key).unwrap_or_else(|| panic!("missing golden key: {key}")).clone();
    t.realize().unwrap();
    t.as_vec::<f32>().unwrap()
}

fn load_golden_i64(key: &str) -> Vec<i64> {
    let golden_path = resolve_file("golden.safetensors");
    let sd = state::load_safetensors(&golden_path).expect("load golden");
    let t = sd.get(key).unwrap_or_else(|| panic!("missing golden key: {key}")).clone();
    t.realize().unwrap();
    t.as_vec::<i64>().unwrap()
}

fn max_abs_delta(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

fn real_token_max_delta(got: &[f32], want: &[f32], mask: &[i64], batch: usize, seq_len: usize, hidden: usize) -> f32 {
    let mut worst = 0.0f32;
    for b in 0..batch {
        for s in 0..seq_len {
            if mask[b * seq_len + s] == 0 {
                continue;
            }
            let off = (b * seq_len + s) * hidden;
            for d in 0..hidden {
                worst = worst.max((got[off + d] - want[off + d]).abs());
            }
        }
    }
    worst
}

#[test]
#[ignore = "heavy: real Qwen3-Embedding-0.6B weights + PyTorch golden"]
fn last_hidden_state_matches_pytorch() {
    let model = load_model();

    let shape_vec = load_golden_i64("input_ids_shape");
    let batch = shape_vec[0] as usize;
    let seq_len = shape_vec[1] as usize;

    let ids_vec = load_golden_i64("input_ids");
    let mask_vec = load_golden_i64("attention_mask");

    let ids = Tensor::from_slice(&ids_vec).try_reshape([batch as isize, seq_len as isize]).unwrap();
    let mask = Tensor::from_slice(&mask_vec).try_reshape([batch as isize, seq_len as isize]).unwrap();

    let out = model.forward(&ids, Some(&mask)).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();

    let want = load_golden_vec("last_hidden_state");
    assert_eq!(got.len(), want.len());

    let mask_vec = load_golden_i64("attention_mask");
    let delta = real_token_max_delta(&got, &want, &mask_vec, batch, seq_len, 1024);
    assert!(delta < 1e-3, "real-token max_abs_delta = {delta:.6} (threshold 1e-3)");
}

#[test]
#[ignore = "heavy: real Qwen3-Embedding-0.6B weights + PyTorch golden"]
fn embeddings_match_pytorch() {
    let model = load_model();
    let emb = Qwen3Embedding { model, normalize: true };

    let shape_vec = load_golden_i64("input_ids_shape");
    let batch = shape_vec[0] as usize;
    let seq_len = shape_vec[1] as usize;

    let ids_vec = load_golden_i64("input_ids");
    let mask_vec = load_golden_i64("attention_mask");

    let ids = Tensor::from_slice(&ids_vec).try_reshape([batch as isize, seq_len as isize]).unwrap();
    let mask = Tensor::from_slice(&mask_vec).try_reshape([batch as isize, seq_len as isize]).unwrap();

    let out = emb.encode(&ids, &mask).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();

    let want = load_golden_vec("embeddings");
    assert_eq!(got.len(), want.len());
    let delta = max_abs_delta(&got, &want);
    assert!(delta < 1e-3, "max_abs_delta = {delta:.6} (threshold 1e-3)");
}

#[test]
#[ignore = "heavy: negative control — all-ones mask must diverge"]
fn ignoring_padding_diverges_from_golden() {
    let model = load_model();

    let shape_vec = load_golden_i64("input_ids_shape");
    let batch = shape_vec[0] as usize;
    let seq_len = shape_vec[1] as usize;

    let ids_vec = load_golden_i64("input_ids");
    let ids = Tensor::from_slice(&ids_vec).try_reshape([batch as isize, seq_len as isize]).unwrap();

    let ones_mask =
        Tensor::from_slice(vec![1i64; batch * seq_len]).try_reshape([batch as isize, seq_len as isize]).unwrap();

    let out = model.forward(&ids, Some(&ones_mask)).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();

    let want = load_golden_vec("last_hidden_state");
    let delta = max_abs_delta(&got, &want);
    assert!(delta > 1e-2, "all-ones mask did NOT diverge (delta={delta:.6})");
}

// --- Reranker parity ---

fn resolve_reranker_file(name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_QWEN3") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/qwen3").join(name);
    if p.exists() {
        return p;
    }
    let repo = crate::hub::HubRepo::open(RERANKER_HUB_REPO, "main").expect("HF Hub API");
    repo.get(name).unwrap_or_else(|_| panic!("download {name} from {RERANKER_HUB_REPO}"))
}

fn load_reranker() -> Qwen3Reranker {
    let cfg_path = resolve_reranker_file("config.json");
    let mut cfg = Qwen3Config::from_json(&cfg_path).expect("config");
    cfg.dtype = DType::Float32;
    let weights_path = resolve_reranker_file("reranker_model.safetensors");
    Qwen3Reranker::from_safetensors(&weights_path, cfg).expect("load reranker")
}

#[test]
#[ignore = "heavy: real Qwen3-Reranker-0.6B weights + PyTorch golden"]
fn reranker_scores_match_pytorch() {
    let reranker = load_reranker();

    let golden_path = resolve_file("golden_reranker.safetensors");
    let sd = state::load_safetensors(&golden_path).expect("load golden reranker");

    let shape_vec = {
        let t = sd.get("input_ids_shape").unwrap().clone();
        t.realize().unwrap();
        t.as_vec::<i64>().unwrap()
    };
    let batch = shape_vec[0] as usize;
    let seq_len = shape_vec[1] as usize;

    let ids_vec = {
        let t = sd.get("input_ids").unwrap().clone();
        t.realize().unwrap();
        t.as_vec::<i64>().unwrap()
    };
    let mask_vec = {
        let t = sd.get("attention_mask").unwrap().clone();
        t.realize().unwrap();
        t.as_vec::<i64>().unwrap()
    };

    let ids = Tensor::from_slice(ids_vec).try_reshape([batch as isize, seq_len as isize]).unwrap();
    let mask = Tensor::from_slice(mask_vec).try_reshape([batch as isize, seq_len as isize]).unwrap();

    let out = reranker.forward(&ids, &mask).unwrap();
    out.realize().unwrap();
    let got = out.as_vec::<f32>().unwrap();

    let want = {
        let t = sd.get("scores").unwrap().clone();
        t.realize().unwrap();
        t.as_vec::<f32>().unwrap()
    };

    assert_eq!(got.len(), want.len());
    let delta = max_abs_delta(&got, &want);
    assert!(delta < 1e-3, "max_abs_delta = {delta:.6} (threshold 1e-3)");
}
