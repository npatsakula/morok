//! Parity against the PyTorch reference (`answerdotai/ModernBERT-base` and its
//! SST-2 / CoNLL-2003 fine-tunes). Heavy: each tier loads the real checkpoint
//! (fetched from HF Hub on first run) plus a golden produced by HuggingFace
//! `transformers`.
//!
//! # Goldens are not committed
//!
//! The `golden*.safetensors` fixtures are generated locally and are **not** on
//! HF Hub — only the model weights / configs are. Generate them before running
//! the `--ignored` tier (output lands in `data/modernbert/`, overridable via
//! `SVOD_MODERNBERT`):
//!
//! ```text
//! uv run model/scripts/generate_modernbert_golden.py            // backbone + MLM + embedder
//! uv run model/scripts/generate_modernbert_classifier_golden.py // SST-2 classifier
//! uv run model/scripts/generate_modernbert_token_golden.py      // CoNLL-2003 NER
//! ```
//!
//! Runs in **f32** (config dtype overridden) so it works on CPU backends
//! without GPU bf16 transcendentals. bf16 numerical parity is implied by the
//! framework's f32-accumulator guarantees in layernorm/matmul/attention.

use std::path::{Path, PathBuf};

use svod_arch::pipelines::text::{EncoderHead, Encoding};

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::modernbert::{
    ModernBert, ModernBertClassificationModel, ModernBertClassifier, ModernBertConfig, ModernBertEmbedder,
    ModernBertForMaskedLm, ModernBertTokenClassificationModel, ModernBertTokenClassifier,
};
use crate::state::StateDict;

const HUB_REPO: &str = "answerdotai/ModernBERT-base";

/// `repo.get` fallback: model weights / configs live on HF Hub, but the
/// `golden*.safetensors` fixtures are generated locally (not committed, not
/// published) — tell the caller which is which instead of a generic "download"
/// message that's wrong for goldens.
fn missing_from_hub(name: &str) -> ! {
    if name.starts_with("golden") {
        panic!(
            "{name} not found in data/modernbert/ and not on HF Hub — goldens are generated \
             locally; run `model/scripts/generate_modernbert*.py` (or point SVOD_MODERNBERT at \
             the output dir)"
        );
    }
    panic!("download {name} from HF Hub")
}

/// Resolve `model.safetensors` / `golden.safetensors` for the real-checkpoint
/// tests: `SVOD_MODERNBERT` dir override → local `data/modernbert/` (output of
/// `model/scripts/generate_modernbert_golden.py`) → HF Hub download.
fn real_file(name: &str) -> PathBuf {
    let dir = std::env::var_os("SVOD_MODERNBERT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/modernbert"));
    let local = dir.join(name);
    if local.exists() {
        local
    } else {
        let api = hf_hub::api::sync::Api::new().expect("HF Hub API");
        let repo = api.repo(hf_hub::Repo::with_revision(HUB_REPO.into(), hf_hub::RepoType::Model, "main".into()));
        repo.get(name).unwrap_or_else(|_| missing_from_hub(name))
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

/// Embedding-pipeline parity: Svod's fused backbone+masked-mean-pool+L2-norm JIT
/// (`ModernBertEmbedder`) vs the PyTorch reference (`expected_embedding` from the
/// golden, produced by `transformers` + the same pooling recipe). Validates the
/// full embed path — not just the backbone forward — against an independent
/// reference. The 1e-12 denominator/norm EPS Svod adds is negligible here.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn embeddings_match_pytorch() {
    let weights = real_file("model.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");
    let cfg_path = real_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;
    let model = ModernBert::from_safetensors(&weights, cfg).expect("load weights");

    let input_ids: Vec<u32> = load_golden_vec::<i64>(&golden, "input_ids").into_iter().map(|x| x as u32).collect();
    let attention_mask: Vec<u32> =
        load_golden_vec::<i64>(&golden, "attention_mask").into_iter().map(|x| x as u32).collect();
    let want: Vec<f32> = load_golden_vec(&golden, "expected_embedding");
    let seq_len = input_ids.len();
    let d = want.len();

    let mut embedder = ModernBertEmbedder::new(model, 1, seq_len).expect("embedder");
    let enc = Encoding {
        input_ids,
        attention_mask,
        token_type_ids: vec![0; seq_len],
        offsets: vec![(0, 0); seq_len],
        special_tokens_mask: vec![0; seq_len],
    };
    let (got, _prof) = embedder.run_batch(&[&enc], false).expect("embed");
    let got = &got[0].values;
    assert_eq!(got.len(), d, "embedding dim mismatch");

    let max_delta = got.iter().zip(want.iter()).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
    eprintln!("embedding max |delta| = {max_delta:.3e}");
    assert!(max_delta < 1e-3, "embedding drifted from PyTorch golden: max |delta| = {max_delta}");
}

// ── classifier parity (SST-2 fine-tuned checkpoint) ─────────────────────────
//
// Tests the fused backbone + classification head (pool → dense → GELU → norm →
// classifier) against `AutoModelForSequenceClassification` from `transformers`.
// The SST-2 model uses mean pooling and `classifier_bias = false` — exercising
// the masked-mean path and the optional-bias code.

const CLASSIFIER_HUB_REPO: &str = "AnkitAI/Sensible-ModernBERT-Sentiment-Analysis";

/// Resolve classifier-parity artifacts: same 3-tier as [`real_file`] but from
/// the SST-2 fine-tuned repo.
fn classifier_file(name: &str) -> PathBuf {
    let dir = std::env::var_os("SVOD_MODERNBERT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/modernbert"));
    let local = dir.join(name);
    if local.exists() {
        local
    } else {
        let api = hf_hub::api::sync::Api::new().expect("HF Hub API");
        let repo =
            api.repo(hf_hub::Repo::with_revision(CLASSIFIER_HUB_REPO.into(), hf_hub::RepoType::Model, "main".into()));
        repo.get(name).unwrap_or_else(|_| missing_from_hub(name))
    }
}

/// Build the classifier + golden encodings. Batch and seq_len are derived from
/// the golden's `input_ids` shape so the JIT preparation matches the golden's
/// padding.
fn load_classifier_fixture() -> (ModernBertClassifier, Vec<Encoding>, Vec<f32>, usize) {
    let golden = crate::state::load_safetensors(&classifier_file("golden_classifier.safetensors")).expect("golden");

    let ids_shape = golden.get("input_ids").expect("input_ids").shape().expect("input_ids shape");
    let batch = ids_shape[0].as_const().expect("concrete batch dim");
    let seq_len = ids_shape[1].as_const().expect("concrete seq_len dim");

    let input_ids: Vec<i64> = load_golden_vec(&golden, "input_ids");
    let attention_mask: Vec<i64> = load_golden_vec(&golden, "attention_mask");
    let expected_logits: Vec<f32> = load_golden_vec(&golden, "expected_logits");
    let num_labels = expected_logits.len() / batch;

    let encodings: Vec<Encoding> = (0..batch)
        .map(|i| {
            let off = i * seq_len;
            Encoding {
                input_ids: input_ids[off..off + seq_len].iter().map(|x| *x as u32).collect(),
                attention_mask: attention_mask[off..off + seq_len].iter().map(|x| *x as u32).collect(),
                token_type_ids: vec![0; seq_len],
                offsets: vec![(0, 0); seq_len],
                special_tokens_mask: vec![0; seq_len],
            }
        })
        .collect();

    let cfg_path = classifier_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;

    let weights_path = classifier_file("model.safetensors");
    let sd = crate::state::load_safetensors(&weights_path).expect("load weights");
    let model = ModernBertClassificationModel::from_state_dict(&sd, &cfg).expect("build model");
    let classifier = ModernBertClassifier::new(model, batch, seq_len).expect("build classifier");

    (classifier, encodings, expected_logits, num_labels)
}

/// Classification parity: Svod's fused backbone+head JIT (`ModernBertClassifier`)
/// vs the PyTorch reference (`expected_logits` from the golden, produced by
/// `transformers` `AutoModelForSequenceClassification`). The SST-2 model uses
/// mean pooling with `classifier_bias = false`.
#[test]
#[ignore = "heavy: real SST-2 ModernBERT-base classifier weights + PyTorch golden (local or HF Hub download)"]
fn classify_logits_match_pytorch() {
    let (mut classifier, encodings, want, num_labels) = load_classifier_fixture();

    let refs: Vec<&Encoding> = encodings.iter().collect();
    let (classifications, _prof) = classifier.run_batch(&refs, false).expect("classify");

    assert_eq!(classifications.len(), encodings.len(), "batch size mismatch");
    for (i, c) in classifications.iter().enumerate() {
        let expected = &want[i * num_labels..(i + 1) * num_labels];
        let max_delta = c.logits.iter().zip(expected.iter()).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
        eprintln!("[{i}] logits max |delta| = {max_delta:.3e}  got={:.4?}  want={:.4?}", c.logits, expected);
        assert!(max_delta < 1e-3, "classification logits [{i}] drifted from PyTorch golden: max |delta| = {max_delta}");
    }
}

/// Negative control: an all-ones mask (ignoring padding) must DIVERGE from the
/// golden — both attention and mean-pooling are contaminated by pad tokens. If
/// this passes, the golden isn't exercising the mask.
#[test]
#[ignore = "heavy: real SST-2 ModernBERT-base classifier weights + PyTorch golden (local or HF Hub download)"]
fn ignoring_padding_diverges_in_classification() {
    let (mut classifier, encodings, want, num_labels) = load_classifier_fixture();

    let unmasked: Vec<Encoding> = encodings
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.attention_mask.fill(1);
            e
        })
        .collect();

    let refs: Vec<&Encoding> = unmasked.iter().collect();
    let (classifications, _prof) = classifier.run_batch(&refs, false).expect("classify unmasked");

    for (i, c) in classifications.iter().enumerate() {
        let expected = &want[i * num_labels..(i + 1) * num_labels];
        let max_delta = c.logits.iter().zip(expected.iter()).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
        assert!(
            max_delta > 1e-2,
            "ignoring padding did NOT diverge from the golden [{i}] (max |delta| = {max_delta:.3e})"
        );
        eprintln!("[{i}] unmasked logits max |delta| = {max_delta:.3e} (diverges as expected)");
    }
}

// ─── token classification parity ───────────────────────────────────────────
//
// Tests the fused backbone + token head (`prediction_head_tail` over the full
// `(B, L, D)` state, no pooling) against `AutoModelForTokenClassification` from
// `transformers`. The CoNLL-2003 NER fine-tune has no pooling; `classifier_bias`
// is exercised by the always-present `classifier.bias`.

const TOKEN_HUB_REPO: &str = "sanketrai/modernbert-base-conll2003-english-ner";

/// Resolve token-parity artifacts: same 3-tier as [`real_file`] but from the
/// CoNLL-2003 NER fine-tuned repo.
fn token_file(name: &str) -> PathBuf {
    let dir = std::env::var_os("SVOD_MODERNBERT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/modernbert"));
    let local = dir.join(name);
    if local.exists() {
        local
    } else {
        let api = hf_hub::api::sync::Api::new().expect("HF Hub API");
        let repo = api.repo(hf_hub::Repo::with_revision(TOKEN_HUB_REPO.into(), hf_hub::RepoType::Model, "main".into()));
        repo.get(name).unwrap_or_else(|_| missing_from_hub(name))
    }
}

/// Build the token classifier + golden encodings. Batch and seq_len come from
/// the golden's `input_ids` shape so the JIT preparation matches the golden's
/// padding. Returns the flat `(B, L, num_labels)` expected logits + num_labels.
fn load_token_fixture() -> (ModernBertTokenClassifier, Vec<Encoding>, Vec<f32>, usize, usize) {
    let golden = crate::state::load_safetensors(&token_file("golden_token.safetensors")).expect("golden");

    let ids_shape = golden.get("input_ids").expect("input_ids").shape().expect("input_ids shape");
    let batch = ids_shape[0].as_const().expect("concrete batch dim");
    let seq_len = ids_shape[1].as_const().expect("concrete seq_len dim");

    let input_ids: Vec<i64> = load_golden_vec(&golden, "input_ids");
    let attention_mask: Vec<i64> = load_golden_vec(&golden, "attention_mask");
    let expected_logits: Vec<f32> = load_golden_vec(&golden, "expected_logits");
    let num_labels = expected_logits.len() / (batch * seq_len);

    let encodings: Vec<Encoding> = (0..batch)
        .map(|i| {
            let off = i * seq_len;
            Encoding {
                input_ids: input_ids[off..off + seq_len].iter().map(|x| *x as u32).collect(),
                attention_mask: attention_mask[off..off + seq_len].iter().map(|x| *x as u32).collect(),
                token_type_ids: vec![0; seq_len],
                offsets: vec![(0, 0); seq_len],
                special_tokens_mask: vec![0; seq_len],
            }
        })
        .collect();

    let cfg_path = token_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;

    let weights_path = token_file("model.safetensors");
    let sd = crate::state::load_safetensors(&weights_path).expect("load weights");
    let model = ModernBertTokenClassificationModel::from_state_dict(&sd, &cfg).expect("build model");
    let recognizer = ModernBertTokenClassifier::new(model, batch, seq_len).expect("build token classifier");

    (recognizer, encodings, expected_logits, num_labels, seq_len)
}

/// Token-classification parity: Svod's fused backbone + per-token head JIT
/// (`ModernBertTokenClassifier`) vs the PyTorch reference (`expected_logits`
/// from the golden, produced by `transformers`
/// `AutoModelForTokenClassification`). Compares real-token positions only (mask
/// = 1); pad positions are a don't-care.
#[test]
#[ignore = "heavy: real CoNLL-2003 ModernBERT-base NER weights + PyTorch golden (local or HF Hub download)"]
fn token_logits_match_pytorch() {
    let (mut recognizer, encodings, want, num_labels, seq_len) = load_token_fixture();

    let refs: Vec<&Encoding> = encodings.iter().collect();
    let (classifications, _prof) = recognizer.run_batch(&refs, false).expect("recognize");

    assert_eq!(classifications.len(), encodings.len(), "batch size mismatch");
    let mut worst = 0.0f32;
    for (i, c) in classifications.iter().enumerate() {
        assert_eq!(c.logits.len(), seq_len * num_labels, "per-chunk grid shape");
        // Compare only real-token rows: pad rows are a don't-care and may drift.
        for t in 0..seq_len {
            if encodings[i].attention_mask[t] == 0 {
                continue;
            }
            let got = &c.logits[t * num_labels..(t + 1) * num_labels];
            let want_off = (i * seq_len + t) * num_labels;
            let expected = &want[want_off..want_off + num_labels];
            let max_delta = got.iter().zip(expected.iter()).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            worst = worst.max(max_delta);
            assert!(max_delta < 1e-3, "token logits [{i}][{t}] drifted from PyTorch golden: max |delta| = {max_delta}");
        }
    }
    eprintln!("token logits real-position max |delta| = {worst:.3e}");
}

/// Negative control: an all-ones mask (ignoring padding) must DIVERGE from the
/// golden on real-token rows — backbone attention is contaminated by pad keys.
/// If this passes, the golden isn't exercising the mask.
#[test]
#[ignore = "heavy: real CoNLL-2003 ModernBERT-base NER weights + PyTorch golden (local or HF Hub download)"]
fn ignoring_padding_diverges_in_token_classification() {
    let (mut recognizer, encodings, want, num_labels, seq_len) = load_token_fixture();

    let unmasked: Vec<Encoding> = encodings
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e.attention_mask.fill(1);
            e
        })
        .collect();

    let refs: Vec<&Encoding> = unmasked.iter().collect();
    let (classifications, _prof) = recognizer.run_batch(&refs, false).expect("recognize unmasked");

    for (i, c) in classifications.iter().enumerate() {
        for t in 0..seq_len {
            if encodings[i].attention_mask[t] == 0 {
                continue; // compare only originally-real positions
            }
            let got = &c.logits[t * num_labels..(t + 1) * num_labels];
            let want_off = (i * seq_len + t) * num_labels;
            let expected = &want[want_off..want_off + num_labels];
            let max_delta = got.iter().zip(expected.iter()).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            assert!(
                max_delta > 1e-2,
                "ignoring padding did NOT diverge from the golden [{i}][{t}] (max |delta| = {max_delta:.3e})"
            );
        }
    }
}
