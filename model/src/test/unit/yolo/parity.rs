//! Golden parity test — downloads real YOLO26n weights from HuggingFace,
//! runs forward, and compares against a PyTorch reference output.
//!
//! ```text
//! SVOD_YOLO=$PWD/data/yolo cargo test -p svod-model --lib yolo::parity -- --ignored
//! ```

use std::path::PathBuf;

use svod_tensor::{Tensor, Variable};

use crate::state::StateDict;
use crate::state::load_safetensors;
use crate::yolo::{Yolo26Detect, YoloConfig, YoloScale};

const HUB_REPO: &str = "ultralytics/yolo26n";

fn resolve_file(name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("SVOD_YOLO") {
        let p = PathBuf::from(dir).join(name);
        if p.exists() {
            return p;
        }
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data/yolo").join(name);
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

fn load_golden_i64(sd: &StateDict, key: &str) -> Vec<i64> {
    load_golden_vec(sd, key)
}

fn max_abs_delta(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

#[test]
#[ignore = "heavy: real YOLO26n weights + PyTorch golden (local or HF Hub download)"]
fn detect_output_matches_pytorch() {
    let weights = resolve_file("model.safetensors");
    let golden_path = resolve_file("golden.safetensors");

    let cfg = YoloConfig::new(YoloScale::Nano, 80);
    let model = Yolo26Detect::from_safetensors(&weights, cfg).expect("load model");

    let golden = load_safetensors(&golden_path).expect("load golden");

    let image_shape = load_golden_i64(&golden, "images_shape");
    let images_vec = load_golden_vec::<f32>(&golden, "images");
    let images = Tensor::from_slice(&images_vec)
        .try_reshape(image_shape.iter().map(|&d| d as isize).collect::<Vec<_>>())
        .unwrap();

    let var = Variable::new("b", 1, 1);
    let b = var.bind(1).unwrap();
    let out = model.forward(&images, &b).expect("forward");
    out.realize().unwrap();

    let got = out.as_vec::<f32>().unwrap();
    let want = load_golden_vec::<f32>(&golden, "output");
    assert_eq!(got.len(), want.len(), "output length mismatch");
    let delta = max_abs_delta(&got, &want);
    assert!(delta < 1e-3, "max |delta| = {delta:.6} exceeds 1e-3");
}
