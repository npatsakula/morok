//! [`Yolo26SemSeg`] — YOLO v26 semantic segmentation.
//!
//! Full backbone (layers 0–10) + partial FPN top-down (layers 11–16) +
//! Conv→Conv2d classifier on P3. Forward returns `[B, nc, H/8, W/8]` logits.

use svod_tensor::nn::{Conv2d, Layer, Module, ResizeMode};
use svod_tensor::{BoundVariable, Tensor};

use crate::state::StateDict;

use super::backbone::YoloBackbone;
use super::blocks::conv::{YoloConv, conv2d_bias};
use super::blocks::csp::C3k2;
use super::config::{YoloConfig, make_depth};
use super::error::Result;

use super::loader;

/// Semantic segmentation classifier: Conv(k3) → Conv2d(k1, bias).
///
/// State-dict keys: `classifier.0.{conv,bn}.*`, `classifier.2.{weight,bias}`.
#[derive(Clone, Module)]
pub struct SemSegClassifier {
    #[module(key = "0")]
    pub conv0: YoloConv,
    #[module(key = "2")]
    pub conv2: Conv2d,
}

impl SemSegClassifier {
    pub fn empty(in_ch: usize, nc: usize) -> Self {
        Self { conv0: YoloConv::empty(in_ch, in_ch, 3, 1, true), conv2: conv2d_bias(in_ch, nc, 1, 1) }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(x)?;
        Ok(self.conv2.forward(&x)?)
    }
}

/// YOLO v26 semantic segmentation model.
///
/// Forward returns `[B, nc, H/8, W/8]` per-pixel logits.
#[derive(Clone, Module)]
pub struct Yolo26SemSeg {
    #[module(skip)]
    pub config: YoloConfig,
    #[module(key = "")]
    pub backbone: YoloBackbone,
    #[module(key = "13")]
    pub c3k2_13: C3k2,
    #[module(key = "16")]
    pub c3k2_16: C3k2,
    #[module(key = "17.classifier")]
    pub classifier: SemSegClassifier,
}

impl Yolo26SemSeg {
    pub fn with_zero_weights(config: YoloConfig) -> Self {
        let scale = config.scale;
        let nc = config.nc;
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [_, _, c2, c3, c4] = super::backbone::scaled_channels(scale);
        Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            c3k2_13: C3k2::empty(c4 + c3, c3, d(2), true, 0.5, true, false),
            c3k2_16: C3k2::empty(c3 + c3, c2, d(2), true, 0.5, true, false),
            classifier: SemSegClassifier::empty(c2, nc),
        }
    }

    pub fn from_hub(model_id: &str, config: YoloConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: YoloConfig) -> Result<Self> {
        let path = loader::download_safetensors(model_id, revision)?;
        Self::from_safetensors(&path, config)
    }

    pub fn from_safetensors(path: &std::path::Path, config: YoloConfig) -> Result<Self> {
        let sd = loader::prepare_state_dict(path)?;
        Self::from_state_dict(&sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: YoloConfig) -> Result<Self> {
        let mut model = Self::with_zero_weights(config);
        model.load_state_dict(sd, "")?;
        Ok(model)
    }

    /// Run the full network. Returns `[B, nc, H/8, W/8]` per-pixel logits.
    pub fn forward(&self, images: &Tensor, batch: &BoundVariable) -> Result<Tensor> {
        let x = loader::shrink_batch(images, batch)?;
        let (l4, l6, l10) = self.backbone.forward(&x)?;

        // Partial FPN top-down (layers 11–16)
        let up = l10.upsample(&[2, 2], ResizeMode::Nearest)?;
        let cat = Tensor::cat(&[&up, &l6], 1)?;
        let l13 = self.c3k2_13.forward(&cat)?;

        let up = l13.upsample(&[2, 2], ResizeMode::Nearest)?;
        let cat = Tensor::cat(&[&up, &l4], 1)?;
        let l16 = self.c3k2_16.forward(&cat)?;

        // Classifier on P3
        self.classifier.forward(&l16)
    }
}
