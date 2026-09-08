//! [`Yolo26Depth`] — YOLO v26 monocular depth estimation.
//!
//! Full backbone+neck + multi-scale fusion decoder. Projects each pyramid
//! level to `c_mid` channels, progressively upsamples+fuses from P5→P3,
//! then runs Conv→ConvTranspose2d→Conv→Conv2d to produce `[B, 1, H/4, W/4]`.

use svod_tensor::{BoundVariable, Tensor};

use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::backbone::YoloBackbone;
use super::blocks::conv::{Conv2dBias, ConvTranspose2dBias, YoloConv};
use super::config::YoloConfig;
use super::error::Result;

use super::loader;
use super::neck::YoloNeck;

/// Bilinear 2× upsample with align_corners=True.
fn resize_bilinear_2x(x: &Tensor) -> Result<Tensor> {
    use svod_tensor::nn::{CoordinateTransformMode, ResizeMode};
    Ok(x.resize()
        .scales(&[1.0, 1.0, 2.0, 2.0])
        .mode(ResizeMode::Linear)
        .coordinate_transformation_mode(CoordinateTransformMode::AlignCorners)
        .call()?)
}

/// Depth fusion decoder head.
///
/// State-dict keys:
/// - `proj.{i}.*` — 1×1 conv per level
/// - `refine.{i}.0.*`, `refine.{i}.1.*` — refinement blocks
/// - `head.0.*`, `head.1.*`, `head.2.*`, `head.3.*`
/// - `cal_a`, `cal_b` — log-affine calibration buffers
#[derive(Clone)]
pub struct DepthHead {
    pub proj: Vec<YoloConv>,
    pub refine: Vec<(YoloConv, YoloConv)>,
    pub head_conv0: YoloConv,
    pub head_deconv: ConvTranspose2dBias,
    pub head_conv1: YoloConv,
    pub head_conv2: Conv2dBias,
    pub cal_a: Tensor,
    pub cal_b: Tensor,
}

impl DepthHead {
    pub fn empty(ch: &[usize], c_mid: usize) -> Self {
        let proj: Vec<YoloConv> = ch.iter().map(|&c| YoloConv::empty(c, c_mid, 1, 1, true)).collect();
        let nl = ch.len();
        let refine: Vec<(YoloConv, YoloConv)> = (0..nl - 1)
            .map(|_| (YoloConv::empty(c_mid, c_mid, 3, 1, true), YoloConv::empty(c_mid, c_mid, 3, 1, true)))
            .collect();
        Self {
            proj,
            refine,
            head_conv0: YoloConv::empty(c_mid, c_mid / 2, 3, 1, true),
            head_deconv: ConvTranspose2dBias::empty(c_mid / 2, c_mid / 2, 2),
            head_conv1: YoloConv::empty(c_mid / 2, c_mid / 4, 3, 1, true),
            head_conv2: Conv2dBias::empty(c_mid / 4, 1, 1, 1),
            cal_a: Tensor::from_slice([1.0f32]),
            cal_b: Tensor::from_slice([0.0f32]),
        }
    }

    pub fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let nl = feats.len();
        let projected: Vec<Tensor> = (0..nl).map(|i| self.proj[i].forward(&feats[i])).collect::<Result<_>>()?;

        let mut out = projected[nl - 1].clone();
        for i in (0..nl - 1).rev() {
            out = resize_bilinear_2x(&out)?;
            out = out.try_add(&projected[i])?;
            out = self.refine[i].0.forward(&out)?;
            out = self.refine[i].1.forward(&out)?;
        }

        let out = self.head_conv0.forward(&out)?;
        let out = self.head_deconv.forward(&out)?;
        let out = self.head_conv1.forward(&out)?;
        let out = self.head_conv2.forward(&out)?;

        // exp(clamp(out, -4, 5))
        let neg4 = Tensor::from_slice([-4.0f32]);
        let pos5 = Tensor::from_slice([5.0f32]);
        let clamped = out.clamp().min(&neg4).max(&pos5).call()?;
        let depth = clamped.try_exp()?;
        // Log-affine calibration: depth = depth^cal_a * exp(cal_b)
        let depth = depth.try_pow(&self.cal_a)?;
        let cal_b_exp = self.cal_b.try_exp()?;
        Ok(depth.try_mul(&cal_b_exp)?)
    }
}

impl HasStateDict for DepthHead {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, p) in self.proj.iter().enumerate() {
            sd.extend(p.state_dict(&prefixed(prefix, &format!("proj.{i}"))));
        }
        for (i, (a, b)) in self.refine.iter().enumerate() {
            sd.extend(a.state_dict(&prefixed(prefix, &format!("refine.{i}.0"))));
            sd.extend(b.state_dict(&prefixed(prefix, &format!("refine.{i}.1"))));
        }
        sd.extend(self.head_conv0.state_dict(&prefixed(prefix, "head.0")));
        sd.extend(self.head_deconv.state_dict(&prefixed(prefix, "head.1")));
        sd.extend(self.head_conv1.state_dict(&prefixed(prefix, "head.2")));
        sd.extend(self.head_conv2.state_dict(&prefixed(prefix, "head.3")));
        sd.insert(prefixed(prefix, "cal_a"), self.cal_a.clone());
        sd.insert(prefixed(prefix, "cal_b"), self.cal_b.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, p) in self.proj.iter_mut().enumerate() {
            p.load_state_dict(sd, &prefixed(prefix, &format!("proj.{i}")))?;
        }
        for (i, (a, b)) in self.refine.iter_mut().enumerate() {
            a.load_state_dict(sd, &prefixed(prefix, &format!("refine.{i}.0")))?;
            b.load_state_dict(sd, &prefixed(prefix, &format!("refine.{i}.1")))?;
        }
        self.head_conv0.load_state_dict(sd, &prefixed(prefix, "head.0"))?;
        self.head_deconv.load_state_dict(sd, &prefixed(prefix, "head.1"))?;
        self.head_conv1.load_state_dict(sd, &prefixed(prefix, "head.2"))?;
        self.head_conv2.load_state_dict(sd, &prefixed(prefix, "head.3"))?;
        self.cal_a = get_tensor(sd, &prefixed(prefix, "cal_a")).unwrap_or_else(|_| Tensor::from_slice([1.0f32]));
        self.cal_b = get_tensor(sd, &prefixed(prefix, "cal_b")).unwrap_or_else(|_| Tensor::from_slice([0.0f32]));
        Ok(())
    }
}

/// YOLO v26 depth model. Forward returns `[B, 1, H/4, W/4]` depth map.
#[derive(Clone)]
pub struct Yolo26Depth {
    pub config: YoloConfig,
    pub backbone: YoloBackbone,
    pub neck: YoloNeck,
    pub head: DepthHead,
}

impl Yolo26Depth {
    pub fn with_zero_weights(config: YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, c2, c3, c4] = super::backbone::scaled_channels(scale);
        Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            neck: YoloNeck::empty(scale),
            head: DepthHead::empty(&[c2, c3, c4], 256),
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

    pub fn forward(&self, images: &Tensor, batch: &BoundVariable) -> Result<Tensor> {
        let x = loader::shrink_batch(images, batch)?;
        let (l4, l6, l10) = self.backbone.forward(&x)?;
        let (p3, p4, p5) = self.neck.forward(&l4, &l6, &l10)?;
        self.head.forward(&[p3, p4, p5])
    }
}

impl HasStateDict for Yolo26Depth {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.backbone.state_dict(prefix);
        sd.extend(self.neck.state_dict(prefix));
        sd.extend(self.head.state_dict(&prefixed(prefix, "23")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.backbone.load_state_dict(sd, prefix)?;
        self.neck.load_state_dict(sd, prefix)?;
        self.head.load_state_dict(sd, &prefixed(prefix, "23"))?;
        Ok(())
    }
}
