//! [`Yolo26Segment`] — YOLO v26 instance segmentation.
//!
//! Detection head + mask coefficient branch + Proto26 prototype generator.
//! Forward returns `(predictions [B, 4+nc+nm, A], protos [B, nm, H/4, W/4])`.

use svod_ir::SInt;
use svod_tensor::{BoundVariable, Tensor};

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::backbone::{YoloBackbone, upsample_nearest_2x};
use super::blocks::conv::ConvTranspose2dBias;
use super::blocks::conv::{Conv2dBias, YoloConv};
use super::config::DETECT_STRIDES;
use super::error::Result;

use super::head::{BoxBranch, ClsBranch, dist2bbox, make_anchors};
use super::loader;
use super::neck::YoloNeck;

/// Mask coefficient branch: Conv(k3) → Conv(k3) → Conv2d(k1, bias).
/// Outputs `nm` channels.
///
/// State-dict keys: `0.*`, `1.*`, `2.weight`, `2.bias`.
#[derive(Clone)]
pub struct MaskBranch {
    pub conv0: YoloConv,
    pub conv1: YoloConv,
    pub conv2: Conv2dBias,
}

impl MaskBranch {
    pub fn empty(in_ch: usize, nm: usize) -> Self {
        let hidden = (16usize).max(in_ch / 4).max(nm);
        Self {
            conv0: YoloConv::empty(in_ch, hidden, 3, 1, true),
            conv1: YoloConv::empty(hidden, hidden, 3, 1, true),
            conv2: Conv2dBias::empty(hidden, nm, 1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(x)?;
        let x = self.conv1.forward(&x)?;
        self.conv2.forward(&x)
    }
}

impl HasStateDict for MaskBranch {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.conv0.state_dict(&prefixed(prefix, "0"));
        sd.extend(self.conv1.state_dict(&prefixed(prefix, "1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "2")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv0.load_state_dict(sd, &prefixed(prefix, "0"))?;
        self.conv1.load_state_dict(sd, &prefixed(prefix, "1"))?;
        self.conv2.load_state_dict(sd, &prefixed(prefix, "2"))?;
        Ok(())
    }
}

/// Proto26: multi-scale feature fusion → ConvTranspose2d → prototype masks.
///
/// State-dict keys (under `proto.`):
/// - `feat_refine.{i}.*` — 1×1 convs projecting P4/P5 → P3 channels
/// - `feat_fuse.*` — Conv(k3)
/// - `cv1.*` — Proto parent cv1 (3×3 conv)
/// - `upsample.weight`, `upsample.bias` — ConvTranspose2d
/// - `cv2.*` — Proto parent cv2 (3×3 conv)
/// - `cv3.*` — Proto parent cv3 (1×1 conv)
#[derive(Clone)]
pub struct Proto26 {
    pub feat_refine: Vec<YoloConv>,
    pub feat_fuse: YoloConv,
    pub cv1: YoloConv,
    pub upsample: ConvTranspose2dBias,
    pub cv2: YoloConv,
    pub cv3: YoloConv,
}

impl Proto26 {
    pub fn empty(ch: &[usize], c_: usize, nm: usize) -> Self {
        let p3_ch = ch[0];
        Self {
            feat_refine: ch[1..].iter().map(|&c| YoloConv::empty(c, p3_ch, 1, 1, true)).collect(),
            feat_fuse: YoloConv::empty(p3_ch, c_, 3, 1, true),
            cv1: YoloConv::empty(c_, c_, 3, 1, true),
            upsample: ConvTranspose2dBias::empty(c_, c_, 2),
            cv2: YoloConv::empty(c_, c_, 3, 1, true),
            cv3: YoloConv::empty(c_, nm, 1, 1, true),
        }
    }

    pub fn forward(&self, feats: &[Tensor]) -> Result<Tensor> {
        let mut feat = feats[0].clone();
        for (i, f) in self.feat_refine.iter().enumerate() {
            let refined = f.forward(&feats[i + 1])?;
            let mut upsampled = refined;
            // Nearest upsample by 2^(i+1) via repeated 2× calls
            for _ in 0..(i + 1) {
                upsampled = upsample_nearest_2x(&upsampled)?;
            }
            feat = feat.try_add(&upsampled)?;
        }
        let feat = self.feat_fuse.forward(&feat)?;
        // Proto parent: cv1 → ConvTranspose2d → cv2 → cv3
        let p = self.cv1.forward(&feat)?;
        let p = self.upsample.forward(&p)?;
        let p = self.cv2.forward(&p)?;
        self.cv3.forward(&p)
    }
}

impl HasStateDict for Proto26 {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, f) in self.feat_refine.iter().enumerate() {
            sd.extend(f.state_dict(&prefixed(prefix, &format!("feat_refine.{i}"))));
        }
        sd.extend(self.feat_fuse.state_dict(&prefixed(prefix, "feat_fuse")));
        sd.extend(self.cv1.state_dict(&prefixed(prefix, "cv1")));
        sd.extend(self.upsample.state_dict(&prefixed(prefix, "upsample")));
        sd.extend(self.cv2.state_dict(&prefixed(prefix, "cv2")));
        sd.extend(self.cv3.state_dict(&prefixed(prefix, "cv3")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, f) in self.feat_refine.iter_mut().enumerate() {
            f.load_state_dict(sd, &prefixed(prefix, &format!("feat_refine.{i}")))?;
        }
        self.feat_fuse.load_state_dict(sd, &prefixed(prefix, "feat_fuse"))?;
        self.cv1.load_state_dict(sd, &prefixed(prefix, "cv1"))?;
        self.upsample.load_state_dict(sd, &prefixed(prefix, "upsample"))?;
        self.cv2.load_state_dict(sd, &prefixed(prefix, "cv2"))?;
        self.cv3.load_state_dict(sd, &prefixed(prefix, "cv3"))?;
        Ok(())
    }
}

/// Segment26 head: Detect box+cls + mask coefficients + Proto26.
///
/// Forward returns `(predictions [B, 4+nc+nm, A], protos [B, nm, H/4, W/4])`.
#[derive(Clone)]
pub struct Segment26 {
    pub cv2: Vec<BoxBranch>,
    pub cv3: Vec<ClsBranch>,
    pub cv4: Vec<MaskBranch>,
    pub proto: Proto26,
    pub nc: usize,
    pub nm: usize,
    pub reg_max: usize,
}

impl Segment26 {
    pub fn empty(ch: &[usize], nc: usize, nm: usize, npr: usize, reg_max: usize) -> Self {
        let hidden_box = (16usize).max(ch[0] / 4).max(reg_max * 4);
        let hidden_cls = ch[0].max(nc.min(100));
        Self {
            cv2: ch.iter().map(|&c| BoxBranch::empty(c, hidden_box, reg_max)).collect(),
            cv3: ch.iter().map(|&c| ClsBranch::empty(c, hidden_cls, nc)).collect(),
            cv4: ch.iter().map(|&c| MaskBranch::empty(c, nm)).collect(),
            proto: Proto26::empty(ch, npr, nm),
            nc,
            nm,
            reg_max,
        }
    }

    pub fn forward(&self, feats: &[Tensor]) -> Result<(Tensor, Tensor)> {
        let proto = self.proto.forward(feats)?;

        let shape = feats[0].shape()?;
        let b = shape[0].clone();

        let mut boxes_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut scores_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut mask_list: Vec<Tensor> = Vec::with_capacity(feats.len());
        let mut feat_sizes: Vec<(usize, usize)> = Vec::with_capacity(feats.len());

        for (i, feat) in feats.iter().enumerate() {
            let h = feat.dim_const(2)?;
            let w = feat.dim_const(3)?;
            feat_sizes.push((h, w));
            let hw = h * w;

            let box_out = self.cv2[i].forward(feat)?;
            boxes_list.push(box_out.try_reshape([b.clone(), SInt::from(4 * self.reg_max), SInt::from(hw)])?);

            let cls_out = self.cv3[i].forward(feat)?;
            scores_list.push(cls_out.try_reshape([b.clone(), SInt::from(self.nc), SInt::from(hw)])?);

            let mask_out = self.cv4[i].forward(feat)?;
            mask_list.push(mask_out.try_reshape([b.clone(), SInt::from(self.nm), SInt::from(hw)])?);
        }

        let boxes_refs: Vec<&Tensor> = boxes_list.iter().collect();
        let scores_refs: Vec<&Tensor> = scores_list.iter().collect();
        let mask_refs: Vec<&Tensor> = mask_list.iter().collect();

        let boxes = Tensor::cat(&boxes_refs, 2)?;
        let scores = Tensor::cat(&scores_refs, 2)?;
        let masks = Tensor::cat(&mask_refs, 2)?;

        let num_anchors: usize = feat_sizes.iter().map(|&(h, w)| h * w).sum();
        let (anchors, strides) = make_anchors(&feat_sizes, &DETECT_STRIDES);

        let dbox = dist2bbox(&boxes, &anchors, &strides, num_anchors)?;
        let scores = scores.sigmoid()?;
        let preds = Tensor::cat(&[&dbox, &scores, &masks], 1)?;

        Ok((preds, proto))
    }
}

impl HasStateDict for Segment26 {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, br) in self.cv2.iter().enumerate() {
            sd.extend(br.state_dict(&prefixed(prefix, &format!("one2one_cv2.{i}"))));
        }
        for (i, br) in self.cv3.iter().enumerate() {
            sd.extend(br.state_dict(&prefixed(prefix, &format!("one2one_cv3.{i}"))));
        }
        for (i, br) in self.cv4.iter().enumerate() {
            sd.extend(br.state_dict(&prefixed(prefix, &format!("one2one_cv4.{i}"))));
        }
        sd.extend(self.proto.state_dict(&prefixed(prefix, "proto")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, br) in self.cv2.iter_mut().enumerate() {
            br.load_state_dict(sd, &prefixed(prefix, &format!("one2one_cv2.{i}")))?;
        }
        for (i, br) in self.cv3.iter_mut().enumerate() {
            br.load_state_dict(sd, &prefixed(prefix, &format!("one2one_cv3.{i}")))?;
        }
        for (i, br) in self.cv4.iter_mut().enumerate() {
            br.load_state_dict(sd, &prefixed(prefix, &format!("one2one_cv4.{i}")))?;
        }
        self.proto.load_state_dict(sd, &prefixed(prefix, "proto"))?;
        Ok(())
    }
}

/// YOLO v26 instance segmentation model.
///
/// Forward returns `(predictions [B, 4+nc+nm, A], protos [B, nm, H/4, W/4])`.
#[derive(Clone)]
pub struct Yolo26Segment {
    pub config: super::config::YoloConfig,
    pub backbone: YoloBackbone,
    pub neck: YoloNeck,
    pub head: Segment26,
}

impl Yolo26Segment {
    pub fn with_zero_weights(config: super::config::YoloConfig) -> Self {
        let scale = config.scale;
        let [_, _, c2, c3, c4] = super::backbone::scaled_channels(scale);
        Self {
            config: config.clone(),
            backbone: YoloBackbone::empty(scale),
            neck: YoloNeck::empty(scale),
            head: Segment26::empty(&[c2, c3, c4], config.nc, 32, 256, config.reg_max),
        }
    }

    pub fn from_hub(model_id: &str, config: super::config::YoloConfig) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main", config)
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str, config: super::config::YoloConfig) -> Result<Self> {
        let path = loader::download_safetensors(model_id, revision)?;
        Self::from_safetensors(&path, config)
    }

    pub fn from_safetensors(path: &std::path::Path, config: super::config::YoloConfig) -> Result<Self> {
        let sd = loader::prepare_state_dict(path)?;
        Self::from_state_dict(&sd, config)
    }

    pub fn from_state_dict(sd: &StateDict, config: super::config::YoloConfig) -> Result<Self> {
        let mut model = Self::with_zero_weights(config);
        model.load_state_dict(sd, "")?;
        Ok(model)
    }

    pub fn forward(&self, images: &Tensor, batch: &BoundVariable) -> Result<(Tensor, Tensor)> {
        let x = loader::shrink_batch(images, batch)?;
        let (l4, l6, l10) = self.backbone.forward(&x)?;
        let (p3, p4, p5) = self.neck.forward(&l4, &l6, &l10)?;
        self.head.forward(&[p3, p4, p5])
    }
}

impl HasStateDict for Yolo26Segment {
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
