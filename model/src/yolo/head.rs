//! Shared detection-head infrastructure: branches, anchor generation, box
//! decoding, and postprocessing.

use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::blocks::conv::{Conv2dBias, YoloConv};
use super::error::Result;

/// Generate anchor points and stride tensor from feature map sizes.
///
/// Returns `(anchors [2, A], strides [1, A])` as constant f32 tensors.
/// Anchor point `(x, y)` = `(col + 0.5, row + 0.5)` in grid coordinates.
pub(crate) fn make_anchors(feat_sizes: &[(usize, usize)], strides: &[usize]) -> (Tensor, Tensor) {
    let mut anchor_vec = Vec::new();
    let mut stride_vec = Vec::new();
    for (&(h, w), &s) in feat_sizes.iter().zip(strides.iter()) {
        for y in 0..h {
            for x in 0..w {
                anchor_vec.push((x as f32) + 0.5);
                anchor_vec.push((y as f32) + 0.5);
                stride_vec.push(s as f32);
            }
        }
    }
    let a = anchor_vec.len() / 2;
    let anchors = Tensor::from_slice(&anchor_vec).try_reshape([2, a]).expect("anchor reshape");
    let strides = Tensor::from_slice(&stride_vec).try_reshape([1, a]).expect("stride reshape");
    (anchors, strides)
}

/// Convert distance (lt, rb) predictions to xyxy boxes, then scale by strides.
///
/// `boxes [B, 4, A]`, `anchors [2, A]`, `strides [1, A]` → `[B, 4, A]`.
pub(crate) fn dist2bbox(boxes: &Tensor, anchors: &Tensor, strides: &Tensor, num_anchors: usize) -> Result<Tensor> {
    let parts = boxes.split(&[2, 2], 1)?;
    let lt = &parts[0];
    let rb = &parts[1];

    let anchors_3d = anchors.try_reshape([SInt::from(1isize), SInt::from(2isize), SInt::from(num_anchors as isize)])?;

    let x1y1 = anchors_3d.try_sub(lt)?;
    let x2y2 = anchors_3d.try_add(rb)?;
    let bbox = Tensor::cat(&[&x1y1, &x2y2], 1)?;

    let strides_3d = strides.try_reshape([SInt::from(1isize), SInt::from(1isize), SInt::from(num_anchors as isize)])?;
    Ok(bbox.try_mul(&strides_3d)?)
}

/// Box-regression branch: `Conv(k3) → Conv(k3) → Conv2d(k1, bias)`.
/// Outputs `4 * reg_max` channels.
///
/// State-dict keys: `0.{conv,bn}.*`, `1.{conv,bn}.*`, `2.weight`, `2.bias`.
#[derive(Clone)]
pub struct BoxBranch {
    pub conv0: YoloConv,
    pub conv1: YoloConv,
    pub conv2: Conv2dBias,
}

impl BoxBranch {
    pub fn empty(in_ch: usize, hidden: usize, reg_max: usize) -> Self {
        Self {
            conv0: YoloConv::empty(in_ch, hidden, 3, 1, true),
            conv1: YoloConv::empty(hidden, hidden, 3, 1, true),
            conv2: Conv2dBias::empty(hidden, 4 * reg_max, 1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(x)?;
        let x = self.conv1.forward(&x)?;
        self.conv2.forward(&x)
    }
}

impl HasStateDict for BoxBranch {
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

/// Classification branch (non-legacy): `(DWConv→Conv) × 2 → Conv2d(bias)`.
///
/// State-dict keys: `0.0.*`, `0.1.*`, `1.0.*`, `1.1.*`, `2.weight`, `2.bias`.
#[derive(Clone)]
pub struct ClsBranch {
    pub dw0: YoloConv,
    pub conv0: YoloConv,
    pub dw1: YoloConv,
    pub conv1: YoloConv,
    pub conv2: Conv2dBias,
}

impl ClsBranch {
    pub fn empty(in_ch: usize, hidden: usize, nc: usize) -> Self {
        Self {
            dw0: YoloConv::empty_dw(in_ch, in_ch, 3, 1, true),
            conv0: YoloConv::empty(in_ch, hidden, 1, 1, true),
            dw1: YoloConv::empty_dw(hidden, hidden, 3, 1, true),
            conv1: YoloConv::empty(hidden, hidden, 1, 1, true),
            conv2: Conv2dBias::empty(hidden, nc, 1, 1),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.dw0.forward(x)?;
        let x = self.conv0.forward(&x)?;
        let x = self.dw1.forward(&x)?;
        let x = self.conv1.forward(&x)?;
        self.conv2.forward(&x)
    }
}

impl HasStateDict for ClsBranch {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.dw0.state_dict(&prefixed(prefix, "0.0"));
        sd.extend(self.conv0.state_dict(&prefixed(prefix, "0.1")));
        sd.extend(self.dw1.state_dict(&prefixed(prefix, "1.0")));
        sd.extend(self.conv1.state_dict(&prefixed(prefix, "1.1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "2")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.dw0.load_state_dict(sd, &prefixed(prefix, "0.0"))?;
        self.conv0.load_state_dict(sd, &prefixed(prefix, "0.1"))?;
        self.dw1.load_state_dict(sd, &prefixed(prefix, "1.0"))?;
        self.conv1.load_state_dict(sd, &prefixed(prefix, "1.1"))?;
        self.conv2.load_state_dict(sd, &prefixed(prefix, "2"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Postprocess — top-k selection (outside JIT graph)
// ---------------------------------------------------------------------------

/// A single detection: `(x1, y1, x2, y2, confidence, class_id)`.
pub type Detection = [f32; 6];

/// Top-k postprocess on raw `[B, 4+nc, A]` f32 data.
///
/// Implements the two-stage selection from Ultralytics:
/// 1. Pre-filter anchors by max class score (top-k).
/// 2. Global top-k across all `(anchor, class)` pairs within those anchors.
pub fn postprocess_raw(data: &[f32], shape: &[usize], nc: usize, max_det: usize) -> Result<Vec<Vec<Detection>>> {
    let batch = shape[0];
    let out_ch = shape[1];
    let anchors = shape[2];

    let stride_b = out_ch * anchors;
    let stride_c = anchors;

    let k = max_det.min(anchors);
    let mut results = Vec::with_capacity(batch);

    for b in 0..batch {
        // Stage 1: top-k anchors by max class score.
        let mut max_scores: Vec<(f32, usize)> = (0..anchors)
            .map(|a| {
                let mut best = 0.0f32;
                for c in 0..nc {
                    let s = data[b * stride_b + (4 + c) * stride_c + a];
                    if s > best {
                        best = s;
                    }
                }
                (best, a)
            })
            .collect();
        max_scores.sort_by(|a, b2| b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        max_scores.truncate(k);

        // Stage 2: top-k (anchor, class) within the pre-filtered anchors.
        let mut candidates: Vec<(f32, usize, usize)> = Vec::with_capacity(k * nc);
        for &(_score, a) in &max_scores {
            for c in 0..nc {
                let s = data[b * stride_b + (4 + c) * stride_c + a];
                candidates.push((s, a, c));
            }
        }
        candidates.sort_by(|a, b2| b2.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        let dets: Vec<Detection> = candidates
            .iter()
            .map(|&(score, a, c)| {
                let base = b * stride_b;
                [
                    data[base + a],
                    data[base + stride_c + a],
                    data[base + 2 * stride_c + a],
                    data[base + 3 * stride_c + a],
                    score,
                    c as f32,
                ]
            })
            .collect();
        results.push(dets);
    }

    Ok(results)
}

/// Top-k postprocess on a realized `[B, 4+nc, A]` tensor. Returns one
/// `Vec<Detection>` per image, sorted by confidence descending.
pub fn postprocess(preds: &mut Tensor, nc: usize, max_det: usize) -> Result<Vec<Vec<Detection>>> {
    preds.realize()?;
    let dims = preds.dims()?;
    let data = preds.as_vec::<f32>()?;
    postprocess_raw(&data, &dims, nc, max_det)
}
