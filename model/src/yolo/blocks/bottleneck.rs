use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::conv::YoloConv;
use crate::yolo::error::Result;

/// Standard YOLO bottleneck: two Conv+BN+SiLU layers with optional residual.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`.
#[derive(Clone)]
pub struct YoloBottleneck {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub add: bool,
}

impl YoloBottleneck {
    /// Default: `k=(3,3)`, `e=0.5`.
    pub fn empty(in_ch: usize, out_ch: usize, shortcut: bool) -> Self {
        Self::empty_full(in_ch, out_ch, shortcut, 3, 3, 0.5)
    }

    /// Full control: separate kernel sizes for cv1/cv2 and expansion ratio.
    pub fn empty_full(in_ch: usize, out_ch: usize, shortcut: bool, k1: usize, k2: usize, e: f64) -> Self {
        let c_ = (out_ch as f64 * e) as usize;
        let add = shortcut && in_ch == out_ch;
        Self { cv1: YoloConv::empty(in_ch, c_, k1, 1, true), cv2: YoloConv::empty(c_, out_ch, k2, 1, true), add }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.cv2.forward(&self.cv1.forward(x)?)?;
        if self.add { Ok(out.try_add(x)?) } else { Ok(out) }
    }
}

impl HasStateDict for YoloBottleneck {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.cv1.state_dict(&prefixed(prefix, "cv1"));
        sd.extend(self.cv2.state_dict(&prefixed(prefix, "cv2")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.cv1.load_state_dict(sd, &prefixed(prefix, "cv1"))?;
        self.cv2.load_state_dict(sd, &prefixed(prefix, "cv2"))?;
        Ok(())
    }
}
