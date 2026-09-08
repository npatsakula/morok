use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::conv::YoloConv;
use crate::yolo::error::Result;

/// Spatial Pyramid Pooling - Fast: 1×1 conv → 3× MaxPool(k=5) → cat → 1×1 conv.
/// When `shortcut` is true and `in_ch == out_ch`, a residual connection is added.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`.
#[derive(Clone)]
pub struct Sppf {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub add: bool,
}

impl Sppf {
    pub fn empty(in_ch: usize, out_ch: usize, _kernel: usize, n: usize, shortcut: bool) -> Self {
        let c_hidden = in_ch / 2;
        Self {
            cv1: YoloConv::empty(in_ch, c_hidden, 1, 1, false),
            cv2: YoloConv::empty(c_hidden * (n + 1), out_ch, 1, 1, true),
            add: shortcut && in_ch == out_ch,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y0 = self.cv1.forward(x)?;
        let y1 = y0.max_pool2d().kernel_size(&[5, 5]).stride(&[1, 1]).padding(&[(2, 2), (2, 2)]).call()?;
        let y2 = y1.max_pool2d().kernel_size(&[5, 5]).stride(&[1, 1]).padding(&[(2, 2), (2, 2)]).call()?;
        let y3 = y2.max_pool2d().kernel_size(&[5, 5]).stride(&[1, 1]).padding(&[(2, 2), (2, 2)]).call()?;
        let cat = Tensor::cat(&[&y0, &y1, &y2, &y3], 1)?;
        let out = self.cv2.forward(&cat)?;
        if self.add { Ok(out.try_add(x)?) } else { Ok(out) }
    }
}

impl HasStateDict for Sppf {
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
