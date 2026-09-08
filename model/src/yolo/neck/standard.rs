//! YOLO v26 FPN+PAN neck (layers 11–22).
//!
//! Takes backbone skip features `(l4, l6, l10)` and produces three detection
//! feature maps `(P3, P4, P5)` at strides 8, 16, 32.

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use crate::yolo::backbone::{scaled_channels, upsample_nearest_2x};
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::config::{YoloScale, make_depth};
use crate::yolo::error::Result;

/// Full FPN+PAN neck (layers 11–22). Shared by detect, segment, obb, pose,
/// and depth models.
#[derive(Clone)]
pub struct YoloNeck {
    pub c3k2_13: C3k2,
    pub c3k2_16: C3k2,
    pub conv17: YoloConv,
    pub c3k2_19: C3k2,
    pub conv20: YoloConv,
    pub c3k2_22: C3k2,
}

impl YoloNeck {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [_, _, c2, c3, c4] = scaled_channels(scale);
        Self {
            // Layer 13: C3k2 [512, True] — input is cat(upsample(c4), c3) = c4+c3
            c3k2_13: C3k2::empty(c4 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 16: C3k2 [256, True] — input is cat(upsample(c3), c3) = c3+c3
            c3k2_16: C3k2::empty(c3 + c3, c2, d(2), true, 0.5, true, false),
            // Layer 17: Conv [256, 3, 2]
            conv17: YoloConv::empty(c2, c2, 3, 2, true),
            // Layer 19: C3k2 [512, True] — input is cat(conv17_out, c3k2_13_out)
            c3k2_19: C3k2::empty(c2 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 20: Conv [512, 3, 2]
            conv20: YoloConv::empty(c3, c3, 3, 2, true),
            // Layer 22: C3k2 [1024, True, 0.5, True] — attn=True
            c3k2_22: C3k2::empty(c3 + c4, c4, d(1), true, 0.5, true, true),
        }
    }

    /// Run the full FPN+PAN. Returns `(P3, P4, P5)` detection features.
    pub fn forward(&self, l4: &Tensor, l6: &Tensor, l10: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // FPN top-down
        let up = upsample_nearest_2x(l10)?;
        let cat = Tensor::cat(&[&up, l6], 1)?;
        let l13 = self.c3k2_13.forward(&cat)?;

        let up = upsample_nearest_2x(&l13)?;
        let cat = Tensor::cat(&[&up, l4], 1)?;
        let l16 = self.c3k2_16.forward(&cat)?;

        // PAN bottom-up
        let l17 = self.conv17.forward(&l16)?;
        let cat = Tensor::cat(&[&l17, &l13], 1)?;
        let l19 = self.c3k2_19.forward(&cat)?;

        let l20 = self.conv20.forward(&l19)?;
        let cat = Tensor::cat(&[&l20, l10], 1)?;
        let l22 = self.c3k2_22.forward(&cat)?;

        Ok((l16, l19, l22))
    }
}

impl HasStateDict for YoloNeck {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        let mut sd = self.c3k2_13.state_dict(&p(13));
        sd.extend(self.c3k2_16.state_dict(&p(16)));
        sd.extend(self.conv17.state_dict(&p(17)));
        sd.extend(self.c3k2_19.state_dict(&p(19)));
        sd.extend(self.conv20.state_dict(&p(20)));
        sd.extend(self.c3k2_22.state_dict(&p(22)));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        self.c3k2_13.load_state_dict(sd, &p(13))?;
        self.c3k2_16.load_state_dict(sd, &p(16))?;
        self.conv17.load_state_dict(sd, &p(17))?;
        self.c3k2_19.load_state_dict(sd, &p(19))?;
        self.conv20.load_state_dict(sd, &p(20))?;
        self.c3k2_22.load_state_dict(sd, &p(22))?;
        Ok(())
    }
}
