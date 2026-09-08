//! [`YoloNeckP2`] — YOLO v26-P2 FPN+PAN neck (layers 11–28).
//!
//! Extends the standard neck with one extra FPN top-down stage (reaching
//! P2/4) and one extra PAN bottom-up stage. Outputs four detection feature
//! maps at strides 4, 8, 16, 32.

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use crate::yolo::backbone::upsample_nearest_2x;
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::config::{YoloScale, make_depth, scale_channels};
use crate::yolo::error::Result;

/// P2 neck (layers 11–28). Takes four backbone features `(l2, l4, l6, l10)`
/// and produces `(P2, P3, P4, P5)` at strides 4, 8, 16, 32.
#[derive(Clone)]
pub struct YoloNeckP2 {
    pub c3k2_13: C3k2,
    pub c3k2_16: C3k2,
    pub c3k2_19: C3k2,
    pub conv20: YoloConv,
    pub c3k2_22: C3k2,
    pub conv23: YoloConv,
    pub c3k2_25: C3k2,
    pub conv26: YoloConv,
    pub c3k2_28: C3k2,
}

impl YoloNeckP2 {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let sc = |yaml_c| scale_channels(yaml_c, scale);
        let c1 = sc(128);
        let c2 = sc(256);
        let c3 = sc(512);
        let c4 = sc(1024);
        Self {
            // Layer 13: C3k2 [512, True] — input cat(upsample(c4), c3) = c4+c3
            c3k2_13: C3k2::empty(c4 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 16: C3k2 [256, True] — input cat(upsample(l13_out=c3), l4=c3) = c3+c3
            c3k2_16: C3k2::empty(c3 + c3, c2, d(2), true, 0.5, true, false),
            // Layer 19: C3k2 [128, True] — input cat(upsample(l16_out=c2), l2=c2) = c2+c2
            c3k2_19: C3k2::empty(c2 + c2, c1, d(2), true, 0.5, true, false),
            // Layer 20: Conv [128, 3, 2]
            conv20: YoloConv::empty(c1, c1, 3, 2, true),
            // Layer 22: C3k2 [256, True] — input cat(conv20_out, c3k2_16_out)
            c3k2_22: C3k2::empty(c1 + c2, c2, d(2), true, 0.5, true, false),
            // Layer 23: Conv [256, 3, 2]
            conv23: YoloConv::empty(c2, c2, 3, 2, true),
            // Layer 25: C3k2 [512, True] — input cat(conv23_out, c3k2_13_out)
            c3k2_25: C3k2::empty(c2 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 26: Conv [512, 3, 2]
            conv26: YoloConv::empty(c3, c3, 3, 2, true),
            // Layer 28: C3k2 [1024, True, 0.5, True] — attn=True
            c3k2_28: C3k2::empty(c3 + c4, c4, d(1), true, 0.5, true, true),
        }
    }

    /// Run the P2 FPN+PAN. Returns `(P2, P3, P4, P5)` detection features.
    pub fn forward(
        &self,
        l2: &Tensor,
        l4: &Tensor,
        l6: &Tensor,
        l10: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // FPN top-down: l10 → up → cat(l6) → c3k2_13 → up → cat(l4) → c3k2_16 → up → cat(l2) → c3k2_19
        let up = upsample_nearest_2x(l10)?;
        let cat = Tensor::cat(&[&up, l6], 1)?;
        let l13 = self.c3k2_13.forward(&cat)?;

        let up = upsample_nearest_2x(&l13)?;
        let cat = Tensor::cat(&[&up, l4], 1)?;
        let l16 = self.c3k2_16.forward(&cat)?;

        let up = upsample_nearest_2x(&l16)?;
        let cat = Tensor::cat(&[&up, l2], 1)?;
        let l19 = self.c3k2_19.forward(&cat)?;

        // PAN bottom-up: l19 → conv20 → cat(l16) → c3k2_22 → conv23 → cat(l13) → c3k2_25 → conv26 → cat(l10) → c3k2_28
        let l20 = self.conv20.forward(&l19)?;
        let cat = Tensor::cat(&[&l20, &l16], 1)?;
        let l22 = self.c3k2_22.forward(&cat)?;

        let l23 = self.conv23.forward(&l22)?;
        let cat = Tensor::cat(&[&l23, &l13], 1)?;
        let l25 = self.c3k2_25.forward(&cat)?;

        let l26 = self.conv26.forward(&l25)?;
        let cat = Tensor::cat(&[&l26, l10], 1)?;
        let l28 = self.c3k2_28.forward(&cat)?;

        Ok((l19, l22, l25, l28))
    }
}

impl HasStateDict for YoloNeckP2 {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        let mut sd = self.c3k2_13.state_dict(&p(13));
        sd.extend(self.c3k2_16.state_dict(&p(16)));
        sd.extend(self.c3k2_19.state_dict(&p(19)));
        sd.extend(self.conv20.state_dict(&p(20)));
        sd.extend(self.c3k2_22.state_dict(&p(22)));
        sd.extend(self.conv23.state_dict(&p(23)));
        sd.extend(self.c3k2_25.state_dict(&p(25)));
        sd.extend(self.conv26.state_dict(&p(26)));
        sd.extend(self.c3k2_28.state_dict(&p(28)));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        self.c3k2_13.load_state_dict(sd, &p(13))?;
        self.c3k2_16.load_state_dict(sd, &p(16))?;
        self.c3k2_19.load_state_dict(sd, &p(19))?;
        self.conv20.load_state_dict(sd, &p(20))?;
        self.c3k2_22.load_state_dict(sd, &p(22))?;
        self.conv23.load_state_dict(sd, &p(23))?;
        self.c3k2_25.load_state_dict(sd, &p(25))?;
        self.conv26.load_state_dict(sd, &p(26))?;
        self.c3k2_28.load_state_dict(sd, &p(28))?;
        Ok(())
    }
}
