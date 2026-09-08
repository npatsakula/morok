//! [`YoloNeckP6`] — YOLO v26-P6 FPN+PAN neck (layers 13–30).
//!
//! Takes four P6 backbone features `(l4, l6, l8, l12)` at strides 8, 16, 32,
//! 64 and produces four detection feature maps `(P3, P4, P5, P6)`.

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use crate::yolo::backbone::p6::p6_scaled_channels;
use crate::yolo::backbone::upsample_nearest_2x;
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::config::{YoloScale, make_depth};
use crate::yolo::error::Result;

/// P6 neck (layers 13–30). Takes four backbone features `(l4, l6, l8, l12)`
/// and produces `(P3, P4, P5, P6)` at strides 8, 16, 32, 64.
#[derive(Clone)]
pub struct YoloNeckP6 {
    pub c3k2_15: C3k2,
    pub c3k2_18: C3k2,
    pub c3k2_21: C3k2,
    pub conv22: YoloConv,
    pub c3k2_24: C3k2,
    pub conv25: YoloConv,
    pub c3k2_27: C3k2,
    pub conv28: YoloConv,
    pub c3k2_30: C3k2,
}

impl YoloNeckP6 {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [_, _, c2, c3, c5, c6] = p6_scaled_channels(scale);
        Self {
            // Layer 15: C3k2 [768, True] — input cat(upsample(c6), c5) = c6+c5
            c3k2_15: C3k2::empty(c6 + c5, c5, d(2), true, 0.5, true, false),
            // Layer 18: C3k2 [512, True] — input cat(upsample(c5), c3) = c5+c3
            c3k2_18: C3k2::empty(c5 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 21: C3k2 [256, True] — input cat(upsample(l18_out=c3), l4=c3) = c3+c3
            c3k2_21: C3k2::empty(c3 + c3, c2, d(2), true, 0.5, true, false),
            // Layer 22: Conv [256, 3, 2]
            conv22: YoloConv::empty(c2, c2, 3, 2, true),
            // Layer 24: C3k2 [512, True] — input cat(conv22_out, c3k2_18_out)
            c3k2_24: C3k2::empty(c2 + c3, c3, d(2), true, 0.5, true, false),
            // Layer 25: Conv [512, 3, 2]
            conv25: YoloConv::empty(c3, c3, 3, 2, true),
            // Layer 27: C3k2 [768, True] — input cat(conv25_out, c3k2_15_out)
            c3k2_27: C3k2::empty(c3 + c5, c5, d(2), true, 0.5, true, false),
            // Layer 28: Conv [768, 3, 2]
            conv28: YoloConv::empty(c5, c5, 3, 2, true),
            // Layer 30: C3k2 [1024, True, 0.5, True] — attn=True
            c3k2_30: C3k2::empty(c5 + c6, c6, d(1), true, 0.5, true, true),
        }
    }

    /// Run the P6 FPN+PAN. Returns `(P3, P4, P5, P6)` detection features.
    pub fn forward(
        &self,
        l4: &Tensor,
        l6: &Tensor,
        l8: &Tensor,
        l12: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        // FPN top-down: l12 → up → cat(l8) → c3k2_15 → up → cat(l6) → c3k2_18 → up → cat(l4) → c3k2_21
        let up = upsample_nearest_2x(l12)?;
        let cat = Tensor::cat(&[&up, l8], 1)?;
        let l15 = self.c3k2_15.forward(&cat)?;

        let up = upsample_nearest_2x(&l15)?;
        let cat = Tensor::cat(&[&up, l6], 1)?;
        let l18 = self.c3k2_18.forward(&cat)?;

        let up = upsample_nearest_2x(&l18)?;
        let cat = Tensor::cat(&[&up, l4], 1)?;
        let l21 = self.c3k2_21.forward(&cat)?;

        // PAN bottom-up: l21 → conv22 → cat(l18) → c3k2_24 → conv25 → cat(l15) → c3k2_27 → conv28 → cat(l12) → c3k2_30
        let l22 = self.conv22.forward(&l21)?;
        let cat = Tensor::cat(&[&l22, &l18], 1)?;
        let l24 = self.c3k2_24.forward(&cat)?;

        let l25 = self.conv25.forward(&l24)?;
        let cat = Tensor::cat(&[&l25, &l15], 1)?;
        let l27 = self.c3k2_27.forward(&cat)?;

        let l28 = self.conv28.forward(&l27)?;
        let cat = Tensor::cat(&[&l28, l12], 1)?;
        let l30 = self.c3k2_30.forward(&cat)?;

        Ok((l21, l24, l27, l30))
    }
}

impl HasStateDict for YoloNeckP6 {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        let mut sd = self.c3k2_15.state_dict(&p(15));
        sd.extend(self.c3k2_18.state_dict(&p(18)));
        sd.extend(self.c3k2_21.state_dict(&p(21)));
        sd.extend(self.conv22.state_dict(&p(22)));
        sd.extend(self.c3k2_24.state_dict(&p(24)));
        sd.extend(self.conv25.state_dict(&p(25)));
        sd.extend(self.c3k2_27.state_dict(&p(27)));
        sd.extend(self.conv28.state_dict(&p(28)));
        sd.extend(self.c3k2_30.state_dict(&p(30)));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        self.c3k2_15.load_state_dict(sd, &p(15))?;
        self.c3k2_18.load_state_dict(sd, &p(18))?;
        self.c3k2_21.load_state_dict(sd, &p(21))?;
        self.conv22.load_state_dict(sd, &p(22))?;
        self.c3k2_24.load_state_dict(sd, &p(24))?;
        self.conv25.load_state_dict(sd, &p(25))?;
        self.c3k2_27.load_state_dict(sd, &p(27))?;
        self.conv28.load_state_dict(sd, &p(28))?;
        self.c3k2_30.load_state_dict(sd, &p(30))?;
        Ok(())
    }
}
