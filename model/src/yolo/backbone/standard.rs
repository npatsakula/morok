//! Shared YOLO v26 backbone (layers 0–10) and classification backbone
//! (layers 0–9, no SPPF).
//!
//! The backbone produces three skip-connection feature maps (P3/8, P4/16,
//! P5/32) consumed by detection-family and depth neck/head. The classify
//! variant stops at the C2PSA layer (layer 9) and omits SPPF.

use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use crate::yolo::blocks::attention::C2PSA;
use crate::yolo::blocks::conv::YoloConv;
use crate::yolo::blocks::csp::C3k2;
use crate::yolo::blocks::sppf::Sppf;
use crate::yolo::config::{YoloScale, make_depth, scale_channels};
use crate::yolo::error::Result;

/// Backbone channels after scaling (used to construct neck/head).
pub fn scaled_channels(scale: YoloScale) -> [usize; 5] {
    let sc = |yaml_c| scale_channels(yaml_c, scale);
    [sc(64), sc(128), sc(256), sc(512), sc(1024)]
}

/// Full YOLO v26 backbone (layers 0–10): Conv→Conv→C3k2→Conv→C3k2→Conv→
/// C3k2→Conv→C3k2→SPPF→C2PSA.
///
/// Forward returns the three skip-connection outputs: `(l4, l6, l10)`.
#[derive(Clone)]
pub struct YoloBackbone {
    pub conv0: YoloConv,
    pub conv1: YoloConv,
    pub c3k2_2: C3k2,
    pub conv3: YoloConv,
    pub c3k2_4: C3k2,
    pub conv5: YoloConv,
    pub c3k2_6: C3k2,
    pub conv7: YoloConv,
    pub c3k2_8: C3k2,
    pub sppf9: Sppf,
    pub c2psa10: C2PSA,
}

impl YoloBackbone {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [c0, c1, c2, c3, c4] = scaled_channels(scale);
        Self {
            conv0: YoloConv::empty(3, c0, 3, 2, true),
            conv1: YoloConv::empty(c0, c1, 3, 2, true),
            c3k2_2: C3k2::empty(c1, c2, d(2), true, 0.25, false, false),
            conv3: YoloConv::empty(c2, c2, 3, 2, true),
            c3k2_4: C3k2::empty(c2, c3, d(2), true, 0.25, false, false),
            conv5: YoloConv::empty(c3, c3, 3, 2, true),
            c3k2_6: C3k2::empty(c3, c3, d(2), true, 0.5, true, false),
            conv7: YoloConv::empty(c3, c4, 3, 2, true),
            c3k2_8: C3k2::empty(c4, c4, d(2), true, 0.5, true, false),
            sppf9: Sppf::empty(c4, c4, 5, 3, true),
            c2psa10: C2PSA::empty(c4, c4, d(2), 0.5),
        }
    }

    /// Run backbone layers 0–10, returning skip features `(l4, l6, l10)`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let x = self.conv0.forward(x)?;
        let x = self.conv1.forward(&x)?;
        let x = self.c3k2_2.forward(&x)?;
        let x = self.conv3.forward(&x)?;
        let l4 = self.c3k2_4.forward(&x)?;
        let x = self.conv5.forward(&l4)?;
        let l6 = self.c3k2_6.forward(&x)?;
        let x = self.conv7.forward(&l6)?;
        let x = self.c3k2_8.forward(&x)?;
        let x = self.sppf9.forward(&x)?;
        let l10 = self.c2psa10.forward(&x)?;
        Ok((l4, l6, l10))
    }

    /// Run backbone layers 0–10, returning four skip features `(l2, l4, l6, l10)`.
    /// Used by the P2 variant neck which taps the P2/4 feature at layer 2.
    pub fn forward_with_p2(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let x = self.conv0.forward(x)?;
        let x = self.conv1.forward(&x)?;
        let l2 = self.c3k2_2.forward(&x)?;
        let x = self.conv3.forward(&l2)?;
        let l4 = self.c3k2_4.forward(&x)?;
        let x = self.conv5.forward(&l4)?;
        let l6 = self.c3k2_6.forward(&x)?;
        let x = self.conv7.forward(&l6)?;
        let x = self.c3k2_8.forward(&x)?;
        let x = self.sppf9.forward(&x)?;
        let l10 = self.c2psa10.forward(&x)?;
        Ok((l2, l4, l6, l10))
    }
}

impl HasStateDict for YoloBackbone {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        let mut sd = self.conv0.state_dict(&p(0));
        sd.extend(self.conv1.state_dict(&p(1)));
        sd.extend(self.c3k2_2.state_dict(&p(2)));
        sd.extend(self.conv3.state_dict(&p(3)));
        sd.extend(self.c3k2_4.state_dict(&p(4)));
        sd.extend(self.conv5.state_dict(&p(5)));
        sd.extend(self.c3k2_6.state_dict(&p(6)));
        sd.extend(self.conv7.state_dict(&p(7)));
        sd.extend(self.c3k2_8.state_dict(&p(8)));
        sd.extend(self.sppf9.state_dict(&p(9)));
        sd.extend(self.c2psa10.state_dict(&p(10)));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        self.conv0.load_state_dict(sd, &p(0))?;
        self.conv1.load_state_dict(sd, &p(1))?;
        self.c3k2_2.load_state_dict(sd, &p(2))?;
        self.conv3.load_state_dict(sd, &p(3))?;
        self.c3k2_4.load_state_dict(sd, &p(4))?;
        self.conv5.load_state_dict(sd, &p(5))?;
        self.c3k2_6.load_state_dict(sd, &p(6))?;
        self.conv7.load_state_dict(sd, &p(7))?;
        self.c3k2_8.load_state_dict(sd, &p(8))?;
        self.sppf9.load_state_dict(sd, &p(9))?;
        self.c2psa10.load_state_dict(sd, &p(10))?;
        Ok(())
    }
}

/// Classification backbone (layers 0–9): same as [`YoloBackbone`] but without
/// SPPF (layer 9 is C2PSA, not SPPF). The classify YAML puts C2PSA at index 9.
#[derive(Clone)]
pub struct YoloBackboneCls {
    pub conv0: YoloConv,
    pub conv1: YoloConv,
    pub c3k2_2: C3k2,
    pub conv3: YoloConv,
    pub c3k2_4: C3k2,
    pub conv5: YoloConv,
    pub c3k2_6: C3k2,
    pub conv7: YoloConv,
    pub c3k2_8: C3k2,
    pub c2psa9: C2PSA,
}

impl YoloBackboneCls {
    pub fn empty(scale: YoloScale) -> Self {
        let d = |yaml_n| make_depth(yaml_n, scale);
        let [c0, c1, c2, c3, c4] = scaled_channels(scale);
        Self {
            conv0: YoloConv::empty(3, c0, 3, 2, true),
            conv1: YoloConv::empty(c0, c1, 3, 2, true),
            c3k2_2: C3k2::empty(c1, c2, d(2), true, 0.25, false, false),
            conv3: YoloConv::empty(c2, c2, 3, 2, true),
            c3k2_4: C3k2::empty(c2, c3, d(2), true, 0.25, false, false),
            conv5: YoloConv::empty(c3, c3, 3, 2, true),
            c3k2_6: C3k2::empty(c3, c3, d(2), true, 0.5, true, false),
            conv7: YoloConv::empty(c3, c4, 3, 2, true),
            c3k2_8: C3k2::empty(c4, c4, d(2), true, 0.5, true, false),
            c2psa9: C2PSA::empty(c4, c4, d(2), 0.5),
        }
    }

    /// Run layers 0–9, returning the final feature map.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv0.forward(x)?;
        let x = self.conv1.forward(&x)?;
        let x = self.c3k2_2.forward(&x)?;
        let x = self.conv3.forward(&x)?;
        let x = self.c3k2_4.forward(&x)?;
        let x = self.conv5.forward(&x)?;
        let x = self.c3k2_6.forward(&x)?;
        let x = self.conv7.forward(&x)?;
        let x = self.c3k2_8.forward(&x)?;
        self.c2psa9.forward(&x)
    }
}

impl HasStateDict for YoloBackboneCls {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        let mut sd = self.conv0.state_dict(&p(0));
        sd.extend(self.conv1.state_dict(&p(1)));
        sd.extend(self.c3k2_2.state_dict(&p(2)));
        sd.extend(self.conv3.state_dict(&p(3)));
        sd.extend(self.c3k2_4.state_dict(&p(4)));
        sd.extend(self.conv5.state_dict(&p(5)));
        sd.extend(self.c3k2_6.state_dict(&p(6)));
        sd.extend(self.conv7.state_dict(&p(7)));
        sd.extend(self.c3k2_8.state_dict(&p(8)));
        sd.extend(self.c2psa9.state_dict(&p(9)));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        let p = |i: usize| prefixed(prefix, &i.to_string());
        self.conv0.load_state_dict(sd, &p(0))?;
        self.conv1.load_state_dict(sd, &p(1))?;
        self.c3k2_2.load_state_dict(sd, &p(2))?;
        self.conv3.load_state_dict(sd, &p(3))?;
        self.c3k2_4.load_state_dict(sd, &p(4))?;
        self.conv5.load_state_dict(sd, &p(5))?;
        self.c3k2_6.load_state_dict(sd, &p(6))?;
        self.conv7.load_state_dict(sd, &p(7))?;
        self.c3k2_8.load_state_dict(sd, &p(8))?;
        self.c2psa9.load_state_dict(sd, &p(9))?;
        Ok(())
    }
}

/// Nearest-neighbour 2× upsample via gather (tinygrad interpolate approach).
pub fn upsample_nearest_2x(x: &Tensor) -> Result<Tensor> {
    use svod_ir::SInt;
    let b = x.dim(0)?;
    let c = x.dim(1)?;
    let h = x.dim_const(2)?;
    let w = x.dim_const(3)?;

    let h_idx: Vec<i64> = (0..h as i64).flat_map(|v| [v, v]).collect();
    let w_idx: Vec<i64> = (0..w as i64).flat_map(|v| [v, v]).collect();

    let h_index = Tensor::from_slice(&h_idx)
        .try_reshape([SInt::from(1usize), SInt::from(1usize), SInt::from(h * 2), SInt::from(1usize)])?
        .try_expand([b.clone(), c.clone(), SInt::from(h * 2), SInt::from(w)])?;
    let x = x.gather(2, &h_index)?;

    let shape = x.shape()?;
    let b = shape[0].clone();
    let c = shape[1].clone();
    let w_index = Tensor::from_slice(&w_idx)
        .try_reshape([SInt::from(1usize), SInt::from(1usize), SInt::from(1usize), SInt::from(w * 2)])?
        .try_expand([b, c, SInt::from(h * 2), SInt::from(w * 2)])?;
    Ok(x.gather(3, &w_index)?)
}
