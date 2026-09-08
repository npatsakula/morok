use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::attention::PSABlock;
use super::bottleneck::YoloBottleneck;
use super::conv::YoloConv;
use crate::yolo::error::Result;

// ---------------------------------------------------------------------------
// C2f — the C2f/C3k2 parent: 1×1 conv → chunk(2) → chain → cat → 1×1 conv
// ---------------------------------------------------------------------------

/// Faster CSP Bottleneck: 1×1 conv splits into two halves; the second half
/// passes through a chain of bottlenecks; all outputs are concatenated and
/// merged by a final 1×1 conv.
///
/// State-dict keys: `cv1.*`, `cv2.*`, `m.{i}.*`.
#[derive(Clone)]
pub struct C2f {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub m: Vec<YoloBottleneck>,
    pub c_hidden: usize,
}

impl C2f {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, shortcut: bool, e: f64) -> Self {
        let c_hidden = (out_ch as f64 * e) as usize;
        Self {
            cv1: YoloConv::empty(in_ch, 2 * c_hidden, 1, 1, true),
            cv2: YoloConv::empty((2 + n) * c_hidden, out_ch, 1, 1, true),
            m: (0..n).map(|_| YoloBottleneck::empty(c_hidden, c_hidden, shortcut)).collect(),
            c_hidden,
        }
    }

    /// C2f forward shared with C3k2: `cv1 → chunk(2) → chain(m) → cat → cv2`.
    pub fn forward_chain(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let chunks = y.chunk(2, 1)?;
        let mut parts = vec![chunks[0].clone(), chunks[1].clone()];
        let mut current = chunks[1].clone();
        for blk in &self.m {
            current = blk.forward(&current)?;
            parts.push(current.clone());
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let cat = Tensor::cat(&refs, 1)?;
        self.cv2.forward(&cat)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_chain(x)
    }
}

impl HasStateDict for C2f {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.cv1.state_dict(&prefixed(prefix, "cv1"));
        sd.extend(self.cv2.state_dict(&prefixed(prefix, "cv2")));
        for (i, blk) in self.m.iter().enumerate() {
            sd.extend(blk.state_dict(&prefixed(prefix, &format!("m.{i}"))));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.cv1.load_state_dict(sd, &prefixed(prefix, "cv1"))?;
        self.cv2.load_state_dict(sd, &prefixed(prefix, "cv2"))?;
        for (i, blk) in self.m.iter_mut().enumerate() {
            blk.load_state_dict(sd, &prefixed(prefix, &format!("m.{i}")))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// C3k — CSP Bottleneck with 3 convs and configurable kernel size
// ---------------------------------------------------------------------------

/// CSP with three 1×1 convs and n inner bottlenecks with kernel `k`.
///
/// State-dict keys: `cv1.*`, `cv2.*`, `cv3.*`, `m.{i}.*`.
#[derive(Clone)]
pub struct C3k {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub cv3: YoloConv,
    pub m: Vec<YoloBottleneck>,
}

impl C3k {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, shortcut: bool, e: f64, k: usize) -> Self {
        let c_ = (out_ch as f64 * e) as usize;
        Self {
            cv1: YoloConv::empty(in_ch, c_, 1, 1, true),
            cv2: YoloConv::empty(in_ch, c_, 1, 1, true),
            cv3: YoloConv::empty(2 * c_, out_ch, 1, 1, true),
            m: (0..n).map(|_| YoloBottleneck::empty_full(c_, c_, shortcut, k, k, 1.0)).collect(),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let left = self.cv1.forward(x)?;
        let mut current = left;
        for blk in &self.m {
            current = blk.forward(&current)?;
        }
        let right = self.cv2.forward(x)?;
        let cat = Tensor::cat(&[&current, &right], 1)?;
        self.cv3.forward(&cat)
    }
}

impl HasStateDict for C3k {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.cv1.state_dict(&prefixed(prefix, "cv1"));
        sd.extend(self.cv2.state_dict(&prefixed(prefix, "cv2")));
        sd.extend(self.cv3.state_dict(&prefixed(prefix, "cv3")));
        for (i, blk) in self.m.iter().enumerate() {
            sd.extend(blk.state_dict(&prefixed(prefix, &format!("m.{i}"))));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.cv1.load_state_dict(sd, &prefixed(prefix, "cv1"))?;
        self.cv2.load_state_dict(sd, &prefixed(prefix, "cv2"))?;
        self.cv3.load_state_dict(sd, &prefixed(prefix, "cv3"))?;
        for (i, blk) in self.m.iter_mut().enumerate() {
            blk.load_state_dict(sd, &prefixed(prefix, &format!("m.{i}")))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// C3k2 — C2f parent with pluggable inner blocks
// ---------------------------------------------------------------------------

/// Inner block for C3k2: plain Bottleneck, C3k, or Bottleneck+PSABlock.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum C3k2Inner {
    Bottleneck(YoloBottleneck),
    C3k(C3k),
    Attn(AttnBottleneck),
}

/// `Sequential(Bottleneck, PSABlock)` — used when `attn=True`.
#[derive(Clone)]
pub struct AttnBottleneck {
    pub bottleneck: YoloBottleneck,
    pub psa: PSABlock,
}

impl C3k2Inner {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            C3k2Inner::Bottleneck(b) => b.forward(x),
            C3k2Inner::C3k(c) => c.forward(x),
            C3k2Inner::Attn(a) => {
                let b = a.bottleneck.forward(x)?;
                a.psa.forward(&b)
            }
        }
    }
}

impl HasStateDict for C3k2Inner {
    fn state_dict(&self, prefix: &str) -> StateDict {
        match self {
            C3k2Inner::Bottleneck(b) => b.state_dict(prefix),
            C3k2Inner::C3k(c) => c.state_dict(prefix),
            C3k2Inner::Attn(a) => {
                let mut sd = a.bottleneck.state_dict(&prefixed(prefix, "0"));
                sd.extend(a.psa.state_dict(&prefixed(prefix, "1")));
                sd
            }
        }
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        match self {
            C3k2Inner::Bottleneck(b) => b.load_state_dict(sd, prefix),
            C3k2Inner::C3k(c) => c.load_state_dict(sd, prefix),
            C3k2Inner::Attn(a) => {
                a.bottleneck.load_state_dict(sd, &prefixed(prefix, "0"))?;
                a.psa.load_state_dict(sd, &prefixed(prefix, "1"))?;
                Ok(())
            }
        }
    }
}

/// C3k2: C2f with pluggable inner blocks (Bottleneck, C3k, or attn).
///
/// State-dict keys: `cv1.*`, `cv2.*`, `m.{i}.*`.
#[derive(Clone)]
pub struct C3k2 {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub m: Vec<C3k2Inner>,
    pub c_hidden: usize,
}

impl C3k2 {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, shortcut: bool, e: f64, c3k: bool, attn: bool) -> Self {
        let c_hidden = (out_ch as f64 * e) as usize;
        let num_heads = (c_hidden / 64).max(1);
        let m: Vec<C3k2Inner> = (0..n)
            .map(|_| {
                if attn {
                    C3k2Inner::Attn(AttnBottleneck {
                        bottleneck: YoloBottleneck::empty(c_hidden, c_hidden, shortcut),
                        psa: PSABlock::empty(c_hidden, num_heads),
                    })
                } else if c3k {
                    C3k2Inner::C3k(C3k::empty(c_hidden, c_hidden, 2, shortcut, 0.5, 3))
                } else {
                    C3k2Inner::Bottleneck(YoloBottleneck::empty(c_hidden, c_hidden, shortcut))
                }
            })
            .collect();
        Self {
            cv1: YoloConv::empty(in_ch, 2 * c_hidden, 1, 1, true),
            cv2: YoloConv::empty((2 + n) * c_hidden, out_ch, 1, 1, true),
            m,
            c_hidden,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let chunks = y.chunk(2, 1)?;
        let mut parts = vec![chunks[0].clone(), chunks[1].clone()];
        let mut current = chunks[1].clone();
        for blk in &self.m {
            current = blk.forward(&current)?;
            parts.push(current.clone());
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let cat = Tensor::cat(&refs, 1)?;
        self.cv2.forward(&cat)
    }
}

impl HasStateDict for C3k2 {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.cv1.state_dict(&prefixed(prefix, "cv1"));
        sd.extend(self.cv2.state_dict(&prefixed(prefix, "cv2")));
        for (i, blk) in self.m.iter().enumerate() {
            sd.extend(blk.state_dict(&prefixed(prefix, &format!("m.{i}"))));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.cv1.load_state_dict(sd, &prefixed(prefix, "cv1"))?;
        self.cv2.load_state_dict(sd, &prefixed(prefix, "cv2"))?;
        for (i, blk) in self.m.iter_mut().enumerate() {
            blk.load_state_dict(sd, &prefixed(prefix, &format!("m.{i}")))?;
        }
        Ok(())
    }
}
