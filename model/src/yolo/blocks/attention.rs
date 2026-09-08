use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::conv::YoloConv;
use crate::yolo::error::Result;

/// Multi-head attention with a depthwise-conv positional encoding.
///
/// Computes full N×N attention but with a reduced key dimension
/// (`key_dim = head_dim * attn_ratio`) to cut QK^T cost.
/// All projections are 1×1 convs (no activation).
///
/// State-dict keys: `qkv.{conv,bn}.*`, `proj.{conv,bn}.*`, `pe.{conv,bn}.*`.
#[derive(Clone)]
pub struct Attention {
    pub qkv: YoloConv,
    pub proj: YoloConv,
    pub pe: YoloConv,
    pub num_heads: usize,
    pub key_dim: usize,
    pub head_dim: usize,
    pub scale: f32,
}

impl Attention {
    pub fn empty(dim: usize, num_heads: usize, attn_ratio: f64) -> Self {
        let head_dim = dim / num_heads;
        let key_dim = (head_dim as f64 * attn_ratio) as usize;
        let scale = (key_dim as f64).powf(-0.5) as f32;
        let nh_kd = key_dim * num_heads;
        let qkv_out = dim + nh_kd * 2;

        Self {
            qkv: YoloConv::empty(dim, qkv_out, 1, 1, false),
            proj: YoloConv::empty(dim, dim, 1, 1, false),
            pe: YoloConv::empty_dw(dim, dim, 3, 1, false),
            num_heads,
            key_dim,
            head_dim,
            scale,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.dim(0)?;
        let nh = self.num_heads;
        let kd = self.key_dim;
        let hd = self.head_dim;
        let kkd = kd * 2 + hd;

        let h = x.dim_const(2)?;
        let w = x.dim_const(3)?;
        let hw = h * w;

        // QKV 1×1 conv: [B, C, H, W] → [B, nh*kkd, H, W]
        let qkv = self.qkv.forward(x)?;

        // Flatten spatial then separate heads:
        // [B, nh*kkd, H, W] → [B, nh*kkd, H*W] → [B, nh, kkd, N]
        let qkv = qkv.try_reshape([b.clone(), SInt::from(nh * kkd), SInt::from(hw)])?.try_reshape([
            b.clone(),
            SInt::from(nh),
            SInt::from(kkd),
            SInt::from(hw),
        ])?;

        // Split into q [B,nh,kd,N], k [B,nh,kd,N], v [B,nh,hd,N]
        let parts = qkv.split(&[kd, kd, hd], 2)?;
        let q = &parts[0];
        let k = &parts[1];
        let v = &parts[2];

        // Scale q
        let scale_t = Tensor::from_slice([self.scale]);
        let q = q.try_mul(&scale_t)?;

        // attn = q^T @ k : [B, nh, N, kd] @ [B, nh, kd, N] = [B, nh, N, N]
        let q_t = q.try_transpose(-2, -1)?;
        let attn = q_t.matmul(k)?;
        let attn = attn.softmax(-1)?;

        // out = v @ attn^T : [B, nh, hd, N] @ [B, nh, N, N] = [B, nh, hd, N]
        let attn_t = attn.try_transpose(-2, -1)?;
        let out = v.matmul(&attn_t)?;

        // Reshape back to spatial: [B, nh, hd, N] → [B, C, H, W]
        let c = nh * hd;
        let out = out.try_reshape([b.clone(), SInt::from(c), SInt::from(hw)])?.try_reshape([
            b.clone(),
            SInt::from(c),
            SInt::from(h),
            SInt::from(w),
        ])?;

        // Positional encoding: depthwise conv on v reshaped to spatial
        let v_spatial = v.try_reshape([b.clone(), SInt::from(c), SInt::from(hw)])?.try_reshape([
            b.clone(),
            SInt::from(c),
            SInt::from(h),
            SInt::from(w),
        ])?;
        let pe = self.pe.forward(&v_spatial)?;

        let out = out.try_add(&pe)?;

        self.proj.forward(&out)
    }
}

impl HasStateDict for Attention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.qkv.state_dict(&prefixed(prefix, "qkv"));
        sd.extend(self.proj.state_dict(&prefixed(prefix, "proj")));
        sd.extend(self.pe.state_dict(&prefixed(prefix, "pe")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.qkv.load_state_dict(sd, &prefixed(prefix, "qkv"))?;
        self.proj.load_state_dict(sd, &prefixed(prefix, "proj"))?;
        self.pe.load_state_dict(sd, &prefixed(prefix, "pe"))?;
        Ok(())
    }
}

/// Position-Sensitive Attention block: attention + FFN, both with residual.
///
/// State-dict keys: `attn.*`, `ffn.0.{conv,bn}.*`, `ffn.1.{conv,bn}.*`.
#[derive(Clone)]
pub struct PSABlock {
    pub attn: Attention,
    pub ffn0: YoloConv,
    pub ffn1: YoloConv,
}

impl PSABlock {
    pub fn empty(c: usize, num_heads: usize) -> Self {
        Self {
            attn: Attention::empty(c, num_heads, 0.5),
            ffn0: YoloConv::empty(c, c * 2, 1, 1, true),
            ffn1: YoloConv::empty(c * 2, c, 1, 1, false),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn_out = self.attn.forward(x)?;
        let x = x.try_add(&attn_out)?;
        let ffn_out = self.ffn1.forward(&self.ffn0.forward(&x)?)?;
        Ok(x.try_add(&ffn_out)?)
    }
}

impl HasStateDict for PSABlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.attn.state_dict(&prefixed(prefix, "attn"));
        sd.extend(self.ffn0.state_dict(&prefixed(prefix, "ffn.0")));
        sd.extend(self.ffn1.state_dict(&prefixed(prefix, "ffn.1")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.attn.load_state_dict(sd, &prefixed(prefix, "attn"))?;
        self.ffn0.load_state_dict(sd, &prefixed(prefix, "ffn.0"))?;
        self.ffn1.load_state_dict(sd, &prefixed(prefix, "ffn.1"))?;
        Ok(())
    }
}

/// CSP with partial self-attention: split channels, run PSABlocks on half,
/// concat, project back.
///
/// State-dict keys: `cv1.{conv,bn}.*`, `cv2.{conv,bn}.*`, `m.{i}.*`.
#[derive(Clone)]
pub struct C2PSA {
    pub cv1: YoloConv,
    pub cv2: YoloConv,
    pub m: Vec<PSABlock>,
    pub c_hidden: usize,
}

impl C2PSA {
    pub fn empty(in_ch: usize, out_ch: usize, n: usize, e: f64) -> Self {
        assert_eq!(in_ch, out_ch, "C2PSA requires c1 == c2");
        let c_hidden = (out_ch as f64 * e) as usize;
        let num_heads = (c_hidden / 64).max(1);
        Self {
            cv1: YoloConv::empty(in_ch, 2 * c_hidden, 1, 1, true),
            cv2: YoloConv::empty(2 * c_hidden, out_ch, 1, 1, true),
            m: (0..n).map(|_| PSABlock::empty(c_hidden, num_heads)).collect(),
            c_hidden,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let parts = y.split(&[self.c_hidden, self.c_hidden], 1)?;
        let a = &parts[0];
        let mut b = parts[1].clone();
        for blk in &self.m {
            b = blk.forward(&b)?;
        }
        let cat = Tensor::cat(&[a, &b], 1)?;
        self.cv2.forward(&cat)
    }
}

impl HasStateDict for C2PSA {
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
