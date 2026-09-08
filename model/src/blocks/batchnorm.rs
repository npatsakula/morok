use svod_dtype::DType;
use svod_tensor::nn::BatchNorm2d;

/// Default PyTorch BatchNorm epsilon. The timm, Ultralytics and WeSpeaker
/// checkpoints we target do not override it.
pub const BN_EPS: f64 = 1e-5;

/// Identity-initialized inference batch norm over the channel axis, keyed with
/// PyTorch's `weight` / `bias` / `running_mean` / `running_var` names.
pub fn batchnorm2d(channels: usize) -> BatchNorm2d {
    BatchNorm2d::with_dims(channels, BN_EPS, DType::Float32)
}
