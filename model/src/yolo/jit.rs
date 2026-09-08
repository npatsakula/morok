//! JIT wrappers for YOLO v26 models.
//!
//! Every wrapper follows the same shape: one `images` input, a `batch_var b`
//! that shrinks it to the live batch on dim 0, and one `predictions` output.

use svod_macros::jit_wrapper;

use super::classify::Yolo26Classify;
use super::depth::Yolo26Depth;
use super::detect::Yolo26Detect;
use super::obb::Yolo26Obb;
use super::pose::Yolo26Pose;
use super::semseg::Yolo26SemSeg;

macro_rules! detect_like_jit {
    ($($jit:ident($model:ty)),* $(,)?) => {
        $(jit_wrapper! {
            $jit($model) {
                inputs { images: Tensor }
                batch_var b: (1, model.config.max_batch_size),
                outputs { predictions }
                build(images) { model.forward(images) }
            }
        })*
    };
}

detect_like_jit! {
    Yolo26DetectJit(Yolo26Detect),
    Yolo26ClassifyJit(Yolo26Classify),
    Yolo26PoseJit(Yolo26Pose),
    Yolo26ObbJit(Yolo26Obb),
    Yolo26DepthJit(Yolo26Depth),
    Yolo26SemSegJit(Yolo26SemSeg),
}
