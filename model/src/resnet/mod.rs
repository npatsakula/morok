//! ResNet image classification / feature backbone.
//!
//! Supports the canonical depth schedules — 18, 34, 50, 101, 152 — via a single
//! factory parameterised by [`ResNetDepth`]. Block type follows depth: shallow
//! variants use [`BasicBlock`](crate::blocks::BasicBlock), deeper ones
//! use [`Bottleneck`](crate::blocks::Bottleneck) with 4× channel
//! expansion. The v1.5 stride placement (3×3 conv of the bottleneck owns the
//! downsample stride) is used everywhere — same as `timm`, `torchvision`, and
//! NVIDIA DeepLearningExamples.
//!
//! ## Recipe
//!
//! ```no_run
//! use svod_model::resnet::{OutputMode, ResNet, ResNetDepth, ResNetJit};
//! use svod_model::jit::InputSpec;
//!
//! let model = ResNet::from_hub("timm/resnet34.a1_in1k", ResNetDepth::R34,
//!     OutputMode::Classification { num_classes: 1000 })?;
//!
//! let max_batch = 4;
//! let mut jit = ResNetJit::new(model);
//! jit.prepare(InputSpec::f32(&[max_batch, 3, 224, 224]))?;
//!
//! // copy NCHW fp32 image batch into `jit.images_mut()?`, then:
//! jit.execute_with_vars(&[("b", 1)])?;
//! let _logits = jit.output()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Only the batch dimension is symbolic; the H/W of the image grid are baked
//! into the plan by [`prepare`](ResNetJit#method.prepare). Re-`prepare` for a
//! different resolution. See [the JIT graphs page] for the wrapper contract.
//!
//! [the JIT graphs page]: https://svod.dev/docs/architecture/jit-graphs

mod config;
mod error;
mod jit;
mod model;

pub use crate::blocks::BlockKind;
pub use config::{OutputMode, ResNetConfig, ResNetDepth};
pub use error::{Error, Result};
pub use jit::ResNetJit;
pub use model::ResNet;
