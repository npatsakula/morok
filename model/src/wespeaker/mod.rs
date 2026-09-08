//! WeSpeaker ResNet34 speaker-embedding model
//! (`pyannote/wespeaker-voxceleb-resnet34-LM`).
//!
//! 1-channel mel-spectrogram input (`[B, T=1598, F=80]`) + per-frame attention
//! weights (`[B, T_w=799]`) → 256-d L2-normalisable speaker embedding
//! (`[B, 256]`).
//!
//! Reuses [`crate::blocks::BasicBlock`] / [`crate::blocks::ResidualStage`]; the
//! WeSpeaker variant differs only in stem (3×3 stride 1, no maxpool), width
//! schedule (32→64→128→256), input modality, and head (TSTP weighted-stats
//! pooling + `Linear(5120 → 256)`).
//!
//! # Loader gotchas — pyannote checkpoint format
//!
//! Two non-obvious things the pyannote-side `WeSpeakerResNet34` wrapper does
//! to its checkpoint that we have to undo on load (see [`pickle`] and
//! `model::rename_shortcut_to_downsample`):
//!
//! - **Nested pickle.** `torch.save({"state_dict": OrderedDict(...),
//!   "pyannote.audio": ..., "pytorch-lightning_version": ...})` does *not*
//!   surface to `repugnant-pickle::torch::RepugnantTorchTensors::new_from_file`
//!   as a flat tensor dict — the latter only handles a single top-level
//!   OrderedDict/Dict and skips entries that aren't `_rebuild_tensor_v2`
//!   calls. We use `parse_ops` + `evaluate` directly and walk the `Value`
//!   tree to descend into the `state_dict` key first.
//!
//! - **`shortcut.{0,1}` naming.** pyannote's `BasicBlock` calls the
//!   downsample sub-module `shortcut` rather than torchvision's `downsample`.
//!   The svod [`crate::blocks::BasicBlock`] uses the torchvision keys, so
//!   the loader renames `.shortcut.` → `.downsample.` in every key on the
//!   way in.
//!
//! Everything else loads key-for-key: the layers read PyTorch's own parameter
//! names, so no value transform happens at load time.

mod error;
mod jit;
mod model;
pub mod pickle;
mod tstp;

pub use error::{Error, Result};
pub use jit::WeSpeakerResNet34Jit;
pub use model::{EMBED_DIM, M_CHANNELS, NUM_BLOCKS, NUM_MEL_BINS, WeSpeakerConfig, WeSpeakerResNet34};
