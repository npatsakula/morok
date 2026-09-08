//! DiariZen wavlm-large-s80-md-v2 segmentation model.
//!
//! V1 scope is the backbone + segmentation forward; no clustering or RTTM.
//! See `/Users/cognito/.claude/plans/alright-plan-this-properly-lovely-scott.md`
//! for the full plan.

mod config;
mod conformer;
mod error;
mod jit;
mod model;
mod powerset;
mod segment;

pub use config::{DiariZenConfig, chunk_plan, hop_samples, powerset_class_count, powerset_table, window_samples};
pub use conformer::{
    ConformerBlock, ConformerEncoder, ConformerMHA, ConvolutionModule, PlainMultiHeadSelfAttention,
    PositionwiseFeedForward,
};
pub use error::{Error, Result};
pub use jit::DiariZenSegmentationJit;
pub use model::{DiariZenSegmentationModel, ForwardIntermediates};
pub use powerset::powerset_to_multilabel;
pub use segment::{DiariZenSegmenter, SegmentOutput, SlidingWindow};
