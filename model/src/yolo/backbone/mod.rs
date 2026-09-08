//! YOLO v26 backbone variants.
//!
//! - [`YoloBackbone`] / [`YoloBackboneCls`] — standard P3-P5 (layers 0-10)
//! - [`YoloBackboneP6`] — deeper P3-P6 (layers 0-12)

pub(crate) mod p6;
pub(crate) mod standard;

pub use p6::{YoloBackboneP6, p6_scaled_channels};
pub use standard::{YoloBackbone, YoloBackboneCls, scaled_channels};
