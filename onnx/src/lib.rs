//! ONNX model frontend for Morok.
//!
//! This crate provides ONNX model parsing and conversion to Morok Tensors.
//!
//! # Example
//!
//! ```ignore
//! use morok_onnx::{OnnxImporter, OnnxGraph};
//!
//! // Two-phase import for runtime inputs
//! let importer = OnnxImporter::new();
//! let graph = importer.prepare(model)?;
//! let outputs = importer.execute(&graph, inputs)?;
//!
//! // Or convenience method for all-initializer models
//! let mut importer = OnnxImporter::new();
//! let outputs = importer.import_path("model.onnx")?;
//! ```

pub mod error;
pub mod importer;
pub mod parser;
pub mod registry;

pub use error::{Error, Result};
pub use importer::{DimValue, InputSpec, OnnxGraph, OnnxImporter};
