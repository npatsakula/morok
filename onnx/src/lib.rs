//! ONNX model frontend for Morok.
//!
//! This crate provides ONNX model parsing and conversion to Morok Tensors.
//!
//! # Example
//!
//! ```ignore
//! use morok_onnx::{OnnxImporter, OnnxGraph};
//!
//! // Trace: build lazy graph with allocated input buffers
//! let importer = OnnxImporter::new();
//! let graph = importer.prepare(model)?;
//! let (inputs, outputs) = importer.trace(&graph)?;
//!
//! // Prepare execution plan, copyin data, execute repeatedly
//! let plan = outputs["output"].prepare()?;
//! plan.execute(&mut executor)?;
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

#[cfg(test)]
mod test;
