//! Error types for ONNX parsing and import.

use snafu::Snafu;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("IO error: {source}"))]
    Io { source: std::io::Error },

    #[snafu(display("Protobuf decode error: {source}"))]
    ProtobufDecode { source: prost::DecodeError },

    #[snafu(display("Unsupported ONNX operator: {op} (domain: {domain})"))]
    UnsupportedOp { op: String, domain: String },

    #[snafu(display("Unsupported ONNX data type: {dtype}"))]
    UnsupportedDType { dtype: i32 },

    #[snafu(display("Missing input '{input}' for node '{node}'"))]
    MissingInput { node: String, input: String },

    #[snafu(display("Shape mismatch for '{context}': expected {expected}, got {actual}"))]
    ShapeMismatch { context: String, expected: String, actual: String },

    #[snafu(display("IR construction error: {details}"))]
    IrConstruction { details: String },

    #[snafu(display("Empty model - no graph found"))]
    EmptyModel,

    #[snafu(display("Tensor operation error: {source}"))]
    Tensor { source: morok_tensor::error::Error },
}

// Implement From for tensor errors
impl From<morok_tensor::error::Error> for Error {
    fn from(source: morok_tensor::error::Error) -> Self {
        Error::Tensor { source }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
