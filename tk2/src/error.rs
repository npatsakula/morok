//! Library error handling (snafu `.context()` + `*Snafu`), per the crate
//! conventions. Backend errors are boxed so the enum stays small.

use snafu::Snafu;

/// Result alias for the lowering / verify / launch surface.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors raised while verifying, compiling, or dispatching a tile-IR program.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    /// Linearizing the lowered SINK failed (CFG / structural error).
    #[snafu(display("linearize {name:?}: {source}"))]
    Linearize {
        name: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// The linearized program was not the expected `PROGRAM(LINEAR)` shape.
    #[snafu(display("linearized program for {name:?} has no LINEAR stage"))]
    NoLinearStage { name: String },

    /// `type_verify` rejected the lowered program (a spec-validity violation).
    #[snafu(display("spec verification of {name:?} failed: {source}"))]
    Verify {
        name: String,
        #[snafu(source(from(svod_schedule::spec::SpecError, Box::new)))]
        source: Box<svod_schedule::spec::SpecError>,
    },

    /// Resolving the concrete `Device` (renderer/compiler/runtime) failed.
    #[snafu(display("resolve device {spec}: {source}"))]
    DeviceResolve {
        spec: String,
        #[snafu(source(from(svod_runtime::Error, Box::new)))]
        source: Box<svod_runtime::Error>,
    },

    /// Rendering / compiling the program through the codegen pipeline failed.
    #[snafu(display("compile {name:?}: {source}"))]
    Compile {
        name: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// Headless IR rendering (no device) failed — the codegen `Renderer::render` error as text.
    #[snafu(display("render {name:?}: {reason}"))]
    Render { name: String, reason: String },

    /// A buffer required by the compiled ABI was not supplied.
    #[snafu(display("buffer slot {slot} not supplied (of {supplied})"))]
    BufferMissing { slot: usize, supplied: usize },

    /// Allocating / preparing a bound buffer failed.
    #[snafu(display("prepare buffer slot {slot}: {source}"))]
    Buffer {
        slot: usize,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// Runtime scalar binding is missing, duplicated, unknown, or outside its declared bounds.
    #[snafu(display("runtime scalar {name:?}: {reason}"))]
    RuntimeScalar { name: String, reason: String },

    /// The kernel dispatch itself failed.
    #[snafu(display("dispatch {name:?}: {source}"))]
    Dispatch {
        name: String,
        #[snafu(source(from(svod_device::Error, Box::new)))]
        source: Box<svod_device::Error>,
    },

    /// Wrapping the tk2 kernel SINK as a `custom_kernel` (`Op::Call`) graph node —
    /// the opaque, schedulable/profilable form ([`crate::graph`]) — failed.
    #[snafu(display("graph kernel {name:?}: {source}"))]
    GraphKernel {
        name: String,
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
}
