use snafu::Snafu;
use svod_dtype::DType;

mod recurrent;
pub use recurrent::{JitRecurrent, LstmState, RecurrentJit, StepTiming};

/// Shape + dtype descriptor for a single JIT input. Used by
/// `jit_wrapper!`-generated `prepare()` calls to allocate zero-initialized
/// placeholder buffers internally — callers no longer construct fake
/// `Tensor::zeros(..).realize()` placeholders.
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Allocate the input device-local (no host mapping). The host can still
    /// reach it through `copyin`/`copyout` (staged over the copy engine), and
    /// on-device `copy_from`/`copy_region_from` stays on-device — for
    /// recurrent state the host shouldn't observe per step.
    pub device_local: bool,
}

impl InputSpec {
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        Self { shape: shape.to_vec(), dtype, device_local: false }
    }

    pub fn f32(shape: &[usize]) -> Self {
        Self::new(shape, DType::Float32)
    }

    pub fn i32(shape: &[usize]) -> Self {
        Self::new(shape, DType::Int32)
    }

    pub fn i64(shape: &[usize]) -> Self {
        Self::new(shape, DType::Int64)
    }

    pub fn device_local(mut self) -> Self {
        self.device_local = true;
        self
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum JitError {
    #[snafu(display("JIT not prepared: call prepare() first"))]
    NotPrepared,

    #[snafu(display("input buffer not found: {name}"))]
    InputBufferNotFound { name: &'static str },

    #[snafu(display("duplicate JIT input buffer: {name} aliases {duplicate_of} with {buffer_id:?}"))]
    DuplicateInputBuffer { name: &'static str, duplicate_of: &'static str, buffer_id: svod_device::BufferId },

    /// The plan buffer an input resolved to is not the buffer that was
    /// realized for it — cross-plan input aliasing (a concurrent prepare
    /// corrupted this input's graph identity).
    #[snafu(display("JIT input {name} resolved to a foreign plan buffer: expected {expected:?}, got {actual:?}"))]
    InputAliased { name: &'static str, expected: svod_device::BufferId, actual: svod_device::BufferId },

    /// Wraps the user-supplied error type returned by a `jit_wrapper!` build
    /// closure. Genuine `Box<dyn>` because the closure's `E` is arbitrary.
    #[snafu(display("{source}"))]
    Build { source: Box<dyn std::error::Error + Send + Sync> },

    #[snafu(display("{source}"), context(false))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },

    #[snafu(display("{source}"), context(false))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },

    /// `JitRecurrent::new` rejected a JIT whose output element count does not
    /// match the declared `head_len + |h| + |c|`. Typically means the `build`
    /// closure was changed and now emits a different layout than the wrapper
    /// expects.
    #[snafu(display(
        "JIT output layout mismatch: declared {declared_head} head + {declared_state} state elements \
         ({}), actual {actual} elements. Check that the `build` closure returns `cat([head, h, c], -1)` \
         with the declared shapes.",
        declared_head + declared_state
    ))]
    OutputLayoutMismatch { declared_head: usize, declared_state: usize, actual: usize },

    /// A multi-output `jit_wrapper!` declared `declared` outputs but the
    /// compiled plan kept `actual`. Means the scheduler fused or elided an
    /// output the `build` closure returned — the positional `output_*()`
    /// accessors would be misaligned, so fail loud at `prepare` instead.
    #[snafu(display("JIT output count mismatch: declared {declared} outputs, plan kept {actual}"))]
    OutputCountMismatch { declared: usize, actual: usize },

    #[snafu(display("{source}"))]
    Runtime { source: svod_runtime::Error },
}

pub type Result<T> = std::result::Result<T, JitError>;
