use std::sync::Arc;

use morok_device::device::Device;
use morok_device::registry::DeviceRegistry;
use morok_ir::DeviceSpec;
use morok_runtime::CpuBackend;
use morok_schedule::OptimizerConfig;
use snafu::ResultExt;

use crate::error::{DeviceFactorySnafu, DeviceSnafu};

/// Resolves a `DeviceSpec` into a concrete `Device` for compilation.
///
/// Implementations control which codegen backend is used for each device type.
/// This enables per-call backend selection instead of relying on the
/// `DEVICE_FACTORIES` singleton (which bakes one backend per device spec).
pub(crate) trait DeviceResolver: Send + Sync {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>>;
}

/// Default resolver: delegates to `DEVICE_FACTORIES` singleton (reads env vars
/// like `MOROK_CPU_BACKEND` at first device creation, then caches).
struct EnvResolver;

impl DeviceResolver for EnvResolver {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        morok_runtime::DEVICE_FACTORIES.device(spec, registry).context(DeviceFactorySnafu)
    }
}

/// Creates CPU devices with a specific backend; delegates other device types
/// to `DEVICE_FACTORIES`. This is the resolver used by `PrepareConfig::for_cpu_backend()`.
struct CpuBackendResolver(CpuBackend);

impl DeviceResolver for CpuBackendResolver {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        match spec {
            DeviceSpec::Cpu => {
                Ok(Arc::new(morok_runtime::create_cpu_device_with_backend(registry, self.0).context(DeviceSnafu)?))
            }
            _ => morok_runtime::DEVICE_FACTORIES.device(spec, registry).context(DeviceFactorySnafu),
        }
    }
}

/// Configuration for `prepare()`/`realize()` that bundles optimizer settings
/// with device resolution (codegen backend selection).
///
/// Instead of relying on the `MOROK_CPU_BACKEND` env var (global mutable state),
/// the backend is selected per-call via a [`DeviceResolver`].
pub struct PrepareConfig {
    pub optimizer: OptimizerConfig,
    pub(crate) resolver: Arc<dyn DeviceResolver>,
}

impl std::fmt::Debug for PrepareConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrepareConfig").field("optimizer", &self.optimizer).finish_non_exhaustive()
    }
}

impl Default for PrepareConfig {
    fn default() -> Self {
        Self { optimizer: OptimizerConfig::default(), resolver: Arc::new(EnvResolver) }
    }
}

impl PrepareConfig {
    /// Read both `MOROK_CPU_BACKEND` and optimizer env vars.
    pub fn from_env() -> Self {
        Self { optimizer: OptimizerConfig::from_env(), resolver: Arc::new(EnvResolver) }
    }

    /// Convenience constructor: specific CPU backend with default optimizer.
    pub fn for_cpu_backend(backend: CpuBackend) -> Self {
        Self { optimizer: OptimizerConfig::default(), resolver: Arc::new(CpuBackendResolver(backend)) }
    }

    /// Resolve a `DeviceSpec` into a `Device` using this config's resolver.
    pub(crate) fn resolve_device(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        self.resolver.resolve(spec, registry)
    }
}

impl From<OptimizerConfig> for PrepareConfig {
    fn from(optimizer: OptimizerConfig) -> Self {
        Self { optimizer, resolver: Arc::new(EnvResolver) }
    }
}

/// Generate one test per codegen backend (Clang, LLVM) from a single test body.
///
/// Supports three forms:
///
/// **Simple test** (config only, no extra params):
/// ```ignore
/// codegen_tests! {
///     fn test_add(config) {
///         let mut a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
///         a.realize_with(&config).unwrap();
///         let result: Vec<f32> = a.as_vec().unwrap();
///     }
/// }
/// // Generates: test_add::clang, test_add::llvm
/// ```
///
/// **Parameterized test** (extra typed params, use with `#[test_case]`):
/// ```ignore
/// codegen_tests! {
///     #[test_case(128, 0.5; "128x128")]
///     fn test_matmul(config, size: usize, tol: f32) {
///         let mut result = run_matmul(size);
///         result.realize_with(&config).unwrap();
///         assert_close(&result, tol);
///     }
/// }
/// // Generates: test_matmul::clang::test_matmul, test_matmul::llvm::test_matmul
/// ```
///
/// **Proptest** (property-based, params use `in` syntax):
/// ```ignore
/// codegen_tests! {
///     #[proptest_config(ProptestConfig::with_cases(50))]
///     fn test_sort_random(config, data in proptest::collection::vec(-100.0f32..100.0, 1..=16)) {
///         let mut t = Tensor::from_slice(&data);
///         let (sorted, _) = t.sort(-1, false).unwrap();
///         // ...
///     }
/// }
/// // Generates: test_sort_random::clang, test_sort_random::llvm
/// ```
#[macro_export]
macro_rules! codegen_tests {
    // Base case
    () => {};

    // Simple test (config only, no extra params)
    ($(#[$meta:meta])* fn $name:ident($config:ident) $body:block $($rest:tt)*) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            $(#[$meta])*
            fn clang() {
                ::morok_schedule::testing::setup_test_tracing();
                let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                $body
            }

            #[test]
            $(#[$meta])*
            fn llvm() {
                ::morok_schedule::testing::setup_test_tracing();
                let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                $body
            }
        }
        $crate::codegen_tests!($($rest)*);
    };

    // Proptest with config: #[proptest_config(...)] fn name(config, param in strategy) { body }
    (#[proptest_config($($pc:tt)*)] $(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident in $strategy:expr),+ $(,)?) $body:block $($rest:tt)*) => {
        $crate::codegen_tests!(@proptest $name, $config, [$($param in $strategy),+], $body,
            ::proptest::test_runner::TestRunner::new($($pc)*), [$(#[$meta])*]);
        $crate::codegen_tests!($($rest)*);
    };

    // Proptest with default config: fn name(config, param in strategy) { body }
    ($(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident in $strategy:expr),+ $(,)?) $body:block $($rest:tt)*) => {
        $crate::codegen_tests!(@proptest $name, $config, [$($param in $strategy),+], $body,
            ::proptest::test_runner::TestRunner::default(), [$(#[$meta])*]);
        $crate::codegen_tests!($($rest)*);
    };

    // Internal: proptest code generation (uses TestRunner API directly)
    (@proptest $name:ident, $config:ident, [$($param:ident in $strategy:expr),+], $body:block, $runner:expr, [$(#[$meta:meta])*]) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn clang() {
                ::morok_schedule::testing::setup_test_tracing();
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                    $body
                    Ok(())
                }).unwrap();
            }

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn llvm() {
                ::morok_schedule::testing::setup_test_tracing();
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                    $body
                    Ok(())
                }).unwrap();
            }
        }
    };

    // Parameterized test (extra typed params — test_case attrs expected, no #[test])
    ($(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident: $ty:ty),+ $(,)?) $body:block $($rest:tt)*) => {
        mod $name {
            mod clang {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::morok_schedule::testing::setup_test_tracing();
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                    $body
                }
            }
            mod llvm {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::morok_schedule::testing::setup_test_tracing();
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                    $body
                }
            }
        }
        $crate::codegen_tests!($($rest)*);
    };
}
