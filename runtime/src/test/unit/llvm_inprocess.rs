use super::*;
use crate::LlvmKernel;
use crate::llvm::{LlvmObjectProducer, llvm_object_flags};
use svod_device::device::{AbiParamDescriptor, AbiParamKind};

/// `out[i] = a[i] * b[i] + sqrt(a[i])` over `n` floats, in the shape codegen
/// emits: opaque pointers, fast-math flags and the CPU attribute set.
const AXPY_SQRT_IR: &str = r#"
define void @axpy_sqrt(ptr noalias %data0, ptr noalias %data1, ptr noalias %data2, i32 %n) #0 {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %pa = getelementptr inbounds float, ptr %data1, i32 %i
  %pb = getelementptr inbounds float, ptr %data2, i32 %i
  %a = load float, ptr %pa, align 4
  %b = load float, ptr %pb, align 4
  %s = call float @llvm.sqrt.f32(float %a)
  %m = fmul nsz arcp contract afn float %a, %b
  %r = fadd nsz arcp contract afn float %m, %s
  %po = getelementptr inbounds float, ptr %data0, i32 %i
  store float %r, ptr %po, align 4
  %next = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %next, %n
  br i1 %done, label %exit, label %loop
exit:
  ret void
}

declare float @llvm.sqrt.f32(float)

attributes #0 = { nounwind "no-builtins" "no-trapping-math"="true" }
"#;

const N: usize = 1000;

fn axpy_abi() -> Vec<AbiParamDescriptor> {
    let storage = |slot| AbiParamDescriptor {
        slot,
        kind: AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
        dtype: svod_dtype::DType::Float32,
        name: None,
    };
    vec![
        storage(0),
        storage(1),
        storage(2),
        AbiParamDescriptor {
            slot: 3,
            kind: AbiParamKind::Scalar,
            dtype: svod_dtype::DType::Int32,
            name: Some("n".into()),
        },
    ]
}

fn axpy_inputs() -> (Vec<f32>, Vec<f32>) {
    let a = (0..N).map(|i| (i as f32) * 0.37 + 0.5).collect();
    let b = (0..N).map(|i| ((i % 17) as f32) - 8.0).collect();
    (a, b)
}

fn run_axpy(object: &[u8]) -> Vec<f32> {
    let kernel = LlvmKernel::load_object_with_abi(object, "axpy_sqrt", "axpy_sqrt", vec!["n".into()], &axpy_abi())
        .expect("object loads through the jit loader");
    let (mut a, mut b) = axpy_inputs();
    let mut out = vec![0.0f32; N];
    let buffers = [out.as_mut_ptr().cast::<u8>(), a.as_mut_ptr().cast(), b.as_mut_ptr().cast()];
    unsafe { kernel.execute_with_vals(&buffers, &[N as i64]).expect("kernel executes") };
    out
}

/// Skip (not fail) on hosts without a loadable libLLVM.
fn host_library() -> Option<&'static LlvmLibrary> {
    match library() {
        Ok(library) => Some(library),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

/// libLLVM emits a relocatable object the existing loader accepts, and the
/// loaded kernel computes the right values, including the `llvm.sqrt`
/// intrinsic lowering.
#[test]
fn in_process_object_loads_and_runs() {
    let Some(library) = host_library() else { return };
    let object = library.compile_ir_to_object(AXPY_SQRT_IR).expect("in-process compile");
    crate::clang::validate_relocatable_object(&object, "axpy_sqrt").expect("relocatable ELF with the entry symbol");

    let out = run_axpy(&object);
    let (a, b) = axpy_inputs();
    for i in 0..N {
        let expected = a[i] * b[i] + a[i].sqrt();
        assert!((out[i] - expected).abs() <= 1e-3 * expected.abs().max(1.0), "out[{i}] = {} != {expected}", out[i]);
    }
}

/// Malformed IR and IR that fails verification are reported as errors rather
/// than aborting the process, and the session stays usable afterwards.
#[test]
fn in_process_reports_bad_ir_and_recovers() {
    let Some(library) = host_library() else { return };
    let parse = library.compile_ir_to_object("this is not LLVM IR").expect_err("parse failure is an error");
    assert!(parse.to_string().contains("IR parse failed"), "{parse}");
    let verify = library
        .compile_ir_to_object("define void @broken() {\nentry:\n  %x = add i32 0, 0\n}\n")
        .expect_err("verification failure is an error");
    assert!(verify.to_string().contains("failed"), "{verify}");
    assert!(library.compile_ir_to_object("define void @fine() {\n  ret void\n}\n").is_ok());
}

/// Objects from libLLVM and from the clang subprocess for the same kernel both
/// run and agree numerically (the pipelines match; byte equality is not
/// required).
#[test]
fn in_process_and_clang_objects_agree() {
    let Some(library) = host_library() else { return };
    let toolchain = crate::clang::ClangToolchain::discover(None).expect("clang toolchain");
    let clang = LlvmObjectProducer::Clang { toolchain, flags: llvm_object_flags() };

    let in_process = run_axpy(&library.compile_ir_to_object(AXPY_SQRT_IR).unwrap());
    let subprocess = run_axpy(&clang.compile(AXPY_SQRT_IR).unwrap());
    for i in 0..N {
        assert!(
            (in_process[i] - subprocess[i]).abs() <= 1e-4 * subprocess[i].abs().max(1.0),
            "out[{i}]: in-process {} vs clang {}",
            in_process[i],
            subprocess[i]
        );
    }
}

/// Every compile thread gets its own context: concurrent compiles of distinct
/// modules must neither interfere nor crash.
#[test]
fn in_process_compiles_concurrently() {
    let Some(library) = host_library() else { return };
    let outputs = std::thread::scope(|scope| {
        let handles = (0..8)
            .map(|thread| {
                scope.spawn(move || {
                    (0..4)
                        .map(|round| {
                            let name = format!("kernel_{thread}_{round}");
                            let ir = AXPY_SQRT_IR.replace("axpy_sqrt", &name);
                            let object = library.compile_ir_to_object(&ir).expect("compile");
                            crate::clang::validate_relocatable_object(&object, &name).expect("valid object");
                        })
                        .count()
                })
            })
            .collect::<Vec<_>>();
        handles.into_iter().map(|handle| handle.join().expect("compile thread")).sum::<usize>()
    });
    assert_eq!(outputs, 32);
}

/// An explicit library path that does not exist is a load error carrying the
/// `libloading` source and the path, not a panic or a flattened string.
#[test]
fn discover_rejects_missing_override() {
    let Err(error) = LlvmLibrary::discover(Some("/nonexistent/libLLVM.so".into())) else {
        panic!("missing library loaded")
    };
    let Error::LlvmUnavailable { failures } = &error else { panic!("{error:?}") };
    let [Error::LibraryLoad { path, source }] = failures.as_slice() else { panic!("{failures:?}") };
    assert_eq!(path, Path::new("/nonexistent/libLLVM.so"));
    assert!(std::error::Error::source(&failures[0]).is_some_and(|s| s.to_string() == source.to_string()));
    let text = error.to_string();
    assert!(text.starts_with("no usable libLLVM: cannot load library /nonexistent/libLLVM.so: "), "{text}");
}

/// A library without the LLVM C API fails to bind on the first missing
/// symbol, naming the symbol and the library.
#[cfg(unix)]
#[test]
fn bind_reports_missing_symbol() {
    let library = Library::from(libloading::os::unix::Library::this());
    let Err(error) = LlvmApi::bind(&library, Path::new("<self>")) else { panic!("bound LLVM against the test binary") };
    let Error::LibrarySymbol { path, symbol, .. } = &error else { panic!("{error:?}") };
    assert_eq!((path.as_path(), symbol.as_str()), (Path::new("<self>"), "LLVMGetVersion"));
    assert!(error.to_string().starts_with("cannot resolve symbol `LLVMGetVersion` in <self>: "), "{error}");
}

/// With `SVOD_LLVM_LIB` pointing at a nonexistent file the LLVM backend falls
/// back to the clang subprocess: it identifies itself as that producer and
/// still compiles and runs kernels. Runs in a child process because library
/// discovery is memoised per process.
#[cfg(unix)]
#[test]
fn fallback_engages_when_library_is_missing() {
    use svod_device::device::ProgramSpec;
    use svod_device::registry::DeviceRegistry;

    const HELPER: &str = "SVOD_TEST_LLVM_FALLBACK_CHILD";
    if std::env::var_os(HELPER).is_some() {
        let registry = DeviceRegistry::default();
        let device = crate::devices::cpu::create_cpu_device_with_backend(&registry, crate::CpuBackend::Llvm).unwrap();
        assert!(device.compiler.cache_key().starts_with("cpu-llvm-clang:"), "{}", device.compiler.cache_key());
        let spec = ProgramSpec::new(
            "axpy_sqrt".into(),
            AXPY_SQRT_IR.into(),
            svod_dtype::DeviceSpec::Cpu,
            svod_ir::UOp::sink(vec![]),
        );
        let compiled = device.compiler.compile(&spec).unwrap();
        let out = run_axpy(&compiled.bytes);
        let (a, b) = axpy_inputs();
        assert!((out[N - 1] - (a[N - 1] * b[N - 1] + a[N - 1].sqrt())).abs() < 1e-2);
        return;
    }

    let test_name = std::thread::current().name().unwrap().to_string();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name, "--nocapture"])
        .env(HELPER, "1")
        .env("SVOD_LLVM_LIB", "/nonexistent/libLLVM.so")
        .env("SVOD_OBJECT_CACHE", "0")
        .status()
        .unwrap();
    assert!(status.success());
}

/// Every file name a distribution installs for one major is generated, the
/// runtime SONAMEs ahead of the dev/compat symlink, so a runtime package
/// without the `libLLVM.so` dev symlink is found.
#[test_case::test_case(18, "so" => vec!["libLLVM.so.18.1", "libLLVM-18.so.1", "libLLVM-18.so", "libLLVM.so.18"]; "elf, newer soname scheme")]
#[test_case::test_case(16, "so" => vec!["libLLVM.so.16.1", "libLLVM-16.so.1", "libLLVM-16.so", "libLLVM.so.16"]; "elf, older soname scheme")]
#[test_case::test_case(20, "dylib" => vec!["libLLVM-20.dylib", "libLLVM.20.dylib"]; "mach-o")]
fn versioned_names_cover_distro_layouts(major: u32, extension: &str) -> Vec<String> {
    versioned_names(major, extension).collect()
}

/// The dev symlink is tried first, then every supported major newest-first.
#[test]
fn candidate_names_try_the_dev_symlink_then_newest_major_first() {
    let names = candidate_names("so");
    assert_eq!(names[0], "libLLVM.so");
    assert_eq!(names[1], format!("libLLVM.so.{MAX_PROBED_MAJOR_VERSION}.1"));
    assert_eq!(names.last().unwrap(), &format!("libLLVM.so.{MIN_MAJOR_VERSION}"));
    assert_eq!(names.len(), 1 + 4 * (MAX_PROBED_MAJOR_VERSION - MIN_MAJOR_VERSION + 1) as usize);
}

/// `llvm-config --libdir` is searched before the loader's default path, which
/// stays in the list for hosts without `llvm-config`.
#[test]
fn llvm_config_libdir_is_searched_before_the_loader_path() {
    let candidates = default_candidates();
    let dev_symlink = format!("libLLVM.{LIBRARY_EXTENSION}");
    let loader_path_at = candidates
        .iter()
        .position(|candidate| candidate.as_os_str() == dev_symlink.as_str())
        .expect("bare name present");
    match llvm_config_libdir() {
        Some(libdir) => {
            assert_eq!(candidates[0], libdir.join(&dev_symlink));
            assert!(loader_path_at > 0, "{candidates:?}");
        }
        None => assert_eq!(loader_path_at, 0, "{candidates:?}"),
    }
}

/// The versioned runtime SONAME of the loaded libLLVM, which a distribution's
/// runtime package installs without the dev symlink, loads on its own.
#[cfg(target_os = "linux")]
#[test]
fn runtime_soname_of_host_library_loads() {
    let Some(library) = host_library() else { return };
    let directory = library.path.parent().filter(|directory| !directory.as_os_str().is_empty());
    let major = library.version[0];
    let loaded = versioned_names(major, "so")
        .map(|name| directory.map_or_else(|| PathBuf::from(&name), |directory| directory.join(&name)))
        .find(|path| LlvmLibrary::discover(Some(path.clone().into_os_string())).is_ok());
    assert!(loaded.is_some(), "no versioned name of LLVM {major} loads beside {}", library.path.display());
}
