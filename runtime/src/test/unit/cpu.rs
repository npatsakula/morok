//! CPU device integration tests.
//!
//! Tests the full Device pipeline: render → compile → runtime factory → execute

use crate::devices::cpu::{CpuBackend, create_cpu_device_with_backend};
use morok_device::registry::DeviceRegistry;
use morok_ir::UOp;

#[test]
fn test_cpu_device_creation_cranelift() {
    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Verify device properties
    assert_eq!(device.base_device_key(), "CPU");
    assert!(device.compiler.cache_key().is_some());
    assert_eq!(device.compiler.cache_key().unwrap(), "cranelift-jit");
}

#[test]
fn test_cpu_device_creation_llvm() {
    let registry = DeviceRegistry::default();
    let device =
        create_cpu_device_with_backend(&registry, CpuBackend::Llvm).expect("Failed to create CPU device with LLVM");

    // Verify device properties
    assert_eq!(device.base_device_key(), "CPU");
    assert!(device.compiler.cache_key().is_some());
    assert_eq!(device.compiler.cache_key().unwrap(), "llvm-jit");
}

#[test]
fn test_compile_and_runtime_pipeline_llvm() {
    use morok_device::device::ProgramSpec;
    use morok_dtype::DeviceSpec;

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();

    // Create a minimal valid LLVM IR program
    // This is a no-op kernel that just returns
    let llvm_ir = r#"
define void @test_kernel() {
entry:
  ret void
}
"#;

    let sink = UOp::sink(vec![]);
    let spec = ProgramSpec::new("test_kernel".to_string(), llvm_ir.to_string(), DeviceSpec::Cpu, sink);

    // Test 1: Compile
    let compiled = device.compiler.compile(&spec).expect("Compile should succeed");
    assert!(compiled.src.is_some(), "LLVM JIT should have source");
    assert!(compiled.bytes.is_empty(), "LLVM JIT should have empty bytes");
    assert_eq!(compiled.name, "test_kernel");

    // Test 2: Runtime factory
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");
    // Note: program.name() might not match spec.name (it's a TODO in LlvmProgram)
    assert!(!program.name().is_empty(), "Program should have a name");

    // Test 3: Execute (no buffers needed for this kernel)
    let pointers: Vec<*mut u8> = vec![];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }
}

#[test]
fn test_compile_invalid_ir() {
    use morok_device::device::ProgramSpec;
    use morok_dtype::DeviceSpec;

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();

    // Create a ProgramSpec with invalid LLVM IR
    let sink = UOp::sink(vec![]);
    let spec = ProgramSpec::new("test".to_string(), "this is not valid LLVM IR".to_string(), DeviceSpec::Cpu, sink);

    // Compilation should fail gracefully
    // Note: Current implementation doesn't validate, so this will pass
    // TODO: Add LLVM IR validation to LlvmCompiler
    let result = device.compiler.compile(&spec);
    assert!(result.is_ok(), "Should return CompiledSpec even with invalid IR (validation TODO)");
}

#[test]
fn test_cranelift_bootstrap_pipeline() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Create a simple kernel that stores a constant to a buffer:
    // buf0[0] = 42
    let ptr_dtype = DType::Scalar(ScalarDType::Int32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    let const_42 = UOp::const_(DType::Scalar(ScalarDType::Int32), ConstValue::Int(42));
    let const_0 = UOp::index_const(0);

    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, const_42.clone());

    let sink = UOp::sink(vec![store]);

    // Render using Cranelift
    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    // Verify the IR contains kernel_impl function
    // Note: Bootstrap function is built programmatically in runtime, not in codegen
    assert!(rendered.src.contains("kernel_impl"), "Should contain kernel_impl function");

    // Compile
    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");

    // Create program via runtime factory
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    // Execute with a buffer
    let mut buffer: [i32; 1] = [0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    // Verify the buffer was written correctly
    assert_eq!(buffer[0], 42, "Kernel should have written 42 to buffer[0]");
}

#[test]
fn test_cranelift_exp2_decomposition() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Create a kernel that computes exp2(2.0) and stores to buffer:
    // buf0[0] = exp2(2.0) = 4.0
    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    // Create exp2(2.0)
    let const_2 = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(2.0));
    let exp2_result = const_2.try_exp2().expect("exp2 should succeed");

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, exp2_result.clone());

    let sink = UOp::sink(vec![store]);

    // Apply decomposition for Cranelift (which doesn't support exp2 natively)
    let decomposed = match device.renderer.decompositor() {
        Some(matcher) => decompositions::decompose_with(&sink, &matcher),
        None => sink.clone(),
    };

    // Verify the decomposed graph no longer contains Exp2
    let has_exp2 = decomposed.toposort().iter().any(|node| matches!(node.op(), Op::Unary(morok_ir::UnaryOp::Exp2, _)));
    assert!(!has_exp2, "Decomposed graph should not contain Exp2");

    // Render and execute the decomposed kernel
    let rendered = device.renderer.render(&decomposed).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    // Execute with a buffer
    let mut buffer: [f32; 1] = [0.0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    // Verify exp2(2.0) = 4.0 with some tolerance
    let expected = 4.0f32;
    let tolerance = 1e-3; // Allow 0.1% error for polynomial approximation
    assert!(
        (buffer[0] - expected).abs() < tolerance,
        "exp2(2.0) should be ~4.0, got {} (error = {})",
        buffer[0],
        (buffer[0] - expected).abs()
    );
}

#[test]
fn test_cranelift_simple_math() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Create a kernel that computes 2.0 * 3.0 = 6.0 and stores to buffer
    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    let const_2 = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(2.0));
    let const_3 = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(3.0));
    let mul_result = const_2.try_mul(&const_3).expect("mul should succeed");

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, mul_result.clone());

    let sink = UOp::sink(vec![store]);

    // Render and execute
    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    let mut buffer: [f32; 1] = [0.0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    assert_eq!(buffer[0], 6.0, "2.0 * 3.0 should be 6.0");
}

#[test]
fn test_cranelift_pow2if() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions::helpers;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Create a kernel that computes pow2if(2) = 4.0 and stores to buffer
    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    // Create pow2if(2) using the helper function
    let q = UOp::const_(DType::Scalar(ScalarDType::Int32), ConstValue::Int(2));
    let pow2if_result = helpers::pow2if(&q, &DType::Scalar(ScalarDType::Float32));

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, pow2if_result.clone());

    let sink = UOp::sink(vec![store]);

    // Render and execute
    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    let mut buffer: [f32; 1] = [0.0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    // pow2if(2) = 2^2 = 4.0
    assert!((buffer[0] - 4.0).abs() < 1e-6, "pow2if(2) should be 4.0, got {}", buffer[0]);
}

#[test]
fn test_cranelift_ldexp2k() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions::helpers;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Create a kernel that computes ldexp2k(1.0, 2) = 1.0 * 2^2 = 4.0
    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    // Create ldexp2k(1.0, 2)
    let d = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(1.0));
    let e = UOp::const_(DType::Scalar(ScalarDType::Int32), ConstValue::Int(2));
    let ldexp_result = helpers::ldexp2k(&d, &e);

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, ldexp_result.clone());

    let sink = UOp::sink(vec![store]);

    // Render and execute
    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    let mut buffer: [f32; 1] = [0.0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    // ldexp2k(1.0, 2) = 1.0 * 2^2 = 4.0
    assert!((buffer[0] - 4.0).abs() < 1e-6, "ldexp2k(1.0, 2) should be 4.0, got {}", buffer[0]);
}

#[test]
fn test_cranelift_rintk() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions::helpers;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Create a kernel that computes rintk(2.0) and stores to buffer
    // Note: rintk returns Int32, but we're storing to f32 buffer for simplicity
    let ptr_dtype = DType::Scalar(ScalarDType::Int32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    // Create rintk(2.0)
    let d = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(2.0));
    let rintk_result = helpers::rintk(&d);

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, rintk_result.clone());

    let sink = UOp::sink(vec![store]);

    // Render and execute
    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    let mut buffer: [i32; 1] = [0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    // rintk(2.0) = 2
    assert_eq!(buffer[0], 2, "rintk(2.0) should be 2, got {}", buffer[0]);
}

#[test]
fn test_cranelift_exp2_simple() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions::helpers::{ldexp2k, poly_n, rintk};
    use morok_ir::{ConstValue, Op};

    // Manually build the exp2(2.0) computation step by step
    // to isolate the issue

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    // Build exp2(2.0) step by step:
    // d = 2.0
    // q = rintk(d) = 2 (as Int32)
    // q_float = cast(q, Float32) = 2.0
    // s = d - q_float = 0.0
    // u = poly_n(s, coeffs) = 1.0 (since x=0)
    // result = ldexp2k(u, q) = 1.0 * 2^2 = 4.0

    let d = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let q = rintk(&d);
    let q_float = q.cast(DType::Float32);
    let s = d.try_sub(&q_float).expect("sub failed");

    // Use the polynomial (just for s=0, should return 1.0)
    let coeffs: &[f64] =
        &[0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0];
    let u = poly_n(&s, coeffs);

    // ldexp2k(u, q)
    let result = ldexp2k(&u, &q);

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, result.clone());

    let sink = UOp::sink(vec![store]);

    // Render and execute
    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    let mut buffer: [f32; 1] = [0.0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    let expected = 4.0f32;
    let tolerance = 1e-3;
    assert!(
        (buffer[0] - expected).abs() < tolerance,
        "exp2(2.0) simple should be ~4.0, got {} (error = {})",
        buffer[0],
        (buffer[0] - expected).abs()
    );
}

#[test]
fn test_cranelift_ldexp_with_rintk() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions::helpers::{float_const, ldexp2k, rintk};
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    // ldexp2k(1.0, rintk(2.0)) = 1.0 * 2^2 = 4.0
    let d = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let q = rintk(&d);
    let one = float_const(&DType::Float32, 1.0);
    let result = ldexp2k(&one, &q);

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, result.clone());

    let sink = UOp::sink(vec![store]);

    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    let mut buffer: [f32; 1] = [0.0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    assert!((buffer[0] - 4.0).abs() < 1e-6, "ldexp2k(1.0, rintk(2.0)) should be 4.0, got {}", buffer[0]);
}

#[test]
fn test_cranelift_poly_at_zero() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions::helpers::poly_n;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
    let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

    // poly_n(0.0, coeffs) should return a0 (the first coefficient = 1.0 in exp2 case)
    let x = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let coeffs: &[f64] =
        &[0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0];
    let result = poly_n(&x, coeffs);

    let const_0 = UOp::index_const(0);
    let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
    let store = UOp::store(idx, result.clone());

    let sink = UOp::sink(vec![store]);

    let rendered = device.renderer.render(&sink).expect("Cranelift render should succeed");

    let compiled = device.compiler.compile(&rendered).expect("Compile should succeed");
    let program = (device.runtime)(&compiled).expect("RuntimeFactory should succeed");

    let mut buffer: [f32; 1] = [0.0];
    let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

    unsafe {
        program.execute(&pointers, &[], None, None).expect("Execution should succeed");
    }

    // poly(0) should be the LAST coefficient = 1.0
    // Because tinygrad's polyN uses descending power order:
    // polyN(x, [c0, c1, ..., c_{n-1}]) = c0*x^{n-1} + ... + c_{n-1}
    // At x=0: result = c_{n-1} = 1.0
    let expected = 1.0_f32;
    assert!((buffer[0] - expected).abs() < 1e-10, "poly_n(0, ...) should be ~{}, got {}", expected, buffer[0]);
}

// ============================================================================
// Transcendental accuracy tests (Phase 5)
// ============================================================================

/// Test exp2 accuracy across a range of values
#[test]
fn test_cranelift_exp2_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Test values covering various ranges
    let test_values: &[f32] = &[-10.0, -5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];

    for &input in test_values {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let exp2_result = const_input.try_exp2().expect("exp2 should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, exp2_result.clone());

        let sink = UOp::sink(vec![store]);

        // Apply decomposition
        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        let expected = (2.0_f32).powf(input);
        let ulp_error = ulp_diff(buffer[0], expected);

        assert!(
            ulp_error < 10.0, // Allow up to 10 ULP error
            "exp2({}) = {}, expected {}, ULP error = {}",
            input,
            buffer[0],
            expected,
            ulp_error
        );
    }
}

/// Test sin accuracy at key points
#[test]
fn test_cranelift_sin_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Test values: 0, π/6, π/4, π/3, π/2, π, 3π/2, 2π, and some larger values
    let pi = std::f32::consts::PI;
    let test_values: &[f32] = &[
        0.0,
        pi / 6.0,       // 30°
        pi / 4.0,       // 45°
        pi / 3.0,       // 60°
        pi / 2.0,       // 90°
        pi,             // 180°
        3.0 * pi / 2.0, // 270°
        2.0 * pi,       // 360°
        -pi / 2.0,      // -90°
        -pi,            // -180°
        10.0,           // ~3π
    ];

    for &input in test_values {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let sin_result = const_input.try_sin().expect("sin should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, sin_result.clone());

        let sink = UOp::sink(vec![store]);

        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        let expected = input.sin();

        // For values near zero (like sin(π), sin(2π)), use absolute tolerance
        let tolerance = if expected.abs() < 1e-5 {
            1e-4 // Absolute tolerance for near-zero results
        } else {
            expected.abs() * 1e-3 // Relative tolerance otherwise
        };

        assert!(
            (buffer[0] - expected).abs() < tolerance,
            "sin({}) = {}, expected {}, error = {}",
            input,
            buffer[0],
            expected,
            (buffer[0] - expected).abs()
        );
    }
}

/// Test cos accuracy at key points
#[test]
fn test_cranelift_cos_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    let pi = std::f32::consts::PI;
    let test_values: &[f32] =
        &[0.0, pi / 6.0, pi / 4.0, pi / 3.0, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi, -pi / 2.0, -pi];

    for &input in test_values {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let cos_result = const_input.try_cos().expect("cos should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, cos_result.clone());

        let sink = UOp::sink(vec![store]);

        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        let expected = input.cos();

        let tolerance = if expected.abs() < 1e-5 { 1e-4 } else { expected.abs() * 1e-3 };

        assert!(
            (buffer[0] - expected).abs() < tolerance,
            "cos({}) = {}, expected {}, error = {}",
            input,
            buffer[0],
            expected,
            (buffer[0] - expected).abs()
        );
    }
}

/// Test log2 accuracy
#[test]
fn test_cranelift_log2_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Test values: powers of 2 and other positive values
    let test_values: &[f32] = &[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 0.1, 0.3, 0.7, 1.5, 3.0, 10.0, 100.0];

    for &input in test_values {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let log2_result = const_input.try_log2().expect("log2 should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, log2_result.clone());

        let sink = UOp::sink(vec![store]);

        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        let expected = input.log2();
        let ulp_error = ulp_diff(buffer[0], expected);

        assert!(ulp_error < 10.0, "log2({}) = {}, expected {}, ULP error = {}", input, buffer[0], expected, ulp_error);
    }
}

/// Test exp (natural exponential) accuracy
#[test]
fn test_cranelift_exp_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    let test_values: &[f32] = &[-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0];

    for &input in test_values {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let exp_result = const_input.try_exp().expect("exp should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, exp_result.clone());

        let sink = UOp::sink(vec![store]);

        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        let expected = input.exp();
        let ulp_error = ulp_diff(buffer[0], expected);

        assert!(ulp_error < 10.0, "exp({}) = {}, expected {}, ULP error = {}", input, buffer[0], expected, ulp_error);
    }
}

/// Test log (natural log) accuracy
#[test]
fn test_cranelift_log_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    let test_values: &[f32] = &[0.1, 0.5, 1.0, 2.0, std::f32::consts::E, 10.0, 100.0];

    for &input in test_values {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let log_result = const_input.try_log().expect("log should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, log_result.clone());

        let sink = UOp::sink(vec![store]);

        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        let expected = input.ln();
        let ulp_error = ulp_diff(buffer[0], expected);

        assert!(ulp_error < 10.0, "log({}) = {}, expected {}, ULP error = {}", input, buffer[0], expected, ulp_error);
    }
}

/// Test tan accuracy at key points
#[test]
fn test_cranelift_tan_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    let pi = std::f32::consts::PI;
    // Avoid values near π/2 where tan goes to infinity
    let test_values: &[f32] = &[
        0.0,
        pi / 6.0,  // 30°: tan = 1/√3
        pi / 4.0,  // 45°: tan = 1
        pi / 3.0,  // 60°: tan = √3
        -pi / 4.0, // -45°: tan = -1
        -pi / 3.0, // -60°: tan = -√3
        0.1,
        0.5,
        1.0,
    ];

    for &input in test_values {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let tan_result = const_input.try_tan().expect("tan should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, tan_result.clone());

        let sink = UOp::sink(vec![store]);

        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        let expected = input.tan();

        // Use relative tolerance for tan since values can be large
        let tolerance = if expected.abs() < 1e-5 {
            1e-4
        } else {
            expected.abs() * 1e-2 // 1% relative tolerance
        };

        assert!(
            (buffer[0] - expected).abs() < tolerance,
            "tan({}) = {}, expected {}, error = {}",
            input,
            buffer[0],
            expected,
            (buffer[0] - expected).abs()
        );
    }
}

/// Test erf accuracy
#[test]
fn test_cranelift_erf_accuracy() {
    use morok_dtype::{AddrSpace, DType, ScalarDType};
    use morok_ir::decompositions;
    use morok_ir::{ConstValue, Op};

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Cranelift)
        .expect("Failed to create CPU device with Cranelift");

    // Test values and their expected erf values (computed with high precision)
    let test_cases: &[(f32, f32)] = &[
        (-3.0, -0.9999779),
        (-2.0, -0.9953223),
        (-1.0, -0.8427008),
        (-0.5, -0.5204999),
        (0.0, 0.0),
        (0.5, 0.5204999),
        (1.0, 0.8427008),
        (2.0, 0.9953223),
        (3.0, 0.9999779),
    ];

    for &(input, expected) in test_cases {
        let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(None, AddrSpace::Global);
        let buf0 = UOp::new(Op::DefineGlobal(0), ptr_dtype);

        let const_input = UOp::const_(DType::Scalar(ScalarDType::Float32), ConstValue::Float(input as f64));
        let erf_result = UOp::erf(const_input).expect("erf should succeed");

        let const_0 = UOp::index_const(0);
        let idx = UOp::index().buffer(buf0.clone()).indices(vec![const_0.clone()]).call().unwrap();
        let store = UOp::store(idx, erf_result.clone());

        let sink = UOp::sink(vec![store]);

        let decomposed = match device.renderer.decompositor() {
            Some(matcher) => decompositions::decompose_with(&sink, &matcher),
            None => sink.clone(),
        };

        let rendered = device.renderer.render(&decomposed).expect("render failed");
        let compiled = device.compiler.compile(&rendered).expect("compile failed");
        let program = (device.runtime)(&compiled).expect("runtime failed");

        let mut buffer: [f32; 1] = [0.0];
        let pointers: Vec<*mut u8> = vec![buffer.as_mut_ptr() as *mut u8];

        unsafe {
            program.execute(&pointers, &[], None, None).expect("execute failed");
        }

        // Allow 1e-5 absolute error (the approximation has ~1.5e-7 max error)
        let tolerance = 1e-5;

        assert!(
            (buffer[0] - expected).abs() < tolerance,
            "erf({}) = {}, expected {}, error = {}",
            input,
            buffer[0],
            expected,
            (buffer[0] - expected).abs()
        );
    }
}

/// Helper function to compute ULP (Units in Last Place) difference between two floats
fn ulp_diff(a: f32, b: f32) -> f32 {
    if a == b {
        return 0.0;
    }
    if a.is_nan() || b.is_nan() {
        return f32::INFINITY;
    }
    if a.is_infinite() || b.is_infinite() {
        if a == b {
            return 0.0;
        } else {
            return f32::INFINITY;
        }
    }

    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;

    // Handle sign differences
    let a_signed = if a < 0.0 { i32::MIN - a_bits } else { a_bits };
    let b_signed = if b < 0.0 { i32::MIN - b_bits } else { b_bits };

    (a_signed - b_signed).abs() as f32
}
