use super::*;

#[test]
fn test_jit_loader_noop() {
    let src = "void test_kernel(void) { }\n";
    let kernel = JitKernel::compile(src, "test_kernel", vec![], 0).unwrap();
    assert_eq!(kernel.name(), "test_kernel");
    unsafe {
        kernel.execute_with_vals(&[], &[]).unwrap();
    }
}

#[test]
fn test_jit_loader_add() {
    let src = r#"
void add_kernel(float* restrict a, float* restrict b, float* restrict out) {
    out[0] = a[0] + b[0];
}
"#;
    let kernel = JitKernel::compile(src, "add_kernel", vec![], 3).unwrap();

    let mut a = [1.0f32];
    let mut b = [2.0f32];
    let mut out = [0.0f32];

    let buffers = vec![a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[]).unwrap();
    }

    assert_eq!(out[0], 3.0);
}

#[test]
fn test_jit_loader_math() {
    let src = r#"
void math_kernel(float* restrict in_buf, float* restrict out) {
    out[0] = __builtin_sqrtf(in_buf[0]);
}
"#;
    let kernel = JitKernel::compile(src, "math_kernel", vec![], 2).unwrap();

    let mut input = [9.0f32];
    let mut out = [0.0f32];

    let buffers = vec![input.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[]).unwrap();
    }

    assert!((out[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_jit_loader_with_vars() {
    let src = r#"
void var_kernel(float* restrict out, const int N) {
    for (int i = 0; i < N; i++) {
        out[i] = (float)i;
    }
}
"#;
    let kernel = JitKernel::compile(src, "var_kernel", vec!["N".to_string()], 1).unwrap();

    let mut out = [0.0f32; 8];
    let buffers = vec![out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[5]).unwrap();
    }

    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 1.0);
    assert_eq!(out[2], 2.0);
    assert_eq!(out[3], 3.0);
    assert_eq!(out[4], 4.0);
}
