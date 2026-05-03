use super::*;

#[test]
fn test_llvm_kernel_no_args() {
    let ir = r#"
        define void @test_kernel() {
            ret void
        }
    "#;

    let kernel = LlvmKernel::compile_ir(ir, "test_kernel", "test_kernel", vec![], 0).unwrap();
    assert_eq!(kernel.name(), "test_kernel");

    unsafe {
        kernel.execute_with_vals(&[], &[]).unwrap();
    }
}

#[test]
fn test_llvm_kernel_with_args() {
    let ir = r#"
        define void @add_kernel(ptr noalias %data0, ptr noalias %data1) {
            ret void
        }
    "#;

    let kernel = LlvmKernel::compile_ir(ir, "add_kernel", "add_kernel", vec![], 2).unwrap();

    let mut data1 = vec![0u8; 16];
    let mut data2 = vec![0u8; 16];
    let buffers = vec![data1.as_mut_ptr(), data2.as_mut_ptr()];

    unsafe {
        kernel.execute_with_vals(&buffers, &[]).unwrap();
    }
}

#[test]
fn test_kernel_drop_order() {
    let ir = r#"
        define void @test() {
            ret void
        }
    "#;

    let kernel = LlvmKernel::compile_ir(ir, "test", "test", vec![], 0).unwrap();
    drop(kernel); // Should not crash
}
