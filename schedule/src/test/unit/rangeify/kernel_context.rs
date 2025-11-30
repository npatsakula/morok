use std::rc::Rc;

use morok_ir::UOp;

use crate::rangeify::KernelContext;

#[test]
fn test_kernel_context_new() {
    let ctx = KernelContext::new();
    assert_eq!(ctx.global_counter, 0);
    assert_eq!(ctx.local_counter, 0);
    assert_eq!(ctx.range_counter, 0);
    assert!(ctx.buffer_map.is_empty());
    assert!(ctx.vars.is_empty());
}

#[test]
fn test_next_global() {
    let mut ctx = KernelContext::new();
    assert_eq!(ctx.next_global(), 0);
    assert_eq!(ctx.next_global(), 1);
    assert_eq!(ctx.next_global(), 2);
}

#[test]
fn test_next_local() {
    let mut ctx = KernelContext::new();
    assert_eq!(ctx.next_local(), 0);
    assert_eq!(ctx.next_local(), 1);
    assert_eq!(ctx.next_local(), 2);
}

#[test]
fn test_next_range() {
    let mut ctx = KernelContext::new();
    assert_eq!(ctx.next_range(), 0);
    assert_eq!(ctx.next_range(), 1);
    assert_eq!(ctx.next_range(), 2);
}

#[test]
fn test_buffer_mapping() {
    use morok_dtype::DType;

    let mut ctx = KernelContext::new();

    let original = UOp::native_const(1.0f32);
    let replacement = UOp::define_global(0, DType::Float32);

    assert!(!ctx.has_buffer(&original));

    ctx.map_buffer(original.clone(), replacement.clone());

    assert!(ctx.has_buffer(&original));
    assert!(Rc::ptr_eq(ctx.get_buffer(&original).unwrap(), &replacement));
}

#[test]
fn test_var_tracking() {
    use morok_dtype::DType;

    let mut ctx = KernelContext::new();
    let var = UOp::define_global(1, DType::Index);

    assert!(!ctx.has_var(&var));

    ctx.add_var(var.clone());

    assert!(ctx.has_var(&var));
}
