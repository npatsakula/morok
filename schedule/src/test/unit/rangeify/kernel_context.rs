use std::sync::Arc;

use morok_ir::UOp;

use crate::rangeify::RangeifyBufferContext;

#[test]
fn test_kernel_context_new() {
    let ctx = RangeifyBufferContext::new();
    assert_eq!(ctx.global_counter, 0);
    assert_eq!(ctx.local_counter, 0);
    assert_eq!(ctx.range_counter, 0);
    assert!(ctx.buffer_map.is_empty());
    assert!(ctx.vars.is_empty());
}

#[test]
fn test_next_global() {
    let mut ctx = RangeifyBufferContext::new();
    assert_eq!(ctx.next_global(), 0);
    assert_eq!(ctx.next_global(), 1);
    assert_eq!(ctx.next_global(), 2);
}

#[test]
fn test_next_local() {
    let mut ctx = RangeifyBufferContext::new();
    assert_eq!(ctx.next_local(), 0);
    assert_eq!(ctx.next_local(), 1);
    assert_eq!(ctx.next_local(), 2);
}

#[test]
fn test_next_range() {
    let mut ctx = RangeifyBufferContext::new();
    assert_eq!(ctx.next_range(), 0);
    assert_eq!(ctx.next_range(), 1);
    assert_eq!(ctx.next_range(), 2);
}

#[test]
fn test_buffer_mapping() {
    use morok_dtype::DType;

    let mut ctx = RangeifyBufferContext::new();

    let original = UOp::native_const(1.0f32);
    let replacement = UOp::param(0, 1, DType::Float32, None);

    assert!(!ctx.has_buffer(&original));

    ctx.map_buffer(original.clone(), replacement.clone());

    assert!(ctx.has_buffer(&original));
    assert!(Arc::ptr_eq(ctx.get_buffer(&original).unwrap(), &replacement));
}

#[test]
fn test_var_tracking() {
    let mut ctx = RangeifyBufferContext::new();
    let var = UOp::define_var("test_var".to_string(), 0, 10);

    assert!(!ctx.vars.contains_key("test_var"));

    ctx.add_var(var.clone(), Some(5));

    assert!(ctx.vars.contains_key("test_var"));
    let (stored_uop, stored_val) = ctx.vars.get("test_var").unwrap();
    assert_eq!(stored_uop.id, var.id);
    assert_eq!(*stored_val, Some(5));
}
