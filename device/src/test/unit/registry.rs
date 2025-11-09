#[test]
fn test_registry_cpu() {
    let allocator = crate::cpu().unwrap();
    assert_eq!(allocator.name(), "CPU");
}
