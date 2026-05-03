use super::*;

#[test]
fn test_function_name_ascii() {
    let info = KernelInfo::new("test_kernel", vec![], false);
    assert_eq!(info.function_name(), "test_kernel");
}

#[test]
fn test_function_name_with_underscores() {
    let info = KernelInfo::new("r_g16l16R32u4", vec![], false);
    assert_eq!(info.function_name(), "r_g16l16R32u4");
}

#[test]
fn test_function_name_unicode() {
    // ANSI escape codes should be converted to hex
    let info = KernelInfo::new("r\x1b[34mg16\x1b[0m", vec![], false);
    let func_name = info.function_name();
    assert!(func_name.contains("1B")); // ESC character (0x1B)
    assert!(func_name.contains("5B")); // '[' character (0x5B)
}

#[test]
fn test_function_name_special_chars() {
    let info = KernelInfo::new("test-kernel+v2", vec![], false);
    let func_name = info.function_name();
    assert!(func_name.contains("2D")); // '-' = 0x2D
    assert!(func_name.contains("2B")); // '+' = 0x2B
}
