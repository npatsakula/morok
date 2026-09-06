use crate::metal::compile::{extract_metallib, pack_request, std_for_major, validate_metallib};

#[test]
fn request_frame_layout() {
    let request = pack_request(b"k", b"-x metal");
    let mut expected = Vec::new();
    expected.extend_from_slice(&4u64.to_le_bytes());
    expected.extend_from_slice(&9u64.to_le_bytes());
    expected.extend_from_slice(b"k\0\0\0");
    expected.extend_from_slice(b"-x metal\0");
    assert_eq!(request, expected);
}

/// The source blob is NUL-terminated and padded to a multiple of 4 bytes.
#[test_case::test_case(0 => 4)]
#[test_case::test_case(3 => 4)]
#[test_case::test_case(4 => 8)]
#[test_case::test_case(5 => 8)]
#[test_case::test_case(7 => 8)]
#[test_case::test_case(8 => 12)]
fn request_source_padding(source_len: usize) -> usize {
    let request = pack_request(&vec![b'x'; source_len], b"");
    let padded = u64::from_le_bytes(request[..8].try_into().unwrap()) as usize;
    assert_eq!(request.len(), 16 + padded + 1);
    assert!(request[16 + source_len..16 + padded].iter().all(|byte| *byte == 0));
    padded
}

#[test]
fn reply_payload_follows_header_and_warnings() {
    let mut reply = vec![0u8; 16];
    reply[8..12].copy_from_slice(&16u32.to_le_bytes());
    reply[12..16].copy_from_slice(&4u32.to_le_bytes());
    reply.extend_from_slice(b"warn");
    reply.extend_from_slice(b"MTLB payload ENDT");
    assert_eq!(extract_metallib(&reply).unwrap(), b"MTLB payload ENDT");
}

#[test_case::test_case(&[0u8; 8], "shorter than its header"; "truncated header")]
#[test_case::test_case(&{ let mut r = vec![0u8; 16]; r[8..12].copy_from_slice(&100u32.to_le_bytes()); r }, "exceeds reply length"; "offset past end")]
fn reply_parsing_rejects_malformed(reply: &[u8], reason: &str) {
    let error = extract_metallib(reply).unwrap_err();
    assert!(error.contains(reason), "{error}");
}

#[test_case::test_case(b"MTLB...kernel_main...ENDT", "kernel_main" => true; "metallib with entry")]
#[test_case::test_case(b"MTLB...other...ENDT", "kernel_main" => false; "metallib without entry")]
#[test_case::test_case(b"MTLB...kernel_main...", "kernel_main" => false; "metallib without trailer")]
#[test_case::test_case(b"kernel void kernel_main() {}", "kernel_main" => true; "source with entry")]
#[test_case::test_case(b"kernel void other() {}", "kernel_main" => false; "source without entry")]
#[test_case::test_case(&[0xff, 0xfe, 0x00], "kernel_main" => false; "neither form")]
fn metallib_validation(bytes: &[u8], name: &str) -> bool {
    validate_metallib(bytes, name).is_ok()
}

#[test_case::test_case(26 => "metal4.0")]
#[test_case::test_case(27 => "metal4.0")]
#[test_case::test_case(14 => "metal3.1")]
#[test_case::test_case(25 => "metal3.1")]
#[test_case::test_case(13 => "metal3.0")]
#[test_case::test_case(12 => "macos-metal2.0")]
#[test_case::test_case(0 => "macos-metal2.0")]
fn msl_standard_by_macos_major(major: u32) -> &'static str {
    std_for_major(major)
}

// ── hardware ─────────────────────────────────────────────────────────────

#[test]
fn compiles_hand_written_msl() {
    let Some(dev) = super::metal_device_or_skip() else { return };
    let bytes = super::compile_for_test(&dev, super::VADD_MSL).expect("compile vadd");
    validate_metallib(&bytes, "vadd").expect("valid payload");
    assert!(validate_metallib(&bytes, "not_there").is_err());
    if crate::metal::compile::codegen_service_available() {
        assert!(bytes.starts_with(b"MTLB") && bytes.ends_with(b"ENDT"), "private service must return a metallib");
    }
}

#[test]
fn compile_error_surfaces_a_diagnostic() {
    let Some(dev) = super::metal_device_or_skip() else { return };
    let error = super::compile_for_test(&dev, "kernel void k() { this is not msl }").expect_err("invalid MSL");
    let message = format!("{error}");
    assert!(message.contains("Metal compile failed"), "{message}");
    assert!(message.contains("error"), "diagnostic expected: {message}");
}

#[test]
fn both_compile_paths_load_and_agree() {
    use crate::metal::MetalProgram;
    use crate::metal::compile::{codegen_service_available, compile_msl, compile_msl_public, metal_std_flag};
    let Some(dev) = super::metal_device_or_skip() else { return };
    if !codegen_service_available() {
        return;
    }
    let params =
        format!("-fno-fast-math -std={} --driver-mode=metal -x metal -fno-caret-diagnostics", metal_std_flag());
    let private = compile_msl(super::VADD_MSL, &params).expect("private compile");
    let public = compile_msl_public(&dev, super::VADD_MSL).expect("public compile");
    assert!(private.starts_with(b"MTLB"));
    assert_eq!(public, super::VADD_MSL.as_bytes());
    let abi = super::program::vadd_abi();
    let a = MetalProgram::load(dev.clone(), &private, "vadd", &abi).expect("load metallib");
    let b = MetalProgram::load(dev, &public, "vadd", &abi).expect("load source");
    assert_eq!(a.max_total_threads_per_threadgroup(), b.max_total_threads_per_threadgroup());
}
