//! Unit tests for AMX helper functions.

use crate::mlir::amx::{fma_opcode_and_flags, validate_amx_dtypes, z_row_stride};
use morok_dtype::DType;
use morok_ir::{WmmaMetadata, WmmaUpcastAxes};

/// Helper to create minimal WmmaMetadata for testing.
fn make_metadata(dtype_in: DType, dtype_out: DType) -> WmmaMetadata {
    WmmaMetadata {
        name: "test_wmma".to_string(),
        dims: (16, 16, 1),
        dtype_in,
        dtype_out,
        device: "AppleAMX".to_string(),
        threads: 1,
        upcast_axes: WmmaUpcastAxes { a: vec![], b: vec![], c: vec![] },
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

/// Test z_row_stride calculation for different dtype combinations.
#[test]
fn test_z_row_stride() {
    // f64 x f64 -> f64: stride = 8
    assert_eq!(z_row_stride(&DType::Float64, &DType::Float64), 8);

    // f32 x f32 -> f32: stride = 4
    assert_eq!(z_row_stride(&DType::Float32, &DType::Float32), 4);

    // f16 x f16 -> f16: stride = 2
    assert_eq!(z_row_stride(&DType::Float16, &DType::Float16), 2);

    // f16 x f16 -> f32 (mixed-precision): stride = 2 (input dtype determines stride)
    assert_eq!(z_row_stride(&DType::Float16, &DType::Float32), 2);

    // i16 x i16 -> i16: stride = 2
    assert_eq!(z_row_stride(&DType::Int16, &DType::Int16), 2);
}

/// Test FMA opcode selection for different dtypes.
#[test]
fn test_fma_opcode_and_flags() {
    // f64 x f64 -> f64: opcode 10, no flags
    let (op, flags) = fma_opcode_and_flags(&make_metadata(DType::Float64, DType::Float64)).unwrap();
    assert_eq!(op, 10);
    assert_eq!(flags, 0);

    // f32 x f32 -> f32: opcode 12, no flags
    let (op, flags) = fma_opcode_and_flags(&make_metadata(DType::Float32, DType::Float32)).unwrap();
    assert_eq!(op, 12);
    assert_eq!(flags, 0);

    // f16 x f16 -> f16: opcode 15, no flags
    let (op, flags) = fma_opcode_and_flags(&make_metadata(DType::Float16, DType::Float16)).unwrap();
    assert_eq!(op, 15);
    assert_eq!(flags, 0);

    // f16 x f16 -> f32 (mixed-precision): opcode 15, bit 62 set
    let (op, flags) = fma_opcode_and_flags(&make_metadata(DType::Float16, DType::Float32)).unwrap();
    assert_eq!(op, 15);
    assert_eq!(flags, 1 << 62);

    // i16 x i16 -> i16: opcode 14, no flags
    let (op, flags) = fma_opcode_and_flags(&make_metadata(DType::Int16, DType::Int16)).unwrap();
    assert_eq!(op, 14);
    assert_eq!(flags, 0);
}

/// Test that unsupported dtype combinations return an error.
#[test]
fn test_validate_amx_dtypes_unsupported() {
    // f32 x f32 -> f16 should fail (output must be f32 for f32 input)
    let result = validate_amx_dtypes(&DType::Float32, &DType::Float16);
    assert!(result.is_err());

    // i32 is not supported
    let result = validate_amx_dtypes(&DType::Int32, &DType::Int32);
    assert!(result.is_err());

    // bf16 is not supported (M2+, future enhancement)
    let result = validate_amx_dtypes(&DType::BFloat16, &DType::BFloat16);
    assert!(result.is_err());
}

/// Test that supported dtype combinations pass validation.
#[test]
fn test_validate_amx_dtypes_supported() {
    // All supported combinations should pass
    assert!(validate_amx_dtypes(&DType::Float32, &DType::Float32).is_ok());
    assert!(validate_amx_dtypes(&DType::Float64, &DType::Float64).is_ok());
    assert!(validate_amx_dtypes(&DType::Float16, &DType::Float16).is_ok());
    assert!(validate_amx_dtypes(&DType::Float16, &DType::Float32).is_ok()); // Mixed-precision
    assert!(validate_amx_dtypes(&DType::Int16, &DType::Int16).is_ok());
}
