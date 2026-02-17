//! C renderer tests for code generation verification.

use morok_dtype::DType;
use morok_ir::{AxisId, AxisType, ConstValue, ReduceOp, UOp, WmmaMetadata, WmmaUpcastAxes};
use smallvec::SmallVec;

use crate::c::render;

#[test]
fn test_range_end_basic() {
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);
    let noop = UOp::noop();
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = noop.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render(&sink, Some("test_loop")).expect("C codegen failed");

    assert!(result.code.contains("for"), "Missing for loop:\n{}", result.code);
    assert!(result.code.contains("ridx0"), "Missing loop variable:\n{}", result.code);
    assert!(result.code.contains("< 10"), "Missing loop bound:\n{}", result.code);
}

#[test]
fn test_reduce_add_basic() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let range =
        UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render(&sink, Some("test_reduce")).expect("C codegen failed");

    assert!(result.code.contains("acc"), "Missing accumulator:\n{}", result.code);
    assert!(result.code.contains("for"), "Missing for loop:\n{}", result.code);
    assert!(result.code.contains("+="), "Missing accumulation:\n{}", result.code);
    assert!(result.code.contains("0.0f"), "Missing identity value:\n{}", result.code);
}

#[test]
fn test_reduce_max() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let range = UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(5)), AxisId::Renumbered(0), AxisType::Reduce);

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Max);
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render(&sink, Some("test_reduce_max")).expect("C codegen failed");

    assert!(result.code.contains("fmaxf"), "Missing fmaxf:\n{}", result.code);
}

#[test]
fn test_reduce_empty_ranges() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let reduce = const_val.reduce(smallvec::smallvec![], ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let result = render(&sink, Some("test_reduce_empty"));
    assert!(result.is_ok(), "C codegen failed: {:?}", result.err());
}

/// Helper to create AMX float32 WMMA metadata matching the APPLE_AMX TcConfig.
fn amx_f32_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_1_float_float".to_string(),
        dims: (16, 16, 1),
        dtype_in: DType::Float32,
        dtype_out: DType::Float32,
        device: "AppleAMX".to_string(),
        threads: 1,
        upcast_axes: WmmaUpcastAxes { a: vec![(2, 256)], b: vec![(2, 256)], c: vec![(2, 256)] },
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

/// Helper to create AMX mixed-precision (f16→f32) WMMA metadata.
fn amx_f16_to_f32_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_1_half_float".to_string(),
        dims: (16, 16, 1),
        dtype_in: DType::Float16,
        dtype_out: DType::Float32,
        device: "AppleAMX".to_string(),
        threads: 1,
        upcast_axes: WmmaUpcastAxes { a: vec![(2, 256)], b: vec![(2, 256)], c: vec![(2, 256)] },
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

/// Helper to create AMX WMMA metadata with 2×2 tile grid.
fn amx_tile_grid_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_1_float_float_tile2x2".to_string(),
        dims: (16, 16, 1),
        dtype_in: DType::Float32,
        dtype_out: DType::Float32,
        device: "AppleAMX".to_string(),
        threads: 1,
        upcast_axes: WmmaUpcastAxes { a: vec![(2, 256)], b: vec![(2, 256)], c: vec![(2, 256)] },
        reduce_axes: vec![],
        tile_grid: (2, 2),
    }
}

#[test]
fn test_wmma_preamble_macros() {
    // Construct a minimal WMMA node: a(float16) × b(float16) + c(float256) → float256
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify AMX macros are emitted
    assert!(result.code.contains("#define AMX_SET"), "Missing AMX_SET macro:\n{}", result.code);
    assert!(result.code.contains("#define AMX("), "Missing AMX macro:\n{}", result.code);
}

#[test]
fn test_wmma_preamble_static_function() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify static wrapper function is emitted with correct signature
    assert!(
        result
            .code
            .contains("static float256 __WMMA_16_16_1_float_float(float16 data1, float16 data2, float256 data0)"),
        "Missing or incorrect static WMMA function signature:\n{}",
        result.code,
    );
    // Verify AMX instructions inside the static function
    assert!(result.code.contains("AMX_SET(0)"), "Missing AMX_SET(0) init:\n{}", result.code);
    assert!(result.code.contains("AMX_SET(1)"), "Missing AMX_SET(1) finalize:\n{}", result.code);
    assert!(result.code.contains("AMX(12,"), "Missing fma32 instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(0,"), "Missing ldx instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(1,"), "Missing ldy instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(4,"), "Missing ldz instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(5,"), "Missing stz instruction:\n{}", result.code);
}

#[test]
fn test_wmma_function_call() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify the kernel body contains a WMMA function call
    assert!(result.code.contains("__WMMA_16_16_1_float_float("), "Missing WMMA function call:\n{}", result.code);
}

#[test]
fn test_wmma_vector_typedefs() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify vector typedefs for float16 and float256 are emitted
    assert!(result.code.contains("typedef float float16"), "Missing float16 typedef:\n{}", result.code);
    assert!(result.code.contains("typedef float float256"), "Missing float256 typedef:\n{}", result.code);
}

#[test]
fn test_wmma_mixed_precision_flag() {
    // f16 × f16 → f32 requires bit 62 set in FMA encoding
    let zero = UOp::const_(DType::Float16, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = UOp::const_(DType::Float32, ConstValue::Float(0.0)).broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f16_to_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render(&sink, Some("test_wmma_mixed")).expect("C codegen failed");

    // Verify fma16 opcode (15) is used
    assert!(result.code.contains("AMX(15,"), "Missing fma16 opcode:\n{}", result.code);

    // Verify bit 62 is set: 1<<62 = 4611686018427387904 in decimal
    assert!(
        result.code.contains("4611686018427387904ull"),
        "Missing mixed-precision bit 62 flag in FMA encoding:\n{}",
        result.code
    );
}

#[test]
fn test_wmma_tile_grid_load_pair() {
    // 2×2 tile grid should enable load-pair mode on LDX/LDY
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_tile_grid_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render(&sink, Some("test_wmma_tile")).expect("C codegen failed");

    // Verify load-pair bit (bit 62) is set on LDX and LDY
    // 1<<62 = 4611686018427387904 in decimal
    assert!(
        result.code.contains("AMX(0, (int *)(&data2), 4611686018427387904ull)"),
        "Missing load-pair bit on LDX (opcode 0):\n{}",
        result.code
    );
    assert!(
        result.code.contains("AMX(1, (int *)(&data1), 4611686018427387904ull)"),
        "Missing load-pair bit on LDY (opcode 1):\n{}",
        result.code
    );
}

#[test]
fn test_wmma_tile_grid_multiple_fma() {
    // 2×2 tile grid should emit 4 FMAs with proper encoding
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_tile_grid_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render(&sink, Some("test_wmma_multi_fma")).expect("C codegen failed");

    // Count FMA calls - should be 4 for a 2x2 tile grid
    let fma_count = result.code.matches("AMX(12,").count();
    assert_eq!(fma_count, 4, "Expected 4 FMAs for 2x2 tile grid, got {}:\n{}", fma_count, result.code);

    // Verify FMA encodings for each tile position
    // encoding = fma_flags | (z_row << 20) | (x_off << 10) | y_off
    // where x_off = tx * 64, y_off = ty * 64

    // Tile (0,0): z_row=0, x_off=0, y_off=0 → encoding = 0
    assert!(result.code.contains("AMX(12, 0, 0ull);"), "Missing FMA for tile (0,0):\n{}", result.code);

    // Tile (0,1): z_row=1, x_off=64, y_off=0 → encoding = (1<<20) | (64<<10) | 0 = 1114112
    assert!(result.code.contains("AMX(12, 0, 1114112ull);"), "Missing FMA for tile (0,1):\n{}", result.code);

    // Tile (1,0): z_row=2, x_off=0, y_off=64 → encoding = (2<<20) | (0<<10) | 64 = 2097216
    assert!(result.code.contains("AMX(12, 0, 2097216ull);"), "Missing FMA for tile (1,0):\n{}", result.code);

    // Tile (1,1): z_row=3, x_off=64, y_off=64 → encoding = (3<<20) | (64<<10) | 64 = 3211328
    assert!(result.code.contains("AMX(12, 0, 3211328ull);"), "Missing FMA for tile (1,1):\n{}", result.code);
}
