//! LLVM type and constant string generation.
//!
//! Provides functions for converting Morok types to LLVM IR text.
//! Shared between CPU and GPU backends.

use morok_dtype::{AddrSpace, DType, ScalarDType};
use morok_ir::ConstValue;

/// Convert a DType to LLVM type string.
pub fn ldt(dtype: &DType) -> String {
    match dtype {
        DType::Vector { scalar, count } => {
            format!("<{} x {}>", count, ldt_scalar(*scalar))
        }
        DType::Ptr { base, vcount, .. } if *vcount > 1 => {
            format!("<{} x {}*>", vcount, ldt(base))
        }
        DType::Ptr { base, .. } => {
            format!("{}*", ldt(base))
        }
        DType::Scalar(s) => ldt_scalar(*s).to_string(),
        DType::Image { .. } => "ptr".to_string(),
    }
}

/// Convert a ScalarDType to LLVM type string.
fn ldt_scalar(s: ScalarDType) -> &'static str {
    match s {
        ScalarDType::Bool => "i1",
        ScalarDType::Int8 | ScalarDType::UInt8 => "i8",
        ScalarDType::Int16 | ScalarDType::UInt16 => "i16",
        ScalarDType::Int32 | ScalarDType::UInt32 => "i32",
        ScalarDType::Int64 | ScalarDType::UInt64 | ScalarDType::Index => "i64",
        ScalarDType::Float16 => "half",
        ScalarDType::BFloat16 => "bfloat",
        ScalarDType::Float32 => "float",
        ScalarDType::Float64 => "double",
        ScalarDType::Void => "void",
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => "i8",
    }
}

/// Convert a constant value to LLVM literal string.
pub fn lconst(val: &ConstValue, dtype: &DType) -> String {
    match val {
        ConstValue::Int(i) => i.to_string(),
        ConstValue::UInt(u) => (*u as i64).to_string(),
        ConstValue::Float(f) => format_float(*f, dtype),
        ConstValue::Bool(b) => if *b { "1" } else { "0" }.to_string(),
    }
}

/// Format a float value for LLVM IR.
fn format_float(f: f64, dtype: &DType) -> String {
    let scalar = dtype.base();

    if f.is_nan() {
        // LLVM expects NaN in double-precision hex format for all float types
        return match scalar {
            ScalarDType::Float64 | ScalarDType::Float32 => "0x7FF8000000000000".to_string(),
            ScalarDType::Float16 => "0xH7E00".to_string(),
            ScalarDType::BFloat16 => "0xR7FC0".to_string(),
            _ => "nan".to_string(),
        };
    }

    if f.is_infinite() {
        // LLVM expects infinity in hex format with sign encoded in bits
        return match scalar {
            ScalarDType::Float64 | ScalarDType::Float32 => {
                // Use bit representation (sign is encoded in the high bit)
                // +inf = 0x7FF0000000000000, -inf = 0xFFF0000000000000
                format!("0x{:016X}", f.to_bits())
            }
            ScalarDType::Float16 => {
                // Half precision: +inf = 0x7C00, -inf = 0xFC00
                if f.is_sign_positive() { "0xH7C00".to_string() } else { "0xHFC00".to_string() }
            }
            ScalarDType::BFloat16 => {
                // BFloat16: +inf = 0x7F80, -inf = 0xFF80
                if f.is_sign_positive() { "0xR7F80".to_string() } else { "0xRFF80".to_string() }
            }
            _ => {
                if f.is_sign_positive() {
                    "inf".to_string()
                } else {
                    "-inf".to_string()
                }
            }
        };
    }

    match scalar {
        ScalarDType::Float64 => {
            format!("0x{:016X}", f.to_bits())
        }
        ScalarDType::Float32 => {
            // LLVM expects float32 constants in double-precision hex format
            // Convert to f32 for precision, then back to f64 for LLVM encoding
            let f32_val = f as f32;
            let f64_val = f32_val as f64;
            format!("0x{:016X}", f64_val.to_bits())
        }
        ScalarDType::Float16 => {
            let f32_val = f as f32;
            let half_bits = f32_to_f16_bits(f32_val);
            format!("0xH{:04X}", half_bits)
        }
        ScalarDType::BFloat16 => {
            let f32_val = f as f32;
            let bf16_bits = (f32_val.to_bits() >> 16) as u16;
            format!("0xR{:04X}", bf16_bits)
        }
        _ => format!("{:e}", f),
    }
}

/// Convert f32 to f16 bits (IEEE 754 half precision).
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007FFFFF;

    if exp == 255 {
        if mant == 0 { sign | 0x7C00 } else { sign | 0x7E00 }
    } else if exp > 142 {
        sign | 0x7C00
    } else if exp < 113 {
        if exp < 103 {
            sign
        } else {
            let mant = mant | 0x00800000;
            let shift = 126 - exp;
            sign | ((mant >> shift) as u16)
        }
    } else {
        let new_exp = ((exp - 127 + 15) as u16) << 10;
        let new_mant = (mant >> 13) as u16;
        sign | new_exp | new_mant
    }
}

/// Get LLVM cast instruction name for a type conversion.
pub fn lcast(from: &DType, to: &DType) -> &'static str {
    let from_scalar = from.base();
    let to_scalar = to.base();

    if matches!(from, DType::Ptr { .. }) || matches!(to, DType::Ptr { .. }) {
        return if matches!(from, DType::Ptr { .. }) && matches!(to, DType::Ptr { .. }) {
            "bitcast"
        } else if matches!(from, DType::Ptr { .. }) {
            "ptrtoint"
        } else {
            "inttoptr"
        };
    }

    if from_scalar.is_float() && to_scalar.is_float() {
        return if to_scalar.bytes() > from_scalar.bytes() { "fpext" } else { "fptrunc" };
    }

    if (from_scalar.is_unsigned() || from_scalar.is_bool()) && to_scalar.is_float() {
        return "uitofp";
    }
    if from_scalar.is_signed() && to_scalar.is_float() {
        return "sitofp";
    }

    if from_scalar.is_float() && to_scalar.is_unsigned() {
        return "fptoui";
    }
    if from_scalar.is_float() && (to_scalar.is_signed() || to_scalar == ScalarDType::Index) {
        return "fptosi";
    }

    // Integer-to-integer casts
    let from_bytes = from_scalar.bytes();
    let to_bytes = to_scalar.bytes();

    // Same size: bitcast (handles signed↔unsigned same-size casts)
    if from_bytes == to_bytes {
        return "bitcast";
    }

    // Narrowing: always trunc
    if to_bytes < from_bytes {
        return "trunc";
    }

    // Widening: use zext for unsigned/bool, sext for signed/Index
    if from_scalar.is_unsigned() || from_scalar.is_bool() {
        return "zext";
    }

    // Index type is treated as signed integer for casting purposes
    if from_scalar.is_signed() || from_scalar == ScalarDType::Index {
        return "sext";
    }

    "bitcast"
}

/// Get LLVM address space number.
pub fn addr_space_num(addrspace: AddrSpace) -> u32 {
    match addrspace {
        AddrSpace::Global => 0,
        AddrSpace::Local => 3,
        AddrSpace::Reg => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ldt_scalar() {
        assert_eq!(ldt(&DType::Float32), "float");
        assert_eq!(ldt(&DType::Int32), "i32");
        assert_eq!(ldt(&DType::Bool), "i1");
        assert_eq!(ldt(&DType::Float64), "double");
    }

    #[test]
    fn test_ldt_vector() {
        assert_eq!(ldt(&DType::Float32.vec(4)), "<4 x float>");
        assert_eq!(ldt(&DType::Int32.vec(8)), "<8 x i32>");
    }

    #[test]
    fn test_ldt_ptr() {
        assert_eq!(ldt(&DType::Float32.ptr(None, AddrSpace::Global)), "float*");
        assert_eq!(ldt(&DType::Int32.vec(4).ptr(None, AddrSpace::Global)), "<4 x i32>*");
    }

    #[test]
    fn test_lconst() {
        assert_eq!(lconst(&ConstValue::Int(42), &DType::Int32), "42");
        assert_eq!(lconst(&ConstValue::Bool(true), &DType::Bool), "1");
        assert_eq!(lconst(&ConstValue::Bool(false), &DType::Bool), "0");
    }

    #[test]
    fn test_lcast() {
        assert_eq!(lcast(&DType::Float32, &DType::Float64), "fpext");
        assert_eq!(lcast(&DType::Float64, &DType::Float32), "fptrunc");
        assert_eq!(lcast(&DType::Int32, &DType::Float32), "sitofp");
        assert_eq!(lcast(&DType::UInt32, &DType::Float32), "uitofp");
        assert_eq!(lcast(&DType::Float32, &DType::Int32), "fptosi");
        assert_eq!(lcast(&DType::Int64, &DType::Int32), "trunc");
        assert_eq!(lcast(&DType::Int32, &DType::Int64), "sext");
        assert_eq!(lcast(&DType::UInt32, &DType::UInt64), "zext");
    }

    #[test]
    fn test_lcast_index_type() {
        // Index type (i64) should be treated as signed integer for casting
        assert_eq!(lcast(&DType::Index, &DType::Int32), "trunc");
        assert_eq!(lcast(&DType::Index, &DType::Int64), "bitcast"); // same size (both i64)
        assert_eq!(lcast(&DType::Int32, &DType::Index), "sext");
        // Float to Index
        assert_eq!(lcast(&DType::Float32, &DType::Index), "fptosi");
    }
}
