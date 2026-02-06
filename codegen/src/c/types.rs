//! C type mapping and constant rendering for the C codegen backend.

use std::collections::BTreeSet;
use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::{ConstValue, UOp};

/// Convert a DType to its C scalar type string.
pub fn c_scalar(s: ScalarDType) -> &'static str {
    match s {
        ScalarDType::Bool => "_Bool",
        ScalarDType::Int8 => "char",
        ScalarDType::UInt8 => "unsigned char",
        ScalarDType::Int16 => "short",
        ScalarDType::UInt16 => "unsigned short",
        ScalarDType::Int32 => "int",
        ScalarDType::UInt32 => "unsigned int",
        ScalarDType::Int64 | ScalarDType::Index => "long long",
        ScalarDType::UInt64 => "unsigned long long",
        ScalarDType::Float16 => "_Float16",
        ScalarDType::BFloat16 => "__bf16",
        ScalarDType::Float32 => "float",
        ScalarDType::Float64 => "double",
        ScalarDType::Void => "void",
        ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => "unsigned char",
    }
}

/// Convert a DType to its C type string.
///
/// For vectors, returns the typedef name (e.g. `float4`).
/// For pointers, returns `T*`.
pub fn c_dtype(dtype: &DType) -> String {
    match dtype {
        DType::Scalar(s) => c_scalar(*s).to_string(),
        DType::Vector { scalar, count } => {
            format!("{}{}", c_scalar(*scalar), count)
        }
        DType::Ptr { base, .. } => format!("{}*", c_dtype(base)),
        DType::Image { .. } => "void*".to_string(),
    }
}

/// Render a constant value as a C literal.
pub fn c_const(val: &ConstValue, dtype: &DType) -> String {
    match val {
        ConstValue::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        ConstValue::Int(i) => {
            let base = dtype.base();
            match base {
                ScalarDType::Int64 | ScalarDType::Index => format!("{i}LL"),
                ScalarDType::UInt64 => format!("{}ULL", *i as u64),
                _ => i.to_string(),
            }
        }
        ConstValue::UInt(u) => {
            let base = dtype.base();
            match base {
                ScalarDType::UInt64 => format!("{u}ULL"),
                ScalarDType::UInt32 => format!("{u}u"),
                _ => u.to_string(),
            }
        }
        ConstValue::Float(f) => c_float(*f, dtype),
    }
}

/// Render a float constant as a C literal.
fn c_float(f: f64, dtype: &DType) -> String {
    let base = dtype.base();

    if f.is_nan() {
        return match base {
            ScalarDType::Float32 => "__builtin_nanf(\"\")".to_string(),
            ScalarDType::Float64 => "__builtin_nan(\"\")".to_string(),
            ScalarDType::Float16 => "((_Float16)__builtin_nanf(\"\"))".to_string(),
            _ => "__builtin_nanf(\"\")".to_string(),
        };
    }

    if f.is_infinite() {
        let sign = if f.is_sign_negative() { "-" } else { "" };
        return match base {
            ScalarDType::Float32 => format!("{sign}__builtin_inff()"),
            ScalarDType::Float64 => format!("{sign}__builtin_inf()"),
            ScalarDType::Float16 => format!("((_Float16){sign}__builtin_inff())"),
            _ => format!("{sign}__builtin_inff()"),
        };
    }

    match base {
        ScalarDType::Float32 => {
            let f32_val = f as f32;
            if f32_val == 0.0 && f.is_sign_negative() {
                "-0.0f".to_string()
            } else if f32_val.fract() == 0.0 && f32_val.abs() < 1e15 {
                format!("{:.1}f", f32_val)
            } else {
                format!("{:e}f", f32_val)
            }
        }
        ScalarDType::Float64 => {
            if f == 0.0 && f.is_sign_negative() {
                "-0.0".to_string()
            } else if f.fract() == 0.0 && f.abs() < 1e15 {
                format!("{:.1}", f)
            } else {
                format!("{:e}", f)
            }
        }
        ScalarDType::Float16 => {
            let f32_val = f as f32;
            format!("((_Float16){}f)", format_f32_literal(f32_val))
        }
        ScalarDType::BFloat16 => {
            let f32_val = f as f32;
            format!("((__bf16){}f)", format_f32_literal(f32_val))
        }
        _ => format!("{:e}f", f as f32),
    }
}

/// Format an f32 value as a simple literal.
fn format_f32_literal(f: f32) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 { format!("{:.1}", f) } else { format!("{:e}", f) }
}

/// Render a vector constant as a C initializer.
pub fn c_vconst(values: &[ConstValue], dtype: &DType) -> String {
    let scalar_dtype = dtype.scalar_dtype();
    let elements: Vec<String> = values.iter().map(|v| c_const(v, &scalar_dtype)).collect();
    format!("({}){{{}}}", c_dtype(dtype), elements.join(", "))
}

/// Collect all vector types used in the linearized instruction stream
/// and return the necessary typedef declarations.
pub fn collect_vector_typedefs(nodes: &[Arc<UOp>]) -> Vec<String> {
    let mut seen = BTreeSet::new();

    for node in nodes {
        collect_vec_dtype(&node.dtype(), &mut seen);
        // Also check child dtypes for cases where vectors appear as operands
        for child in node.op().children() {
            collect_vec_dtype(&child.dtype(), &mut seen);
        }
    }

    seen.into_iter()
        .map(|(scalar, count)| {
            let scalar_name = c_scalar(scalar);
            let vec_name = format!("{}{}", scalar_name, count);
            let alignment = scalar.bytes() * count;
            // Use next power of two for alignment
            let alignment = alignment.next_power_of_two();
            format!("typedef {scalar_name} {vec_name} __attribute__((aligned({alignment}),ext_vector_type({count})));",)
        })
        .collect()
}

fn collect_vec_dtype(dtype: &DType, seen: &mut BTreeSet<(ScalarDType, usize)>) {
    if let DType::Vector { scalar, count } = dtype {
        seen.insert((*scalar, *count));
    }
}

/// Get the C math function name for the given unary op suffix and dtype.
/// Returns function name with type suffix (e.g. `sqrtf` for float, `sqrt` for double).
pub fn c_math_fn(name: &str, dtype: &DType) -> String {
    let base = dtype.base();
    match base {
        ScalarDType::Float32 => format!("{name}f"),
        ScalarDType::Float64 => name.to_string(),
        // For half/bfloat, cast through float
        _ => format!("{name}f"),
    }
}

/// Get the identity element for a reduce operation as a C literal.
pub fn c_reduce_identity(op: morok_ir::ReduceOp, dtype: &DType) -> String {
    use morok_ir::ReduceOp;
    let is_f64 = matches!(dtype.base(), ScalarDType::Float64);
    match op {
        ReduceOp::Add => {
            if dtype.is_float() {
                if is_f64 { "0.0" } else { "0.0f" }.to_string()
            } else {
                "0".to_string()
            }
        }
        ReduceOp::Mul => {
            if dtype.is_float() {
                if is_f64 { "1.0" } else { "1.0f" }.to_string()
            } else {
                "1".to_string()
            }
        }
        ReduceOp::Max => {
            if dtype.is_float() {
                format!("-{}", c_math_fn("__builtin_inf", dtype))
            } else if dtype.is_signed() {
                match dtype.base() {
                    ScalarDType::Int64 | ScalarDType::Index => format!("{}LL", i64::MIN),
                    ScalarDType::Int32 => format!("{}", i32::MIN),
                    ScalarDType::Int16 => format!("{}", i16::MIN),
                    ScalarDType::Int8 => format!("{}", i8::MIN),
                    _ => "0".to_string(),
                }
            } else {
                "0".to_string()
            }
        }
        ReduceOp::Min => {
            if dtype.is_float() {
                c_math_fn("__builtin_inf", dtype)
            } else if dtype.is_signed() {
                match dtype.base() {
                    ScalarDType::Int64 | ScalarDType::Index => format!("{}LL", i64::MAX),
                    ScalarDType::Int32 => format!("{}", i32::MAX),
                    ScalarDType::Int16 => format!("{}", i16::MAX),
                    ScalarDType::Int8 => format!("{}", i8::MAX),
                    _ => "0".to_string(),
                }
            } else {
                match dtype.base() {
                    ScalarDType::UInt64 => format!("{}ULL", u64::MAX),
                    ScalarDType::UInt32 => format!("{}u", u32::MAX),
                    ScalarDType::UInt16 => format!("{}", u16::MAX),
                    ScalarDType::UInt8 => format!("{}", u8::MAX),
                    _ => "0".to_string(),
                }
            }
        }
    }
}

/// Get the C cast expression for converting between types.
pub fn c_cast(val: &str, from: &DType, to: &DType) -> String {
    let to_str = c_dtype(to);
    // For pointer casts, use void* intermediate
    if matches!(from, DType::Ptr { .. }) && !matches!(to, DType::Ptr { .. }) {
        return format!("({})(long long){}", to_str, val);
    }
    format!("({}){}", to_str, val)
}
