//! C type mapping and constant rendering for the C-family codegen backend.
//!
//! Every function takes the [`CDialect`]; the `Clang` arms are the original
//! clang C spellings, the `Metal` arms the MSL ones.

use std::collections::BTreeSet;
use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, ScalarDType, cast::committed_float_bits};

use super::dialect::{CDialect, addr_qualifier};
use crate::common::value_width;
use crate::{Error, Result};
use svod_ir::{ConstValue, UOp};

/// Convert a scalar dtype to its C / MSL type name.
pub fn c_scalar(s: ScalarDType, dialect: CDialect) -> &'static str {
    match dialect {
        CDialect::Clang => match s {
            ScalarDType::WeakInt | ScalarDType::WeakFloat => panic!("weak dtype reached C rendering"),
            ScalarDType::Bool => "_Bool",
            ScalarDType::Int8 => "signed char",
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
            ScalarDType::FP8E4M3FNUZ | ScalarDType::FP8E5M2FNUZ => panic!("FNUZ reached C rendering"),
        },
        CDialect::Metal => match s {
            ScalarDType::WeakInt | ScalarDType::WeakFloat => panic!("weak dtype reached MSL rendering"),
            ScalarDType::Bool => "bool",
            ScalarDType::Int8 => "char",
            ScalarDType::UInt8 => "uchar",
            ScalarDType::Int16 => "short",
            ScalarDType::UInt16 => "ushort",
            ScalarDType::Int32 => "int",
            ScalarDType::UInt32 => "uint",
            ScalarDType::Int64 | ScalarDType::Index => "long",
            ScalarDType::UInt64 => "ulong",
            ScalarDType::Float16 => "half",
            ScalarDType::BFloat16 => "bfloat",
            ScalarDType::Float32 => "float",
            ScalarDType::Void => "void",
            ScalarDType::Float64
            | ScalarDType::FP8E4M3
            | ScalarDType::FP8E5M2
            | ScalarDType::FP8E4M3FNUZ
            | ScalarDType::FP8E5M2FNUZ => {
                panic!("{s:?} reached MSL rendering; reject_unsupported_metal_dtypes must run first")
            }
        },
    }
}

/// Space-free identifier base for vector type names (e.g. `uchar4`, `llong2`).
/// MSL vector names are the scalar name plus the lane count.
fn c_vector_base(s: ScalarDType, dialect: CDialect) -> &'static str {
    match dialect {
        CDialect::Metal => c_scalar(s, dialect),
        CDialect::Clang => match s {
            ScalarDType::WeakInt | ScalarDType::WeakFloat => panic!("weak dtype reached C vector rendering"),
            ScalarDType::Bool => "bool",
            ScalarDType::Int8 => "schar",
            ScalarDType::UInt8 | ScalarDType::FP8E4M3 | ScalarDType::FP8E5M2 => "uchar",
            ScalarDType::FP8E4M3FNUZ | ScalarDType::FP8E5M2FNUZ => panic!("FNUZ reached C vector rendering"),
            ScalarDType::Int16 => "short",
            ScalarDType::UInt16 => "ushort",
            ScalarDType::Int32 => "int",
            ScalarDType::UInt32 => "uint",
            ScalarDType::Int64 | ScalarDType::Index => "llong",
            ScalarDType::UInt64 => "ullong",
            ScalarDType::Float16 => "half",
            ScalarDType::BFloat16 => "bhalf",
            ScalarDType::Float32 => "float",
            ScalarDType::Float64 => "double",
            ScalarDType::Void => "void",
        },
    }
}

/// Convert a DType to its C type string.
///
/// For vectors, returns the vector type name (e.g. `float4`). For pointers,
/// returns the unqualified `T*`; use [`pointer_type`] where the address space
/// matters.
pub fn c_dtype(dtype: &DType, dialect: CDialect) -> String {
    match dtype {
        DType::Scalar(s) => c_scalar(*s, dialect).to_string(),
        DType::Vector { scalar, count } => format!("{}{}", c_vector_base(*scalar, dialect), count),
        DType::Ptr { base, .. } => format!("{}*", c_dtype(base, dialect)),
        DType::Image { .. } => "void*".to_string(),
    }
}

/// Pointer type to `elem` in `addrspace`: `T*` for C, `device T*` etc. for MSL.
pub fn pointer_type(elem: &DType, addrspace: Option<AddrSpace>, dialect: CDialect) -> String {
    format!("{}{}*", addr_qualifier(dialect, addrspace), c_dtype(elem, dialect))
}

/// Render a constant value as a C literal.
pub fn c_const(val: &ConstValue, dtype: &DType, dialect: CDialect) -> String {
    let (ll, ull) = match dialect {
        CDialect::Clang => ("LL", "ULL"),
        CDialect::Metal => ("L", "UL"),
    };
    match val {
        ConstValue::Invalid => panic!("Invalid reached C constant rendering"),
        ConstValue::Bool(b) => if *b { "1" } else { "0" }.to_string(),
        ConstValue::Int(i) => match dtype.base() {
            ScalarDType::Int64 | ScalarDType::Index => format!("{i}{ll}"),
            ScalarDType::UInt64 => format!("{}{ull}", *i as u64),
            _ => i.to_string(),
        },
        ConstValue::UInt(u) => match dtype.base() {
            ScalarDType::UInt64 => format!("{u}{ull}"),
            ScalarDType::UInt32 => format!("{u}u"),
            _ => u.to_string(),
        },
        ConstValue::Float(f) => c_float(*f, dtype, dialect),
    }
}

/// Render a float constant as a C literal.
fn c_float(f: f64, dtype: &DType, dialect: CDialect) -> String {
    let base = dtype.base();

    if base.is_fp8() {
        return committed_float_bits(f, base).expect("FP8 constant was committed by IR construction").to_string();
    }

    if f.is_nan() {
        return match (dialect, base) {
            (CDialect::Clang, ScalarDType::Float64) => "__builtin_nan(\"\")".to_string(),
            (CDialect::Clang, ScalarDType::Float16) => "((_Float16)__builtin_nanf(\"\"))".to_string(),
            (CDialect::Clang, _) => "__builtin_nanf(\"\")".to_string(),
            (CDialect::Metal, ScalarDType::Float16) => "((half)NAN)".to_string(),
            (CDialect::Metal, ScalarDType::BFloat16) => "((bfloat)NAN)".to_string(),
            (CDialect::Metal, _) => "NAN".to_string(),
        };
    }

    if f.is_infinite() {
        let sign = if f.is_sign_negative() { "-" } else { "" };
        return match (dialect, base) {
            (CDialect::Clang, ScalarDType::Float64) => format!("{sign}__builtin_inf()"),
            (CDialect::Clang, ScalarDType::Float16) => format!("((_Float16){sign}__builtin_inff())"),
            (CDialect::Clang, _) => format!("{sign}__builtin_inff()"),
            (CDialect::Metal, ScalarDType::Float16) => format!("((half){sign}INFINITY)"),
            (CDialect::Metal, ScalarDType::BFloat16) => format!("((bfloat){sign}INFINITY)"),
            (CDialect::Metal, _) => format!("{sign}INFINITY"),
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
        ScalarDType::Float16 | ScalarDType::BFloat16 => {
            format!("(({}){}f)", c_scalar(base, dialect), format_f32_literal(f as f32))
        }
        _ => format!("{:e}f", f as f32),
    }
}

/// Format an f32 value as a simple literal.
fn format_f32_literal(f: f32) -> String {
    if f.fract() == 0.0 && f.abs() < 1e15 { format!("{:.1}", f) } else { format!("{:e}", f) }
}

/// Render a vector constant: a C compound literal or an MSL constructor call.
pub fn c_vconst(values: &[ConstValue], dtype: &DType, dialect: CDialect) -> String {
    let scalar_dtype = dtype.scalar_dtype();
    let elements: Vec<String> = values.iter().map(|v| c_const(v, &scalar_dtype, dialect)).collect();
    vector_literal(&c_dtype(dtype, dialect), &elements.join(", "), dialect)
}

/// `(float4){a, b}` in C; `float4(a, b)` in MSL.
pub fn vector_literal(vector_type: &str, elements: &str, dialect: CDialect) -> String {
    match dialect {
        CDialect::Clang => format!("({vector_type}){{{elements}}}"),
        CDialect::Metal => format!("{vector_type}({elements})"),
    }
}

/// Collect all vector types used in the linearized instruction stream
/// and return the necessary clang typedef declarations (MSL vectors are
/// native, so this is Clang-only).
pub fn collect_vector_typedefs(nodes: &[Arc<UOp>]) -> Vec<String> {
    let dialect = CDialect::Clang;
    let mut seen = BTreeSet::new();

    for node in nodes {
        collect_vec_dtype(&node.dtype(), &mut seen);
        // Grouped memory keeps a scalar dtype and carries its lane count in
        // shape. Only LOAD and STACK synthesize a shape-derived C vector type
        // after devectorization; probing control-flow shapes can recurse
        // through RANGE/END dependencies.
        let count = value_width(node);
        if count > 1 && node.dtype().base() != ScalarDType::Void && !matches!(node.dtype(), DType::Ptr { .. }) {
            seen.insert((node.dtype().base(), count));
        }
        // Also check child dtypes for cases where vectors appear as operands
        for child in node.op().children() {
            collect_vec_dtype(&child.dtype(), &mut seen);
        }
    }

    seen.into_iter()
        .map(|(scalar, count)| {
            // Bool can't be used as ext_vector_type base; store as unsigned char
            let storage_scalar = if scalar == ScalarDType::Bool { "unsigned char" } else { c_scalar(scalar, dialect) };
            let vec_name = format!("{}{}", c_vector_base(scalar, dialect), count);
            let bytes = scalar.bytes() * count;
            let alignment = if scalar == ScalarDType::Bool { 1 } else { 1usize << bytes.ilog2() };
            format!(
                "typedef {storage_scalar} {vec_name} __attribute__((aligned({alignment}),ext_vector_type({count})));",
            )
        })
        .collect()
}

fn collect_vec_dtype(dtype: &DType, seen: &mut BTreeSet<(ScalarDType, usize)>) {
    match dtype {
        DType::Vector { scalar, count } => {
            seen.insert((*scalar, *count));
        }
        DType::Ptr { base, .. } => collect_vec_dtype(base, seen),
        _ => {}
    }
}

/// MSL has no `double` and no fp8 storage type, and only 2/3/4-lane vectors.
pub fn reject_unsupported_metal_dtypes(nodes: &[Arc<UOp>]) -> Result<()> {
    for node in nodes {
        let dtype = node.dtype();
        let base = dtype.base();
        if base == ScalarDType::Float64 {
            return Err(Error::TypeError {
                reason: format!("Metal renderer does not support Float64 on uop {}; MSL has no double type", node.id),
            });
        }
        if base.is_fp8() {
            return Err(Error::TypeError {
                reason: format!("Metal renderer does not support {base:?} on uop {}; MSL has no fp8 type", node.id),
            });
        }
        // An empty STACK has width 0 and renders nothing.
        let width = value_width(node);
        if width > 4 {
            return Err(Error::TypeError {
                reason: format!(
                    "Metal renderer supports only 2-, 3- and 4-component vectors; got {width} lanes on uop {}",
                    node.id
                ),
            });
        }
    }
    Ok(())
}

/// Math builtins with a C-family spelling in both dialects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathFn {
    Sqrt,
    Fma,
    Fmod,
    Fmax,
    Fmin,
    Pow,
    Fabs,
    Exp,
    Exp2,
    Log,
    Log2,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
    Trunc,
    Rint,
    Erf,
}

impl MathFn {
    fn stem(self) -> &'static str {
        match self {
            MathFn::Sqrt => "sqrt",
            MathFn::Fma => "fma",
            MathFn::Fmod => "fmod",
            MathFn::Fmax => "fmax",
            MathFn::Fmin => "fmin",
            MathFn::Pow => "pow",
            MathFn::Fabs => "fabs",
            MathFn::Exp => "exp",
            MathFn::Exp2 => "exp2",
            MathFn::Log => "log",
            MathFn::Log2 => "log2",
            MathFn::Sin => "sin",
            MathFn::Cos => "cos",
            MathFn::Tan => "tan",
            MathFn::Floor => "floor",
            MathFn::Ceil => "ceil",
            MathFn::Trunc => "trunc",
            MathFn::Rint => "rint",
            MathFn::Erf => "erf",
        }
    }
}

/// `__builtin_{stem}f` (f32/half/bfloat) or `__builtin_{stem}` (f64) for
/// clang; the overloaded `metal::` function for MSL, where only `sin` takes the
/// `precise::` spelling (tinygrad parity; everything else is IEEE under
/// `-fno-fast-math`).
pub fn math_fn(f: MathFn, dtype: &DType, dialect: CDialect) -> String {
    match dialect {
        CDialect::Clang => format!("__builtin_{}{}", f.stem(), clang_float_suffix(dtype)),
        CDialect::Metal if f == MathFn::Sin => "precise::sin".to_string(),
        CDialect::Metal => f.stem().to_string(),
    }
}

fn clang_float_suffix(dtype: &DType) -> &'static str {
    if dtype.base() == ScalarDType::Float64 { "" } else { "f" }
}

/// Get the identity element for a reduce operation as a C literal.
pub fn c_reduce_identity(op: svod_ir::ReduceOp, dtype: &DType, dialect: CDialect) -> String {
    use svod_ir::ReduceOp;
    let is_f64 = matches!(dtype.base(), ScalarDType::Float64);
    let inf = match dialect {
        CDialect::Clang => format!("__builtin_inf{}()", clang_float_suffix(dtype)),
        CDialect::Metal => "INFINITY".to_string(),
    };
    let (ll, ull) = match dialect {
        CDialect::Clang => ("LL", "ULL"),
        CDialect::Metal => ("L", "UL"),
    };
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
                format!("-{inf}")
            } else if dtype.is_signed() {
                match dtype.base() {
                    ScalarDType::Int64 | ScalarDType::Index => format!("{}{ll}", i64::MIN),
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
                inf
            } else if dtype.is_signed() {
                match dtype.base() {
                    ScalarDType::Int64 | ScalarDType::Index => format!("{}{ll}", i64::MAX),
                    ScalarDType::Int32 => format!("{}", i32::MAX),
                    ScalarDType::Int16 => format!("{}", i16::MAX),
                    ScalarDType::Int8 => format!("{}", i8::MAX),
                    _ => "0".to_string(),
                }
            } else {
                match dtype.base() {
                    ScalarDType::UInt64 => format!("{}{ull}", u64::MAX),
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
pub fn c_cast(val: &str, from: &DType, to: &DType, dialect: CDialect) -> String {
    let to_str = c_dtype(to, dialect);
    // Pointer-to-integer casts go through the pointer-width integer type.
    if matches!(from, DType::Ptr { .. }) && !matches!(to, DType::Ptr { .. }) {
        let word = match dialect {
            CDialect::Clang => "long long",
            CDialect::Metal => "long",
        };
        return format!("({to_str})({word}){val}");
    }
    format!("({to_str}){val}")
}
