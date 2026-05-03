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
    assert_eq!(ldt(&DType::Float32.ptr(None, AddrSpace::Global)), "ptr");
    assert_eq!(ldt(&DType::Int32.vec(4).ptr(None, AddrSpace::Global)), "ptr");
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
    // Float ↔ Index
    assert_eq!(lcast(&DType::Float32, &DType::Index), "fptosi");
    assert_eq!(lcast(&DType::Index, &DType::Float32), "sitofp");
    assert_eq!(lcast(&DType::Index, &DType::Float64), "sitofp");
}
