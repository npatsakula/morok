use super::*;

#[test]
fn test_compute_row_major_strides() {
    // 3D tensor [2, 3, 4]: strides should be [12, 4, 1]
    assert_eq!(compute_row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);

    // 2D matrix [5, 10]: strides should be [10, 1]
    assert_eq!(compute_row_major_strides(&[5, 10]), vec![10, 1]);

    // 1D: stride is [1]
    assert_eq!(compute_row_major_strides(&[100]), vec![1]);
}

#[test]
fn test_build_linear_index() {
    let i = UOp::index_const(2);
    let j = UOp::index_const(3);
    let linear = build_linear_index(&[i, j], &[10, 1]);

    // Should produce: 2*10 + 3 = Add(Mul(2, 10), 3)
    assert!(matches!(linear.op(), Op::Binary(BinaryOp::Add, _, _)));
}

#[test]
fn test_extract_index_dimension_range() {
    use morok_ir::AxisId;
    // Create a RANGE with size 10
    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), morok_ir::AxisType::Loop);

    let dim = extract_index_dimension(&range);
    assert_eq!(dim, Some(10));
}

#[test]
fn test_extract_index_dimension_complex_expression() {
    use morok_ir::AxisId;
    // Create Add(Mul(Range(4), stride), Range(8))
    // Should multiply all range sizes: 4 * 8 = 32
    let r1 = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), morok_ir::AxisType::Loop);
    let r2 = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(1), morok_ir::AxisType::Loop);
    let stride = UOp::index_const(8);
    let mul = UOp::new(Op::Binary(BinaryOp::Mul, r1, stride), DType::Index);
    let add = UOp::new(Op::Binary(BinaryOp::Add, mul, r2), DType::Index);

    let dim = extract_index_dimension(&add);
    assert_eq!(dim, Some(32));
}
