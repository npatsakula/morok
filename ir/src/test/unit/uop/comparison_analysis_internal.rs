use super::*;

#[test]
fn test_lt_analysis() {
    // x in [100, 200], y in [50, 50]
    // x < y should be false
    assert_eq!(
        ComparisonAnalyzer::analyze_with_ranges(
            BinaryOp::Lt,
            ConstValue::Int(100),
            ConstValue::Int(200),
            ConstValue::Int(50),
            ConstValue::Int(50)
        ),
        Some(false)
    );

    // x in [0, 10], y in [20, 30]
    // x < y should be true
    assert_eq!(
        ComparisonAnalyzer::analyze_with_ranges(
            BinaryOp::Lt,
            ConstValue::Int(0),
            ConstValue::Int(10),
            ConstValue::Int(20),
            ConstValue::Int(30)
        ),
        Some(true)
    );
}

#[test]
fn test_eq_analysis() {
    // Non-overlapping ranges
    assert_eq!(
        ComparisonAnalyzer::analyze_with_ranges(
            BinaryOp::Eq,
            ConstValue::Int(0),
            ConstValue::Int(10),
            ConstValue::Int(20),
            ConstValue::Int(30)
        ),
        Some(false)
    );

    // Same constant
    assert_eq!(
        ComparisonAnalyzer::analyze_with_ranges(
            BinaryOp::Eq,
            ConstValue::Int(5),
            ConstValue::Int(5),
            ConstValue::Int(5),
            ConstValue::Int(5)
        ),
        Some(true)
    );
}
