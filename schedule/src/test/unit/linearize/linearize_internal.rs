use super::*;
use morok_dtype::DType;
use morok_ir::types::ConstValue;
use smallvec::smallvec;

#[test]
fn test_linearize_single_const() {
    let c = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let sink = UOp::sink(vec![c.clone()]);

    let result = linearize(sink.clone());

    assert_eq!(result.len(), 2); // const + sink
    // Const should come before sink
    assert!(matches!(result[0].op(), Op::Const(_)));
    assert!(matches!(result[1].op(), Op::Sink { .. }));
}

#[test]
fn test_linearize_simple_computation() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let sum = a.try_add(&b).unwrap();
    let sink = UOp::sink(vec![sum]);

    let result = linearize(sink);

    // Should have: const, const, add, sink
    assert_eq!(result.len(), 4);
    // Constants should come first (priority -10)
    assert!(matches!(result[0].op(), Op::Const(_)));
    assert!(matches!(result[1].op(), Op::Const(_)));
    // Then binary op
    assert!(matches!(result[2].op(), Op::Binary(_, _, _)));
    // Then sink
    assert!(matches!(result[3].op(), Op::Sink { .. }));
}

#[test]
fn test_linearize_with_range() {
    // Create: for i in range(10): end(value)
    let end_val = UOp::index_const(10);
    let range = UOp::range(end_val, 0);
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let end = value.end(smallvec![range.clone()]);
    let sink = UOp::sink(vec![end]);

    let result = linearize(sink);

    // Verify RANGE comes before END (RANGE priority 5, END priority -5)
    // But RANGE should come after its sources
    let range_pos = result.iter().position(|u| matches!(u.op(), Op::Range { .. }));
    let end_pos = result.iter().position(|u| matches!(u.op(), Op::End { .. }));

    assert!(range_pos.is_some());
    assert!(end_pos.is_some());
    // END depends on RANGE, so RANGE must come before END
    assert!(range_pos.unwrap() < end_pos.unwrap());
}

#[test]
fn test_linearize_preserves_dependencies() {
    // Create a diamond dependency: a + b, where both depend on c
    let c = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let c2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c3 = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let a = c.try_add(&c2).unwrap();
    let b = c.try_add(&c3).unwrap();
    let sum = a.try_add(&b).unwrap();
    let sink = UOp::sink(vec![sum.clone()]);

    let result = linearize(sink);

    // c should appear before both a and b
    let c_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &c));
    let a_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &a));
    let b_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &b));
    let sum_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &sum));

    assert!(c_pos.is_some());
    assert!(a_pos.is_some());
    assert!(b_pos.is_some());
    assert!(sum_pos.is_some());

    // Dependencies: c < a, c < b, a < sum, b < sum
    assert!(c_pos.unwrap() < a_pos.unwrap());
    assert!(c_pos.unwrap() < b_pos.unwrap());
    assert!(a_pos.unwrap() < sum_pos.unwrap());
    assert!(b_pos.unwrap() < sum_pos.unwrap());
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn test_priority_ordering() {
    // Test that priority order is respected: PARAM < default < Range
    assert!(priority::PARAM < priority::DEFAULT);
    assert!(priority::DEFAULT < priority::RANGE);
    assert!(priority::END < priority::DEFAULT);
    assert!(priority::LOAD < priority::DEFAULT);
    assert!(priority::DEFAULT < priority::STORE);
}
