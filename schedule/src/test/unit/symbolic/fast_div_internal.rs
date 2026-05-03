use super::*;

#[test]
fn test_magic_unsigned_div_3() {
    // x / 3 for x in 0..=100
    let result = magic_unsigned(100, 3);
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    // Verify for some values
    for x in 0..=100 {
        let expected = x / 3;
        let actual = ((x as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_magic_unsigned_div_7() {
    // x / 7 for x in 0..=1000
    let result = magic_unsigned(1000, 7);
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in 0..=1000 {
        let expected = x / 7;
        let actual = ((x as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_magic_unsigned_div_10() {
    // x / 10 for x in 0..=10000
    let result = magic_unsigned(10000, 10);
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in (0..=10000).step_by(100) {
        let expected = x / 10;
        let actual = ((x as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_is_power_of_two() {
    assert!(is_power_of_two(1));
    assert!(is_power_of_two(2));
    assert!(is_power_of_two(4));
    assert!(is_power_of_two(8));
    assert!(is_power_of_two(1024));

    assert!(!is_power_of_two(0));
    assert!(!is_power_of_two(-1));
    assert!(!is_power_of_two(3));
    assert!(!is_power_of_two(6));
    assert!(!is_power_of_two(7));
}

#[test]
fn test_magic_unsigned_invalid() {
    // Zero divisor
    assert!(magic_unsigned(100, 0).is_none());

    // Negative divisor
    assert!(magic_unsigned(100, -5).is_none());
}

#[test]
fn test_magic_unsigned_div_6_factorization() {
    // x / 6 for x in 0..=1000
    // Tests power-of-two factorization: 6 = 2 * 3
    // Division by 6 should become: (x >> 1) / 3
    let result = magic_unsigned(500, 3); // After shift, max is 500
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in 0..=1000 {
        let expected = x / 6;
        // Simulate factorization: (x >> 1) then magic divide by 3
        let shifted = x >> 1;
        let actual = ((shifted as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_magic_unsigned_div_12_factorization() {
    // x / 12 for x in 0..=1200
    // Tests power-of-two factorization: 12 = 4 * 3
    // Division by 12 should become: (x >> 2) / 3
    let result = magic_unsigned(300, 3); // After shift by 2, max is 300
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in 0..=1200 {
        let expected = x / 12;
        // Simulate factorization: (x >> 2) then magic divide by 3
        let shifted = x >> 2;
        let actual = ((shifted as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}
