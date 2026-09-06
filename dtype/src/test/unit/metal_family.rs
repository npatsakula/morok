use test_case::test_case;

use super::*;

#[test_case(MetalFamily::Apple(9), "Apple9")]
#[test_case(MetalFamily::Apple(7), "Apple7")]
#[test_case(MetalFamily::Mac2, "Mac2")]
#[test_case(MetalFamily::Unknown, "Unknown")]
fn label_round_trips(family: MetalFamily, label: &str) {
    assert_eq!(family.to_string(), label);
    assert_eq!(MetalFamily::parse(label), Some(family));
}

#[test_case("Apple", None; "missing generation")]
#[test_case("apple9", None; "case sensitive")]
#[test_case("Mac1", None; "unknown mac family")]
#[test_case("Apple12", Some(MetalFamily::Apple(12)); "future generation")]
fn parse_rejects_malformed_labels(label: &str, expected: Option<MetalFamily>) {
    assert_eq!(MetalFamily::parse(label), expected);
}

#[test]
fn simdgroup_matrix_needs_apple7() {
    assert!(MetalFamily::Apple(7).has_simdgroup_matrix());
    assert!(MetalFamily::Apple(9).has_simdgroup_matrix());
    assert!(!MetalFamily::Apple(6).has_simdgroup_matrix());
    assert!(!MetalFamily::Mac2.has_simdgroup_matrix());
    assert!(!MetalFamily::Unknown.has_simdgroup_matrix());
}

#[test]
fn families_order_by_capability() {
    assert!(MetalFamily::Unknown < MetalFamily::Mac2);
    assert!(MetalFamily::Mac2 < MetalFamily::Apple(1));
    assert!(MetalFamily::Apple(8) < MetalFamily::Apple(9));
}
