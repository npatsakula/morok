use proptest::prelude::*;

use crate::metal::device::PointerRegistry;

fn registry() -> PointerRegistry<&'static str> {
    let mut registry = PointerRegistry::default();
    registry.insert(0x1000, 0x100, "first");
    registry.insert(0x4000, 0x40, "second");
    registry.insert(0x9000, 0x10, "third");
    registry
}

#[test_case::test_case(0x1000 => ("first", 0); "base")]
#[test_case::test_case(0x1001 => ("first", 1); "base plus one")]
#[test_case::test_case(0x10ff => ("first", 0xff); "last byte")]
#[test_case::test_case(0x4020 => ("second", 0x20); "middle range")]
#[test_case::test_case(0x900f => ("third", 0xf); "last range")]
fn resolves_addresses_inside_live_ranges(address: usize) -> (&'static str, usize) {
    let registry = registry();
    let (record, offset) = registry.resolve(address).expect("inside a range");
    (*record, offset)
}

#[test_case::test_case(0x0fff, "nearest below none", "nearest above [0x1000, 0x1100)"; "before first")]
#[test_case::test_case(0x1100, "nearest below [0x1000, 0x1100)", "nearest above [0x4000, 0x4040)"; "one past first")]
#[test_case::test_case(0x9010, "nearest below [0x9000, 0x9010)", "nearest above none"; "past last")]
fn rejects_addresses_outside_live_ranges(address: usize, below: &str, above: &str) {
    let error = registry().resolve(address).expect_err("outside every range");
    let message = format!("{error}");
    assert!(matches!(error, crate::Error::Runtime { .. }));
    assert!(message.contains(below) && message.contains(above), "{message}");
}

#[test]
fn remove_is_idempotent_and_unregisters() {
    let mut registry = registry();
    assert_eq!(registry.remove(0x4000), Some("second"));
    assert_eq!(registry.remove(0x4000), None);
    assert!(registry.resolve(0x4000).is_err());
    assert!(registry.resolve(0x1000).is_ok());
}

proptest! {
    /// Resolution agrees with a linear scan over disjoint ranges.
    #[test]
    fn resolve_matches_linear_scan(
        lens in proptest::collection::vec(1usize..64, 1..8),
        gaps in proptest::collection::vec(0usize..16, 8),
        probe in 0usize..1024,
    ) {
        let mut registry = PointerRegistry::default();
        let mut ranges = Vec::new();
        let mut base = 1usize;
        for (i, len) in lens.iter().enumerate() {
            base += gaps[i];
            registry.insert(base, *len, i);
            ranges.push((base, *len, i));
            base += *len;
        }
        let expected = ranges.iter().find(|(b, len, _)| *b <= probe && probe < b + len).map(|(b, _, i)| (*i, probe - b));
        let actual = registry.resolve(probe).ok().map(|(i, off)| (*i, off));
        prop_assert_eq!(actual, expected);
    }
}
