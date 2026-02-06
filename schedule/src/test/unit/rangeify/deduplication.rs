//! Tests for computation deduplication and caching in rangeify.
//!
//! Validates that:
//! - Identical computations are detected and deduplicated
//! - Cache lookups work correctly for binary ops
//! - Reduce results are properly reused
//!
//! Based on Tinygrad's schedule deduplication tests.

use std::sync::Arc;

use morok_ir::{UOp, UOpKey};

// ===== Hash Consing Tests =====

#[test]
fn test_identical_const_dedup() {
    // Same constant created twice should be the same UOp
    let c1 = UOp::native_const(42.0f32);
    let c2 = UOp::native_const(42.0f32);

    // Hash consing should make these pointer-equal
    assert!(Arc::ptr_eq(&c1, &c2), "Identical constants should be deduplicated");
}

#[test]
fn test_identical_binary_op_dedup() {
    // Same binary op with same inputs should deduplicate
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    let add1 = a.try_add(&b).unwrap();
    let add2 = a.try_add(&b).unwrap();

    // Should be pointer-equal due to hash consing
    assert!(Arc::ptr_eq(&add1, &add2), "Identical binary ops should be deduplicated");
}

#[test]
fn test_different_binary_op_not_dedup() {
    // Different operations should not be deduplicated
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    let add = a.try_add(&b).unwrap();
    let mul = a.try_mul(&b).unwrap();

    assert!(!Arc::ptr_eq(&add, &mul), "Different ops should not be deduplicated");
}

// ===== UOpKey Tests =====

#[test]
fn test_uopkey_equality() {
    // UOpKey should compare by Arc pointer
    let a = UOp::native_const(42.0f32);
    let b = a.clone();
    let c = UOp::native_const(42.0f32); // Same value, should be same UOp

    let key_a = UOpKey(a.clone());
    let key_b = UOpKey(b);
    let key_c = UOpKey(c);

    assert_eq!(key_a, key_b, "Clone should have same key");
    assert_eq!(key_a, key_c, "Same value should have same key (hash consing)");
}

#[test]
fn test_uopkey_hash_consistency() {
    use std::collections::HashMap;

    let a = UOp::native_const(42.0f32);
    #[allow(clippy::mutable_key_type)]
    let mut map: HashMap<UOpKey, i32> = HashMap::new();

    map.insert(UOpKey(a.clone()), 100);

    // Lookup with clone
    assert_eq!(map.get(&UOpKey(a.clone())), Some(&100));

    // Lookup with same-value UOp
    let a2 = UOp::native_const(42.0f32);
    assert_eq!(map.get(&UOpKey(a2)), Some(&100));
}

// ===== Computation Graph Deduplication =====

#[test]
fn test_diamond_pattern_dedup() {
    // Diamond pattern: a → [add1, add2] → result
    // The shared 'a' should be computed once
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    let add1 = a.try_add(&b).unwrap();
    let add2 = a.try_add(&c).unwrap();

    // Both reference 'a' - in toposort, 'a' should appear once
    let sink = UOp::sink(vec![add1, add2]);
    let topo = sink.toposort();

    // Count occurrences of 'a'
    let a_count = topo.iter().filter(|u| Arc::ptr_eq(u, &a)).count();
    assert_eq!(a_count, 1, "Shared input should appear once in toposort");
}

#[test]
fn test_reused_intermediate() {
    // Intermediate result used multiple times
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    let sum = a.try_add(&b).unwrap();

    // Use sum twice
    let double = sum.try_add(&sum).unwrap();

    // Toposort should include sum only once
    let topo = double.toposort();
    let sum_count = topo.iter().filter(|u| Arc::ptr_eq(u, &sum)).count();
    assert_eq!(sum_count, 1, "Reused intermediate should appear once");
}

// ===== Cache Behavior =====

#[test]
fn test_cache_different_dtypes() {
    // Same operation but different dtypes should NOT deduplicate
    let a_f32 = UOp::native_const(1.0f32);
    let b_f32 = UOp::native_const(2.0f32);
    let a_f64 = UOp::native_const(1.0f64);
    let b_f64 = UOp::native_const(2.0f64);

    let add_f32 = a_f32.try_add(&b_f32).unwrap();
    let add_f64 = a_f64.try_add(&b_f64).unwrap();

    assert!(!Arc::ptr_eq(&add_f32, &add_f64), "Different dtypes should not deduplicate");
    assert_ne!(add_f32.dtype(), add_f64.dtype());
}

#[test]
fn test_cache_order_matters() {
    // a + b should be same as a + b but not b + a (for non-commutative)
    // Note: Add IS commutative, but sub is not
    let a = UOp::native_const(3.0f32);
    let b = UOp::native_const(1.0f32);

    let sub1 = a.try_sub(&b).unwrap();
    let sub2 = b.try_sub(&a).unwrap();

    assert!(!Arc::ptr_eq(&sub1, &sub2), "Order should matter for non-commutative ops");
}

// ===== Toposort Uniqueness =====

#[test]
fn test_toposort_no_duplicates() {
    // Complex graph should have unique entries in toposort
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    let ab = a.try_add(&b).unwrap();
    let abc = ab.try_add(&c).unwrap();
    let result = abc.try_mul(&ab).unwrap(); // Reuses ab

    let topo = result.toposort();

    // Check no duplicates
    let unique_count = topo.len();
    let mut seen: Vec<Arc<UOp>> = Vec::new();
    for node in &topo {
        if !seen.iter().any(|s| Arc::ptr_eq(s, node)) {
            seen.push(node.clone());
        }
    }
    assert_eq!(seen.len(), unique_count, "Toposort should have no duplicates");
}
