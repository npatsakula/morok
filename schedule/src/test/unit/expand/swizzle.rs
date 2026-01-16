//! Tests for swizzle algorithm helpers.
//!
//! These test the core algorithms from Tinygrad's expander.py:8-20:
//! - expand_arg_to_idx: linear index from axis positions
//! - choices_from_args: generate all axis combinations
//! - swizzle_args: compute GEP indices for axis remapping

use std::collections::HashMap;

use crate::expand::{choices_from_args, expand_arg_to_idx, swizzle_args};

// =============================================================================
// expand_arg_to_idx Tests
// =============================================================================

#[test]
fn test_expand_arg_to_idx_single() {
    // Single axis: [(0, 4)], position {0: 2} → index 2
    let args = vec![(0, 4)];
    let mut rpk = HashMap::new();
    rpk.insert(0, 2);
    assert_eq!(expand_arg_to_idx(&args, &rpk), 2);
}

#[test]
fn test_expand_arg_to_idx_multi() {
    // Multi-axis row-major: [(0, 2), (1, 3)]
    // Position {0: 1, 1: 2} → 1*3 + 2 = 5
    let args = vec![(0, 2), (1, 3)];
    let mut rpk = HashMap::new();
    rpk.insert(0, 1);
    rpk.insert(1, 2);
    assert_eq!(expand_arg_to_idx(&args, &rpk), 5);
}

#[test]
fn test_expand_arg_to_idx_missing_axis() {
    // Missing axis defaults to 0: [(0, 4), (1, 3)], {0: 2} → 2*3 + 0 = 6
    let args = vec![(0, 4), (1, 3)];
    let mut rpk = HashMap::new();
    rpk.insert(0, 2);
    assert_eq!(expand_arg_to_idx(&args, &rpk), 6);
}

#[test]
fn test_expand_arg_to_idx_three_axes() {
    // Three axes: [(0, 2), (1, 3), (2, 4)]
    // Position {0: 1, 1: 2, 2: 3} → 1*12 + 2*4 + 3 = 23
    let args = vec![(0, 2), (1, 3), (2, 4)];
    let mut rpk = HashMap::new();
    rpk.insert(0, 1);
    rpk.insert(1, 2);
    rpk.insert(2, 3);
    assert_eq!(expand_arg_to_idx(&args, &rpk), 23);
}

// =============================================================================
// choices_from_args Tests
// =============================================================================

#[test]
fn test_choices_from_args_single() {
    // Single axis [(0, 3)] → 3 combinations
    let args = vec![(0, 3)];
    let choices = choices_from_args(&args);
    assert_eq!(choices.len(), 3);
    assert_eq!(choices[0].get(&0), Some(&0));
    assert_eq!(choices[1].get(&0), Some(&1));
    assert_eq!(choices[2].get(&0), Some(&2));
}

#[test]
fn test_choices_from_args_multi() {
    // Two axes [(0, 2), (1, 3)] → 6 combinations (cartesian product)
    let args = vec![(0, 2), (1, 3)];
    let choices = choices_from_args(&args);
    assert_eq!(choices.len(), 6);

    // Verify it's cartesian product in order: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    let expected: Vec<(usize, usize)> = vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)];
    for (i, (ax0, ax1)) in expected.into_iter().enumerate() {
        assert_eq!(choices[i].get(&0), Some(&ax0), "mismatch at index {}", i);
        assert_eq!(choices[i].get(&1), Some(&ax1), "mismatch at index {}", i);
    }
}

#[test]
fn test_choices_from_args_empty() {
    // Empty args → single empty choice
    let args: Vec<(usize, usize)> = vec![];
    let choices = choices_from_args(&args);
    assert_eq!(choices.len(), 1);
    assert!(choices[0].is_empty());
}

// =============================================================================
// swizzle_args Tests
// =============================================================================

#[test]
fn test_swizzle_identity() {
    // Same cargs and eargs → identity indices: 0, 1, 2, 3
    let args = vec![(0, 4)];
    let indices = swizzle_args(&args, &args, &[]);
    assert_eq!(indices, vec![0, 1, 2, 3]);
}

#[test]
fn test_swizzle_with_exclude() {
    // Exclude args get zeroed: cargs=[(0,2),(1,2)], eargs=[(0,2),(1,2)], exclude=[1]
    // This zeros axis 1, so indices = [0, 0, 2, 2]
    let cargs = vec![(0, 2), (1, 2)];
    let eargs = vec![(0, 2), (1, 2)];
    let indices = swizzle_args(&cargs, &eargs, &[1]);
    assert_eq!(indices, vec![0, 0, 2, 2]);
}

#[test]
fn test_swizzle_different_axes() {
    // Different axis ordering: cargs=[(1,2)], eargs=[(0,2),(1,2)]
    // cargs iterates axis 1: {1:0}, {1:1}
    // With eargs layout, axis 0 defaults to 0:
    // {1:0} → idx=0, {1:1} → idx=1
    let cargs = vec![(1, 2)];
    let eargs = vec![(0, 2), (1, 2)];
    let indices = swizzle_args(&cargs, &eargs, &[]);
    assert_eq!(indices, vec![0, 1]);
}

#[test]
fn test_swizzle_subset_axes() {
    // Contract subset: cargs=[(0,4)], eargs=[(0,2),(1,2)]
    // cargs enumerates 0..4, but eargs has 2×2=4 elements with different layout
    // cargs position 0: {0:0} → eargs index = 0*2 + 0 = 0
    // cargs position 1: {0:1} → eargs index = 1*2 + 0 = 2 (axis 1 defaults to 0)
    // cargs position 2: {0:2} → out of bounds for eargs[0,2], wraps to 0
    // cargs position 3: {0:3} → out of bounds, wraps to 2
    //
    // Actually, this test doesn't make semantic sense since axis layouts differ.
    // Let's test a sensible case instead.
    let cargs = vec![(0, 2)];
    let eargs = vec![(0, 2), (1, 2)];
    let indices = swizzle_args(&cargs, &eargs, &[]);
    // {0:0} → 0*2+0 = 0
    // {0:1} → 1*2+0 = 2
    assert_eq!(indices, vec![0, 2]);
}

#[test]
fn test_swizzle_contract_middle_axis() {
    // Contract middle axis: cargs=[(1,2)], eargs=[(0,2),(1,2),(2,2)]
    // Axis 0 and 2 default to 0
    // {1:0} → idx with eargs = 0*(2*2) + 0*2 + 0 = 0
    // {1:1} → idx = 0 + 1*2 + 0 = 2
    let cargs = vec![(1, 2)];
    let eargs = vec![(0, 2), (1, 2), (2, 2)];
    let indices = swizzle_args(&cargs, &eargs, &[]);
    assert_eq!(indices, vec![0, 2]);
}
