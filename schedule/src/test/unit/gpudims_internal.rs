use super::*;

#[test]
fn test_group_dims_already_fits() {
    // Dims already fit, no grouping needed
    let result = group_dims(&[4, 4], &[16, 16, 16]);
    assert_eq!(result, Some(vec![4, 4]));
}

#[test]
fn test_group_dims_needs_grouping() {
    // 4 dims need to be grouped to fit into 3 max_sizes
    // [4, 4, 4, 4] can't fit directly into [256, 256, 256] (4 dims > 3 max_sizes)
    // Should group: [4*4, 4, 4] = [16, 4, 4]
    let result = group_dims(&[4, 4, 4, 4], &[256, 256, 256]);
    assert!(result.is_some());
    let result = result.unwrap();
    assert!(result.len() <= 3);
}

#[test]
fn test_group_dims_no_change() {
    // Dims already fit
    let result = group_dims(&[8, 8, 8], &[256, 256, 256]);
    assert_eq!(result, Some(vec![8, 8, 8]));
}

#[test]
fn test_group_dims_impossible() {
    // Can't fit 1000 into max 10
    let result = group_dims(&[1000], &[10]);
    assert_eq!(result, None);
}

#[test]
fn test_split_dims_simple() {
    // 100 exceeds 64, should split
    let result = split_dims(&[100], &[64, 64, 64]);
    // 100 / 2 = 50, then 50 / 2 = 25 fits
    assert!(result.iter().all(|&d| d <= 64));
}

#[test]
fn test_find_smallest_divisor() {
    assert_eq!(find_smallest_divisor(1), 1);
    assert_eq!(find_smallest_divisor(2), 2); // 2 is the smallest divisor of 2 (excluding 1)
    assert_eq!(find_smallest_divisor(3), 1); // 3 is prime
    assert_eq!(find_smallest_divisor(4), 2);
    assert_eq!(find_smallest_divisor(9), 3);
    assert_eq!(find_smallest_divisor(100), 2);
}

#[test]
fn test_get_contraction_non_consecutive() {
    // [2, 5, 2] → [10, 2]: dims 0,1 fuse to 10; dim 2 stays as 2
    // acc_old = [2, 10, 20], acc_new = [10, 20]
    let result = get_contraction(&[2, 5, 2], &[10, 2]);
    assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
}

#[test]
fn test_get_contraction_identity() {
    // [4, 4, 4] → [4, 4, 4]: no grouping
    let result = get_contraction(&[4, 4, 4], &[4, 4, 4]);
    assert_eq!(result, Some(vec![vec![0], vec![1], vec![2]]));
}

#[test]
fn test_get_contraction_all_fused() {
    // [2, 3, 4] → [24]: all dims fuse to one
    let result = get_contraction(&[2, 3, 4], &[24]);
    assert_eq!(result, Some(vec![vec![0, 1, 2]]));
}

#[test]
fn test_get_contraction_empty() {
    let result = get_contraction(&[], &[]);
    assert_eq!(result, Some(vec![]));
}

#[test]
fn test_get_contraction_invalid() {
    // [2, 3, 4] → [5, 4]: 2*3 = 6 != 5
    let result = get_contraction(&[2, 3, 4], &[5, 4]);
    assert_eq!(result, None);
}

#[test]
fn test_get_contraction_partial() {
    // [2, 4, 3] → [8, 3]: dims 0,1 fuse to 8; dim 2 stays as 3
    let result = get_contraction(&[2, 4, 3], &[8, 3]);
    assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
}
