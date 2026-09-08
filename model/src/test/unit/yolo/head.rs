use crate::yolo::head::make_anchors;

/// Anchors come out as `[2, A]` — the whole x row, then the whole y row —
/// with the per-level grids concatenated in feature-map order. Ultralytics
/// builds `[A, 2]` and transposes; reshaping the interleaved buffer instead
/// would put x and y values side by side in the same row.
#[test]
fn anchors_are_an_x_row_then_a_y_row() {
    let (anchors, strides) = make_anchors(&[(2, 2), (1, 1)], &[8, 16]).expect("make anchors");

    assert_eq!(anchors.dims().expect("anchor dims"), vec![2, 5]);
    #[rustfmt::skip]
    let expected = vec![
        0.5, 1.5, 0.5, 1.5, 0.5, // x of the 2×2 grid, then of the 1×1 grid
        0.5, 0.5, 1.5, 1.5, 0.5, // y of the same anchors, in the same order
    ];
    assert_eq!(anchors.to_vec::<f32>().expect("anchor values"), expected);
    assert_eq!(strides.to_vec::<f32>().expect("stride values"), vec![8.0, 8.0, 8.0, 8.0, 16.0]);
}
