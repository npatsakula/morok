use crate::*;

/// Regression test for ScatterND: rows not targeted by scatter indices
/// must preserve original data. Reproduces the ONNX test_scatternd inputs.
///
/// Root cause was a rewrite engine bug where `handle_link` resolved stale
/// results before index and stack rewrites completed.
#[test]
fn test_scatternd_debug() {
    let cfg = PrepareConfig::from(svod_schedule::OptimizerConfig::default());

    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    let indices: Vec<i64> = vec![0, 2];
    let updates: Vec<f32> = vec![
        5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0,
        2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0,
    ];
    let expected: Vec<f32> = vec![
        5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0,
        4.0, 4.0, 4.0, 4.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];

    // All tensors use the active default device so the kernel isn't split
    // across devices (CPU index + GPU data would trip KernelSplitMixedDevices).
    let x = Tensor::from_slice(&data).try_reshape([4, 4, 4]).unwrap();
    let idx = Tensor::from_slice(&indices).try_reshape([2, 1]).unwrap();
    let upd = Tensor::from_slice(&updates).try_reshape([2, 4, 4]).unwrap();

    let result = x.scatter_nd(&idx, &upd, None).unwrap();
    result.realize_with(&cfg).unwrap();
    let actual = result.as_vec::<f32>().unwrap();
    assert_eq!(actual, expected);
}
