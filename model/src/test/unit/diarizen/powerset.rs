//! Unit tests for powerset → multilabel decoding.

use crate::diarizen::powerset_to_multilabel;
use svod_tensor::Tensor;

/// Each frame's argmax selects a known powerset subset; the decoded multilabel
/// must equal that subset's speaker membership. `powerset_table(4,4)` order:
/// `0:[] 1:[0] 2:[1] 3:[2] 4:[3] 5:[0,1] 6:[0,2] … 15:[0,1,2,3]`.
#[test]
fn powerset_decode_maps_winning_subset_to_speakers() {
    let k = 16;
    let mut data = vec![0f32; 3 * k];
    data[5] = 10.0; // frame 0 → subset 5 = {0,1}
    data[k + 1] = 10.0; // frame 1 → subset 1 = {0}
    data[2 * k] = 10.0; // frame 2 → subset 0 = {} (no speakers)
    let logits = Tensor::from_slice(&data).try_reshape([1, 3, k]).unwrap();

    let ml = powerset_to_multilabel(&logits, 4, 4).unwrap();
    ml.realize().unwrap();

    // (1, 3, 4): {0,1}→[1,1,0,0], {0}→[1,0,0,0], {}→[0,0,0,0].
    let expected = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(ml.as_vec::<f32>().unwrap(), expected);
}
