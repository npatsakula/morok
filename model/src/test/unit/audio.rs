use crate::audio::reflect_pad;

#[test]
fn test_reflect_pad_matches_pytorch_for_pad_lt_len() {
    // signal=[a,b,c,d], pad=2 → [c,b, a,b,c,d, c,b]
    let signal = vec![1.0f32, 2.0, 3.0, 4.0];
    let padded = reflect_pad(&signal, 2);
    assert_eq!(padded, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
}

#[test]
fn test_reflect_pad_pad_one() {
    let signal = vec![10.0f32, 20.0, 30.0];
    let padded = reflect_pad(&signal, 1);
    assert_eq!(padded, vec![20.0, 10.0, 20.0, 30.0, 20.0]);
}

#[test]
fn test_reflect_pad_pad_zero_is_identity() {
    let signal = vec![1.0f32, 2.0, 3.0];
    let padded = reflect_pad(&signal, 0);
    assert_eq!(padded, signal);
}

#[test]
fn test_reflect_pad_max_valid_pad() {
    // pad = len - 1 is the largest single-bounce reflection.
    let signal = vec![1.0f32, 2.0, 3.0, 4.0];
    let padded = reflect_pad(&signal, 3);
    assert_eq!(padded, vec![4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
#[should_panic(expected = "reflect_pad requires pad")]
fn test_reflect_pad_panics_when_pad_equals_len() {
    let signal = vec![1.0f32, 2.0, 3.0];
    reflect_pad(&signal, 3);
}

#[test]
#[should_panic(expected = "reflect_pad requires pad")]
fn test_reflect_pad_panics_when_pad_exceeds_len() {
    let signal = vec![1.0f32, 2.0];
    reflect_pad(&signal, 5);
}
