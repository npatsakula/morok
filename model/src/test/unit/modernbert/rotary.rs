use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::modernbert::RotaryTable;

/// Position 0 has zero rotation (cos=1, sin=0): the vector is unchanged.
#[test]
fn rope_identity_at_position_zero() {
    // x shape (1, 1, 2, 4): batch=1, head=1, seq=2, head_dim=4.
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).try_reshape([1isize, 1, 2, 4]).unwrap();
    let table = RotaryTable::new(10_000.0, 2, 4, DType::Float32).unwrap();
    let out = table.apply(&x).unwrap();
    out.realize().unwrap();
    let v = out.as_vec::<f32>().unwrap();
    // Position 0 (first 4 elems) is unchanged.
    assert!((v[0] - 1.0).abs() < 1e-5, "rot pos0 changed: {}", v[0]);
    assert!((v[1] - 2.0).abs() < 1e-5);
    assert!((v[2] - 3.0).abs() < 1e-5);
    assert!((v[3] - 4.0).abs() < 1e-5);
    // Position 1 is rotated — values differ from input.
    assert!((v[4] - 5.0).abs() > 1e-3, "rot pos1 unchanged");
}

/// Different theta bases yield different cos/sin tables (global vs local).
#[test]
fn global_vs_local_theta_differ() {
    let global = RotaryTable::new(160_000.0, 16, 64, DType::Float32).unwrap();
    let local = RotaryTable::new(10_000.0, 16, 64, DType::Float32).unwrap();
    let gc = global.cos.clone();
    gc.realize().unwrap();
    let lc = local.cos.clone();
    lc.realize().unwrap();
    let gv = gc.as_vec::<f32>().unwrap();
    let lv = lc.as_vec::<f32>().unwrap();
    // They agree at position 0 (cos=1 everywhere) but diverge at position >0.
    assert!((gv[0] - 1.0).abs() < 1e-5);
    assert!((lv[0] - 1.0).abs() < 1e-5);
    let diffs = gv.iter().zip(&lv).filter(|(a, b)| (**a - **b).abs() > 1e-3).count();
    assert!(diffs > 0, "global and local cos tables identical");
}
