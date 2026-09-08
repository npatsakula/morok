mod unit;

use svod_tensor::Tensor;

/// Dims with a symbolic axis resolved to its upper bound. For graphs built with
/// a rebindable batch, where [`Tensor::dims`] rejects the symbolic axis.
fn max_dims(t: &Tensor) -> Vec<usize> {
    t.shape()
        .unwrap()
        .iter()
        .map(|s| s.as_const().or_else(|| s.vmax()).expect("concrete or symbolic-max dim"))
        .collect()
}
