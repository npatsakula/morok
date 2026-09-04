//! Call frames captured at the public op entry points.

use std::collections::BTreeSet;

use svod_ir::origin::{self, OriginFrame, OriginScope};

use crate::Tensor;

/// Every distinct frame in the tensor's cone, leaves and their ancestors.
fn frames(tensor: &Tensor) -> Vec<OriginFrame> {
    tensor
        .uop()
        .toposort()
        .iter()
        .filter_map(|node| node.origin())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .flat_map(origin::chain)
        .filter_map(origin::get)
        .map(|origin| origin.frame)
        .collect()
}

/// The `(op, file, line)` of every call frame in the cone, deduplicated.
fn calls(tensor: &Tensor) -> BTreeSet<(&'static str, String, u32)> {
    frames(tensor)
        .into_iter()
        .filter_map(|frame| match frame {
            OriginFrame::Call { op, at } => Some((op, at.file.to_string(), at.line)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_binary_op_records_the_caller_site() {
    let _capture = origin::capture_for_thread(true);
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let b = Tensor::from_slice([3.0f32, 4.0]);

    let line = line!() + 1;
    let c = &a + &b;

    let calls: Vec<_> = calls(&c).into_iter().collect();
    assert_eq!(calls.len(), 1, "one public entry point, one frame: {calls:?}");
    let (op, file, at) = &calls[0];
    // The operator impl and `try_add` are both `#[track_caller]`, so the frame
    // names the test, not the library.
    assert_eq!(*op, "add");
    assert!(file.ends_with("tensor/src/test/unit/origin.rs"), "unexpected file {file}");
    assert_eq!(*at, line);
}

#[test]
fn capture_is_off_by_default() {
    let _capture = origin::capture_for_thread(false);
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let c = &a + &a;
    assert!(calls(&c).is_empty());
    assert!(c.uop().toposort().iter().all(|node| node.origin().is_none()));
}

#[test]
fn a_public_op_built_from_public_ops_keeps_only_the_outer_frame() {
    let _capture = origin::capture_for_thread(true);
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape([1isize, 3]).unwrap();
    let weight = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).try_reshape([2isize, 3]).unwrap();
    let bias = Tensor::from_slice([0.1f32, 0.2]);

    let line = line!() + 1;
    let out = x.linear().weight(&weight).bias(&bias).call().unwrap();

    // `linear` runs `matmul`, `sum` and `add` internally; outermost wins.
    let calls: Vec<_> = calls(&out).into_iter().collect();
    assert_eq!(calls.len(), 1, "nested public ops must not stack frames: {calls:?}");
    assert_eq!(calls[0].0, "linear");
    assert_eq!(calls[0].2, line);
}

#[test]
fn a_call_frame_nests_under_the_enclosing_module() {
    let _capture = origin::capture_for_thread(true);
    let a = Tensor::from_slice([1.0f32, 2.0]);
    let b = Tensor::from_slice([3.0f32, 4.0]);

    let c = {
        let _outer = OriginScope::module("encoder");
        let _inner = OriginScope::module("layers.0");
        &a * &b
    };

    let leaf = c.uop().toposort().iter().find_map(|node| node.origin()).expect("the product carries an origin");
    let rendered = origin::path(leaf);
    assert!(rendered.starts_with("encoder.layers.0 @ mul "), "unexpected path {rendered}");
}

#[test]
fn sibling_scopes_split_an_otherwise_identical_graph() {
    let _capture = origin::capture_for_thread(true);
    let a = Tensor::from_slice([5.0f32, 6.0]);

    let left = {
        let _scope = OriginScope::module("left");
        a.try_mul(&a).unwrap()
    };
    let right = {
        let _scope = OriginScope::module("right");
        a.try_mul(&a).unwrap()
    };

    assert_ne!(left.uop().origin(), right.uop().origin());
    assert!(origin::path(left.uop().origin().unwrap()).starts_with("left @ mul"));
    assert!(origin::path(right.uop().origin().unwrap()).starts_with("right @ mul"));
}
