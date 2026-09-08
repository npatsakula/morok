//! Call frames captured at the public op entry points.

use std::collections::BTreeSet;

use svod_dtype::DType;
use svod_ir::origin::{self, OriginFrame, OriginScope};
use test_case::test_case;

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
    let c = (&a + &b).unwrap();

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
    let c = (&a + &a).unwrap();
    assert!(calls(&c).is_empty());
    assert!(c.uop().toposort().iter().all(|node| node.origin().is_none()));
}

#[test]
fn a_public_op_built_from_public_ops_keeps_only_the_outer_frame() {
    let _capture = origin::capture_for_thread(true);
    let (x, weight, bias) = (operand(&[1, 3]), operand(&[2, 3]), operand(&[2]));

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

    let leaf =
        c.unwrap().uop().toposort().iter().find_map(|node| node.origin()).expect("the product carries an origin");
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

/// Operands are built with capture off, so the op under test is the only frame
/// in the cone it produces.
fn operand(shape: &[usize]) -> Tensor {
    let _off = origin::capture_for_thread(false);
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = (0..numel).map(|i| i as f32 + 1.0).collect();
    let dims: Vec<isize> = shape.iter().map(|&dim| dim as isize).collect();
    Tensor::from_slice(data).try_reshape(dims).expect("operand reshape")
}

/// An all-true boolean operand, likewise built with capture off.
fn mask(shape: &[usize]) -> Tensor {
    let _off = origin::capture_for_thread(false);
    let dims: Vec<isize> = shape.iter().map(|&dim| dim as isize).collect();
    Tensor::from_slice(vec![true; shape.iter().product()]).try_reshape(dims).expect("mask reshape")
}

/// One public call, one frame, named after the op and located in this file.
#[test_case("mean", &|| operand(&[2, 3]).mean(()).unwrap() ; "mean")]
#[test_case("max", &|| operand(&[2, 3]).max(()).unwrap() ; "max")]
#[test_case("min", &|| operand(&[2, 3]).min(()).unwrap() ; "min")]
#[test_case("prod", &|| operand(&[2, 3]).prod(()).unwrap() ; "prod")]
#[test_case("sum", &|| operand(&[2, 3]).sum_with().axes(0isize).keepdim(true).call().unwrap() ; "sum_with")]
#[test_case("var", &|| operand(&[2, 3]).var(()).unwrap() ; "var")]
#[test_case("all", &|| operand(&[2, 3]).all(()).unwrap() ; "all")]
#[test_case("any", &|| operand(&[2, 3]).any(()).unwrap() ; "any")]
#[test_case("argmax", &|| operand(&[2, 3]).argmax(Some(1isize)).unwrap() ; "argmax")]
#[test_case("cumsum", &|| operand(&[2, 3]).cumsum(1).unwrap() ; "cumsum")]
#[test_case("reshape", &|| operand(&[2, 3]).try_reshape([6isize]).unwrap() ; "reshape")]
#[test_case("permute", &|| operand(&[2, 3]).try_permute(&[1, 0]).unwrap() ; "permute")]
#[test_case("pad", &|| operand(&[2, 3]).try_pad(&[(1, 1), (0, 0)]).unwrap() ; "pad")]
#[test_case("cat", &|| Tensor::cat(&[&operand(&[2, 3]), &operand(&[2, 3])], 0).unwrap() ; "cat")]
#[test_case("where", &|| operand(&[2, 3]).where_(&mask(&[2, 3]), operand(&[2, 3])).unwrap() ; "where_op")]
#[test_case("exp", &|| operand(&[2, 3]).try_exp().unwrap() ; "exp")]
#[test_case("sigmoid", &|| operand(&[2, 3]).sigmoid().unwrap() ; "sigmoid")]
#[test_case("relu", &|| operand(&[2, 3]).relu().unwrap() ; "relu")]
#[test_case("rms_norm", &|| operand(&[2, 3]).rms_norm(-1, 1e-5).unwrap() ; "rms_norm")]
#[test_case("layernorm", &|| operand(&[2, 3]).layernorm(-1, 1e-5).unwrap() ; "layernorm")]
#[test_case("scaled_dot_product_attention", &|| {
    operand(&[1, 1, 2, 2])
        .scaled_dot_product_attention()
        .key(&operand(&[1, 1, 2, 2]))
        .value(&operand(&[1, 1, 2, 2]))
        .call()
        .unwrap()
} ; "sdpa")]
#[test_case("arange", &|| Tensor::arange(4, None, None).unwrap() ; "arange")]
#[test_case("full", &|| Tensor::full(&[2, 2], 1.0f64, DType::Float32) ; "full")]
#[test_case("matmul", &|| operand(&[2, 3]).matmul(&operand(&[3, 2])).unwrap() ; "matmul")]
fn a_public_op_records_exactly_one_frame(op: &str, build: &dyn Fn() -> Tensor) {
    let _capture = origin::capture_for_thread(true);
    let out = build();

    let calls: Vec<_> = calls(&out).into_iter().collect();
    assert_eq!(calls.len(), 1, "{op}: one public call must yield one frame, got {calls:?}");
    assert_eq!(calls[0].0, op, "frame names the public op, not an internal one");
    assert!(calls[0].1.ends_with("tensor/src/test/unit/origin.rs"), "{op}: frame points into svod at {:?}", calls[0]);
}

/// A user helper composing two public ops: two frames, at the helper's own lines.
fn user_helper(x: &Tensor) -> Tensor {
    let scaled = x.try_mul(x).expect("mul");
    scaled.relu().expect("relu")
}

#[test]
fn each_public_call_in_a_user_helper_gets_its_own_frame() {
    let _capture = origin::capture_for_thread(true);
    let x = operand(&[2, 3]);

    let out = user_helper(&x);

    let calls: Vec<_> = calls(&out).into_iter().collect();
    assert_eq!(calls.len(), 2, "two sibling public calls, two frames: {calls:?}");
    assert_eq!(calls.iter().map(|call| call.0).collect::<BTreeSet<_>>(), BTreeSet::from(["mul", "relu"]));
    assert!(calls.iter().all(|call| call.1.ends_with("tensor/src/test/unit/origin.rs")));
    assert_ne!(calls[0].2, calls[1].2, "each call frame names its own line: {calls:?}");
}
