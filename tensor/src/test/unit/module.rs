//! Tests: module.

use std::collections::BTreeSet;

use svod_dtype::DType;
use test_case::test_case;

use crate::Tensor;
use crate::error::ErrorKind;
use crate::nn::{Module, StateDict, get_tensor, prefixed};

/// A scalar tensor, so a parameter's identity is one readable number.
fn t(v: f32) -> Tensor {
    Tensor::from_slice([v])
}

fn val(x: &Tensor) -> f32 {
    x.to_vec::<f32>().unwrap()[0]
}

fn keys(sd: &StateDict) -> BTreeSet<String> {
    sd.keys().cloned().collect()
}

fn expect<'a>(names: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    names.into_iter().map(str::to_string).collect()
}

#[derive(Module)]
#[module(crate = "crate")]
struct Inner {
    weight: Tensor,
    #[module(optional)]
    bias: Option<Tensor>,
}

impl Inner {
    fn new(base: f32, bias: bool) -> Self {
        Self { weight: t(base), bias: bias.then(|| t(base + 0.5)) }
    }
}

#[derive(Module)]
#[module(crate = "crate")]
enum Head {
    Empty,
    Proj(Inner),
    Gated {
        gate: Tensor,
        #[module(key = "proj")]
        inner: Inner,
    },
}

/// Exercises every field shape the derive classifies: a bare tensor, a `Vec`
/// of children, a flattened child, an `Option` child, an enum, a `Box`, a
/// tuple, an array, an auto-skipped primitive and an explicitly skipped one.
#[derive(Module)]
#[module(crate = "crate")]
struct Net {
    embed: Tensor,
    blocks: Vec<Inner>,
    #[module(key = "")]
    flat: Inner,
    tail: Option<Inner>,
    head: Head,
    boxed: Box<Tensor>,
    pair: (Tensor, Tensor),
    arr: [Tensor; 2],
    eps: f64,
    #[module(skip)]
    dtype: DType,
}

impl Net {
    /// Every parameter is `base + <a distinct offset>`, so a round-trip that
    /// mixes two nets up is visible in any single value.
    fn new(base: f32) -> Self {
        Self {
            embed: t(base),
            blocks: vec![Inner::new(base + 10.0, true), Inner::new(base + 20.0, false)],
            flat: Inner::new(base + 30.0, true),
            tail: Some(Inner::new(base + 40.0, false)),
            head: Head::Gated { gate: t(base + 50.0), inner: Inner::new(base + 60.0, true) },
            boxed: Box::new(t(base + 70.0)),
            pair: (t(base + 80.0), t(base + 81.0)),
            arr: [t(base + 90.0), t(base + 91.0)],
            eps: 1e-5,
            dtype: DType::Float32,
        }
    }
}

const NET_KEYS: &[&str] = &[
    "embed",
    "blocks.0.weight",
    "blocks.0.bias",
    "blocks.1.weight",
    "weight",
    "bias",
    "tail.weight",
    "head.gate",
    "head.proj.weight",
    "head.proj.bias",
    "boxed",
    "pair.0",
    "pair.1",
    "arr.0",
    "arr.1",
];

#[test]
fn root_prefix_writes_bare_keys() {
    assert_eq!(keys(&Net::new(0.0).state_dict("")), expect(NET_KEYS.iter().copied()));
}

#[test]
fn nested_prefix_dots_every_key() {
    let expected: BTreeSet<String> = NET_KEYS.iter().map(|k| format!("m.{k}")).collect();
    assert_eq!(keys(&Net::new(0.0).state_dict("m")), expected);
}

#[test_case("" ; "root")]
#[test_case("m.sub" ; "nested")]
fn round_trip_restores_every_parameter(prefix: &str) {
    let src = Net::new(1.0);
    let sd = src.state_dict(prefix);

    let mut dst = Net::new(1000.0);
    dst.load_state_dict(&sd, prefix).unwrap();

    assert_eq!(dst.state_dict(prefix).len(), sd.len());
    for (key, tensor) in &sd {
        assert_eq!(val(&dst.state_dict(prefix)[key]), val(tensor), "key {key}");
    }
}

#[test]
fn non_parameter_fields_survive_a_load() {
    let mut dst = Net::new(1000.0);
    dst.eps = 1e-3;
    dst.dtype = DType::Float16;
    dst.load_state_dict(&Net::new(1.0).state_dict(""), "").unwrap();

    assert_eq!(dst.eps, 1e-3);
    assert_eq!(dst.dtype, DType::Float16);
}

#[test]
fn a_missing_required_key_names_itself() {
    let mut sd = Net::new(1.0).state_dict("m");
    sd.remove("m.head.proj.weight");

    let err = Net::new(0.0).load_state_dict(&sd, "m").unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::MissingKey { key } if key == "m.head.proj.weight"), "{err}");
}

#[test]
fn an_absent_optional_clears_the_field() {
    let mut sd = StateDict::new();
    sd.insert("weight".to_string(), t(7.0));

    let mut inner = Inner::new(0.0, true);
    inner.load_state_dict(&sd, "").unwrap();

    assert_eq!(val(&inner.weight), 7.0);
    assert!(inner.bias.is_none());
}

#[test]
fn a_present_optional_is_written_and_read() {
    let sd = Inner::new(3.0, true).state_dict("");
    assert_eq!(keys(&sd), expect(["bias", "weight"]));

    let mut inner = Inner::new(0.0, false);
    inner.load_state_dict(&sd, "").unwrap();
    assert_eq!(inner.bias.map(|b| val(&b)), Some(3.5));
}

#[test_case(Head::Empty, &[] ; "unit variant carries nothing")]
#[test_case(Head::Proj(Inner::new(1.0, false)), &["head.weight"] ; "newtype variant is transparent")]
#[test_case(
    Head::Gated { gate: t(1.0), inner: Inner::new(2.0, false) },
    &["head.gate", "head.proj.weight"] ;
    "struct variant keys its child"
)]
fn enum_variants_key_their_own_fields(head: Head, expected: &[&str]) {
    assert_eq!(keys(&head.state_dict("head")), expect(expected.iter().copied()));
}

#[test]
fn visit_params_sees_exactly_the_state_dict() {
    let net = Net::new(2.0);
    let mut seen = Vec::new();
    net.visit_params("m", &mut |key, tensor| seen.push((key.to_string(), val(tensor))));

    let sd = net.state_dict("m");
    assert_eq!(seen.len(), sd.len());
    for (key, v) in seen {
        assert_eq!(val(&sd[&key]), v, "key {key}");
    }
}

#[test]
fn module_fields_lists_the_weight_carrying_fields_only() {
    assert_eq!(
        Net::MODULE_FIELDS,
        &[
            ("embed", "embed"),
            ("blocks", "blocks"),
            ("flat", ""),
            ("tail", "tail"),
            ("head", "head"),
            ("boxed", "boxed"),
            ("pair", "pair"),
            ("arr", "arr"),
        ]
    );
}

#[test]
fn a_vec_child_keeps_the_receivers_length() {
    let sd = Net::new(1.0).state_dict("");
    let mut short = Net::new(0.0);
    short.blocks.pop();
    short.load_state_dict(&sd, "").unwrap();

    assert_eq!(short.blocks.len(), 1);
    assert_eq!(val(&short.blocks[0].weight), 11.0);
}

#[test_case("", "weight", "weight" ; "root drops the dot")]
#[test_case("m", "weight", "m.weight" ; "child joins with a dot")]
#[test_case("a.b", "0", "a.b.0" ; "index is just a segment")]
#[test_case("m", "", "m." ; "an empty name still joins")]
fn prefixed_joins_a_path(prefix: &str, name: &str, expected: &str) {
    assert_eq!(prefixed(prefix, name), expected);
}

#[test]
fn get_tensor_reports_the_key_it_wanted() {
    let sd = StateDict::new();
    let err = get_tensor(&sd, "a.b").unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::MissingKey { key } if key == "a.b"), "{err}");
}
