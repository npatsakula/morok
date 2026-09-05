//! Unit tests for the origin arena, scopes, rendering and UOp carriage.

use std::sync::Arc;

use proptest::prelude::*;
use test_case::test_case;

use crate::origin::{self, Origin, OriginFrame, OriginId, OriginScope, SourceLocation};
use crate::{ConstValue, DType, Op, ParamArg, UOp, ops};

fn module(name: &str) -> Origin {
    Origin { parent: None, frame: OriginFrame::Module { name: name.to_owned() } }
}

// =========================================================================
// Arena
// =========================================================================

#[test]
fn interning_is_idempotent_and_round_trips() {
    let root = origin::intern(module("interning-root"));
    assert_eq!(root, origin::intern(module("interning-root")));

    let child = Origin { parent: Some(root), frame: OriginFrame::Label { text: String::from("child") } };
    let child_id = origin::intern(child.clone());
    assert_ne!(root, child_id);
    assert_eq!(child_id, origin::intern(child.clone()));
    assert_eq!(origin::get(child_id), Some(child));
    assert_eq!(origin::get(root), Some(module("interning-root")));
}

#[test]
fn snapshot_indexes_by_id_and_only_grows() {
    let before = origin::snapshot();
    let id = origin::intern(module("snapshot-probe"));
    let after = origin::snapshot();

    // The table is append-only: an earlier prefix never changes, and the id indexes it.
    assert!(after.len() >= before.len());
    assert_eq!(after[..before.len()], before[..]);
    assert_eq!(after[id.get() as usize - 1], module("snapshot-probe"));
    // Re-interning an existing origin allocates no new entry.
    assert_eq!(origin::intern(module("snapshot-probe")), id);
}

#[test]
fn unknown_ids_resolve_to_nothing() {
    assert_eq!(origin::get(OriginId::from_raw(u32::MAX).unwrap()), None);
    assert_eq!(OriginId::from_raw(0), None);
}

proptest! {
    /// Interning a random frame tree twice yields the same ids, and every id
    /// resolves back to the origin it was interned from.
    #[test]
    fn interning_a_random_tree_is_idempotent(names in prop::collection::vec("[a-z]{1,6}", 1..8)) {
        let mut first = Vec::new();
        let mut parent = None;
        for name in &names {
            let origin = Origin { parent, frame: OriginFrame::Module { name: format!("prop.{name}") } };
            let id = origin::intern(origin.clone());
            prop_assert_eq!(origin::get(id), Some(origin));
            first.push(id);
            parent = Some(id);
        }

        let mut parent = None;
        for (name, expected) in names.iter().zip(&first) {
            let origin = Origin { parent, frame: OriginFrame::Module { name: format!("prop.{name}") } };
            prop_assert_eq!(origin::intern(origin), *expected);
            parent = Some(*expected);
        }
    }
}

// =========================================================================
// Scopes
// =========================================================================

#[test]
fn scopes_nest_and_restore_on_drop() {
    let _capture = crate::origin::capture_for_thread(true);
    assert_eq!(origin::current(), None);

    let outer = OriginScope::module("encoder");
    let outer_id = origin::current().expect("outer scope installs an origin");
    {
        let _inner = OriginScope::module("layers.3");
        let inner_id = origin::current().expect("inner scope installs an origin");
        assert_ne!(inner_id, outer_id);
        assert_eq!(origin::get(inner_id).unwrap().parent, Some(outer_id));
    }
    assert_eq!(origin::current(), Some(outer_id));
    drop(outer);
    assert_eq!(origin::current(), None);
}

#[test]
fn a_panic_unwinding_through_a_scope_restores_the_previous_one() {
    let _capture = crate::origin::capture_for_thread(true);
    let _outer = OriginScope::module("unwind-outer");
    let outer_id = origin::current();

    let panicked = std::panic::catch_unwind(|| {
        let _inner = OriginScope::module("unwind-inner");
        panic!("unwinding through the scope guard");
    });

    assert!(panicked.is_err());
    assert_eq!(origin::current(), outer_id);
}

#[test]
fn suspend_detaches_and_install_reattaches() {
    let _capture = crate::origin::capture_for_thread(true);
    let _outer = OriginScope::module("suspend-outer");
    let outer_id = origin::current().expect("scope installed");

    {
        let _suspended = OriginScope::suspend();
        assert_eq!(origin::current(), None);
        let _nested = OriginScope::module("suspend-nested");
        assert_eq!(origin::get(origin::current().unwrap()).unwrap().parent, None);
    }
    assert_eq!(origin::current(), Some(outer_id));

    let worker = std::thread::spawn(move || {
        assert_eq!(origin::current(), None, "a fresh thread starts unscoped");
        let _capture = crate::origin::capture_for_thread(true);
        let _installed = origin::install(Some(outer_id));
        let seen = origin::current();
        let _nested = OriginScope::label("worker");
        (seen, origin::get(origin::current().unwrap()).unwrap().parent)
    })
    .join()
    .unwrap();
    assert_eq!(worker, (Some(outer_id), Some(outer_id)));
}

#[test_case("module"; "module frame")]
#[test_case("label"; "label frame")]
#[test_case("onnx"; "onnx frame")]
#[test_case("call"; "call frame")]
fn constructors_are_no_ops_while_capture_is_off(kind: &str) {
    let _capture = crate::origin::capture_for_thread(false);

    let scope = match kind {
        "module" => OriginScope::module("off"),
        "label" => OriginScope::label("off"),
        "onnx" => OriginScope::onnx(0, None, "Add", "ai.onnx", 17),
        _ => OriginScope::call("add", std::panic::Location::caller()),
    };
    let captured = origin::current();
    drop(scope);

    assert_eq!(captured, None, "{kind} scope must not capture while disabled");
}

/// Two guards dropped out of order are two tasks interleaving on one thread.
#[test]
#[cfg_attr(debug_assertions, should_panic(expected = "origin scopes must nest"))]
fn interleaved_scopes_are_rejected() {
    let _capture = crate::origin::capture_for_thread(true);
    let outer = OriginScope::module("interleave-outer");
    let inner = OriginScope::module("interleave-inner");
    drop(outer);
    drop(inner);
}

// =========================================================================
// Rendering
// =========================================================================

/// Build `encoder.layers.3` + an ONNX node + a call frame, root-first.
fn rendering_chain() -> Vec<OriginId> {
    let encoder = origin::intern(module("render.encoder"));
    let layers =
        origin::intern(Origin { parent: Some(encoder), frame: OriginFrame::Module { name: String::from("layers.3") } });
    let node = origin::intern(Origin {
        parent: Some(layers),
        frame: OriginFrame::Onnx {
            index: 7,
            name: None,
            op_type: String::from("MatMul"),
            domain: String::from("ai.onnx"),
            version: 17,
        },
    });
    let call = origin::intern(Origin {
        parent: Some(node),
        frame: OriginFrame::Call { op: "mul", at: SourceLocation::new("tensor/src/arithmetic.rs", 31, 5) },
    });
    vec![encoder, layers, node, call]
}

#[test]
fn path_renders_named_segments_dotted_and_the_call_frame_trailing() {
    let chain = rendering_chain();
    assert_eq!(origin::path(chain[1]), "render.encoder.layers.3");
    assert_eq!(origin::path(chain[2]), "render.encoder.layers.3.#7:MatMul");
    assert_eq!(origin::path(chain[3]), "render.encoder.layers.3.#7:MatMul @ mul tensor/src/arithmetic.rs:31");
}

#[test]
fn a_named_onnx_node_renders_by_name() {
    let id = origin::intern(Origin {
        parent: None,
        frame: OriginFrame::Onnx {
            index: 2,
            name: Some(String::from("/encoder/Conv")),
            op_type: String::from("Conv"),
            domain: String::from("ai.onnx"),
            version: 17,
        },
    });
    assert_eq!(origin::path(id), "/encoder/Conv");
}

#[test]
fn chain_is_root_first_and_truncate_walks_it() {
    let chain = rendering_chain();
    let leaf = *chain.last().unwrap();
    assert_eq!(origin::chain(leaf), chain);

    assert_eq!(origin::truncate(leaf, 0), None);
    for (depth, expected) in (1..=chain.len()).zip(&chain) {
        assert_eq!(origin::truncate(leaf, depth), Some(*expected));
    }
    // Past the leaf a rollup keeps the whole path rather than losing the row.
    assert_eq!(origin::truncate(leaf, chain.len() + 5), Some(leaf));
}

// =========================================================================
// Serde
// =========================================================================

fn frames() -> Vec<OriginFrame> {
    vec![
        OriginFrame::Module { name: String::from("encoder") },
        OriginFrame::Label { text: String::from("initializer") },
        OriginFrame::Call { op: "add", at: SourceLocation::new("tensor/src/traits.rs", 24, 9) },
        OriginFrame::Onnx {
            index: 3,
            name: None,
            op_type: String::from("Gemm"),
            domain: String::from(""),
            version: 21,
        },
        OriginFrame::Onnx {
            index: 4,
            name: Some(String::from("/head/Gemm")),
            op_type: String::from("Gemm"),
            domain: String::from("ai.onnx"),
            version: 21,
        },
    ]
}

#[test_case(0; "module")]
#[test_case(1; "label")]
#[test_case(2; "call")]
#[test_case(3; "onnx anonymous")]
#[test_case(4; "onnx named")]
fn origin_frames_serialize_with_their_parent(index: usize) {
    let frame = frames()[index].clone();
    let origin = Origin { parent: OriginId::from_raw(9), frame: frame.clone() };
    let json = serde_json::to_value(&origin).unwrap();
    assert_eq!(json["parent"], 9);
    let variant = json["frame"].as_object().and_then(|frame| frame.keys().next().cloned()).unwrap();
    assert!(["Module", "Label", "Call", "Onnx"].contains(&variant.as_str()), "{json}");
    assert_eq!(json["frame"], serde_json::to_value(&frame).unwrap());
}

// =========================================================================
// UOp carriage
// =========================================================================

/// A node built with no origin hashes exactly as it did before origins existed.
/// Captured from `main` before `content_hash` learned about origins.
const CONST_I32_5_CONTENT_HASH: u64 = 0x782e_4138_6be3_2c13;
const ADD_CONST_CONTENT_HASH: u64 = 0x21f2_95ab_f3b3_3883;

#[test]
fn origin_free_content_hashes_are_unchanged() {
    let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    let seven = UOp::const_(DType::Int32, ConstValue::Int(7));
    assert_eq!(five.origin(), None);
    assert_eq!(five.content_hash, CONST_I32_5_CONTENT_HASH);
    assert_eq!(five.add(&seven).content_hash, ADD_CONST_CONTENT_HASH);
}

fn expression() -> Arc<UOp> {
    let lhs = UOp::const_(DType::Int32, ConstValue::Int(11));
    let rhs = UOp::const_(DType::Int32, ConstValue::Int(13));
    lhs.add(&rhs)
}

#[test]
fn distinct_scopes_split_a_shared_expression() {
    let _capture = crate::origin::capture_for_thread(true);

    let (first, first_origin) = {
        let _scope = OriginScope::module("split.a");
        (expression(), origin::current())
    };
    let (second, second_origin) = {
        let _scope = OriginScope::module("split.b");
        (expression(), origin::current())
    };
    let (again, _) = {
        let _scope = OriginScope::module("split.a");
        (expression(), origin::current())
    };

    assert_eq!(first.origin(), first_origin);
    assert_eq!(second.origin(), second_origin);
    assert_ne!(first.content_hash, second.content_hash);
    assert!(!Arc::ptr_eq(&first, &second));
    assert!(Arc::ptr_eq(&first, &again), "one scope must yield one allocation");
    assert!(Arc::ptr_eq(&expression(), &expression()), "unscoped nodes still dedup");
}

#[test]
fn rewrites_and_rebuilds_keep_the_origin() {
    let _capture = crate::origin::capture_for_thread(true);
    let _scope = OriginScope::module("carriage");
    let scope_id = origin::current();

    let node = expression();
    assert_eq!(node.origin(), scope_id);

    let sources = vec![UOp::const_(DType::Int32, ConstValue::Int(1)), UOp::const_(DType::Int32, ConstValue::Int(2))];
    assert_eq!(node.with_sources(sources).origin(), scope_id);
    assert_eq!(node.with_dtype(DType::Int64).origin(), scope_id);
    assert_eq!(node.rtag(Some(smallvec::smallvec![3])).origin(), scope_id);
}

#[test]
fn rorigin_reinterns_and_is_identity_when_equal() {
    let _capture = crate::origin::capture_for_thread(true);
    let outer = OriginScope::module("rorigin.outer");
    let outer_id = origin::current();
    let node = expression();
    drop(outer);

    assert!(Arc::ptr_eq(&node.rorigin(outer_id), &node));

    let _inner = OriginScope::module("rorigin.inner");
    let moved = node.rorigin(origin::current());
    assert_eq!(moved.origin(), origin::current());
    assert_ne!(moved.content_hash, node.content_hash);
    assert_eq!(moved.rorigin(None).origin(), None);
}

#[test]
fn identity_ops_never_carry_an_origin() {
    let _capture = crate::origin::capture_for_thread(true);
    let _scope = OriginScope::module("identity");

    let shape = UOp::new(Op::Stack(ops::Stack { sources: smallvec::smallvec![UOp::index_const(4)] }), DType::Index);
    let arg = ParamArg::buffer(0, DType::Float32, crate::AddrSpace::Global, Some(svod_dtype::DeviceSpec::Cpu));
    let buffer = UOp::new(Op::Buffer(ops::Buffer { shape: shape.clone(), arg: arg.clone().into() }), DType::Float32);
    let param = UOp::new(Op::Param(ops::Param { shape: shape.clone(), arg: arg.into() }), DType::Float32);
    let unique = UOp::new(Op::Unique(7), DType::Void);
    let lunique = UOp::new(Op::LUnique(7), DType::Void);

    assert_eq!(buffer.origin(), None);
    assert_eq!(param.origin(), None);
    assert_eq!(unique.origin(), None);
    assert_eq!(lunique.origin(), None);
    // A literal is the one value here: two scopes build it independently and
    // identically, so an origin on it would only split a node the kernel cut
    // re-merges (see `origin_opaque`).
    assert_eq!(UOp::const_(DType::Int32, ConstValue::Int(17)).origin(), None);
    // Shape algebra and bindings are structure, not work: a variable built in two
    // scopes must stay one variable, and so must the shapes that name it.
    assert_eq!(shape.origin(), None);
    let var = UOp::variable("identity.t".into(), 1, 8, DType::WeakInt);
    assert_eq!(var.origin(), None);
    assert_eq!(var.bind(UOp::index_const(4)).origin(), None);
    let scaled = var.try_mul(&UOp::index_const(3)).expect("symbolic product");
    assert!(matches!(scaled.op(), Op::Binary(..)), "a symbolic product is a real node, not a folded literal");
    assert_eq!(scaled.origin(), None);
}

#[test]
fn a_variable_built_in_two_scopes_is_one_variable() {
    let _capture = crate::origin::capture_for_thread(true);
    let build = || UOp::variable("shared.t".into(), 1, 8, DType::WeakInt);
    let left = {
        let _scope = OriginScope::module("variable.left");
        build()
    };
    let right = {
        let _scope = OriginScope::module("variable.right");
        build()
    };
    assert!(Arc::ptr_eq(&left, &right));
    assert!(Arc::ptr_eq(&left.bind(UOp::index_const(4)), &right.bind(UOp::index_const(4))));
}

#[test]
fn the_optimizer_wire_format_round_trips_the_origin() {
    let _capture = crate::origin::capture_for_thread(true);
    let _scope = OriginScope::module("wire");

    let node = expression();
    let graph = crate::OptimizerWireGraph::from_root(&node).unwrap();
    let decoded = graph.decode_root().unwrap();

    assert_eq!(decoded.origin(), node.origin());
    assert!(Arc::ptr_eq(&decoded, &node));
}

#[test]
fn canonical_default_form_ignores_origins_while_verbose_reports_them() {
    let _capture = crate::origin::capture_for_thread(true);
    let _scope = OriginScope::module("canonical");

    let scoped = expression();
    let plain = scoped.rorigin(None);

    let default_scoped = crate::CanonicalGraph::from_root("tensor", &scoped).unwrap().to_pretty_json().unwrap();
    let default_plain = crate::CanonicalGraph::from_root("tensor", &plain).unwrap().to_pretty_json().unwrap();
    assert_eq!(default_scoped, default_plain);
    assert!(!default_scoped.contains("origin"));

    let verbose = crate::CanonicalGraph::from_root_verbose("tensor", &scoped).unwrap();
    let rendered = verbose.verbose.unwrap();
    assert!(rendered.iter().any(|node| node.origin.as_deref() == Some("canonical")));
}
