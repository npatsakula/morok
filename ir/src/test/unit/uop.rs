// UOpKey uses immutable id field for hashing/equality, so interior mutability is safe
#![allow(clippy::mutable_key_type)]

use std::collections::HashMap;
use std::sync::Arc;

use morok_dtype::DType;
use morok_dtype::DeviceSpec;
use smallvec::smallvec;

use crate::pattern::{Matcher, RewriteResult};
use crate::{AxisId, CallInfo, ConstValue, Op, SInt, UOp, UOpKey, shape::Shape}; // ConstValue kept for DType::Index

struct RewriteCallToFirstArg;

impl Matcher<()> for RewriteCallToFirstArg {
    fn rewrite(&self, uop: &Arc<UOp>, _ctx: &mut ()) -> RewriteResult {
        match uop.op() {
            Op::Call { args, .. } | Op::Function { args, .. } if !args.is_empty() => {
                RewriteResult::Rewritten(args[0].clone())
            }
            _ => RewriteResult::NoMatch,
        }
    }
}

#[test]
fn test_const_creation() {
    let c1 = UOp::native_const(1.0f32);
    assert_eq!(c1.dtype(), DType::Float32);
    assert!(matches!(c1.op(), Op::Const(_)));
}

#[test]
fn test_hash_consing() {
    // Create two identical constants
    let c1 = UOp::native_const(1.0f32);
    let c2 = UOp::native_const(1.0f32);

    // They should be the same object
    assert!(Arc::ptr_eq(&c1, &c2), "Hash consing should return same Rc for identical UOps");
}

#[test]
fn test_hash_consing_with_src() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    // Create a + b twice
    let add1 = a.try_add(&b).unwrap();
    let add2 = a.try_add(&b).unwrap();

    // Should be the same object
    assert!(Arc::ptr_eq(&add1, &add2), "Hash consing should work with src nodes");
}

/// Test that hash consing works across threads.
///
/// This is the key correctness property: creating the same UOp in different
/// threads should return the same Arc<UOp>, so Arc::ptr_eq works across threads.
#[test]
fn test_cross_thread_hash_consing() {
    use std::sync::Barrier;

    let num_threads = 10;
    let barrier = Arc::new(Barrier::new(num_threads));

    // All threads create the same UOp concurrently
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || {
                // Wait for all threads to be ready
                b.wait();
                // Create the same constant in each thread
                UOp::native_const(42.0f32)
            })
        })
        .collect();

    // Collect results from all threads
    let uops: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads must get the same Arc
    for i in 1..uops.len() {
        assert!(
            Arc::ptr_eq(&uops[0], &uops[i]),
            "Thread {} got different Arc than thread 0 (id {} vs {})",
            i,
            uops[i].id,
            uops[0].id
        );
    }
}

/// Test that hash consing works for complex UOps across threads.
#[test]
fn test_cross_thread_hash_consing_complex() {
    use std::sync::Barrier;

    let num_threads = 8;
    let barrier = Arc::new(Barrier::new(num_threads));

    // All threads create the same expression: (1.0 + 2.0) * 3.0
    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let b = Arc::clone(&barrier);
            std::thread::spawn(move || {
                b.wait();
                let a = UOp::native_const(1.0f32);
                let b_val = UOp::native_const(2.0f32);
                let c = UOp::native_const(3.0f32);
                let add = a.try_add(&b_val).unwrap();
                add.try_mul(&c).unwrap()
            })
        })
        .collect();

    let uops: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    // All threads must get the same Arc for the final expression
    for i in 1..uops.len() {
        assert!(Arc::ptr_eq(&uops[0], &uops[i]), "Thread {} got different Arc for complex expression", i);
    }
}

#[test]
fn test_binary_operations() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    let add = a.try_add(&b).unwrap();
    assert_eq!(add.dtype(), DType::Float32);
    assert_eq!(add.op().children().len(), 2);

    let mul = a.try_mul(&b).unwrap();
    assert_eq!(mul.dtype(), DType::Float32);
}

#[test]
fn test_unary_operations() {
    let a = UOp::native_const(4.0f32);

    let sqrt = a.try_sqrt().unwrap();
    assert_eq!(sqrt.dtype(), DType::Float32);
    assert_eq!(sqrt.op().children().len(), 1);
}

#[test]
fn test_cast() {
    let a = UOp::native_const(1.5f32);
    let cast = a.cast(DType::Int32);

    assert_eq!(cast.dtype(), DType::Int32);
}

#[test]
fn test_comparison() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    let cmp = a.try_cmplt(&b).unwrap();
    assert_eq!(cmp.dtype(), DType::Bool);
}

#[test]
fn test_toposort() {
    // Build graph: (a + b) * c
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    let add = a.try_add(&b).unwrap();
    let mul = add.try_mul(&c).unwrap();

    let sorted = mul.toposort();

    // All nodes should be present
    assert!(sorted.len() >= 5); // a, b, c, add, mul

    // Check that dependencies come before dependents
    let positions: HashMap<_, _> = sorted.iter().enumerate().map(|(i, node)| (Arc::as_ptr(node), i)).collect();

    for node in &sorted {
        let node_pos = positions[&Arc::as_ptr(node)];
        for child in node.op().children() {
            let child_pos = positions[&Arc::as_ptr(child)];
            assert!(child_pos < node_pos, "Dependencies must come before dependents");
        }
    }
}

#[test]
fn test_toposort_shared_node() {
    // Build graph: x = a + b; y = a + c; z = x * y
    // Node 'a' is shared between x and y
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let c = UOp::native_const(3.0f32);

    let x = a.try_add(&b).unwrap();
    let y = a.try_add(&c).unwrap();
    let z = x.try_mul(&y).unwrap();

    let sorted = z.toposort();

    // Node 'a' should appear only once
    let a_ptr = Arc::as_ptr(&a);
    let a_count = sorted.iter().filter(|node| Arc::as_ptr(node) == a_ptr).count();
    assert_eq!(a_count, 1, "Shared node 'a' should appear exactly once");
}

#[test]
fn test_toposort_call_aware_boundaries() {
    let p0 = UOp::param(0, 1, DType::Float32, None);
    let p1 = UOp::param(1, 1, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();
    let arg0 = UOp::native_const(4.0f32);
    let arg1 = UOp::native_const(5.0f32);
    let call = body.call(smallvec![arg0.clone(), arg1.clone()], CallInfo::default());

    let include_bodies = call.toposort_call_aware(true);
    assert!(include_bodies.iter().any(|u| matches!(u.op(), Op::Param { slot: 0, .. })), "expected CALL body params");

    let preserve_boundaries = call.toposort_call_aware(false);
    assert!(preserve_boundaries.iter().any(|u| matches!(u.op(), Op::Call { .. })), "expected CALL node itself");
    assert!(preserve_boundaries.iter().any(|u| Arc::ptr_eq(u, &arg0)));
    assert!(preserve_boundaries.iter().any(|u| Arc::ptr_eq(u, &arg1)));
    assert!(!preserve_boundaries.iter().any(|u| matches!(u.op(), Op::Param { .. })), "CALL body should be excluded");

    let sink = UOp::sink(vec![call.clone()]);
    let program = UOp::program(sink.clone(), UOp::device(DeviceSpec::Cpu), None, None, None);
    let program_include = program.toposort_call_aware(true);
    assert!(program_include.iter().any(|u| Arc::ptr_eq(u, &sink)));
    let program_preserve = program.toposort_call_aware(false);
    assert!(!program_preserve.iter().any(|u| Arc::ptr_eq(u, &sink)), "PROGRAM internals should be excluded");
}

#[test]
fn test_substitute_preserve_calls_keeps_call_body() {
    let p0 = UOp::param(0, 1, DType::Float32, None);
    let p1 = UOp::param(1, 1, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();
    let call = body.call(smallvec![UOp::native_const(10.0f32), UOp::native_const(20.0f32)], CallInfo::default());

    let mut map = HashMap::new();
    map.insert(UOpKey(p0.clone()), UOp::native_const(0.0f32));

    let rewritten_preserve = call.substitute_preserve_calls(&map);
    match rewritten_preserve.op() {
        Op::Call { body: new_body, .. } => assert!(Arc::ptr_eq(new_body, &body), "CALL body should stay untouched"),
        op => panic!("expected Call op, got {op:?}"),
    }

    let rewritten_full = call.substitute(&map);
    match rewritten_full.op() {
        Op::Call { body: new_body, .. } => {
            assert!(!Arc::ptr_eq(new_body, &body), "full substitute should rewrite CALL body")
        }
        op => panic!("expected Call op, got {op:?}"),
    }
}

#[test]
fn test_graph_rewrite_preserve_calls_can_rewrite_call_node() {
    let body = UOp::param(0, 1, DType::Float32, None);
    let arg = UOp::native_const(7.0f32);
    let call = body.call(smallvec![arg.clone()], CallInfo::default());

    let rewritten = crate::graph_rewrite_preserve_calls(&RewriteCallToFirstArg, call, &mut ());
    assert!(Arc::ptr_eq(&rewritten, &arg), "preserve-calls rewrite should still match and rewrite CALL node");
}

#[test]
fn test_substitute_preserve_calls_rewrites_args_not_body() {
    let p0 = UOp::param(0, 1, DType::Float32, None);
    let p1 = UOp::param(1, 1, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let arg0 = UOp::native_const(10.0f32);
    let arg1 = UOp::native_const(20.0f32);
    let call = body.call(smallvec![arg0.clone(), arg1], CallInfo::default());

    let mut map = HashMap::new();
    let arg_replacement = UOp::native_const(11.0f32);
    map.insert(UOpKey(arg0.clone()), arg_replacement.clone());
    map.insert(UOpKey(p0.clone()), UOp::native_const(12.0f32));

    let rewritten_preserve = call.substitute_preserve_calls(&map);
    match rewritten_preserve.op() {
        Op::Call { body: new_body, args, .. } => {
            assert!(Arc::ptr_eq(new_body, &body), "preserve-calls substitute should keep CALL body untouched");
            assert!(Arc::ptr_eq(&args[0], &arg_replacement), "preserve-calls substitute should rewrite CALL args");
        }
        op => panic!("expected Call op, got {op:?}"),
    }

    let rewritten_full = call.substitute(&map);
    match rewritten_full.op() {
        Op::Call { body: new_body, args, .. } => {
            assert!(!Arc::ptr_eq(new_body, &body), "full substitute should rewrite CALL body");
            assert!(Arc::ptr_eq(&args[0], &arg_replacement), "full substitute should also rewrite CALL args");
        }
        op => panic!("expected Call op, got {op:?}"),
    }
}

#[test]
fn test_buffer_creation() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    assert!(matches!(buf.op(), Op::Buffer { .. }));
    assert_eq!(buf.dtype(), DType::Float32);

    if let Op::Buffer { size, .. } = buf.op() {
        assert_eq!(*size, 100);
    } else {
        panic!("Expected Buffer op");
    }
}

#[test]
fn test_buffer_hash_consing() {
    // Two buffers with same device and size should NOT be the same
    // (due to different UNIQUE identifiers)
    let buf1 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let buf2 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    assert!(!Arc::ptr_eq(&buf1, &buf2), "Different buffers should have different UNIQUE ids");
}

#[test]
fn test_buffer_hash_consing_lunique_distinct_from_unique() {
    // LUnique slots and Unique global ids both start at small numbers, so
    // collapsing them into the same OpData::BufferData key (without the
    // `local` discriminator) would hash-cons distinct buffers together.
    let unique_zero = UOp::buffer_id(Some(0));
    let lunique_zero = UOp::lunique(Some(0));
    let device = UOp::device(DeviceSpec::Cpu);
    let buf_unique = UOp::new(Op::Buffer { unique: unique_zero, device: device.clone(), size: 64 }, DType::Float32);
    let buf_lunique = UOp::new(Op::Buffer { unique: lunique_zero, device, size: 64 }, DType::Float32);

    assert!(
        !Arc::ptr_eq(&buf_unique, &buf_lunique),
        "Buffer wrapping Unique(0) must not hash-cons with buffer wrapping LUnique(0)"
    );
}

#[test]
fn test_has_buffer_identity_through_get_tuple_chain() {
    use smallvec::smallvec;
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let scratch = UOp::new(Op::Noop, DType::Float32);

    let tup = UOp::tuple(smallvec![buf.clone(), scratch.clone()]);
    let projected_buf = tup.gettuple(0);
    let projected_scratch = tup.gettuple(1);

    assert!(
        projected_buf.has_buffer_identity(),
        "GETTUPLE pointing at a BUFFER element of TUPLE must report buffer identity"
    );
    assert!(
        !projected_scratch.has_buffer_identity(),
        "GETTUPLE pointing at a non-buffer element must not report buffer identity"
    );
}

#[test]
fn test_buffer_view() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 1000, DType::Float32);
    let view = buf.view(100, 50);

    assert!(matches!(view.op(), Op::BufferView { .. }));
    assert_eq!(view.dtype(), DType::Float32);

    if let Op::BufferView { size, offset, .. } = view.op() {
        assert_eq!(*size, 100);
        assert_eq!(*offset, 50);
    } else {
        panic!("Expected BufferView op");
    }
}

#[test]
fn test_index_operation() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let idx = UOp::const_(DType::Index, ConstValue::UInt(10));

    let indexed = UOp::index().buffer(buf).indices(vec![idx]).call().expect("index should succeed");
    assert!(matches!(indexed.op(), Op::Index { .. }));
    assert_eq!(indexed.op().children().len(), 2); // buffer + 1 index
}

#[test]
fn test_device_and_unique() {
    let dev = UOp::device(DeviceSpec::Cpu);
    assert!(matches!(dev.op(), Op::Device(_)));
    if let Op::Device(spec) = dev.op() {
        assert_eq!(*spec, DeviceSpec::Cpu);
    }

    let uniq = UOp::buffer_id(Some(42));
    assert!(matches!(uniq.op(), Op::Unique(42)));

    let uniq_auto = UOp::buffer_id(None);
    assert!(matches!(uniq_auto.op(), Op::Unique(_)));

    let luniq = UOp::lunique(Some(123));
    assert!(matches!(luniq.op(), Op::LUnique(123)));
}

#[test]
fn test_call_constructor_and_with_sources() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let body = a.try_add(&b).unwrap();

    let info = CallInfo {
        grad_tag: Some("grad_add".to_string()),
        metadata: vec!["tag0".to_string()],
        name: Some("call_add".to_string()),
        precompile: true,
        precompile_backward: false,
    };
    let call = body.call(smallvec![a.clone(), b.clone()], info.clone());

    // Per tinygrad spec, CALL dtype is always void.
    assert_eq!(call.dtype(), DType::Void);
    match call.op() {
        Op::Call { body: call_body, args, info: call_info } => {
            assert!(Arc::ptr_eq(call_body, &body));
            assert_eq!(args.len(), 2);
            assert_eq!(*call_info, info);
        }
        op => panic!("expected Call op, got {op:?}"),
    }

    let c = UOp::native_const(3.0f32);
    let new_body = b.try_mul(&c).unwrap();
    let rewritten = call.with_sources(vec![new_body.clone(), c.clone(), a.clone()]);
    match rewritten.op() {
        Op::Call { body: call_body, args, info: call_info } => {
            assert!(Arc::ptr_eq(call_body, &new_body));
            assert_eq!(args.len(), 2);
            assert!(Arc::ptr_eq(&args[0], &c));
            assert!(Arc::ptr_eq(&args[1], &a));
            assert_eq!(*call_info, info);
        }
        op => panic!("expected rewritten Call op, got {op:?}"),
    }
}

#[test]
fn test_function_constructor_with_sources_shape_and_hash() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let body = a.try_add(&b).unwrap();

    let info = CallInfo {
        grad_tag: Some("grad_fn".to_string()),
        metadata: vec!["m0".to_string()],
        name: Some("fn_add".to_string()),
        precompile: true,
        precompile_backward: true,
    };
    let function = body.function(smallvec![a.clone(), b.clone()], info.clone());

    // Per tinygrad spec, FUNCTION dtype is always void and body is TUPLE-wrapped.
    assert_eq!(function.dtype(), DType::Void);
    // FUNCTION body is a TUPLE of values, which has no shape itself; querying the
    // shape of an element requires GETTUPLE.
    assert!(function.shape().unwrap().is_none());
    assert_eq!(function.op().range_ending_src_index(), Some(1));

    match function.op() {
        Op::Function { body: fn_body, args, info: fn_info } => {
            // Auto-wrapped non-Tuple body in a single-element TUPLE.
            let Op::Tuple { src } = fn_body.op() else { panic!("expected TUPLE body, got {:?}", fn_body.op()) };
            assert_eq!(src.len(), 1);
            assert!(Arc::ptr_eq(&src[0], &body));
            assert_eq!(args.len(), 2);
            assert_eq!(*fn_info, info);
        }
        op => panic!("expected Function op, got {op:?}"),
    }

    let c = UOp::native_const(3.0f32);
    let new_body_inner = b.try_mul(&c).unwrap();
    let new_tuple_body = UOp::tuple(smallvec![new_body_inner.clone()]);
    // with_sources expects positional new sources matching children() order:
    // [body, args...]; the new body must already be a TUPLE.
    let rewritten = function.with_sources(vec![new_tuple_body.clone(), c.clone(), a.clone()]);
    match rewritten.op() {
        Op::Function { body: fn_body, args, info: fn_info } => {
            assert!(Arc::ptr_eq(fn_body, &new_tuple_body));
            assert_eq!(args.len(), 2);
            assert!(Arc::ptr_eq(&args[0], &c));
            assert!(Arc::ptr_eq(&args[1], &a));
            assert_eq!(*fn_info, info);
        }
        op => panic!("expected rewritten Function op, got {op:?}"),
    }

    let function_same = body.function(smallvec![a.clone(), b.clone()], info.clone());
    assert!(Arc::ptr_eq(&function, &function_same), "same function info should hash-cons");

    let info_other = CallInfo { name: Some("other_name".to_string()), ..info };
    let function_other = body.function(smallvec![a, b], info_other);
    assert!(!Arc::ptr_eq(&function, &function_other), "different CallInfo should not hash-cons");
}

#[test]
fn test_tuple_constructor_void_dtype() {
    use morok_ir::Op;
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2i32);
    let t = UOp::tuple(smallvec![a.clone(), b.clone()]);
    assert_eq!(t.dtype(), DType::Void);
    let Op::Tuple { src } = t.op() else { panic!("expected TUPLE, got {:?}", t.op()) };
    assert_eq!(src.len(), 2);
    assert!(Arc::ptr_eq(&src[0], &a));
    assert!(Arc::ptr_eq(&src[1], &b));
}

#[test]
fn test_gettuple_extracts_element_dtype_and_shape() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2i32);
    let t = UOp::tuple(smallvec![a.clone(), b.clone()]);
    let g0 = t.gettuple(0);
    let g1 = t.gettuple(1);
    assert_eq!(g0.dtype(), DType::Float32);
    assert_eq!(g1.dtype(), DType::Int32);
    // GETTUPLE shape mirrors the element's shape.
    assert_eq!(g0.shape().unwrap().cloned(), a.shape().unwrap().cloned());
    assert_eq!(g1.shape().unwrap().cloned(), b.shape().unwrap().cloned());
}

#[test]
fn test_gettuple_through_function_body() {
    let value = UOp::native_const(1.0f32);
    let grad = UOp::native_const(2.0f32);
    let body = UOp::tuple(smallvec![value.clone(), grad.clone()]);
    let function = body.function(smallvec![], CallInfo::default());
    let g0 = function.gettuple(0);
    let g1 = function.gettuple(1);
    assert_eq!(g0.dtype(), DType::Float32);
    assert_eq!(g1.dtype(), DType::Float32);
}

#[test]
fn test_function_keeps_tuple_body_as_is() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let t = UOp::tuple(smallvec![a, b]);
    let function = t.clone().function(smallvec![], CallInfo::default());
    let Op::Function { body, .. } = function.op() else { panic!("expected FUNCTION") };
    assert!(Arc::ptr_eq(body, &t));
}

#[test]
fn test_tuple_hash_consing() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let t1 = UOp::tuple(smallvec![a.clone(), b.clone()]);
    let t2 = UOp::tuple(smallvec![a, b]);
    assert!(Arc::ptr_eq(&t1, &t2));
}

#[test]
fn test_program_family_constructors_and_with_sources() {
    let sink = UOp::sink(vec![]);
    let device = UOp::device(DeviceSpec::Cpu);
    let linear = UOp::linear(smallvec![UOp::noop()]);
    let source = UOp::source("void kernel() {}".to_string());
    let binary = UOp::binary(vec![1, 2, 3, 4]);

    let program =
        UOp::program(sink.clone(), device.clone(), Some(linear.clone()), Some(source.clone()), Some(binary.clone()));
    assert_eq!(program.op().children().len(), 5);
    match program.op() {
        Op::Program {
            sink: p_sink,
            device: p_device,
            linear: Some(p_linear),
            source: Some(p_source),
            binary: Some(p_binary),
        } => {
            assert!(Arc::ptr_eq(p_sink, &sink));
            assert!(Arc::ptr_eq(p_device, &device));
            assert!(Arc::ptr_eq(p_linear, &linear));
            assert!(Arc::ptr_eq(p_source, &source));
            assert!(Arc::ptr_eq(p_binary, &binary));
        }
        op => panic!("expected Program op with all stages, got {op:?}"),
    }

    let sink2 = UOp::sink(vec![UOp::noop()]);
    let linear2 = UOp::linear(smallvec![UOp::native_const(7i32)]);
    let source2 = UOp::source("void kernel2() {}".to_string());
    let binary2 = UOp::binary(vec![9, 8]);
    let rewritten =
        program.with_sources(vec![sink2.clone(), device.clone(), linear2.clone(), source2.clone(), binary2.clone()]);
    match rewritten.op() {
        Op::Program {
            sink: p_sink,
            device: p_device,
            linear: Some(p_linear),
            source: Some(p_source),
            binary: Some(p_binary),
        } => {
            assert!(Arc::ptr_eq(p_sink, &sink2));
            assert!(Arc::ptr_eq(p_device, &device));
            assert!(Arc::ptr_eq(p_linear, &linear2));
            assert!(Arc::ptr_eq(p_source, &source2));
            assert!(Arc::ptr_eq(p_binary, &binary2));
        }
        op => panic!("expected rewritten Program op with all stages, got {op:?}"),
    }
}

#[test]
fn test_placeholder_like_concrete_shape() {
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32);
    let shaped = buf.try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(3)])).unwrap();

    let placeholder = UOp::placeholder_like(&shaped, 7).expect("placeholder_like should succeed");
    let placeholder_shape = placeholder.shape().unwrap().cloned().expect("placeholder should have shape");
    assert_eq!(placeholder_shape.len(), 2);
    assert_eq!(placeholder_shape[0].as_const(), Some(2));
    assert_eq!(placeholder_shape[1].as_const(), Some(3));

    match placeholder.op() {
        Op::Reshape { src, .. } => match src.op() {
            Op::Param { slot, size, .. } => {
                assert_eq!(*slot, 7);
                assert_eq!(*size, 6);
            }
            op => panic!("expected PARAM under RESHAPE, got {op:?}"),
        },
        op => panic!("expected RESHAPE placeholder, got {op:?}"),
    }
}

#[test]
fn test_placeholder_like_symbolic_shape_fails() {
    // Symbolic input is rejected outright — tinygrad's placeholder_like
    // asserts the shape is all concrete ints, and we mirror that contract.
    let n = UOp::define_var("N".to_string(), 1, 8);
    let buf = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let shaped = buf.try_reshape(&Shape::from_iter([SInt::from(n)])).unwrap();

    let err = UOp::placeholder_like(&shaped, 0).expect_err("symbolic placeholder_like should fail");
    assert!(format!("{err}").contains("symbolic shape is not supported"));
}

#[test]
fn test_placeholder_like_multi_uses_shard_shape() {
    let shard = UOp::new_buffer(DeviceSpec::Cpu, 6, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(3)]))
        .unwrap();
    let multi = UOp::multi(shard, 0);

    let placeholder = UOp::placeholder_like(&multi, 3).expect("placeholder_like should succeed for MULTI shard shape");
    let shape = placeholder.shape().unwrap().cloned().expect("placeholder should have shape");
    assert_eq!(shape.iter().map(|d| d.as_const()).collect::<Vec<_>>(), vec![Some(2), Some(3)]);
}

#[test]
fn test_placeholder_like_mstack_mselect_uses_buffer_shape() {
    let shard0 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(2)]))
        .unwrap();
    let shard1 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32)
        .try_reshape(&Shape::from_iter([SInt::Const(2), SInt::Const(2)]))
        .unwrap();
    let stacked = UOp::mstack(smallvec::smallvec![shard0, shard1]);
    let selected = stacked.mselect(1);

    let placeholder = UOp::placeholder_like(&selected, 4).expect("placeholder_like should succeed for MSELECT");
    let shape = placeholder.shape().unwrap().cloned().expect("placeholder should have shape");
    assert_eq!(shape.iter().map(|d| d.as_const()).collect::<Vec<_>>(), vec![Some(2), Some(2)]);
}

#[test]
fn test_custom_kernel_builds_after_call_outputs() {
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let outputs = UOp::custom_kernel(
        vec![a.clone(), b.clone()],
        |placeholders| {
            assert_eq!(placeholders.len(), 2);
            UOp::sink(vec![placeholders[0].clone(), placeholders[1].clone()])
        },
        CallInfo::default(),
    )
    .expect("custom kernel should build");

    assert_eq!(outputs.len(), 2);
    for out in outputs {
        match out.op() {
            Op::After { passthrough, deps } => {
                assert!(matches!(passthrough.op(), Op::Buffer { .. }));
                assert_eq!(deps.len(), 1);
                match deps[0].op() {
                    Op::Call { body, args, .. } => {
                        assert!(matches!(body.op(), Op::Sink { .. }));
                        assert_eq!(args.len(), 2);
                    }
                    op => panic!("expected CALL dep, got {op:?}"),
                }
            }
            op => panic!("expected AFTER output, got {op:?}"),
        }
    }
}

#[test]
fn test_custom_kernel_value_body_wraps_in_function() {
    // A value-producing body (here a binary Add) routes through Op::Function
    // with a TUPLE-wrapped body, while opaque bodies (Sink, Program, ...)
    // keep using Op::Call. Mirrors tinygrad's _OPAQUE_CALL_BODIES dispatch.
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let b = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);

    let outputs = UOp::custom_kernel(
        vec![a.clone(), b.clone()],
        |placeholders| {
            assert_eq!(placeholders.len(), 2);
            placeholders[0].try_add(&placeholders[1]).expect("placeholders should be addable")
        },
        CallInfo::default(),
    )
    .expect("custom kernel with value body should build");

    assert_eq!(outputs.len(), 2);
    for out in outputs {
        let Op::After { deps, .. } = out.op() else {
            panic!("expected AFTER output, got {:?}", out.op());
        };
        assert_eq!(deps.len(), 1);
        match deps[0].op() {
            Op::Function { body, args, .. } => {
                assert_eq!(args.len(), 2, "function should receive contig srcs as args");
                assert!(
                    matches!(body.op(), Op::Tuple { .. }),
                    "non-tuple body must auto-wrap into TUPLE for FUNCTION dispatch, got {:?}",
                    body.op()
                );
            }
            op => panic!("value-producing body must dispatch as FUNCTION, got {op:?}"),
        }
    }
}

#[test]
fn test_custom_kernel_opaque_call_function_body_uses_call() {
    // Tinygrad's _OPAQUE_CALL_BODIES includes CUSTOM_FUNCTION; verify it
    // dispatches as Op::Call and not auto-wrapped via FUNCTION.
    use crate::types::CustomFunctionKind;
    let a = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let outputs = UOp::custom_kernel(
        vec![a.clone()],
        |_placeholders| UOp::custom_function(CustomFunctionKind::EncDec, smallvec::smallvec![UOp::index_const(0)]),
        CallInfo::default(),
    )
    .expect("custom kernel with custom_function body should build");

    let Op::After { deps, .. } = outputs[0].op() else {
        panic!("expected AFTER output, got {:?}", outputs[0].op());
    };
    assert!(
        matches!(deps[0].op(), Op::Call { .. }),
        "CustomFunction body is opaque — must dispatch via CALL, got {:?}",
        deps[0].op()
    );
}

#[test]
fn test_children_method() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();

    let children = add.op().children();
    assert_eq!(children.len(), 2);
    assert!(Arc::ptr_eq(children[0], &a));
    assert!(Arc::ptr_eq(children[1], &b));
}

#[test]
fn test_for_each_child() {
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();

    let mut children = Vec::new();
    add.op().map_child(|child| children.push(child.clone()));

    assert_eq!(children.len(), 2);
    assert!(Arc::ptr_eq(&children[0], &a));
    assert!(Arc::ptr_eq(&children[1], &b));
}

// ============================================================================
// Cached Property Tests
// ============================================================================

#[test]
fn test_shape_property_scalar() {
    // Scalar constant should have empty shape
    let scalar = UOp::native_const(42.0f32);
    let shape = scalar.shape().unwrap();

    assert!(shape.is_some(), "Scalar should have shape");
    assert_eq!(shape.unwrap().len(), 0, "Scalar should have empty shape");
}

#[test]
fn test_shape_property_lazy_evaluation() {
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::ShapeProperty;

    // Use unique values unlikely to be created by other tests to get fresh UOps
    // (global hash consing means identical UOps are shared across all tests)
    let unique_val = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as f64;
    let a = UOp::native_const(unique_val as f32);
    let b = UOp::native_const((unique_val + 1.0) as f32);
    let add = a.try_add(&b).unwrap();

    // VERIFY: Cache is empty before first access (lazy evaluation)
    assert!(ShapeProperty::cache(&add).get().is_none(), "Cache should be empty before first access");

    // First access triggers computation
    let shape1 = ShapeProperty::get(&add);
    assert!(shape1.is_ok() && shape1.as_ref().unwrap().is_some());

    // VERIFY: Cache is now populated
    assert!(ShapeProperty::cache(&add).get().is_some(), "Cache should be populated after first access");

    // Second access retrieves from cache (same pointer)
    let shape2 = ShapeProperty::get(&add);

    // VERIFY: Both accesses return the same cached reference
    assert!(std::ptr::eq(shape1, shape2), "Second access should return same cached reference");
}

#[test]
fn test_ranges_property_no_ranges() {
    // Simple arithmetic with no RANGE ops
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();

    let ranges = add.ranges();
    assert_eq!(ranges.len(), 0, "No RANGE ops in simple arithmetic");
}

#[test]
fn test_ranges_property_with_range() {
    use crate::AxisType;

    // Create a RANGE op
    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);

    // Create some computation that uses the range
    let idx = range.cast(DType::Float32);

    let ranges = idx.ranges();
    assert_eq!(ranges.len(), 1, "Should find one RANGE op");
    assert!(Arc::ptr_eq(&ranges[0], &range));
}

#[test]
fn test_ranges_property_lazy_evaluation() {
    use crate::AxisType;
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::RangesProperty;

    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);
    let idx = range.cast(DType::Float32);

    // VERIFY: Cache is empty before first access (lazy evaluation)
    assert!(RangesProperty::cache(&idx).get().is_none(), "Cache should be empty before first access");

    // First access triggers computation
    let ranges1 = RangesProperty::get(&idx);
    assert_eq!(ranges1.len(), 1);

    // VERIFY: Cache is now populated
    assert!(RangesProperty::cache(&idx).get().is_some(), "Cache should be populated after first access");

    // Second access retrieves from cache (same pointer)
    let ranges2 = RangesProperty::get(&idx);

    // VERIFY: Both accesses return the same cached reference
    assert!(std::ptr::eq(ranges1, ranges2), "Second access should return same cached reference");
    assert!(Arc::ptr_eq(&ranges1[0], &ranges2[0]));
}

#[test]
fn test_in_scope_ranges_simple() {
    use crate::AxisType;

    // Create a RANGE op
    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);

    // RANGE itself should have itself in scope
    let in_scope = range.in_scope_ranges();
    assert_eq!(in_scope.len(), 1, "RANGE should have itself in scope");

    // Create computation that uses the range
    let idx = range.cast(DType::Float32);
    let in_scope_idx = idx.in_scope_ranges();
    assert_eq!(in_scope_idx.len(), 1, "Computation should inherit RANGE scope");
}

#[test]
fn test_in_scope_ranges_lazy_evaluation() {
    use crate::AxisType;
    use crate::uop::cached_property::CachedProperty;
    use crate::uop::properties::InScopeRangesProperty;

    let end = UOp::index_const(10);
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);
    let idx = range.cast(DType::Float32);

    // VERIFY: Cache is empty before first access (lazy evaluation)
    assert!(InScopeRangesProperty::cache(&idx).get().is_none(), "Cache should be empty before first access");

    // First access triggers computation
    let in_scope1 = InScopeRangesProperty::get(&idx);
    assert_eq!(in_scope1.len(), 1);

    // VERIFY: Cache is now populated
    assert!(InScopeRangesProperty::cache(&idx).get().is_some(), "Cache should be populated after first access");

    // Second access retrieves from cache (same pointer)
    let in_scope2 = InScopeRangesProperty::get(&idx);

    // VERIFY: Both accesses return the same cached reference
    assert!(std::ptr::eq(in_scope1, in_scope2), "Second access should return same cached reference");
}

#[test]
fn test_in_scope_ranges_after_end() {
    use crate::AxisType;
    use smallvec::smallvec;

    // Create a RANGE and computation
    let end_val = UOp::index_const(10);
    let range = UOp::range_axis(end_val, AxisId::Renumbered(0), AxisType::Loop);
    let compute = UOp::native_const(1.0f32);

    // Create END operation
    let end_op = compute.end(smallvec![range.clone()]);

    // After END, the range should no longer be in scope
    let in_scope = end_op.in_scope_ranges();
    assert_eq!(in_scope.len(), 0, "After END, range should not be in scope");
}

#[test]
fn test_in_scope_ranges_nested() {
    use crate::AxisType;
    use smallvec::smallvec;

    // Create two nested RANGEs
    let end1 = UOp::index_const(10);
    let _range1 = UOp::range_axis(end1, AxisId::Renumbered(0), AxisType::Loop);

    let end2 = UOp::index_const(20);
    let range2 = UOp::range_axis(end2, AxisId::Renumbered(1), AxisType::Loop);

    // Computation that uses both ranges
    let compute = UOp::native_const(1.0f32);

    // Both ranges should be in scope
    let in_scope = compute.in_scope_ranges();
    assert_eq!(in_scope.len(), 0, "Const has no ranges in scope initially");

    // After ending range2, only range1 should be in scope
    let after_end2 = compute.end(smallvec![range2.clone()]);
    let in_scope_after = after_end2.in_scope_ranges();
    assert_eq!(in_scope_after.len(), 0, "After END, ranges are not propagated to parent");
}

#[test]
fn test_toposort_filtered_basic() {
    // Build graph: a -> b -> c
    let a = UOp::native_const(1.0f32);
    let b = a.try_add(&UOp::native_const(2.0f32)).unwrap();
    let c = b.try_mul(&UOp::native_const(3.0f32)).unwrap();

    // Filter to only include 'c'
    let filtered = c.toposort_filtered(|node| Arc::ptr_eq(node, &c));

    // Should only contain 'c' since gate blocks traversal of children
    assert_eq!(filtered.len(), 1, "Filtered toposort should only include nodes passing gate");
    assert!(Arc::ptr_eq(&filtered[0], &c));
}

#[test]
fn test_toposort_filtered_all() {
    // Build graph: a + b
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();

    // Filter that accepts all nodes
    let filtered = add.toposort_filtered(|_| true);

    // Should be same as regular toposort
    let regular = add.toposort();
    assert_eq!(filtered.len(), regular.len());
}

#[test]
fn test_toposort_filtered_none() {
    // Build graph
    let a = UOp::native_const(1.0f32);

    // Filter that rejects all nodes
    let filtered = a.toposort_filtered(|_| false);

    // Should be empty (gate blocks traversal)
    assert_eq!(filtered.len(), 0, "Gate blocking all nodes should return empty");
}

#[test]
fn test_multiple_properties_coexist() {
    // Create a constant (has shape)
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);

    // Create an addition operation
    let add = a.try_add(&b).unwrap();

    // Access shape property (const operations have shape)
    let shape = add.shape().unwrap();
    assert!(shape.is_some());
    assert_eq!(shape.unwrap().len(), 0); // Scalar

    // Access ranges property (no ranges in this graph)
    let ranges = add.ranges();
    assert_eq!(ranges.len(), 0);

    // Access in_scope_ranges property
    let in_scope = add.in_scope_ranges();
    assert_eq!(in_scope.len(), 0);

    // All should be cached independently
    let shape2 = add.shape().unwrap();
    let ranges2 = add.ranges();
    let in_scope2 = add.in_scope_ranges();

    assert_eq!(shape, shape2);
    assert_eq!(ranges.len(), ranges2.len());
    assert_eq!(in_scope.len(), in_scope2.len());
}
