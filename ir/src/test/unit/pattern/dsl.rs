//! Surface tests for the `patterns!` proc-macro DSL: every pattern form the macro
//! accepts, and the `RewriteResult` it produces for a matching and a non-matching node.

use std::sync::Arc;

use crate::ops;
use crate::pattern::RewriteResult;
use crate::rewrite::graph_rewrite;
use crate::types::{AddrSpace, BufferizeOpts, ReduceOp};
use crate::{BinaryOp, ConstValue, Op, UOp, UnaryOp};
use smallvec::smallvec;
use svod_dtype::DType;
use svod_macros::patterns;

fn binary(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    let dtype = lhs.dtype();
    UOp::new(Op::Binary(op, lhs, rhs), dtype)
}

fn stage_opts() -> BufferizeOpts {
    BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Global, removable: true }
}

/// `INDEX(const, [offset])` — a well-formed address for the Load/Store tests.
fn address(offset: i64) -> Arc<UOp> {
    UOp::index().buffer(UOp::native_const(42.0f32)).indices(vec![UOp::index_const(offset)]).call().unwrap()
}

fn bool_gate() -> Arc<UOp> {
    UOp::const_(DType::Bool, ConstValue::Int(1))
}

#[track_caller]
fn assert_rewrites_to(result: RewriteResult, expected: &Arc<UOp>) {
    match result {
        RewriteResult::Rewritten(r) => assert!(Arc::ptr_eq(&r, expected), "rewrote to the wrong node"),
        other => panic!("expected a rewrite, got {other:?}"),
    }
}

#[track_caller]
fn assert_no_match(result: RewriteResult) {
    assert!(matches!(result, RewriteResult::NoMatch), "expected NoMatch");
}

#[test]
fn op_patterns_with_literal_constants() {
    let matcher = patterns! {
        GetAddr(src) ~> src,
        Add(x, Const(0)) ~> x,
        Mul(x, Const(1)) ~> x,
        Mul(_, zero @ Const(0)) ~> zero,
    };

    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::UInt8);
    assert_rewrites_to(matcher.rewrite(&buffer.getaddr(None), &mut ()), &buffer);

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);
    let two = UOp::native_const(2i32);

    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Add, x.clone(), zero.clone()), &mut ()), &x);
    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Mul, x.clone(), one.clone()), &mut ()), &x);
    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Mul, x.clone(), zero.clone()), &mut ()), &zero);
    assert_no_match(matcher.rewrite(&binary(BinaryOp::Add, x.clone(), one), &mut ()));
    assert_no_match(matcher.rewrite(&binary(BinaryOp::Mul, x, two), &mut ()));
}

/// `@zero`/`@one` match the int and the float spelling of the constant, in either operand
/// position, both as an anonymous source and behind a `name @ ...` binding.
#[test]
fn special_constant_patterns() {
    let matcher = patterns! {
        Add(x, @zero) ~> x,
        Add(@zero, x) ~> x,
        Mul(x, @one) ~> x,
        Mul(@one, x) ~> x,
        Mul(_, zero @ @zero) ~> zero,
        Mul(zero @ @zero, _) ~> zero,
    };

    for (x, zero, one, two) in [
        (UOp::native_const(42i32), UOp::native_const(0i32), UOp::native_const(1i32), UOp::native_const(2i32)),
        (UOp::native_const(42.0f32), UOp::native_const(0.0f32), UOp::native_const(1.0f32), UOp::native_const(2.0f32)),
    ] {
        assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Add, x.clone(), zero.clone()), &mut ()), &x);
        assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Add, zero.clone(), x.clone()), &mut ()), &x);
        assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Mul, x.clone(), one.clone()), &mut ()), &x);
        assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Mul, one, x.clone()), &mut ()), &x);
        assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Mul, x.clone(), zero.clone()), &mut ()), &zero);
        assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Mul, zero.clone(), x.clone()), &mut ()), &zero);
        assert_no_match(matcher.rewrite(&binary(BinaryOp::Add, x.clone(), two.clone()), &mut ()));
        assert_no_match(matcher.rewrite(&binary(BinaryOp::Mul, x, two), &mut ()));
    }
}

/// Guards run after a structural match and veto it: a block guard inspecting the bound
/// const, and an expression guard using `Arc::ptr_eq`.
#[test]
fn guards_veto_a_structural_match() {
    let matcher = patterns! {
        And(x, y) if Arc::ptr_eq(x, y) ~> x,
        Add(x, c) if {
            match c.op() {
                Op::Const(cv) => {
                    matches!(cv.0, ConstValue::Int(0)) || matches!(cv.0, ConstValue::Float(f) if f == 0.0)
                }
                _ => false,
            }
        } ~> x,
    };

    let a = UOp::native_const(42i32);
    let b = UOp::native_const(99i32);
    assert_rewrites_to(matcher.rewrite(&a.try_and_op(&a).unwrap(), &mut ()), &a);
    assert_no_match(matcher.rewrite(&a.try_and_op(&b).unwrap(), &mut ()));

    let a_f32 = UOp::native_const(42.0f32);
    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Add, a.clone(), UOp::native_const(0i32)), &mut ()), &a);
    assert_rewrites_to(
        matcher.rewrite(&binary(BinaryOp::Add, a_f32.clone(), UOp::native_const(0.0f32)), &mut ()),
        &a_f32,
    );
    assert_no_match(matcher.rewrite(&binary(BinaryOp::Add, a, UOp::native_const(1i32)), &mut ()));
}

/// Repeating a binding name generates a `ptr_eq` check per extra occurrence. Each position
/// must be checked independently — a `DuplicateTracker` that only compared first vs. last
/// let `Where(a, b, a)` through.
#[test]
fn repeated_binding_requires_pointer_equality() {
    let matcher = patterns! {
        Where(x, x, x) ~> x
    };

    let a = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let b = UOp::const_(DType::Bool, ConstValue::Bool(false));

    assert_rewrites_to(matcher.rewrite(&UOp::try_where(a.clone(), a.clone(), a.clone()).unwrap(), &mut ()), &a);
    for (c, t, f) in
        [(a.clone(), a.clone(), b.clone()), (a.clone(), b.clone(), a.clone()), (b.clone(), a.clone(), a.clone())]
    {
        assert_no_match(matcher.rewrite(&UOp::try_where(c, t, f).unwrap(), &mut ()));
    }
}

/// Struct patterns bind non-source fields (`dtype`, `axes`) for use in the guard.
#[test]
fn struct_patterns_bind_non_source_fields() {
    let matcher = patterns! {
        Cast { src: x, dtype } if *dtype == DType::Float32 ~> x,
        Permute { src: x, axes } if axes.len() == 2 ~> x,
    };

    let x_int = UOp::native_const(42i32);
    assert_rewrites_to(matcher.rewrite(&x_int.cast(DType::Float32), &mut ()), &x_int);
    assert_no_match(matcher.rewrite(&x_int.cast(DType::Int64), &mut ()));

    let x = UOp::native_const(1.0f32);
    let permute = |axes: Vec<usize>| UOp::new(Op::Permute(ops::Permute { src: x.clone(), axes }), DType::Float32);
    assert_rewrites_to(matcher.rewrite(&permute(vec![1, 0]), &mut ()), &x);
    assert_no_match(matcher.rewrite(&permute(vec![2, 0, 1]), &mut ()));
}

/// A struct pattern nested inside another binds fields from both levels.
#[test]
fn nested_struct_patterns() {
    let matcher = patterns! {
        Cast { src: Cast { src: x, .. }, dtype } if *dtype == DType::Float32 ~> x,
        Index { buffer: Stage { compute, ranges, .. }, indices } if ranges.len() == indices.len() ~> compute,
    };

    let x_int = UOp::native_const(42i32);
    assert_rewrites_to(matcher.rewrite(&x_int.cast(DType::Int64).cast(DType::Float32), &mut ()), &x_int);
    assert_no_match(matcher.rewrite(&x_int.cast(DType::Float32), &mut ()));

    let compute = UOp::native_const(42.0f32);
    let r0 = UOp::range(UOp::index_const(10), 0);
    let r1 = UOp::range(UOp::index_const(20), 1);
    let stage = UOp::stage(compute.clone(), vec![r0.clone(), r1.clone()], stage_opts());
    let indexed = |indices: Vec<Arc<UOp>>| UOp::index().buffer(stage.clone()).indices(indices).call().unwrap();
    assert_rewrites_to(matcher.rewrite(&indexed(vec![r0.clone(), r1]), &mut ()), &compute);
    assert_no_match(matcher.rewrite(&indexed(vec![r0]), &mut ()));
}

/// `for op in <arity> [...]` emits one rule per listed op, and coexists with the plain
/// rules written before and after it.
#[test]
fn for_loop_expands_the_listed_ops() {
    #[allow(unused_variables)]
    let matcher = patterns! {
        Add(x, @zero) ~> x,
        for op in unary [Sqrt, Exp2] {
            op(c) ~> Arc::clone(c)
        },
        for op in binary [Mul, Sub] {
            op(x, @zero) ~> x
        },
        for op in ternary [Where, MulAcc] {
            op(a, _b, _c) ~> Arc::clone(a)
        },
        Mul(x, @one) ~> x,
    };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Sub] {
        assert_rewrites_to(matcher.rewrite(&binary(op, x.clone(), zero.clone()), &mut ()), &x);
    }
    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Mul, x.clone(), UOp::native_const(1i32)), &mut ()), &x);
    assert_no_match(matcher.rewrite(&x.try_and_op(&zero).unwrap(), &mut ()));

    let f = UOp::native_const(42.0f32);
    assert_rewrites_to(matcher.rewrite(&f.try_sqrt().unwrap(), &mut ()), &f);
    assert_rewrites_to(matcher.rewrite(&f.try_exp2().unwrap(), &mut ()), &f);
    assert_no_match(matcher.rewrite(&f.try_sin().unwrap(), &mut ()));

    let cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let where_op = UOp::try_where(cond.clone(), f.clone(), UOp::native_const(3.0f32)).unwrap();
    assert_rewrites_to(matcher.rewrite(&where_op, &mut ()), &cond);
    let mulacc = UOp::try_mulacc(f.clone(), UOp::native_const(2.0f32), UOp::native_const(3.0f32)).unwrap();
    assert_rewrites_to(matcher.rewrite(&mulacc, &mut ()), &f);
}

/// `[*]` expands to every op of the arity, not just the commonly-used ones.
#[test]
fn for_loop_wildcard_expands_every_op() {
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in unary [*] {
            op(c) if matches!(c.op(), Op::Const(_)) ~> Arc::clone(c)
        },
        for op in binary [*] {
            op(x, @zero) ~> x
        },
        for op in ternary [*] {
            op(a, _b, _c) ~> Arc::clone(a)
        },
    };

    let f = UOp::native_const(42.0f32);
    // `.neg()` lowers to MUL(x, -1), so build the raw Unary(Neg) the wildcard should cover.
    assert_rewrites_to(matcher.rewrite(&UOp::new(Op::Unary(UnaryOp::Neg, f.clone()), f.dtype()), &mut ()), &f);
    assert_rewrites_to(matcher.rewrite(&f.try_sqrt().unwrap(), &mut ()), &f);
    assert_rewrites_to(matcher.rewrite(&f.try_exp2().unwrap(), &mut ()), &f);

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    for op in [BinaryOp::Add, BinaryOp::Mul] {
        assert_rewrites_to(matcher.rewrite(&binary(op, x.clone(), zero.clone()), &mut ()), &x);
    }
    assert_rewrites_to(matcher.rewrite(&x.try_xor_op(&zero).unwrap(), &mut ()), &x);

    let cond = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let where_op = UOp::try_where(cond.clone(), f.clone(), UOp::native_const(3.0f32)).unwrap();
    assert_rewrites_to(matcher.rewrite(&where_op, &mut ()), &cond);
    let mulacc = UOp::try_mulacc(f.clone(), UOp::native_const(2.0f32), UOp::native_const(3.0f32)).unwrap();
    assert_rewrites_to(matcher.rewrite(&mulacc, &mut ()), &f);
}

/// The loop variable `op` is a real `UnaryOp` value inside the rewrite body.
#[test]
fn for_loop_body_can_read_the_op_variable() {
    let matcher = patterns! {
        for op in unary [Sqrt, Exp2] {
            op(x) ~> match op {
                UnaryOp::Sqrt => x.try_exp2().unwrap(),
                _ => x.try_sqrt().unwrap(),
            }
        }
    };

    let x = UOp::native_const(42.0f32);
    assert_rewrites_to(matcher.rewrite(&x.try_sqrt().unwrap(), &mut ()), &x.try_exp2().unwrap());
    assert_rewrites_to(matcher.rewrite(&x.try_exp2().unwrap(), &mut ()), &x.try_sqrt().unwrap());
}

/// Guards and `name @ ...` bindings work inside a for-loop body.
#[test]
fn for_loop_body_supports_guards_and_bindings() {
    #[allow(unused_variables)]
    let matcher = patterns! {
        for op in unary [Sqrt] {
            op(c) if matches!(c.op(), Op::Const(_)) ~> Arc::clone(c)
        },
        for op in unary [Exp2] {
            op(inner @ @const) ~> inner
        },
    };

    let c = UOp::native_const(42.0f32);
    let non_const = binary(BinaryOp::Add, UOp::native_const(1.0f32), UOp::native_const(2.0f32));

    assert_rewrites_to(matcher.rewrite(&c.try_sqrt().unwrap(), &mut ()), &c);
    assert_rewrites_to(matcher.rewrite(&c.try_exp2().unwrap(), &mut ()), &c);
    assert_no_match(matcher.rewrite(&non_const.try_sqrt().unwrap(), &mut ()));
    assert_no_match(matcher.rewrite(&non_const.try_exp2().unwrap(), &mut ()));
}

/// `name @ const(cv)` binds the `ConstValue` itself, for both the infallible (`~>`) and
/// the fallible (`=>`) rewrite arm.
#[test]
fn const_value_is_extracted_from_the_binding() {
    let matcher = patterns! {
        Add(x, _c@const(cv)) if cv == ConstValue::Int(0) ~> x,
        Sqrt(_c@const(cv)) => cv.cast(&DType::Float32).map(|casted| UOp::const_(DType::Float32, casted)),
    };

    let x = UOp::native_const(42i32);
    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Add, x.clone(), UOp::native_const(0i32)), &mut ()), &x);
    assert_no_match(matcher.rewrite(&binary(BinaryOp::Add, x, UOp::native_const(1i32)), &mut ()));

    let c = UOp::native_const(42.0f32);
    assert_rewrites_to(matcher.rewrite(&c.try_sqrt().unwrap(), &mut ()), &c);
}

/// `..` in a tuple pattern makes the rule arity-independent for variable-arity ops.
#[test]
fn rest_pattern_matches_any_arity() {
    let matcher = patterns! {
        end_op @ End(_, ..) ~> {
            let Op::End(ops::End { computation, .. }) = end_op.op() else { unreachable!() };
            Arc::clone(computation)
        },
        reduce_op @ Reduce(_, ..) if matches!(reduce_op.op(), Op::Reduce(ops::Reduce { reduce_op: ReduceOp::Add, .. }))
            ~> UOp::const_(reduce_op.dtype(), ConstValue::Int(99)),
    };

    let src = UOp::native_const(42i32);
    let r0 = UOp::range(UOp::index_const(10), 0);
    let r1 = UOp::range(UOp::index_const(20), 1);

    assert_rewrites_to(matcher.rewrite(&src.end(smallvec![r0.clone()]), &mut ()), &src);
    assert_rewrites_to(matcher.rewrite(&src.end(smallvec![r0.clone(), r1.clone()]), &mut ()), &src);

    let ninety_nine = UOp::const_(src.dtype(), ConstValue::Int(99));
    assert_rewrites_to(matcher.rewrite(&src.reduce(smallvec![r0.clone()], ReduceOp::Add), &mut ()), &ninety_nine);
    assert_rewrites_to(matcher.rewrite(&src.reduce(smallvec![r0.clone(), r1], ReduceOp::Add), &mut ()), &ninety_nine);
    assert_no_match(matcher.rewrite(&src.reduce(smallvec![r0], ReduceOp::Mul), &mut ()));
}

/// `..` in a struct pattern matches a prefix of the sources (Tinygrad's `zip()` semantics)
/// and ignores the remaining fields, whatever their number.
#[test]
fn struct_rest_pattern_ignores_the_remaining_fields() {
    use crate::DeviceSpec;

    let matcher = patterns! {
        Stage { compute: c, .. } ~> c,
        Index { buffer: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
        Copy { src: c, .. } if matches!(c.op(), Op::Const(_)) ~> c,
    };

    let c = UOp::native_const(42.0f32);
    let ranges =
        [UOp::range(UOp::index_const(10), 0), UOp::range(UOp::index_const(20), 1), UOp::range(UOp::index_const(30), 2)];
    for n in 0..=ranges.len() {
        let stage = UOp::stage(c.clone(), ranges[..n].to_vec(), stage_opts());
        assert_rewrites_to(matcher.rewrite(&stage, &mut ()), &c);
    }

    let indices = [UOp::index_const(0), UOp::index_const(1)];
    for n in 1..=indices.len() {
        let index = UOp::index().buffer(c.clone()).indices(indices[..n].to_vec()).call().unwrap();
        assert_rewrites_to(matcher.rewrite(&index, &mut ()), &c);
    }

    assert_rewrites_to(matcher.rewrite(&c.copy_to_device(DeviceSpec::Cuda { device_id: 0 }), &mut ()), &c);
}

/// The four ways to write an `Option` field: `None`, `Some(g)`, `_`, and a bare name that
/// binds the `Option` itself.
#[test]
fn option_field_patterns() {
    let none_only = patterns! { Load { index, alt: None, gate: None } ~> index };
    let some_only = patterns! { Load { index: _, alt: _, gate: Some(g) } ~> g };
    let either = patterns! { Load { index, alt: _, gate: _ } ~> index };
    let bound = patterns! {
        Store { index, value: _, gate } => {
            match gate {
                Some(g) => Some(g.clone()),
                None => Some(index.clone()),
            }
        }
    };

    let idx = address(0);
    let gate = bool_gate();
    let ungated = UOp::load().index(idx.clone()).call();
    let gated = UOp::load().index(idx.clone()).alt(UOp::native_const(0.0f32)).gate(gate.clone()).call();

    assert_rewrites_to(none_only.rewrite(&ungated, &mut ()), &idx);
    assert_no_match(none_only.rewrite(&gated, &mut ()));
    assert_rewrites_to(some_only.rewrite(&gated, &mut ()), &gate);
    assert_no_match(some_only.rewrite(&ungated, &mut ()));
    assert_rewrites_to(either.rewrite(&ungated, &mut ()), &idx);
    assert_rewrites_to(either.rewrite(&gated, &mut ()), &idx);

    let value = UOp::native_const(1.0f32);
    assert_rewrites_to(bound.rewrite(&idx.store(value.clone()), &mut ()), &idx);
    assert_rewrites_to(bound.rewrite(&idx.store_gated(value, gate.clone()), &mut ()), &gate);
}

/// `gate: None` must hold at both levels of a nested pattern.
#[test]
fn nested_option_field_patterns() {
    let matcher = patterns! {
        Store { index: _, value: Load { index: source, alt: None, gate: None }, gate: None } ~> source
    };

    let target = address(0);
    let source = address(1);
    let load = UOp::load().index(source.clone()).call();
    let gated_load = UOp::load().index(source.clone()).alt(UOp::native_const(0.0f32)).gate(bool_gate()).call();

    assert_rewrites_to(matcher.rewrite(&target.store(load.clone()), &mut ()), &source);
    assert_no_match(matcher.rewrite(&target.store_gated(load, bool_gate()), &mut ()));
    assert_no_match(matcher.rewrite(&target.store(gated_load), &mut ()));
}

/// `|` alternatives: over whole patterns, over op names alone, and combined with `@zero`/`@one`.
#[test]
fn alternative_patterns() {
    let whole = patterns! { (Add(x, _y) | Mul(x, _y)) ~> x };
    let op_names = patterns! { (Add | Mul)(x, @zero) ~> x };
    let special = patterns! { (Add(x, @zero) | Add(x, @one)) ~> x };

    let x = UOp::native_const(42i32);
    let zero = UOp::native_const(0i32);
    let one = UOp::native_const(1i32);
    let two = UOp::native_const(2i32);

    for op in [BinaryOp::Add, BinaryOp::Mul] {
        assert_rewrites_to(whole.rewrite(&binary(op, x.clone(), one.clone()), &mut ()), &x);
        assert_rewrites_to(op_names.rewrite(&binary(op, x.clone(), zero.clone()), &mut ()), &x);
    }
    assert_no_match(whole.rewrite(&binary(BinaryOp::Sub, x.clone(), one.clone()), &mut ()));
    assert_no_match(op_names.rewrite(&binary(BinaryOp::Sub, x.clone(), zero.clone()), &mut ()));

    assert_rewrites_to(special.rewrite(&binary(BinaryOp::Add, x.clone(), zero), &mut ()), &x);
    assert_rewrites_to(special.rewrite(&binary(BinaryOp::Add, x.clone(), one), &mut ()), &x);
    assert_no_match(special.rewrite(&binary(BinaryOp::Add, x, two), &mut ()));
}

/// `Op[a, b]` tries both source orderings — including under `graph_rewrite`, which is where
/// a permuted match first failed.
#[test]
fn permutation_pattern_matches_both_orderings() {
    let matcher = patterns! {
        Add[x, @zero] ~> x
    };

    let x = UOp::var("a", DType::Int32, 0, i64::MAX);
    let zero = UOp::native_const(0i32);

    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Add, x.clone(), zero.clone()), &mut ()), &x);
    assert_rewrites_to(matcher.rewrite(&binary(BinaryOp::Add, zero.clone(), x.clone()), &mut ()), &x);

    let folded = graph_rewrite(&matcher, binary(BinaryOp::Add, zero, x.clone()), &mut ());
    assert!(Arc::ptr_eq(&folded, &x));
}

/// `=> |captures| expr` names the bindings the fallible rewrite body needs.
#[test]
fn explicit_capture_list_in_a_fallible_rewrite() {
    let matcher = patterns! {
        Index {
            buffer: Index { buffer: real_buffer, indices: inner_indices },
            indices: outer_indices
        } if outer_indices.len() == 1 && inner_indices.len() == 1 => |real_buffer, inner_indices| {
            UOp::index().buffer(real_buffer.clone()).indices(vec![inner_indices[0].clone()]).call().ok()
        }
    };

    let real_buffer = UOp::native_const(42.0f32);
    let inner = UOp::index().buffer(real_buffer).indices(vec![UOp::index_const(5)]).call().unwrap();
    let outer = UOp::index().buffer(inner.clone()).indices(vec![UOp::index_const(10)]).call().unwrap();

    assert_rewrites_to(matcher.rewrite(&outer, &mut ()), &inner);
}

#[derive(Default)]
struct TestContext {
    counter: u32,
}

impl TestContext {
    fn increment(&mut self) -> u32 {
        self.counter += 1;
        self.counter
    }
}

/// `@context T` makes `ctx: &mut T` available to every rewrite body, both on a direct
/// `rewrite` call and through `graph_rewrite`.
#[test]
fn context_is_threaded_through_rewrite_bodies() {
    let counting = patterns! {
        @context TestContext;

        x if matches!(x.op(), Op::Const(_)) => {
            ctx.increment();
            Some(Arc::clone(x))
        }
    };

    let c = UOp::native_const(42i32);
    let mut ctx = TestContext::default();
    for expected in 1..=2 {
        assert_rewrites_to(counting.rewrite(&c, &mut ctx), &c);
        assert_eq!(ctx.counter, expected);
    }

    let folder = patterns! {
        @context TestContext;

        Add(x, @zero) => {
            ctx.increment();
            Some(Arc::clone(x))
        }
    };

    let x = UOp::native_const(5i32);
    let add = binary(BinaryOp::Add, x.clone(), UOp::native_const(0i32));
    let mut ctx = TestContext::default();
    assert!(Arc::ptr_eq(&graph_rewrite(&folder, add, &mut ctx), &x));
    assert_eq!(ctx.counter, 1);
}

/// `PatternMatcher<C> + PatternMatcher<C>` keeps both halves reachable and sharing one context.
#[test]
fn context_matchers_compose() {
    let adds = patterns! {
        @context TestContext;
        Add(x, @zero) => { ctx.increment(); Some(Arc::clone(x)) }
    };
    let muls = patterns! {
        @context TestContext;
        Mul(x, @one) => { ctx.increment(); ctx.increment(); Some(Arc::clone(x)) }
    };
    let combined = adds + muls;

    let x = UOp::native_const(5i32);
    let mut ctx = TestContext::default();

    let add = binary(BinaryOp::Add, x.clone(), UOp::native_const(0i32));
    assert_rewrites_to(combined.rewrite(&add, &mut ctx), &x);
    assert_eq!(ctx.counter, 1);

    let mul = binary(BinaryOp::Mul, x.clone(), UOp::native_const(1i32));
    assert_rewrites_to(combined.rewrite(&mul, &mut ctx), &x);
    assert_eq!(ctx.counter, 3);
}
