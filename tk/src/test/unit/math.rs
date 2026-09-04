//! Pure graph-shape tests for the register-tile math ops (`TileMathMixin`).

use svod_dtype::DType;
use svod_ir::{BinaryOp, Op, UnaryOp};

use crate::Kernel;
use crate::tiles::{RT_16X16, TileLayout, VecLayout};

fn probe() -> Kernel {
    Kernel::new("math_probe", [1, 1, 1], 64, vec![], crate::ArchCaps::GFX942)
}

/// `exp2` maps an `Exp2` unary over the tile and stores it back.
#[test]
fn test_exp2_emits_unary_exp2() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let out = warp.exp2(a);
    assert!(
        out.uop().toposort().iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Exp2, _))),
        "exp2 emits a Unary(Exp2)"
    );
}

/// `mul_scalar` folds the scalar into a `Mul` against a constant.
#[test]
fn test_mul_scalar_emits_mul_const() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let out = warp.mul_scalar(a, 1.5);
    let topo = out.uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Mul, _, _))), "mul_scalar emits a Mul");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Const(_))), "mul_scalar references a constant operand");
}

/// `div` is `mul(reciprocal)`, not a raw `Fdiv` (faithful to the mixin).
#[test]
fn test_div_is_mul_reciprocal() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let b = warp.ones(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let out = warp.div(a, &b);
    let topo = out.uop().toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Reciprocal, _))), "div lowers to mul(reciprocal)");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Mul, _, _))), "div uses a Mul");
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Fdiv, _, _))), "div is not a raw Fdiv");
}

/// `sub_rv` broadcasts the register vector into the RT (`add(neg)`), reading the
/// vector once per output element.
#[test]
fn test_sub_rv_broadcast_is_add_neg() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((32, 32), DType::Float32, TileLayout::Row, RT_16X16));
    let v = warp.zero_rv(ker.rv(32, DType::Float32, VecLayout::Ortho, RT_16X16));
    let out = warp.sub_rv(a, &v);
    let topo = out.uop().toposort();
    // sub = add(neg): the neg is MUL(-1); the combine is an Add.
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Add, _, _))), "sub_rv combines with Add");
    // The RV buffer is loaded (broadcast) inside the map body.
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Load(..))), "sub_rv loads the broadcast vector element");
}

/// `maximum` on same-shape tiles emits a `Max`.
#[test]
fn test_maximum_emits_max() {
    let ker = probe();
    let warp = ker.warp();
    let a = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let b = warp.zero(ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16));
    let out = warp.maximum(a, &b);
    assert!(
        out.uop().toposort().iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::Max, _, _))),
        "maximum emits a Max"
    );
}
