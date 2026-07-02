//! UOp decomposition framework.
//!
//! This module provides conditional decomposition of complex operations into
//! simpler primitives that all backends can handle. Backends that don't support
//! certain transcendental operations can use the pattern-based decompositor
//! to transform them into equivalent primitive operations.
//!
//! # Architecture
//!
//! 1. **Backend provides decomposition patterns** via `Renderer::decompositor()`
//! 2. **Decomposition pass** uses `graph_rewrite_bottom_up` to apply patterns
//! 3. **Each pattern** transforms one op into a subtree of primitive ops
//!
//! # Example
//!
//! ```ignore
//! // In tensor realization, before rendering:
//! if let Some(decompositor) = renderer.decompositor() {
//!     let ast = decompose_with(&kernel.ast, &decompositor);
//! }
//! let rendered = renderer.render(&ast)?;
//! ```

pub mod helpers;
pub mod ptrcat;
pub mod transcendentals;

use std::sync::Arc;

use crate::pattern::TypedPatternMatcher;
use crate::rewrite::graph_rewrite_bottom_up;
use crate::uop::UOp;
use svod_macros::patterns;

use transcendentals::{xcos, xerf, xexp, xexp2, xlog, xlog2, xpow, xrsqrt, xsin, xsqrt, xtan};

/// Vector-of-pointer decomposition for MLIR backend.
///
/// MLIR's LLVM dialect doesn't support `vector<N x ptr>` types. This pattern
/// eliminates VECTORIZE and PtrCat operations on pointer types that weren't
/// consumed by LOAD/STORE patterns during devectorization.
///
/// # Example
///
/// ```ignore
/// impl Renderer for MlirRenderer {
///     fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
///         Some(ptrcat_decomposition_patterns())
///     }
/// }
/// ```
pub fn ptrcat_decomposition_patterns() -> TypedPatternMatcher<()> {
    use crate::DType;

    patterns! {
        // Eliminate VECTORIZE on pointers by returning first element
        // (VECTORIZE on pointers that isn't consumed by GEP is dead code)
        Vectorize { elements } if matches!(elements[0].dtype(), DType::Ptr { .. }) ~> |elements| elements[0].clone(),

        // Eliminate bare PtrCat by returning first pointer
        // (PtrCat not consumed by LOAD/STORE is dead code)
        PtrCat { sources } ~> |sources| sources[0].clone(),
    }
}

/// All decomposition patterns for transcendental operations.
///
/// Returns a `TypedPatternMatcher` that decomposes:
/// - Unary: Exp2, Log2, Exp, Log, Sin, Cos, Tan, Sqrt, Rsqrt, Erf
/// - Binary: Pow
///
/// Backends that don't support these operations natively can use this
/// matcher with `decompose_with()` to decompose them into primitives.
///
/// # Example
///
/// ```ignore
/// impl Renderer for CpuRenderer {
///     fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
///         Some(all_decomposition_patterns())
///     }
/// }
/// ```
pub fn all_decomposition_patterns() -> TypedPatternMatcher<()> {
    patterns! {
        // Transcendental unary ops
        Exp2(src) ~> |src| xexp2(src),
        Log2(src) ~> |src| xlog2(src),
        Exp(src)  ~> |src| xexp(src),
        Log(src)  ~> |src| xlog(src),
        Sin(src)  ~> |src| xsin(src),
        Cos(src)  ~> |src| xcos(src),
        Tan(src)  ~> |src| xtan(src),
        Sqrt(src) ~> |src| xsqrt(src),
        Rsqrt(src) ~> |src| xrsqrt(src),
        Erf(src)  ~> |src| xerf(src),

        // Binary pow: x^y = exp2(y * log2(x))
        Pow(base, exp) ~> |base, exp| xpow(base, exp),
    }
}

/// f32 → bf16 round-to-nearest-even done in the integer domain, emitting no
/// `fptrunc`. amdgcn (LLVM 18) cannot select the vectorized bf16 truncstore that
/// `-O3` forms by fusing `fptrunc float to bfloat` + `store bfloat`; routing the
/// bits through integers and a final `bitcast i16 → bfloat` keeps `fptrunc` away
/// from the store. Port of Tinygrad's `cast_float_to_bf16` (`renderer/cstyle.py`),
/// bit-exact with the native conversion and vector-count-preserving.
fn cast_float_to_bf16(x: &Arc<UOp>) -> Arc<UOp> {
    use crate::DType;
    use svod_dtype::ScalarDType;

    let n = x.dtype().vcount();
    let vec = |s: ScalarDType| DType::Scalar(s).vec(n).expect("scalar dtype is vectorizable");

    // The XLA/Tinygrad round-half-to-even encoding. The two branches don't split
    // cleanly along finite/NaN lines (most NaN and Inf take the `rnd` branch); the
    // whole expression is opaque on purpose and is verified bit-exact, so the
    // bindings below are named after their arithmetic, not a semantic gloss.
    let u = x.bitcast(vec(ScalarDType::UInt32));
    // rnd = u + ((u >> 16) & 1) + 0x7fff.
    let lsb = u.try_shr_op(&u.const_like(16)).and_then(|s| s.try_and_op(&u.const_like(1))).expect("bf16: rne lsb");
    let rnd = u.try_add(&lsb).and_then(|r| r.try_add(&u.const_like(0x7fff))).expect("bf16: rne bias");
    // alt = (u & 0xffff) != 0 ? (u | 0x10000) : u.
    let low_nz =
        u.try_and_op(&u.const_like(0xffff)).and_then(|lo| lo.try_cmpne(&u.const_like(0))).expect("bf16: low16 != 0");
    let or_bit = u.try_or_op(&u.const_like(0x10000)).expect("bf16: or 0x10000");
    let alt = UOp::try_where(low_nz, or_bit, u.clone()).expect("bf16: alt select");
    // bits = ((0 - u) & 0x7f800000) != 0 ? rnd : alt.
    let exp_nz = u
        .neg()
        .try_and_op(&u.const_like(0x7f80_0000))
        .and_then(|e| e.try_cmpne(&u.const_like(0)))
        .expect("bf16: exponent test");
    let bits = UOp::try_where(exp_nz, rnd, alt).expect("bf16: rnd/alt select");
    // High 16 bits are the bf16 payload: truncate to u16, reinterpret as bf16.
    bits.try_shr_op(&bits.const_like(16))
        .expect("bf16: extract high half")
        .cast(vec(ScalarDType::UInt16))
        .bitcast(vec(ScalarDType::BFloat16))
}

/// Decomposition patterns for the AMD backend.
///
/// AMD's hardware `v_exp_f32`/`v_log_f32` (emitted as `@llvm.exp2`/`@llvm.log2`)
/// are lower precision than CPU libm, so the exp/log/trig family is routed
/// through the SLEEF `~1 ULP` polynomials instead. This mirrors tinygrad's
/// `TRANSCENDENTAL=2` force mode (`uop/decompositions.py`), and uses the same
/// coefficients (`transcendentals.rs`).
///
/// `Sqrt`/`Rsqrt` are deliberately **omitted** — AMD's `@llvm.sqrt` is
/// IEEE-correct (~0.5 ULP), better than the polynomial, and tinygrad likewise
/// keeps `SQRT` native in `AMDLLVMRenderer.code_for_op`.
///
/// Every pattern is guarded to `f16`/`f32`/`f64` (tinygrad's
/// `TRANSCENDENTAL_DTYPES`): the polynomials are only defined for those, and
/// integer `Pow` (ONNX `test_pow_types_*`) / `bf16` / `fp8` must keep their
/// native lowering.
pub fn amd_decomposition_patterns() -> TypedPatternMatcher<()> {
    use crate::DType;
    fn transc(d: &DType) -> bool {
        use svod_dtype::ScalarDType::{Float16, Float32, Float64};
        matches!(d.base(), Float16 | Float32 | Float64)
    }
    // `exp2` renders natively on amdgcn for **f32** (`@llvm.exp2.f32` → the hardware
    // `v_exp_f32`; the f64 path is blocked by `guard_unsupported_f64_transcendental`,
    // so only f64 truly needs the Sleef polynomial). Leaving f32 `exp2` native mirrors
    // tinygrad's LLVM renderer (native `EXP2` intrinsic, decompose only f64) and avoids
    // the ~140-`v_cndmask` polynomial on softmax-VALU-bound kernels (flash-attention).
    // f16 keeps the polynomial (no reliable `@llvm.exp2.f16` lowering). f32-only guard,
    // so f16/bf16/f64 `exp2` and every other transcendental are unchanged.
    fn exp2_native(d: &DType) -> bool {
        matches!(d.base(), svod_dtype::ScalarDType::Float32)
    }
    patterns! {
        Exp2(src) if transc(&src.dtype()) && !exp2_native(&src.dtype()) ~> |src| xexp2(src),
        Log2(src) if transc(&src.dtype()) ~> |src| xlog2(src),
        Exp(src)  if transc(&src.dtype()) ~> |src| xexp(src),
        Log(src)  if transc(&src.dtype()) ~> |src| xlog(src),
        Sin(src)  if transc(&src.dtype()) ~> |src| xsin(src),
        Cos(src)  if transc(&src.dtype()) ~> |src| xcos(src),
        Tan(src)  if transc(&src.dtype()) ~> |src| xtan(src),
        Erf(src)  if transc(&src.dtype()) ~> |src| xerf(src),

        // Binary pow: x^y = exp2(y * log2(x))
        Pow(base, exp) if transc(&base.dtype()) ~> |base, exp| xpow(base, exp),

        // bf16/fp8/int fall back to f32 then cast back (tinygrad's cast arm).
        // Int `Pow` would otherwise hit `@llvm.pow.f64`, which amdgcn can't
        // select; bf16/fp8 transcendentals have no native intrinsic either.
        Exp2(src) if !exp2_native(&src.dtype()) ~> |src| xexp2(&src.cast(DType::Float32)).cast(src.dtype()),
        Log2(src) ~> |src| xlog2(&src.cast(DType::Float32)).cast(src.dtype()),
        Exp(src)  ~> |src| xexp(&src.cast(DType::Float32)).cast(src.dtype()),
        Log(src)  ~> |src| xlog(&src.cast(DType::Float32)).cast(src.dtype()),
        Sin(src)  ~> |src| xsin(&src.cast(DType::Float32)).cast(src.dtype()),
        Cos(src)  ~> |src| xcos(&src.cast(DType::Float32)).cast(src.dtype()),
        Tan(src)  ~> |src| xtan(&src.cast(DType::Float32)).cast(src.dtype()),
        Erf(src)  ~> |src| xerf(&src.cast(DType::Float32)).cast(src.dtype()),
        Pow(base, exp) ~> |base, exp| xpow(&base.cast(DType::Float32), &exp.cast(DType::Float32)).cast(base.dtype()),

        // f32 → bf16: integer round (see `cast_float_to_bf16`) instead of the
        // `fptrunc` whose vectorized truncstore amdgcn can't select. The result
        // is a BitCast, never a matching Cast, so the rewrite can't recurse.
        node @ Cast { src, .. }
            if node.dtype().base() == svod_dtype::ScalarDType::BFloat16
                && src.dtype().base() == svod_dtype::ScalarDType::Float32
            ~> cast_float_to_bf16(src),
    }
}

/// Apply decomposition to a UOp graph using the provided pattern matcher.
///
/// Uses `graph_rewrite_bottom_up` to traverse the graph and apply decomposition
/// patterns. This ensures children are processed before parents, which is
/// important for recursive decomposition (e.g., when a decomposition result
/// contains more operations that need decomposition).
///
/// # Arguments
///
/// * `root` - The root UOp of the graph to decompose
/// * `matcher` - The pattern matcher containing decomposition rules
///
/// # Returns
///
/// A new UOp graph with matched operations replaced by their decompositions.
///
/// # Example
///
/// ```ignore
/// let matcher = all_decomposition_patterns();
/// let decomposed = decompose_with(&kernel.ast, &matcher);
/// ```
pub fn decompose_with(root: &Arc<UOp>, matcher: &TypedPatternMatcher<()>) -> Arc<UOp> {
    graph_rewrite_bottom_up(matcher, root.clone(), &mut ())
}
