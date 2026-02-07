//! C source code rendering for individual UOp operations.
//!
//! Generates C expressions/statements for each Op variant.
//! Uses SSA inlining: single-use values are inlined as expressions,
//! multi-use values get local variable declarations.

use std::collections::HashMap;
use std::sync::Arc;

use morok_dtype::{DType, ScalarDType};
use morok_ir::{AxisType, BinaryOp, Op, ReduceOp, TernaryOp, UnaryOp, prelude::*};

use super::types::{c_cast, c_dtype, c_math_fn};

/// Context for C code generation, tracking variable names and SSA inlining.
pub struct CContext {
    /// UOp ID -> C expression or variable name
    names: HashMap<u64, String>,
    /// UOp ID -> reference count (how many times used)
    ref_counts: HashMap<u64, usize>,
    /// Variable counter for generating unique names
    counter: usize,
    /// Current indentation depth
    depth: usize,
    /// Pending reduce accumulator info: reduce_id -> (acc_name, dtype)
    pending_reduces: HashMap<u64, (String, DType)>,
}

impl CContext {
    pub fn new(ref_counts: HashMap<u64, usize>) -> Self {
        Self { names: HashMap::new(), ref_counts, counter: 0, depth: 1, pending_reduces: HashMap::new() }
    }

    /// Get the C expression for a UOp. Panics if not registered.
    pub fn get(&self, uop: &Arc<UOp>) -> &str {
        self.names
            .get(&uop.id)
            .map(|s| s.as_str())
            .unwrap_or_else(|| panic!("UOp {} ({:?}) not in C context", uop.id, uop.op()))
    }

    /// Register a name/expression for a UOp ID.
    pub fn register(&mut self, id: u64, expr: String) {
        self.names.insert(id, expr);
    }

    /// Check if a value should be inlined (single-use, expression-safe).
    pub fn should_inline(&self, id: u64) -> bool {
        self.ref_counts.get(&id).copied().unwrap_or(0) <= 1
    }

    /// Generate a unique variable name with given prefix.
    pub fn next_name(&mut self, prefix: &str) -> String {
        let name = format!("{}{}", prefix, self.counter);
        self.counter += 1;
        name
    }

    /// Get current indentation string.
    pub fn indent(&self) -> String {
        "  ".repeat(self.depth)
    }

    /// Increase indentation depth.
    pub fn push_indent(&mut self) {
        self.depth += 1;
    }

    /// Decrease indentation depth.
    pub fn pop_indent(&mut self) {
        self.depth = self.depth.saturating_sub(1);
    }

    /// Register a pending reduce final load.
    pub fn register_reduce_pending(&mut self, reduce_id: u64, acc_name: String, dtype: DType) {
        self.pending_reduces.insert(reduce_id, (acc_name, dtype));
    }

    /// Take all pending reduces.
    pub fn take_pending_reduces(&mut self) -> HashMap<u64, (String, DType)> {
        std::mem::take(&mut self.pending_reduces)
    }

    /// Emit a C expression, either as an inline expression or a variable declaration.
    /// Returns the name/expression to reference this value.
    pub fn emit_expr(&mut self, uop: &Arc<UOp>, expr: String, prefix: &str, kernel: &mut Vec<String>) -> String {
        if self.should_inline(uop.id) {
            self.register(uop.id, expr.clone());
            expr
        } else {
            let name = self.next_name(prefix);
            let dtype = c_dtype(&uop.dtype());
            let indent = self.indent();
            kernel.push(format!("{indent}{dtype} {name} = {expr};"));
            self.register(uop.id, name.clone());
            name
        }
    }
}

/// Render a single UOp to C source code.
///
/// Returns `Some(())` if code was emitted, `None` for meta-ops.
pub fn render_uop(uop: &Arc<UOp>, ctx: &mut CContext, kernel: &mut Vec<String>) -> Option<()> {
    match uop.op() {
        // Meta-ops: no code emitted
        Op::Const(_)
        | Op::VConst { .. }
        | Op::DefineGlobal(_)
        | Op::DefineLocal(_)
        | Op::DefineVar { .. }
        | Op::Noop
        | Op::Sink { .. }
        | Op::Group { .. }
        | Op::Buffer { .. }
        | Op::Unique(_)
        | Op::Device(_)
        | Op::Kernel { .. }
        | Op::Barrier { .. } => None,

        Op::DefineReg { size } => {
            let base_dtype = match uop.dtype() {
                DType::Ptr { base, .. } => base.as_ref().clone(),
                other => other,
            };
            let name = ctx.next_name("reg");
            let indent = ctx.indent();
            kernel.push(format!("{indent}{} {name}[{size}];", c_dtype(&base_dtype)));
            ctx.register(uop.id, name);
            Some(())
        }

        Op::Index { buffer, indices, gate } => {
            let buf = ctx.get(buffer).to_string();

            if indices.is_empty() {
                // No index - just alias the buffer pointer
                ctx.register(uop.id, buf);
            } else {
                let idx = ctx.get(&indices[0]).to_string();

                if let Some(gate_uop) = gate {
                    let gate_val = ctx.get(gate_uop).to_string();
                    let elem_type = match uop.dtype() {
                        DType::Ptr { ref base, .. } => c_dtype(base),
                        ref other => c_dtype(other),
                    };
                    let expr = format!("({gate_val} ? {buf} + {idx} : ({elem_type}*)0)");
                    ctx.emit_expr(uop, expr, "idx", kernel);
                } else {
                    let expr = format!("{buf} + {idx}");
                    ctx.emit_expr(uop, expr, "idx", kernel);
                }
            }
            Some(())
        }

        Op::PointerIndex { ptr, offset } => {
            let ptr_val = ctx.get(ptr).to_string();
            let off_val = ctx.get(offset).to_string();
            let expr = format!("{ptr_val} + {off_val}");
            ctx.emit_expr(uop, expr, "pidx", kernel);
            Some(())
        }

        Op::Load { index, .. } => {
            let idx = ctx.get(index).to_string();
            let load_dtype = uop.dtype();
            // Buffer pointers are declared as scalar types (e.g., float*) in C,
            // so vector loads need an explicit pointer cast (e.g., *((float4*)(ptr))).
            let expr = if load_dtype.vcount() > 1 {
                let cast_type = c_dtype(&load_dtype);
                format!("*(({cast_type}*)({idx}))")
            } else {
                format!("*({idx})")
            };
            ctx.emit_expr(uop, expr, "val", kernel);
            Some(())
        }

        Op::Store { index, value, .. } => {
            let idx = ctx.get(index).to_string();
            let val = ctx.get(value).to_string();
            let indent = ctx.indent();
            let val_dtype = value.dtype();
            // Buffer pointers are declared as scalar types (e.g., float*) in C,
            // so vector stores need an explicit pointer cast.
            if val_dtype.vcount() > 1 {
                let cast_type = c_dtype(&val_dtype);
                kernel.push(format!("{indent}*(({cast_type}*)({idx})) = {val};"));
            } else {
                kernel.push(format!("{indent}*({idx}) = {val};"));
            }
            Some(())
        }

        Op::Binary(op, lhs, rhs) => {
            let l = ctx.get(lhs).to_string();
            let r = ctx.get(rhs).to_string();
            let expr = render_binary(*op, &l, &r, &lhs.dtype());
            ctx.emit_expr(uop, expr, "alu", kernel);
            Some(())
        }

        Op::Unary(op, src) => {
            let s = ctx.get(src).to_string();
            let expr = render_unary(*op, &s, &src.dtype());
            ctx.emit_expr(uop, expr, "alu", kernel);
            Some(())
        }

        Op::Ternary(TernaryOp::Where, cond, t, f) => {
            let c = ctx.get(cond).to_string();
            let tv = ctx.get(t).to_string();
            let fv = ctx.get(f).to_string();
            let expr = format!("({c} ? {tv} : {fv})");
            ctx.emit_expr(uop, expr, "alu", kernel);
            Some(())
        }

        Op::Ternary(TernaryOp::MulAcc, a, b, c) => {
            let av = ctx.get(a).to_string();
            let bv = ctx.get(b).to_string();
            let cv = ctx.get(c).to_string();
            let expr = if a.dtype().is_float() {
                format!("{}({av}, {bv}, {cv})", c_math_fn("fma", &a.dtype()))
            } else {
                format!("(({av} * {bv}) + {cv})")
            };
            ctx.emit_expr(uop, expr, "alu", kernel);
            Some(())
        }

        Op::Cast { src, dtype } => {
            let s = ctx.get(src).to_string();

            // INDEX to Ptr is a no-op in C (INDEX already produces a pointer)
            if matches!(src.op(), Op::Index { .. }) && matches!(dtype, DType::Ptr { .. }) {
                ctx.register(uop.id, s);
                return Some(());
            }

            // Vector casts use __builtin_convertvector for element-wise conversion
            // (a plain C cast would reinterpret bits, not convert values)
            let expr = if dtype.vcount() > 1 && !matches!(dtype, DType::Ptr { .. }) {
                format!("__builtin_convertvector({s}, {})", c_dtype(dtype))
            } else {
                c_cast(&s, &src.dtype(), dtype)
            };
            ctx.emit_expr(uop, expr, "cast", kernel);
            Some(())
        }

        Op::BitCast { src, dtype } => {
            let s = ctx.get(src).to_string();
            let from_type = c_dtype(&src.dtype());
            let to_type = c_dtype(dtype);
            if from_type == to_type {
                ctx.register(uop.id, s);
            } else {
                let expr = format!("__builtin_bit_cast({to_type}, ({from_type})({s}))");
                ctx.emit_expr(uop, expr, "cast", kernel);
            }
            Some(())
        }

        Op::Range { end, axis_id, axis_type, .. } => {
            if matches!(axis_type, AxisType::Thread) {
                return None;
            }
            let end_val = ctx.get(end).to_string();
            let id = axis_id.value();
            let range_dtype = c_dtype(&uop.dtype());
            let var_name = format!("ridx{id}");
            let indent = ctx.indent();
            kernel.push(format!("{indent}for ({range_dtype} {var_name} = 0; {var_name} < {end_val}; {var_name}++) {{"));
            ctx.register(uop.id, var_name);
            ctx.push_indent();
            Some(())
        }

        Op::End { ranges, .. } => {
            for range in ranges.iter() {
                if let Op::Range { axis_type, .. } = range.op() {
                    if matches!(axis_type, AxisType::Thread) {
                        continue;
                    }
                    ctx.pop_indent();
                    let indent = ctx.indent();
                    kernel.push(format!("{indent}}}"));
                }
            }

            // After closing loops, resolve pending reduces.
            // In C, the accumulator variable already holds the final value
            // (unlike LLVM where we need to load from alloca).
            let pending = ctx.take_pending_reduces();
            for (reduce_id, (acc_name, _dtype)) in pending {
                // Re-register the reduce with the accumulator name
                // so downstream users reference the accumulated value.
                ctx.register(reduce_id, acc_name);
            }
            Some(())
        }

        Op::Reduce { src, ranges, reduce_op } => {
            let src_val = ctx.get(src).to_string();
            let dtype = &uop.dtype();

            if ranges.is_empty() {
                // Passthrough reduce
                ctx.register(uop.id, src_val);
            } else {
                // Accumulator was pre-declared in mod.rs with name acc{uop.id}
                let acc_name = ctx.get(uop).to_string();
                let indent = ctx.indent();

                let acc_expr = render_reduce_accumulate(*reduce_op, &acc_name, &src_val, dtype);
                kernel.push(format!("{indent}{acc_expr}"));

                // Register pending for End to emit the final value
                ctx.register_reduce_pending(uop.id, acc_name, dtype.clone());
            }
            Some(())
        }

        Op::Gep { vector, indices } => {
            let vec = ctx.get(vector).to_string();
            if indices.len() == 1 {
                let expr = format!("{vec}[{}]", indices[0]);
                ctx.emit_expr(uop, expr, "gep", kernel);
            } else {
                // Multi-element GEP: build a new vector from extracted elements
                let out_dtype = c_dtype(&uop.dtype());
                let elements: Vec<String> = indices.iter().map(|&i| format!("{vec}[{i}]")).collect();
                let expr = format!("({out_dtype}){{{}}}", elements.join(", "));
                ctx.emit_expr(uop, expr, "gep", kernel);
            }
            Some(())
        }

        Op::Vectorize { elements } => {
            let vals: Vec<String> = elements.iter().map(|e| ctx.get(e).to_string()).collect();
            let out_dtype = c_dtype(&uop.dtype());
            let expr = format!("({out_dtype}){{{}}}", vals.join(", "));
            ctx.emit_expr(uop, expr, "vec", kernel);
            Some(())
        }

        Op::Cat { sources } => {
            render_cat(uop, sources, ctx, kernel);
            Some(())
        }

        Op::PtrCat { sources } => {
            // PtrCat is not typical in C, render as array
            render_cat(uop, sources, ctx, kernel);
            Some(())
        }

        Op::Wmma { a, b, c, metadata } => {
            let a_val = ctx.get(a).to_string();
            let b_val = ctx.get(b).to_string();
            let c_val = ctx.get(c).to_string();
            let expr = format!("__{name}({a_val}, {b_val}, {c_val})", name = metadata.name);
            ctx.emit_expr(uop, expr, "wmma", kernel);
            Some(())
        }

        Op::Contract { src, .. } | Op::Unroll { src, .. } | Op::Detach { src } => {
            let s = ctx.get(src).to_string();
            ctx.register(uop.id, s);
            None
        }

        Op::After { passthrough, .. } => {
            let s = ctx.get(passthrough).to_string();
            ctx.register(uop.id, s);
            None
        }

        Op::Bind { var, value } => {
            let v = ctx.get(value).to_string();
            ctx.register(var.id, v);
            None
        }

        Op::If { condition, .. } => {
            let cond = ctx.get(condition).to_string();
            let indent = ctx.indent();
            kernel.push(format!("{indent}if ({cond}) {{"));
            ctx.push_indent();
            Some(())
        }

        Op::EndIf { .. } => {
            ctx.pop_indent();
            let indent = ctx.indent();
            kernel.push(format!("{indent}}}"));
            Some(())
        }

        _ => {
            let indent = ctx.indent();
            kernel.push(format!("{indent}/* UNSUPPORTED: {:?} */", uop.op().as_ref()));
            None
        }
    }
}

/// Render a binary operation as a C expression.
fn render_binary(op: BinaryOp, l: &str, r: &str, dtype: &DType) -> String {
    match op {
        BinaryOp::Add => format!("({l} + {r})"),
        BinaryOp::Sub => format!("({l} - {r})"),
        BinaryOp::Mul => format!("({l} * {r})"),
        BinaryOp::Fdiv => format!("({l} / {r})"),
        BinaryOp::Idiv => format!("({l} / {r})"),
        BinaryOp::Mod => {
            if dtype.is_float() {
                format!("{}({l}, {r})", c_math_fn("fmod", dtype))
            } else {
                format!("({l} % {r})")
            }
        }
        BinaryOp::Max => {
            if dtype.is_float() {
                format!("{}({l}, {r})", c_math_fn("fmax", dtype))
            } else {
                format!("({l} > {r} ? {l} : {r})")
            }
        }
        BinaryOp::Lt => format!("({l} < {r})"),
        BinaryOp::Le => format!("({l} <= {r})"),
        BinaryOp::Gt => format!("({l} > {r})"),
        BinaryOp::Ge => format!("({l} >= {r})"),
        BinaryOp::Eq => format!("({l} == {r})"),
        BinaryOp::Ne => format!("({l} != {r})"),
        BinaryOp::And => format!("({l} & {r})"),
        BinaryOp::Or => format!("({l} | {r})"),
        BinaryOp::Xor => format!("({l} ^ {r})"),
        BinaryOp::Shl => format!("({l} << {r})"),
        BinaryOp::Shr => format!("({l} >> {r})"),
        BinaryOp::Pow => {
            if dtype.is_float() {
                format!("{}({l}, {r})", c_math_fn("pow", dtype))
            } else {
                // Integer pow via cast to double
                format!("(({})pow((double){l}, (double){r}))", c_dtype(&DType::Scalar(dtype.base())))
            }
        }
        BinaryOp::Threefry => format!("({l} ^ {r})"),
    }
}

/// Render a unary operation as a C expression.
fn render_unary(op: UnaryOp, s: &str, dtype: &DType) -> String {
    match op {
        UnaryOp::Neg => {
            format!("(-{s})")
        }
        UnaryOp::Not => {
            if dtype.is_bool() {
                format!("(!{s})")
            } else {
                format!("(~{s})")
            }
        }
        UnaryOp::Abs => {
            if dtype.is_float() {
                format!("{}({s})", c_math_fn("fabs", dtype))
            } else {
                format!("({s} < 0 ? -{s} : {s})")
            }
        }
        UnaryOp::Sqrt => format!("{}({s})", c_math_fn("sqrt", dtype)),
        UnaryOp::Rsqrt => {
            let one = if matches!(dtype.base(), ScalarDType::Float64) { "1.0" } else { "1.0f" };
            format!("({one} / {}({s}))", c_math_fn("sqrt", dtype))
        }
        UnaryOp::Reciprocal => {
            let one = if matches!(dtype.base(), ScalarDType::Float64) { "1.0" } else { "1.0f" };
            format!("({one} / {s})")
        }
        UnaryOp::Exp => format!("{}({s})", c_math_fn("exp", dtype)),
        UnaryOp::Exp2 => format!("{}({s})", c_math_fn("exp2", dtype)),
        UnaryOp::Log => format!("{}({s})", c_math_fn("log", dtype)),
        UnaryOp::Log2 => format!("{}({s})", c_math_fn("log2", dtype)),
        UnaryOp::Sin => format!("{}({s})", c_math_fn("sin", dtype)),
        UnaryOp::Cos => format!("{}({s})", c_math_fn("cos", dtype)),
        UnaryOp::Tan => format!("{}({s})", c_math_fn("tan", dtype)),
        UnaryOp::Floor => format!("{}({s})", c_math_fn("floor", dtype)),
        UnaryOp::Ceil => format!("{}({s})", c_math_fn("ceil", dtype)),
        UnaryOp::Trunc => format!("{}({s})", c_math_fn("trunc", dtype)),
        UnaryOp::Round => format!("{}({s})", c_math_fn("round", dtype)),
        UnaryOp::Erf => format!("{}({s})", c_math_fn("erf", dtype)),
        UnaryOp::Sign => {
            if dtype.is_float() {
                let zero = if matches!(dtype.base(), ScalarDType::Float64) { "0.0" } else { "0.0f" };
                format!("(({s} > {zero}) - ({s} < {zero}))")
            } else {
                format!("(({s} > 0) - ({s} < 0))")
            }
        }
        UnaryOp::Square => format!("({s} * {s})"),
    }
}

/// Render a reduce accumulation statement.
fn render_reduce_accumulate(op: ReduceOp, acc: &str, val: &str, dtype: &DType) -> String {
    match op {
        ReduceOp::Add => format!("{acc} += {val};"),
        ReduceOp::Mul => format!("{acc} *= {val};"),
        ReduceOp::Max => {
            if dtype.is_float() {
                format!("{acc} = {}({acc}, {val});", c_math_fn("fmax", dtype))
            } else {
                format!("{acc} = ({acc} > {val} ? {acc} : {val});")
            }
        }
        ReduceOp::Min => {
            if dtype.is_float() {
                format!("{acc} = {}({acc}, {val});", c_math_fn("fmin", dtype))
            } else {
                format!("{acc} = ({acc} < {val} ? {acc} : {val});")
            }
        }
    }
}

/// Render a Cat operation (concatenate vectors).
fn render_cat(uop: &Arc<UOp>, sources: &[Arc<UOp>], ctx: &mut CContext, kernel: &mut Vec<String>) {
    let out_dtype = c_dtype(&uop.dtype());
    let mut elements = Vec::new();

    for src in sources {
        let src_val = ctx.get(src).to_string();
        let src_vcount = src.dtype().vcount();
        if src_vcount == 1 {
            elements.push(src_val);
        } else {
            for i in 0..src_vcount {
                elements.push(format!("{src_val}[{i}]"));
            }
        }
    }

    let expr = format!("({out_dtype}){{{}}}", elements.join(", "));
    ctx.emit_expr(uop, expr, "cat", kernel);
}

/// Count references for each UOp ID in the linearized stream.
/// Used to determine which values should be inlined vs declared.
pub fn count_references(nodes: &[Arc<UOp>]) -> HashMap<u64, usize> {
    let mut counts: HashMap<u64, usize> = HashMap::new();
    for node in nodes {
        for child in node.op().children() {
            *counts.entry(child.id).or_insert(0) += 1;
        }
    }
    counts
}
