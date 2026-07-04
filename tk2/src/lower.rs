//! The verified lowering: tile-IR → a device-UOp SINK, then through svod's
//! existing `program_from_sink → do_linearize → type_verify → render/compile`
//! path (DESIGN.md §D; the reference is `tk/src/launch.rs`, whose lower+launch
//! building blocks we reuse from `schedule`/`codegen`/`device` — NOT tk's wrappers).
//!
//! The tile-IR carries ordering as first-class [`After`](crate::ir::Node::After) /
//! [`End`](crate::ir::Node::End) edges, so lowering is a mechanical structural map;
//! the completeness of those edges is the correctness obligation the builder
//! discharges, and `type_verify` is the backstop that the result is spec-valid
//! (integer addresses, matched ALU dtypes, one RANGE per END, movement lowered
//! away) before the renderer can turn a malformed kernel into a GPU fault.

use std::sync::Arc;

use smallvec::SmallVec;
use snafu::ResultExt;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, ConstValue, KernelInfo, Op, UOp};

use crate::error::{self, Result};
use crate::ir::{BinOp, IndexOp, Node, Scalar, ScopeAxis, TileId, TileIr};
use crate::kernels::Program;

/// An `Index`-typed integer constant (pre-lowered; `pm_lower_index_dtype` narrows
/// it to a concrete int width before `type_verify`).
fn cidx(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(v))
}

/// Lower the tile-IR reachable from `sink` to a device-UOp SINK. Children have
/// strictly smaller ids than their parents (the arena interns bottom-up), so a
/// single ascending pass suffices — every operand is already lowered.
pub fn lower(ir: &TileIr, sink: TileId, name: &str) -> Arc<UOp> {
    let mut low: Vec<Option<Arc<UOp>>> = vec![None; ir.len()];
    for i in 0..ir.len() {
        let uop = lower_node(ir, TileId(i as u32), &low, name);
        low[i] = Some(uop);
    }
    low[sink.0 as usize].clone().expect("sink was lowered")
}

fn get(low: &[Option<Arc<UOp>>], id: TileId) -> Arc<UOp> {
    low[id.0 as usize].clone().expect("operand lowered before use")
}

fn lower_node(ir: &TileIr, id: TileId, low: &[Option<Arc<UOp>>], name: &str) -> Arc<UOp> {
    match ir.node(id).clone() {
        Node::Global { slot, dtype, len } => {
            let ptr = dtype.ptr(Some(len), AddrSpace::Global).expect("global element is a scalar");
            UOp::param(slot as usize, len, ptr, None)
        }
        Node::DefineReg { id: rid, dtype, len } => UOp::define_reg_typed_with_id(len, dtype, rid as usize),
        Node::Axis { axis, bound } => {
            let n = match axis {
                ScopeAxis::Grid(a) => format!("gidx{a}"),
                ScopeAxis::Block => "lidx0".to_string(),
            };
            UOp::special(cidx(bound), n)
        }
        Node::Range { id: rid, trips } => {
            UOp::range_axis(cidx(trips), AxisId::Renumbered(rid as usize), AxisType::Loop)
        }
        Node::Const { scalar, dtype } => match scalar {
            Scalar::Int(v) => UOp::const_(dtype, ConstValue::Int(v)),
            Scalar::F32(bits) => UOp::const_(dtype, ConstValue::Float(f32::from_bits(bits) as f64)),
        },
        Node::IndexAlu { op, a, b } => {
            let (a, b) = (get(low, a), get(low, b));
            match op {
                IndexOp::Add => a.add(&b),
                IndexOp::Mul => a.mul(&b),
            }
        }
        Node::LoadGlobal { buf, offset, .. } => {
            let (buf, off) = (get(low, buf), get(low, offset));
            let idx =
                UOp::index().buffer(buf.clone()).indices(vec![off]).ptr(true).call().expect("LOAD index construction");
            UOp::load().buffer(buf).index(idx).call()
        }
        Node::EltwiseBinary { op, a, b } => {
            let (a, b) = (get(low, a), get(low, b));
            match op {
                BinOp::Add => a.add(&b),
                BinOp::Sub => a.sub(&b),
                BinOp::Mul => a.mul(&b),
                BinOp::Max => a.max(&b),
            }
        }
        Node::StoreGlobal { buf, offset, value } => {
            let (buf, off, val) = (get(low, buf), get(low, offset), get(low, value));
            let idx = UOp::index().buffer(buf).indices(vec![off]).ptr(true).call().expect("STORE index construction");
            idx.store(val)
        }
        Node::After { val, deps } => {
            let deps: SmallVec<[Arc<UOp>; 4]> = deps.iter().map(|d| get(low, *d)).collect();
            get(low, val).after(deps)
        }
        Node::End { body, ranges } => {
            let ranges: SmallVec<[Arc<UOp>; 4]> = ranges.iter().map(|r| get(low, *r)).collect();
            get(low, body).end(ranges)
        }
        Node::Sink { roots } => {
            let roots: Vec<Arc<UOp>> = roots.iter().map(|r| get(low, *r)).collect();
            let info = KernelInfo { opts_to_apply: Some(vec![]), name: Some(name.to_string()) };
            UOp::sink_with_info(roots, info)
        }
    }
}

/// Lower + run the pre-render rewrites the direct-launch path needs (the tensor
/// optimizer, which we bypass, normally does these): `Index`-dtype → concrete int
/// (`rule_index_integer`) and the dead-loop-free symbolic fold (which preserves the
/// hand-built END/AFTER loop carries). Mirrors `tk/src/launch.rs::compile`.
pub fn lower_and_prepare(program: &Program) -> Arc<UOp> {
    let sink = lower(&program.ir, program.sink, &program.name);
    let sink = svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), sink, &mut ());
    svod_schedule::graph_rewrite(svod_schedule::symbolic::symbolic_no_dead_loop(), sink, &mut ())
}

/// Lower `program`, linearize it, and run svod's `type_verify` against
/// `spec_program()` — the "verified lowering" contract (DESIGN.md §2.7). Returns
/// the linearized `PROGRAM` UOp on success so callers can inspect it.
pub fn verify(program: &Program) -> Result<Arc<UOp>> {
    let sink = lower_and_prepare(program);
    let prog = svod_codegen::program_pipeline::program_from_sink(sink, svod_dtype::default_device::default_device());
    let linearized = svod_codegen::program_pipeline::do_linearize(&prog)
        .context(error::LinearizeSnafu { name: program.name.clone() })?;

    let Op::Program { linear: Some(lin), .. } = linearized.op() else {
        return error::NoLinearStageSnafu { name: program.name.clone() }.fail();
    };
    svod_schedule::spec::type_verify(lin, &svod_schedule::spec::spec_program())
        .context(error::VerifySnafu { name: program.name.clone() })?;
    Ok(linearized.clone())
}
