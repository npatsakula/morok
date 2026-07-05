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
use svod_dtype::{AddrSpace, AmdArch, DType};
use svod_ir::{AxisId, AxisType, ConstValue, KernelInfo, Op, UOp, WmmaMetadata, WmmaUpcastAxes};
use svod_schedule::optimizer::Renderer;

use crate::error::{self, Result};
use crate::ir::{BinOp, IndexOp, Node, Scalar, ScopeAxis, TileId, TileIr};
use crate::kernels::Program;

/// The gfx942 16×16×16 bf16→f32 MFMA descriptor, reproduced verbatim from tk's
/// `wmma_desc`/`wmma_from_tc` (`tk/src/group/mma.rs`): the per-arch×dtype
/// [`TensorCore`](svod_schedule::optimizer::TensorCore) table is the single source
/// of truth, bridged into the IR [`WmmaMetadata`] a hand-built [`Op::Wmma`] consumes.
/// `upcast_axes` are `log2(elements_per_thread)` size-2 entries per operand
/// (descending axis ids from 4 — the values are cosmetic on the expander-free direct
/// path; codegen reads only the sizes). `reduce_axes` is empty (the K reduce is the
/// kernel's own K-loop range). Intrinsic path only — `asm = false`.
///
/// Hardcoded to gfx942: this is the correctness scaffold's only target (per-arch
/// gating is a later pass concern, DESIGN.md §2.8).
fn wmma_desc(dtype_in: &DType) -> WmmaMetadata {
    let ren = Renderer::for_amd_arch(AmdArch::Gfx942);
    let tc = ren
        .tensor_cores
        .iter()
        .find(|tc| &tc.dtype_in == dtype_in && tc.dims == (16, 16, 16))
        .expect("gfx942 has a 16×16×16 WMMA for the operand dtype (bf16/f16)");
    let axes = |ept: usize| -> Vec<(usize, usize)> { (0..(ept as f64).log2() as usize).map(|i| (4 - i, 2)).collect() };
    WmmaMetadata {
        name: format!("WMMA_{}_{}_{}_{:?}_{:?}", tc.dims.0, tc.dims.1, tc.dims.2, tc.dtype_in, tc.dtype_out),
        dims: tc.dims,
        dtype_in: tc.dtype_in.clone(),
        dtype_out: tc.dtype_out.clone(),
        device: ren.device,
        threads: tc.threads,
        upcast_axes: WmmaUpcastAxes {
            a: axes(tc.elements_per_thread.0),
            b: axes(tc.elements_per_thread.1),
            c: axes(tc.elements_per_thread.2),
        },
        reduce_axes: vec![],
        tile_grid: tc.tile_grid,
        asm: false,
    }
}

/// An `Index`-typed integer constant (pre-lowered; `pm_lower_index_dtype` narrows
/// it to a concrete int width before `type_verify`).
fn cidx(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(v))
}

/// Lower the tile-IR reachable from `sink` to a device-UOp SINK, minting a fresh
/// `PARAM` per [`Node::Global`] (the direct-launch ABI). Children have strictly
/// smaller ids than their parents (the arena interns bottom-up), so a single
/// ascending pass suffices — every operand is already lowered.
pub fn lower(ir: &TileIr, sink: TileId, name: &str) -> Arc<UOp> {
    lower_with_globals(ir, sink, name, None)
}

/// Lower `program` against externally-supplied PARAM placeholders — the
/// `custom_kernel` graph-node path ([`crate::graph`]). Each [`Node::Global`]`{slot}`
/// binds `placeholders[slot]` (unwrapped to its flat PARAM) instead of minting a
/// fresh param, so the kernel's ABI globals ARE the graph's buffers. Returns the
/// **raw** SINK: unlike the direct path this re-enters the tensor `prepare()`
/// pipeline, which runs the `Index`-dtype lowering + backend decompose itself (so
/// `lower_and_prepare`'s pre-render rewrites must NOT be applied here — mirrors tk's
/// `graph_launch`, whose closure likewise returns the un-rewritten `finish()` SINK).
pub fn lower_as_graph_node(program: &Program, placeholders: &[Arc<UOp>]) -> Arc<UOp> {
    lower_with_globals(&program.ir, program.sink, &program.name, Some(placeholders))
}

fn lower_with_globals(ir: &TileIr, sink: TileId, name: &str, globals: Option<&[Arc<UOp>]>) -> Arc<UOp> {
    let mut low: Vec<Option<Arc<UOp>>> = vec![None; ir.len()];
    for i in 0..ir.len() {
        let uop = lower_node(ir, TileId(i as u32), &low, name, globals);
        low[i] = Some(uop);
    }
    low[sink.0 as usize].clone().expect("sink was lowered")
}

/// Unwrap a `custom_kernel` placeholder (`PARAM` or `RESHAPE(PARAM)`) to its flat
/// 1-D pointer buffer — hand-built kernels index the flat PARAM directly, never the
/// multi-dim reshape view (mirrors `tk/src/index.rs::flat_ptr`; tk2 stays tk-free).
fn flat_param(placeholder: &Arc<UOp>) -> Arc<UOp> {
    match placeholder.op() {
        Op::Reshape { src, .. } => src.clone(),
        _ => placeholder.clone(),
    }
}

fn get(low: &[Option<Arc<UOp>>], id: TileId) -> Arc<UOp> {
    low[id.0 as usize].clone().expect("operand lowered before use")
}

fn lower_node(ir: &TileIr, id: TileId, low: &[Option<Arc<UOp>>], name: &str, globals: Option<&[Arc<UOp>]>) -> Arc<UOp> {
    match ir.node(id).clone() {
        // Graph-node path: bind the supplied placeholder (flattened) as this global;
        // direct path: mint a fresh PARAM for the slot.
        Node::Global { slot, dtype, len } => match globals {
            Some(ph) => flat_param(&ph[slot as usize]),
            None => {
                let ptr = dtype.ptr(Some(len), AddrSpace::Global).expect("global element is a scalar");
                UOp::param(slot as usize, len, ptr, None)
            }
        },
        Node::DefineReg { id: rid, dtype, len } => UOp::define_reg_typed_with_id(len, dtype, rid as usize),
        Node::DefineLocal { id: lid, dtype, len } => {
            let ptr = dtype.ptr(Some(len), AddrSpace::Local).expect("LDS element is a scalar");
            UOp::define_local(lid as usize, ptr)
        }
        // A fragment reg is a `DefineReg` of `frag.ept` per-lane elements; the lane-map
        // is consumed by the builder-side `lane_rc` addressing, not the lowering.
        Node::DefineFrag { id: rid, dtype, frag } => UOp::define_reg_typed_with_id(frag.ept, dtype, rid as usize),
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
                IndexOp::Mod => a.mod_(&b),
                IndexOp::Div => a.idiv(&b),
                IndexOp::Xor => a.xor(&b),
                IndexOp::Shr => a.shr(&b),
                IndexOp::Shl => a.shl(&b),
            }
        }
        // Identity (PassThrough): the base kernel's LDS col is flat. `SwizzlePass`
        // materialises the XOR before lowering when the swizzle layout is applied, so a
        // surviving `LdsCol` here means the un-swizzled path.
        Node::LdsCol { col, .. } => get(low, col),
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
        // ONE `<ept × dtype>` vector load of the whole per-lane fragment run (offset 0)
        // — the WMMA operand (mirrors tk's `load_vec_at`).
        Node::LoadRegVec { buf, ept, dtype } => {
            let buf = get(low, buf);
            let idx = UOp::index()
                .buffer(buf.clone())
                .indices(vec![cidx(0)])
                .ptr(true)
                .call()
                .expect("LOAD_VEC index construction");
            UOp::load().buffer(buf).index(idx).dtype(dtype.vec(ept).expect("fragment element is a scalar")).call()
        }
        // ONE `<ept×dtype>` vector load of a contiguous run at flat `base` — a scalar-index
        // vector-dtype LOAD (the LDS analog of `LoadRegVec`), which the AMD renderer lowers
        // to `ds_read_b64`/`b128`. `base` must start an aligned contiguous `ept`-run.
        Node::LoadVecAt { buf, base, ept, dtype } => {
            let (buf, base) = (get(low, buf), get(low, base));
            let idx = UOp::index()
                .buffer(buf.clone())
                .indices(vec![base])
                .ptr(true)
                .call()
                .expect("VEC_AT index construction");
            UOp::load().buffer(buf).index(idx).dtype(dtype.vec(ept).expect("vec element is a scalar")).call()
        }
        // ONE vector store of the WMMA result back into the accumulator fragment reg.
        Node::StoreRegVec { buf, value } => {
            let (buf, val) = (get(low, buf), get(low, value));
            let idx =
                UOp::index().buffer(buf).indices(vec![cidx(0)]).ptr(true).call().expect("STORE_VEC index construction");
            idx.store(val)
        }
        // ONE `<ept×dtype>` vector store of a contiguous run at flat `base` — the store
        // mirror of `LoadVecAt`, which the AMD renderer lowers to `ds_write_b64`/`b128`
        // (the vectorised fill). `base` must start an aligned contiguous `ept`-run.
        Node::StoreVecAt { buf, base, value } => {
            let (buf, base, val) = (get(low, buf), get(low, base), get(low, value));
            let idx =
                UOp::index().buffer(buf).indices(vec![base]).ptr(true).call().expect("VEC_AT store index construction");
            idx.store(val)
        }
        // Extract one scalar lane from a vector (`gep`); build a vector from scalars
        // (`vectorize`) — the register-transpose pair (read a column, pack it).
        Node::VecExtract { vec, index, .. } => get(low, vec).gep(vec![index]),
        Node::VecBuild { elements, .. } => {
            let elems: SmallVec<[Arc<UOp>; 4]> = elements.iter().map(|e| get(low, *e)).collect();
            UOp::vectorize(elems)
        }
        // One K-fragment MFMA: `D = A·B + C`, the gfx942 16×16×16 bf16→f32 intrinsic.
        Node::Mma { a, b, c, .. } => {
            let (a, b, c) = (get(low, a), get(low, b), get(low, c));
            let dtype_in = a.dtype().scalar_dtype();
            UOp::wmma(a, b, c, wmma_desc(&dtype_in))
        }
        Node::Barrier { body, deps } => {
            let deps: SmallVec<[Arc<UOp>; 4]> = deps.iter().map(|d| get(low, *d)).collect();
            get(low, body).barrier(deps)
        }
        // A machine-scheduler fence (`sched.barrier`) → a void `Op::Custom` whose `deps` are the
        // ordering anchors (the mask is Rust-substituted, so no `{N}` placeholders → deps are
        // pure happens-after). The AMDGPU backend hoists+dedups the `declare`.
        Node::SchedFence { mask, deps } => {
            let deps: SmallVec<[Arc<UOp>; 4]> = deps.iter().map(|d| get(low, *d)).collect();
            let code = format!(
                "declare void @llvm.amdgcn.sched.barrier(i32)\ncall void @llvm.amdgcn.sched.barrier(i32 {mask})"
            );
            UOp::custom(deps, code, DType::Void)
        }
        // `s_setprio level` as inline `asm sideeffect` — schedule-opaque, pins its own position.
        Node::SetPrio { level, deps } => {
            let deps: SmallVec<[Arc<UOp>; 4]> = deps.iter().map(|d| get(low, *d)).collect();
            UOp::custom(deps, format!("call void asm sideeffect \"s_setprio {level}\", \"\"()"), DType::Void)
        }
        // The wave-phase asymmetric barrier (`if warp_row==eq: s_barrier`) — the predicated
        // `readfirstlane`+`s_cmp`+`s_cbranch`+`s_barrier` asm block (mirrors tk's `wave_phase_barrier`);
        // `deps[0]` is the warp_row operand (cast to i32; value ∈ {0,1} so exact). The skip label is
        // minted per construction so clang `-O3` unroll never duplicates it.
        Node::WaveBarrier { eq, deps } => {
            let mut lowered: SmallVec<[Arc<UOp>; 4]> = deps.iter().map(|d| get(low, *d)).collect();
            lowered[0] = lowered[0].cast(DType::Int32);
            // The node's own interned id is a unique, deterministic label — hash-consing collapses
            // structurally-identical wave barriers to one node (lowered once), so no global counter.
            let label = format!(".Lwpb{}", id.0);
            UOp::custom(
                lowered,
                format!(
                    "call i32 asm sideeffect \"v_readfirstlane_b32 $0, $1\\0A\\09\
                     s_cmp_eq_u32 $0, {eq}\\0A\\09s_cbranch_scc0 {label}\\0A\\09s_barrier\\0A\\09\
                     {label}:\", \"=s,v,~{{{{scc}}}}\"(i32 {{0}})"
                ),
                DType::Int32,
            )
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
