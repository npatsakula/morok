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

use smallvec::{SmallVec, smallvec};
use snafu::ResultExt;
use svod_dtype::{AddrSpace, AmdArch, DType};
use svod_ir::{AxisId, AxisType, ConstValue, KernelInfo, Op, UOp, WmmaMetadata, WmmaUpcastAxes};
use svod_schedule::optimizer::renderer::AMD_CDNA_323208;
use svod_schedule::optimizer::{Renderer, TensorCore};

use crate::error::{self, Result};
use crate::ir::{BinOp, IndexOp, Node, Scalar, ScopeAxis, TileId, TileIr, UnOp};
use crate::kernels::Program;
use crate::shape::{Mfma16x16x16Bf16, Mfma32x32x8Bf16, MfmaShape};

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
fn wmma_desc(dtype_in: &DType, ept_c: usize) -> WmmaMetadata {
    let ren = Renderer::for_amd_arch(AmdArch::Gfx942);
    // The MFMA shape is SELECTED by the accumulator width (`Node::Mma.ept == EPT_C`): bf16 `EPT_C 4`
    // ⇒ 16×16×16 (the registered core), `16` ⇒ 32×32×8 (a direct-path core, deliberately NOT in the
    // BEAM optimizer list — see `AMD_CDNA_323208`). Deriving from the existing `ept` field adds no
    // `Node` variant/field, so the 16×16×16 tile-IR AND its lowered metadata stay byte-identical.
    let tc: TensorCore = if ept_c == Mfma16x16x16Bf16::EPT_C {
        ren.tensor_cores
            .iter()
            .find(|tc| &tc.dtype_in == dtype_in && tc.dims == Mfma16x16x16Bf16::dims())
            .cloned()
            .expect("gfx942 has a 16×16×16 WMMA for the operand dtype (bf16/f16)")
    } else if ept_c == Mfma32x32x8Bf16::EPT_C {
        AMD_CDNA_323208.build(dtype_in.clone(), DType::Float32)
    } else {
        panic!("tk2 wmma_desc: no MFMA shape for accumulator width {ept_c} (bf16 EPT_C ∈ {{4, 16}})")
    };
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
        // Elementwise unary math → `Op::Unary` (the whole transcendental table is native
        // below the boundary; FA's softmax is tk2's first consumer). `try_*` is fallible only
        // on a non-float dtype — a kernel-authoring bug here, so `.expect`.
        Node::Unary { op, x } => {
            let x = get(low, x);
            match op {
                UnOp::Exp2 => x.try_exp2().expect("exp2: float operand"),
                UnOp::Recip => UOp::try_reciprocal(&x).expect("reciprocal: float operand"),
            }
        }
        // Cross-lane gather (`ds_bpermute`): the ONLY inter-lane primitive svod-ir can't
        // express natively, so it is a hand-written inline-LLVM `Op::Custom` (verbatim tk1's
        // `shuffle_lane`). `addr` is the byte lane-address (`src_lane·4`), cast to i32; `data`
        // is bitcast f32→i32 for transport, and the i32 result bitcast back to f32.
        Node::DsBpermute { addr, data } => {
            let addr = get(low, addr).cast(DType::Int32);
            let data = get(low, data).bitcast(DType::Int32);
            let sh = UOp::custom(
                smallvec![addr, data],
                "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)\n\
                 call i32 @llvm.amdgcn.ds.bpermute(i32 {0}, i32 {1})"
                    .to_string(),
                DType::Int32,
            );
            sh.bitcast(DType::Float32)
        }
        // Intra-lane byte permute (`v_perm_b32` / `llvm.amdgcn.perm`) — the register-level 2×2 bf16
        // transpose. `hi`/`lo` are the two source dwords (bitcast to i32); `selector` is baked into the
        // call as an immediate (a compile-time byte-select, not an operand). The i32 result is bitcast to
        // `<2×bf16>` (the two gathered bf16 halves). Hand-written `Op::Custom`, mirroring `DsBpermute`.
        Node::VPerm { hi, lo, selector } => {
            let hi = get(low, hi).bitcast(DType::Int32);
            let lo = get(low, lo).bitcast(DType::Int32);
            let p = UOp::custom(
                smallvec![hi, lo],
                format!(
                    "declare i32 @llvm.amdgcn.perm(i32, i32, i32)\n\
                     call i32 @llvm.amdgcn.perm(i32 {{0}}, i32 {{1}}, i32 {selector})"
                ),
                DType::Int32,
            );
            p.bitcast(DType::BFloat16.vec(2).expect("v_perm: <2×bf16>"))
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
        // One K-fragment MFMA `D = A·B + C` (gfx942 16×16×16 bf16→f32). `asm` flips the shared
        // `render_wmma_amd` to the inline `asm sideeffect` form (schedule-opaque; the `=v,v,v,0` acc
        // tie is identical either way) — the renderer already handles both.
        Node::Mma { a, b, c, ept, asm } => {
            let (a, b, c) = (get(low, a), get(low, b), get(low, c));
            // `ept` (= the accumulator EPT_C) selects the shape in `wmma_desc`; 16×16×16 (ept 4) is
            // byte-identical, 32×32×8 (ept 16) picks the wide core.
            let mut desc = wmma_desc(&a.dtype().scalar_dtype(), ept);
            desc.asm = asm;
            UOp::wmma(a, b, c, desc)
        }
        // The shared base VGPR of an asm gather: INDEX the (After-wrapped) LDS buffer at the
        // lane's `elem==0` flat offset, then `addrspacecast` it to `addrspace(3)` — one custom,
        // one `$1` VGPR that every fragment's `ds_read_b64 offset:N` reads from (mirrors tk's
        // `base_as3`). The `After` on `buf` carries the RAW-barrier ordering (the address depends
        // on the barrier), so the reads can't observe stale LDS.
        Node::LdsPtrAs3 { buf, base } => {
            let (buf, base) = (get(low, buf), get(low, base));
            let idx =
                UOp::index().buffer(buf).indices(vec![base]).ptr(true).call().expect("LdsPtrAs3 INDEX construction");
            UOp::custom(smallvec![idx], "addrspacecast ptr {0} to ptr addrspace(3)".to_string(), DType::Int32)
        }
        // ONE `ds_read_b64 $0, $1 offset:N` gather (gfx942 §5c). The `sideeffect` asm reads
        // `<ept×i16>` from `base_ptr + off_bytes`; `prev` (the prior fragment's store) rides as an
        // ordering-only operand so the reads stay in program order under one loop `END` and cannot
        // hoist across the `s_barrier`s. Bitcast the `<ept×i16>` result to the `<ept×bf16>` operand.
        Node::DsReadB64 { base_ptr, off_bytes, ept, dtype, prev, hk_form } => {
            let base_ptr = get(low, base_ptr);
            let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![base_ptr];
            if let Some(p) = prev {
                deps.push(get(low, p));
            }
            if hk_form {
                // HK's literal IR: i32 raw-address operand, offset as an `i` immediate operand, a
                // `~{memory}` clobber, `i64` result (bitcast to the `<ept×bf16>` operand). `\0A`, `{{...}}`
                // escape to `{...}` and `{{{{memory}}}}`→`~{memory}` under Rust format! then codegen.
                let read = UOp::custom(
                    deps,
                    format!(
                        "call i64 asm sideeffect \"ds_read_b64 $0, $1 offset:$2\\0A\", \
                         \"=v,v,i,~{{{{memory}}}}\"(i32 {{0}}, i64 {off_bytes})"
                    ),
                    DType::Int64,
                );
                read.bitcast(dtype.vec(ept).expect("ds_read_b64: bf16 vec"))
            } else {
                let i16v = DType::Int16.vec(ept).expect("ds_read_b64: i16 vec");
                let read = UOp::custom(
                    deps,
                    format!(
                        "call <{ept} x i16> asm sideeffect \"ds_read_b64 $0, $1 offset:{off_bytes}\", \"=v,v\"\
                         (ptr addrspace(3) {{0}})"
                    ),
                    i16v,
                );
                read.bitcast(dtype.vec(ept).expect("ds_read_b64: bf16 vec"))
            }
        }
        // The `srsrc` buffer descriptor: `make.buffer.rsrc.p0` from the buffer's base pointer (element 0).
        // `num_bytes` is the SRD extent (bounds); `1114112` = 0x110000 is HK's `make_srsrc` config word.
        // The buffer-resource descriptor via `make.buffer.rsrc.p0` of `&buf[base_off]`; `num_bytes` = the
        // SRD byte extent, `1114112` = 0x110000 (HK's `make_srsrc` config, row_stride 0). base_off = 0 in the
        // shipped fixed-base form (the base is loop-invariant, hoisted, materialised once — no advancing
        // base-high to sink below its loads). Lowers to a pointer-typed `Op::Custom`.
        Node::MakeBufferRsrc { buf, base_off, num_bytes } => {
            let (buf, base_off) = (get(low, buf), get(low, base_off));
            let base = UOp::index()
                .buffer(buf)
                .indices(vec![base_off])
                .ptr(true)
                .call()
                .expect("MakeBufferRsrc base-ptr construction");
            UOp::custom(
                smallvec![base],
                format!(
                    "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr, i16, i32, i32)\n\
                     call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr {{0}}, i16 0, i32 {num_bytes}, i32 1114112)"
                ),
                DType::Int64,
            )
        }
        // ONE MUBUF `raw.ptr.buffer.load` (HK's DRAM prefetch): reads the `ept`-element run at
        // `rsrc[voffset]` (bytes), `soffset = 0` (a non-zero soffset is mishandled by config `0x110000`).
        // `rsrc`/`voffset` are the referenced `{0}/{1}` operands; `order` rides as ordering-only operands
        // (the authoring cluster) exactly as `DsReadB64`'s `prev`. The result is `<dwords × i32>` (the
        // canonical `buffer_load_dwordx{dwords}` type; LLVM 18 cannot select `v{ept}i16`), bitcast to the
        // `<ept × elem>` fill chunk. `dwords = ept · sizeof(elem)/4B` from the node's OWN dtype (f16/bf16/f32
        // all size correctly — never a hard-coded width).
        Node::BufferLoadRaw { rsrc, voffset, ept, dtype, order } => {
            let rsrc = get(low, rsrc);
            // Offsets are `i32`; `Index` narrows to i32 OR i64 (data-dependent), so a width-adaptive `cast`
            // to Int32 renders as a noop alias (already i32) or a `trunc` (from i64).
            let vo = get(low, voffset).cast(DType::Int32);
            let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![rsrc, vo];
            for o in order {
                deps.push(get(low, o));
            }
            let chunk_bytes = ept * dtype.bytes();
            assert_eq!(
                chunk_bytes % DType::Int32.bytes(),
                0,
                "buffer_load chunk ({chunk_bytes}B) must be dword-aligned"
            );
            let dwords = chunk_bytes / DType::Int32.bytes();
            let i32v = DType::Int32.vec(dwords).expect("buffer_load: i32 vec");
            let load = UOp::custom(
                deps,
                format!(
                    "declare <{dwords} x i32> @llvm.amdgcn.raw.ptr.buffer.load.v{dwords}i32(ptr addrspace(8), i32, i32, i32)\n\
                     call <{dwords} x i32> @llvm.amdgcn.raw.ptr.buffer.load.v{dwords}i32(ptr addrspace(8) {{0}}, i32 {{1}}, i32 0, i32 0)"
                ),
                i32v,
            );
            load.bitcast(dtype.vec(ept).expect("buffer_load: element vec"))
        }
        // ONE `ds_write_b64 $0, $1 offset:N` LDS store (gfx942 §5c — the commit twin of `DsReadB64`).
        // The `sideeffect` asm writes the bf16 operand (bitcast to `<ept×i16>`) to `base_ptr + off_bytes`;
        // `prev` (the prior fragment's write) rides as an ordering-only operand so the writes stay in
        // program order and cannot hoist across the barriers. Waitcnt-opaque by construction — an
        // `s_barrier` no longer auto-drains it (that is the point), so a `SWaitLgkmcnt` drains it manually.
        Node::DsWriteB64 { base_ptr, off_bytes, value, ept, prev, hk_form } => {
            let base_ptr = get(low, base_ptr);
            if hk_form {
                // HK's literal IR: i32 raw-address (offset folded into the address, NO `offset:`), an
                // `i64` value (the `<4×bf16>` half bitcast to i64), a `~{memory}` clobber. The `{{...}}`
                // are plain-string codegen placeholders/escapes (no Rust format!): `{0}`/`{1}` → operands,
                // `{{memory}}` → `~{memory}`.
                let val = get(low, value).bitcast(DType::Int64);
                let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![base_ptr, val];
                if let Some(p) = prev {
                    deps.push(get(low, p));
                }
                UOp::custom(
                    deps,
                    "call void asm sideeffect \"ds_write_b64 $0, $1\\0A\", \"v,v,~{{memory}}\"(i32 {0}, i64 {1})"
                        .to_string(),
                    DType::Void,
                )
            } else {
                let val = get(low, value).bitcast(DType::Int16.vec(ept).expect("ds_write_b64: i16 vec"));
                let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![base_ptr, val];
                if let Some(p) = prev {
                    deps.push(get(low, p));
                }
                UOp::custom(
                    deps,
                    format!(
                        "call void asm sideeffect \"ds_write_b64 $0, $1 offset:{off_bytes}\", \"v,v\"\
                         (ptr addrspace(3) {{0}}, <{ept} x i16> {{1}})"
                    ),
                    DType::Void,
                )
            }
        }
        // The manual LDS drain (`s_waitcnt lgkmcnt(0)`) — a void `asm sideeffect` ordered after the last
        // commit write (`prev`), re-establishing the store→barrier→load order the waitcnt-opaque asm
        // commit would otherwise lose.
        Node::SWaitLgkmcnt { prev } => UOp::custom(
            smallvec![get(low, prev)],
            "call void asm sideeffect \"s_waitcnt lgkmcnt(0)\", \"\"()".to_string(),
            DType::Void,
        ),
        // The VMEM drain (`s_waitcnt vmcnt(0)`) — the lgkmcnt twin for HK's cooperative G::load.
        Node::SWaitVmcnt { prev } => UOp::custom(
            smallvec![get(low, prev)],
            "call void asm sideeffect \"s_waitcnt vmcnt(0)\", \"\"()".to_string(),
            DType::Void,
        ),
        // `ptrtoint ptr addrspace(3) → i32`: the raw i32 LDS address HK's ds_read/ds_write asm takes.
        // The `ptr` operand is an `LdsPtrAs3` (its RHS is a `ptr addrspace(3)`, so the source type is
        // named literally here — the node's Int32 meta is bookkeeping and never type-annotated).
        Node::PtrToI32 { ptr } => {
            UOp::custom(smallvec![get(low, ptr)], "ptrtoint ptr addrspace(3) {0} to i32".to_string(), DType::Int32)
        }
        // HK's legacy `<4 x i32>` SRD (`make_srsrc`): the ptrtoint→bitcast→shuffle→insertelement chain
        // building `{ptr_lo, ptr_hi, range=num_bytes, config=0x110000}`. Each step is ONE typed custom
        // instruction (the typed-CUSTOM single-instruction rule); the `<4 x i32>` result feeds
        // `raw.buffer.load.i128`. HK inserts config (w3) then range (w2) — mirrored for IR fidelity.
        Node::MakeSrsrc { buf, base_off, num_bytes } => {
            let (buf, base_off) = (get(low, buf), get(low, base_off));
            let base = UOp::index()
                .buffer(buf)
                .indices(vec![base_off])
                .ptr(true)
                .call()
                .expect("MakeSrsrc base-ptr construction");
            let p = UOp::custom(smallvec![base], "ptrtoint ptr {0} to i64".to_string(), DType::Int64);
            let i32x2 = DType::Int32.vec(2).expect("make_srsrc: <2 x i32>");
            let i32x4 = DType::Int32.vec(4).expect("make_srsrc: <4 x i32>");
            let v2 = UOp::custom(smallvec![p], "bitcast i64 {0} to <2 x i32>".to_string(), i32x2);
            let v4 = UOp::custom(
                smallvec![v2],
                "shufflevector <2 x i32> {0}, <2 x i32> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>"
                    .to_string(),
                i32x4.clone(),
            );
            let cfg = UOp::custom(
                smallvec![v4],
                "insertelement <4 x i32> {0}, i32 1114112, i64 3".to_string(),
                i32x4.clone(),
            );
            UOp::custom(smallvec![cfg], format!("insertelement <4 x i32> {{0}}, i32 {num_bytes}, i64 2"), i32x4)
        }
        // ONE `raw.buffer.load.i128` over the legacy `<4 x i32>` SRD (HK's load_global_to_register_buffer).
        // The i128 call result is bookkept as Int64 (never type-annotated — only the following bitcast
        // custom references it, naming `i128` literally); bitcast `i128 → <ept×bf16>` is the fill chunk.
        Node::BufferLoadI128 { rsrc, voffset, ept, dtype, order } => {
            let rsrc = get(low, rsrc);
            let vo = get(low, voffset).cast(DType::Int32);
            let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![rsrc, vo];
            for o in order {
                deps.push(get(low, o));
            }
            let raw = UOp::custom(
                deps,
                "declare i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32>, i32, i32, i32)\n\
                 call i128 @llvm.amdgcn.raw.buffer.load.i128(<4 x i32> {0}, i32 {1}, i32 0, i32 0)"
                    .to_string(),
                DType::Int64,
            );
            let chunk = dtype.vec(ept).expect("buffer_load_i128: element vec");
            UOp::custom(smallvec![raw], format!("bitcast i128 {{0}} to <{ept} x bfloat>"), chunk)
        }
        // HK's fp32→bf16 truncation `(uint16_t)(bits(f) >> 16)`: bitcast float→i32; lshr 16; trunc i16;
        // bitcast bfloat — each ONE typed custom instruction (matches HK's IR + truncating numerics,
        // distinct from svod's default RNE f32→bf16 cast).
        Node::Bf16Trunc { val } => {
            let v = get(low, val);
            let iv = UOp::custom(smallvec![v], "bitcast float {0} to i32".to_string(), DType::Int32);
            let sh = UOp::custom(smallvec![iv], "lshr i32 {0}, 16".to_string(), DType::Int32);
            let tr = UOp::custom(smallvec![sh], "trunc i32 {0} to i16".to_string(), DType::Int16);
            UOp::custom(smallvec![tr], "bitcast i16 {0} to bfloat".to_string(), DType::BFloat16)
        }
        Node::Barrier { body, deps } => {
            let deps: SmallVec<[Arc<UOp>; 4]> = deps.iter().map(|d| get(low, *d)).collect();
            get(low, body).barrier(deps)
        }
        // A bare workgroup barrier (`s.barrier()`) → a void `Op::Custom` with NO `fence acq/rel`, so
        // it neither drains `lgkmcnt` nor acts as a machine-scheduler barrier throttling MFMA overlap.
        // `body` + `deps` are pure happens-after operands (no `{N}` refs). A positional `sched.barrier(0)`
        // is baked in to reproduce the `wall_after_barriers` cluster wall (that codegen pass keys on
        // `Op::Barrier` and would miss this Custom) — pinning the asm `ds_read`/`ds_write` in their cluster.
        Node::BareBarrier { body, deps } => {
            let mut ops: SmallVec<[Arc<UOp>; 4]> = smallvec![get(low, body)];
            ops.extend(deps.iter().map(|d| get(low, *d)));
            UOp::custom(
                ops,
                "declare void @llvm.amdgcn.s.barrier()\ndeclare void @llvm.amdgcn.sched.barrier(i32)\n\
                 call void @llvm.amdgcn.s.barrier()\ncall void @llvm.amdgcn.sched.barrier(i32 0)"
                    .to_string(),
                DType::Void,
            )
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
        // `s_setprio level` via the native intrinsic (NOT inline `asm sideeffect`): the AMDGPU
        // scheduler models the intrinsic form, so all prio pairs survive + stay positioned (the asm
        // form gets DCE'd/merged — HK uses the intrinsic).
        Node::SetPrio { level, deps } => {
            let deps: SmallVec<[Arc<UOp>; 4]> = deps.iter().map(|d| get(low, *d)).collect();
            let code =
                format!("declare void @llvm.amdgcn.s.setprio(i16)\ncall void @llvm.amdgcn.s.setprio(i16 {level})");
            UOp::custom(deps, code, DType::Void)
        }
        // The HK barrier-wall opt-in → the codegen sentinel `wall_after_barriers` keys on.
        Node::SchedWallMarker => svod_codegen::llvm::sched::wall_marker(),
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
