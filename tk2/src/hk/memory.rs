//! HipKittens memory ops (`ops/warp/memory/…`) as thin tk2 wrappers over the builder + movement
//! primitives, in HK's literal IR forms:
//! - `make_srsrc` / `load_global_to_register_buffer` — the legacy `<4 x i32>` SRD + `raw.buffer.load.i128`
//!   DRAM prefetch (GAP-1).
//! - `store_register_buffer_to_shared` / `load` — the asm `ds_write_b64` / `ds_read_b64` commit + gather,
//!   in HK's `i32`-address + `~{memory}` form (Tier-B, `hk_form`).
//! - `G::load` — the cooperative global→shared fill (`buffer_load` → `vmcnt(0)` → `ds_write` → `lgkmcnt(0)`).
//! - `store` — the register→global C store with HK's truncating `fp32 → bf16` (`bits >> 16`).

#![allow(non_snake_case)]

use crate::build::{BF16, Buf, Builder, Effect, F32, Idx, Lds, Val};
use crate::hk::types::{rt_bf, rt_fl, st_bf, st_subtile};
use crate::ir::{FragMap, TileId};
use crate::kernels::offset_by;

/// gfx942 MFMA edge (16×16 fragment) + bf16 item size.
const EDGE: usize = 16;
const ITEMSIZE: usize = 2;
/// bf16 elements in one 128-bit (`float4`) chunk / half.
const CHUNK_ELEMS: usize = 8;
const HALF_ELEMS: usize = 4;

/// The swizzled i32 LDS **byte** address of element `(r, c)` in a `cols`-wide tile — the raw address
/// HK's `ds_read_b64`/`ds_write_b64` asm takes. `col ^ delta` rides on [`Builder::lds_col`] (the
/// composable swizzle hole `SwizzlePass` fills); `raw` orders it after the RAW barrier.
fn lds_addr_i32(b: &mut Builder, lds: Lds<BF16>, r: Idx, c: Idx, cols: usize, raw: &[TileId]) -> Idx {
    let col_part = b.lds_col(r, c, cols);
    let cols_c = b.idx_const(cols as i64);
    let row_off = b.idx_mul(r, cols_c);
    let off = b.idx_add(row_off, col_part);
    let as3 = b.lds_ptr_as3(lds, off, raw);
    b.ptr_to_i32(as3)
}

/// HK's `make_srsrc(ptr, range, 0)` (`util.cuh:79`) — the legacy `<4 x i32>` buffer-resource
/// descriptor `{ptr, range = whole-buffer bytes, config = 0x110000}`. Feeds
/// [`load_global_to_register_buffer`].
pub fn make_srsrc(b: &mut Builder, src: Buf<BF16>, base_off: Idx) -> Idx {
    b.make_srsrc(src, base_off)
}

/// HK's `load_global_to_register_buffer` (`global_to_shared.cuh:103`) — the mainloop DRAM prefetch:
/// build ONE SRD, then one `raw.buffer.load.i128` per 128-bit (`float4`) chunk at each `voffset`
/// (bytes). Returns the loaded `<8×bf16>` chunks (VGPR register buffer). `order` pins the loads into
/// their authoring cluster.
pub fn load_global_to_register_buffer(
    b: &mut Builder,
    src: Buf<BF16>,
    base_off: Idx,
    voffsets: &[Idx],
    order: &[TileId],
) -> Vec<Val<BF16>> {
    let rsrc = make_srsrc(b, src, base_off);
    voffsets.iter().map(|&vo| b.buffer_load_i128::<BF16>(rsrc, vo, CHUNK_ELEMS, order)).collect()
}

/// HK's `store_register_buffer_to_shared` (`global_to_shared.cuh:150`) — the C6 reg→LDS commit: each
/// `float4` chunk is split into two `{x,y}` / `{z,w}` halves (4 bf16 = i64) and written to the
/// swizzled LDS addresses `dst.idx({row, col})` / `dst.idx({row, col+4})` via HK's asm `ds_write_b64`.
/// The per-thread element position is `lane·epl + chunk·8 + half·4` (HK's `chunk_idx`). `prev0` chains
/// the waitcnt-opaque writes in program order (the caller threads a prior drain / write in). Returns
/// the write effects (the RAW barrier / manual `lgkmcnt(0)` drain fences the last one).
pub fn store_register_buffer_to_shared(
    b: &mut Builder,
    dst: st_bf,
    chunks: &[Val<BF16>],
    lane: Idx,
    prev0: Option<TileId>,
) -> Vec<Effect> {
    let cols = dst.cols;
    let epl = chunks.len() * CHUNK_ELEMS;
    let epl_c = b.idx_const(epl as i64);
    let lane_epl = b.idx_mul(lane, epl_c);
    let cols_c = b.idx_const(cols as i64);
    let mut prev = prev0;
    let mut out = Vec::with_capacity(chunks.len() * 2);
    for (i, &chunk) in chunks.iter().enumerate() {
        for half in 0..2 {
            let elems: Vec<Val<BF16>> = (0..HALF_ELEMS).map(|e| b.vec_extract(chunk, half * HALF_ELEMS + e)).collect();
            let hv = b.vec_build(&elems);
            let flat = offset_by(b, lane_epl, i * CHUNK_ELEMS + half * HALF_ELEMS);
            let r = b.idx_div(flat, cols_c);
            let c = b.idx_mod(flat, cols_c);
            let addr = lds_addr_i32(b, dst.lds, r, c, cols, &[]);
            let w = b.ds_write_b64_hk(addr, hv, prev);
            prev = Some(w.dep());
            out.push(w);
        }
    }
    out
}

/// HK's shared→register gather `load(rt, subtile)` (`shared_to_register.cuh:27`) — the operand read:
/// compute the lane's swizzled LDS base ONCE (fragment 0, element 0), then read each of the
/// `n_frags` 16×16 fragments with `ds_read_b64 offset:(i·EDGE·underlying_cols·2)` (= `i·2048` for
/// `underlying_cols = 64`) in HK's `i32`-address asm form. Each read is stored into `dst`'s fragment
/// and returned as the WMMA operand. `raw` orders the base after the RAW barrier.
pub fn load(b: &mut Builder, dst: &mut rt_bf, sub: st_subtile, lane: Idx, raw: &[TileId]) -> Vec<Val<BF16>> {
    let map = FragMap::gfx942_16x16(false); // Row / A operand
    let inner = sub.underlying_cols();
    assert_eq!(dst.frags.len(), sub.n_frags(), "load: rt fragment count must match the subtile's");
    // Base LDS element offset at (fragment 0, element 0): lane_rc + the subtile block offset.
    let zero = b.idx_const(0);
    let (frag_row, frag_col) = b.lane_rc(map, lane, zero);
    let outer = offset_by(b, frag_row, sub.row_offset());
    // HK's `{warp_row, slice}` subtile selector: add the per-warp runtime row-block `blk_row·sub_rows`
    // (uniform per warp) on top of the lane's fragment row + the compile-time block offset.
    let outer = match sub.blk_row_dyn {
        Some(bd) => {
            let sr = b.idx_const(sub.sub_rows as i64);
            let dyn_off = b.idx_mul(bd, sr);
            b.idx_add(outer, dyn_off)
        }
        None => outer,
    };
    let run = offset_by(b, frag_col, sub.col_offset());
    let inner_c = b.idx_const(inner as i64);
    let col_part = b.lds_col(outer, run, inner);
    let row_off = b.idx_mul(outer, inner_c);
    let base_off = b.idx_add(row_off, col_part);
    let as3 = b.lds_ptr_as3(sub.parent.lds, base_off, raw);
    let addr = b.ptr_to_i32(as3);
    let step_bytes = (EDGE * inner * ITEMSIZE) as i64; // fragment-row `offset:` step (= 2048)
    let mut prev: Option<TileId> = None;
    let mut ops = Vec::with_capacity(dst.frags.len());
    for (i, &frag) in dst.frags.iter().enumerate() {
        let off = i as i64 * step_bytes;
        let v: Val<BF16> = b.ds_read_b64_hk(addr, off, map.ept, prev);
        let st = b.store_frag_vec(frag, v);
        prev = Some(st.dep());
        ops.push(b.load_frag_vec_after(frag, &[st.dep()]));
    }
    ops
}

/// HK's cooperative `G::load` (`group/…/global_to_shared.cuh:10` → `load<2,false,…,512>`) — the
/// prologue global→shared fill: `buffer_load` (global→VGPR, here the legacy `raw.buffer.load.i128`) →
/// `s_waitcnt vmcnt(0)` → asm `ds_write_b64` (VGPR→swizzled LDS) → `s_waitcnt lgkmcnt(0)`. Returns the
/// final LDS drain effect (the token the prologue `s_barrier` fences).
pub fn G_load(
    b: &mut Builder,
    dst: st_bf,
    src: Buf<BF16>,
    base_off: Idx,
    voffsets: &[Idx],
    lane: Idx,
    order: &[TileId],
) -> Effect {
    let chunks = load_global_to_register_buffer(b, src, base_off, voffsets, order);
    let last_load = chunks.last().expect("G::load: at least one chunk").id;
    let vm = b.swait_vmcnt(last_load); // drain VMEM before the LDS commit
    let stores = store_register_buffer_to_shared(b, dst, &chunks, lane, Some(vm.dep()));
    let last_store = stores.last().expect("G::load: at least one store").dep();
    b.swait_lgkmcnt(last_store) // drain LDS before any reader
}

/// HK's register→global C `store` (`global_to_register.cuh:173`, Col layout) — the epilogue: for each
/// 16×16 accumulator fragment, `ept` scalar bf16 stores at `lane_rc`-derived `(row, col)` (`+` the
/// fragment's subtile offset), truncating `fp32 → bf16` with HK's `(uint16_t)(bits(f) >> 16)`
/// ([`Builder::bf16_trunc`], NOT RNE). `base` is the tile's element origin; `row_stride` = C's width.
pub fn store(b: &mut Builder, c: Buf<BF16>, acc: &rt_fl, base: Idx, row_stride: i64, lane: Idx) -> Vec<Effect> {
    let (rows, cols) = (acc.rows / EDGE, acc.cols / EDGE);
    let rs = b.idx_const(row_stride);
    let mut out = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let frag = acc.frags[i * cols + j];
            for inner in 0..frag.map.ept {
                let inner_idx = b.idx_const(inner as i64);
                let (row, col) = b.lane_rc(frag.map, lane, inner_idx);
                let row = offset_by(b, row, i * EDGE);
                let col = offset_by(b, col, j * EDGE);
                let row_off = b.idx_mul(row, rs);
                let off = b.idx_add(base, row_off);
                let off = b.idx_add(off, col);
                let v = b.load_frag_elem::<F32>(frag, inner_idx);
                let bf = b.bf16_trunc(v);
                out.push(b.store(c, off, bf));
            }
        }
    }
    out
}
