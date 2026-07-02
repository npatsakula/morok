//! The load/store path: the public [`Group::load`](super::Group)/`store` entry
//! points (their legal address-space pairs resolved at compile time via
//! `LoadInto`/`StoreInto`), the coalesced GLOBAL↔LDS fills (scalar and vectorized,
//! plus the register-staged prefetch), and the GLOBAL/LOCAL↔REG fragment
//! gather/scatter hops.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_ir::{AxisType, ConstValue, UOp};

use super::{Group, MoveIdx, iadd, idiv, idx_mul, imod, imul, lane_rc, wave_offset};
use crate::index::{
    Idx, cidx, flat_index, flat_offset, index_off, index_off_gated, load_at, load_off, load_off_gated, load_vec,
};
use crate::swizzle::Swizzle;
use crate::tile::{GL, RT, ST};
use crate::tiles::TileLayout;

/// Scalar geometry of the coalesced GLOBAL↔LDS fill for one ST tile (the part
/// independent of the global source / tile position). Shared by the direct fill
/// and the register-staged prefetch so both address LDS identically.
struct LdsGeom {
    ept: i64,
    st_cols: i64,
    memcpy_per_row: i64,
    base_rows: i64,
    base_cols: i64,
    total_calls: i64,
    num_valid: i64,
    clamp: bool,
}

/// Shared 128-bit (`vw = 8` bf16) coalesced addressing of the vectorized prefetch —
/// the fields the GLOBAL→VGPR stage (which adds the global base) and the VGPR→LDS
/// commit (which swizzles) both recompute identically. Mirrors [`LdsGeom`].
struct PrefetchVecGeom {
    /// Lane-pass count.
    total_calls: i64,
    /// 128-bit global-load width (8 bf16).
    vw: i64,
    /// Swizzle-order-safe `ds_write_b64` width (4 bf16).
    sw: i64,
    /// Lane source row / within-tile col of the `vw`-run.
    row0: Arc<UOp>,
    col0: Arc<UOp>,
    /// Fragment coordinate of the run (`height`, `width`, in-fragment `row`).
    height: Arc<UOp>,
    width: Arc<UOp>,
    row: Arc<UOp>,
    /// The lane-pass loop range.
    outer: Arc<UOp>,
}

/// Scale a GLOBAL block-index vector for a tile hop: multiply the `axis` index by
/// `row_scale` (the tile's global row span) and index 3 by `col_scale` (its col
/// span), leaving the rest. Shared by the GLOBAL↔LDS fills (row/col span = the ST
/// `rows`/`cols`) and the GLOBAL↔REG fragment gathers (span = the fragment grid
/// `s3*base_rows` / `s2*base_cols`).
fn scaled_idxs(idxs: &[Idx], axis: usize, row_scale: i64, col_scale: i64) -> Vec<Idx> {
    idxs.iter()
        .enumerate()
        .map(|(i, idx)| {
            let mut e = idx.clone();
            if i == axis {
                e = idx_mul(&e, row_scale);
            }
            if i == 3 {
                e = idx_mul(&e, col_scale);
            }
            e
        })
        .collect()
}

impl<'k> Group<'k> {
    /// Move data into `dst` (tinygrad `Group.load`), with the legal (dst, src)
    /// address-space pair resolved at **compile time** via [`LoadInto`](super::LoadInto):
    /// ST←GL (coalesced fill + barrier), RT←ST / RT←GL (fragment gather). An illegal pair
    /// (e.g. RT←RT) has no impl, so it is a compile error — not a runtime panic:
    ///
    /// ```compile_fail
    /// # use svod_tk::{ArchCaps, Kernel, MoveIdx};
    /// # use svod_tk::tiles::{RT_16X16, TileLayout};
    /// # use svod_dtype::DType;
    /// let ker = Kernel::new("x", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    /// let g = ker.warp();
    /// let a = ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16);
    /// let b = ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16);
    /// let _ = g.load(a, b, MoveIdx::default()); // RT ← RT: no LoadInto impl ⇒ won't compile
    /// ```
    ///
    /// # Panics
    /// A LOCAL→REG (`ST` → `RT`) load panics unless the REG tile fits within the
    /// ST tile — its fragment-grid rows and cols must each be `<=` the ST's.
    pub fn load<Dst, Src>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
    where
        Src: super::LoadInto<'k, Dst>,
    {
        src.load_into(self, dst, ix)
    }

    /// Stage one tile of `src` (GLOBAL) into a fresh per-lane register buffer —
    /// the GLOBAL→VGPR half of the register prefetch. Uses the *same*
    /// coalesced per-lane addressing as [`Self::load_global_to_local`], but lands
    /// the loaded (unswizzled) values in a flat `[total_calls, ept]` DEFINE_REG
    /// instead of LDS, so the load can be issued ahead of the consuming MFMAs.
    /// Commit it with [`Self::commit_reg_to_local`] (same `st`/`idxs`/`axis`).
    ///
    /// # Panics
    /// Panics if `axis` is out of range for the GLOBAL source's rank (the
    /// row-stride is the product of the dims after `axis`).
    pub fn stage_global_to_reg(&self, st: &ST, src: &GL, idxs: &[Idx], axis: usize) -> Arc<UOp> {
        let geom = self.lds_fill_geom(st);
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let idxs_t = scaled_idxs(idxs, axis, st.rows as i64, st.cols as i64);
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let stage = self.ker.alloc_reg((geom.total_calls * geom.ept) as usize, st.elem().clone());
        let outer = self.ker.raw_range(geom.total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(geom.ept, AxisType::Upcast);
        let (height, width, row, col) = self.fill_lane_rc(&geom, &outer, &inner);

        let off = iadd(
            &src_i_base,
            &iadd(
                &iadd(&imul(&height, geom.base_rows * row_stride), &imul(&width, geom.base_cols)),
                &iadd(&imul(&row, row_stride), &col),
            ),
        );
        let mut load = load_off(src.uop(), off);
        if src.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        let stage_shape = [geom.total_calls as usize, geom.ept as usize];
        let stored = flat_index(&stage, &stage_shape, &[Idx::from(&outer), Idx::from(&inner)])
            .store(load)
            .end(smallvec![outer, inner]);
        self.ker.push_store(stored.clone(), stage.clone());
        stage.after(smallvec![stored])
    }

    /// Commit a staged register buffer (from [`Self::stage_global_to_reg`]) into
    /// the swizzled LDS tile — the VGPR→LDS `ds_write` half of the prefetch.
    /// Recomputes the identical per-lane addressing. Ends in a workgroup barrier
    /// when `barrier` (the single-buffer commit); the double-buffered pipeline
    /// passes `false` and shares one barrier per iteration.
    pub fn commit_reg_to_local(&self, st: ST, stage: &Arc<UOp>, barrier: bool) -> ST {
        // The LDS destination geometry is fully determined by the tile shape (the
        // global tile position only mattered when *staging* into the registers).
        let geom = self.lds_fill_geom(&st);
        let outer = self.ker.raw_range(geom.total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(geom.ept, AxisType::Upcast);
        let (height, width, row, col) = self.fill_lane_rc(&geom, &outer, &inner);

        let stage_shape = [geom.total_calls as usize, geom.ept as usize];
        let load = load_at(stage, &stage_shape, &[Idx::from(&outer), Idx::from(&inner)]);
        let off = swizzled_st_offset(&st, &Idx::Uop(height), &Idx::Uop(width), &row, &col);
        let stored = index_off(st.uop(), off).store(load).end(smallvec![outer, inner]);
        let stored = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, stored)
    }

    /// The shared 128-bit (`vw = 8` bf16) coalesced addressing of the vectorized
    /// prefetch: for the lane's pass `outer`, the source-row `row0`, the within-tile
    /// `col0`, and the `(height, width, row)` fragment coordinate — identical to
    /// [`Self::load_global_to_local_vec`]. Returns a [`PrefetchVecGeom`] the stage
    /// (which adds the global base) and the commit (which swizzles) reuse.
    fn prefetch_vec_geom(&self, st: &ST) -> PrefetchVecGeom {
        let itemsize = st.elem().base().bytes() as i64;
        assert_eq!(itemsize, 2, "asm prefetch: bf16-only (128-bit = vec8)");
        let vw = 16 / itemsize; // 8 bf16 — the global_load_dwordx4 width
        let sw = 8 / itemsize; // 4 bf16 — the swizzle-order-safe ds_write_b64 width
        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let st_cols = st.cols as i64;
        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let total_elems = st.shape()[n - 4] as i64 * st.shape()[n - 3] as i64 * num_elements;
        let memcpy_per_row = st_cols / vw;
        let slots = self.group_threads() as i64 * vw;
        let total_calls = (total_elems + slots - 1) / slots;
        let num_valid = total_elems / vw;
        let clamp = total_calls * slots != total_elems;

        let outer = self.ker.raw_range(total_calls, AxisType::Loop);
        let mut load_idx = iadd(&imul(&outer, self.group_threads() as i64), &self.laneid());
        if clamp {
            let cond = load_idx.try_cmplt(&cidx(num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(num_valid - 1)).expect("clamp load_idx");
        }
        let row0 = idiv(&load_idx, memcpy_per_row);
        let col0 = imod(&imul(&load_idx, vw), st_cols);
        let height = idiv(&row0, base_rows);
        let row = imod(&row0, base_rows);
        let width = idiv(&col0, base_cols);
        PrefetchVecGeom { total_calls, vw, sw, row0, col0, height, width, row, outer }
    }

    /// **Inline-`asm` vectorized** GLOBAL→VGPR stage (the register-staged prefetch, half 1): one
    /// `global_load_dwordx4` per lane-pass into a flat `[total_calls, vw]` DEFINE_REG,
    /// as an opaque `asm sideeffect` so the load issues at the loop top and its
    /// ~300-cycle latency overlaps the MFMA clusters (the machine scheduler can't sink
    /// it). Same 128-bit coalesced addressing as [`Self::load_global_to_local_vec`].
    /// Commit with [`Self::commit_reg_to_local_vec_asm`] (same `st`/`idxs`/`axis`).
    pub fn stage_global_to_reg_vec_asm(
        &self,
        st: &ST,
        src: &GL,
        idxs: &[Idx],
        axis: usize,
        anchor: Option<&Arc<UOp>>,
    ) -> Arc<UOp> {
        assert_eq!(src.elem(), st.elem(), "asm prefetch: cast unsupported (bf16→bf16 only)");
        let PrefetchVecGeom { total_calls, vw, row0, col0, outer, .. } = self.prefetch_vec_geom(st);
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let idxs_t = scaled_idxs(idxs, axis, st.rows as i64, st.cols as i64);
        let src_i_base = flat_offset(src.shape(), &idxs_t);
        let off = iadd(&src_i_base, &iadd(&imul(&row0, row_stride), &col0));

        // COMPILER-TRACKED 128-bit global load (`global_load_dwordx4`): a plain
        // vectorized LLVM load, NOT opaque asm. Because the backend tracks the
        // outstanding VMEM, the commit's `ds_write` (which reads this staged
        // register as an input operand) gets a PRECISE auto-inserted `s_waitcnt
        // vmcnt(N)` — instead of the conservative full `vmcnt(0) lgkmcnt(0)` drain
        // the opaque-asm load forced (8 per K-iteration). The enclosing per-cluster
        // `s_barrier`s pin the load to its issue cluster so its ~300-cycle latency
        // still overlaps the MFMA clusters.
        let mut loaded = load_vec(src.uop(), off, vw as usize);
        // Anchor the tracked load to its issue cluster: ordering it AFTER `anchor`
        // (e.g. cluster-3's barrier for the B-prefetch) stops the linearizer's
        // toposort from floating the minimally-dependent load to the loop top
        // bunched with the A-prefetch — it lands mid-loop so its latency overlaps
        // the intervening MFMA clusters (the interleaved mid-loop global-load placement).
        if let Some(a) = anchor {
            loaded = loaded.after(smallvec![a.clone()]);
        }

        let stage = self.ker.alloc_reg((total_calls * vw) as usize, st.elem().clone());
        let stage_shape = [total_calls as usize, vw as usize];
        let stored =
            flat_index(&stage, &stage_shape, &[Idx::from(&outer), Idx::Const(0)]).store(loaded).end(smallvec![outer]);
        self.ker.push_store(stored.clone(), stage.clone());
        // PIN the tracked load at this stage cluster: a `sched.barrier(0)` after the
        // staging store forbids the machine scheduler from sinking the load down to
        // (or bunching it with) the commit. Without it the loads hoist/bunch at the
        // loop top adjacent to the commit and their latency is no longer overlapped
        // by the intervening MFMA clusters.
        let pinned = crate::arch::gfx9::sched_barrier(0, stored);
        stage.after(smallvec![pinned])
    }

    /// **Inline-`asm` vectorized** VGPR→LDS commit (the register-staged prefetch, half 2): reads the
    /// staged vec8 and writes it to the XOR-swizzled LDS half as `vw/sw` opaque
    /// `ds_write_b64`s, recomputing the identical addressing as
    /// [`Self::stage_global_to_reg_vec_asm`]. Opaque `asm sideeffect` so the writes stay
    /// *after* the MFMA clusters (the caller threads `st.after([last_mfma])`) instead of
    /// being hoisted. No barrier — the caller fences once at the loop tail.
    pub fn commit_reg_to_local_vec_asm(&self, st: ST, stage: &Arc<UOp>) -> ST {
        let PrefetchVecGeom { vw, sw, col0, height, width, row, outer, .. } = self.prefetch_vec_geom(&st);
        let base_cols = st.base.base.cols as i64;

        // The `vw/sw` swizzle-safe `ds_write_b64`s, chained (each carries the prior as an
        // ordering dep — void asm side-effects can't sit in a GROUP, so they thread into a
        // single terminal the loop `END` scopes).
        let mut prev: Option<Arc<UOp>> = None;
        for j in 0..vw / sw {
            let col = imod(&iadd(&col0, &cidx(j * sw)), base_cols);
            // The sw-wide run staged at reg offset `outer*vw + j*sw`.
            let roff = iadd(&imul(&outer, vw), &cidx(j * sw));
            let val =
                load_vec(stage, roff, sw as usize).bitcast(svod_dtype::DType::Int16.vec(sw as usize).expect("i16 vec"));
            // Swizzled LDS address (generic `ptr`) → addrspace(3) → asm ds_write_b64.
            let dst = index_off(
                st.uop(),
                swizzled_st_offset(&st, &Idx::Uop(height.clone()), &Idx::Uop(width.clone()), &row, &col),
            );
            let as3 = UOp::custom(
                smallvec![dst],
                "addrspacecast ptr {0} to ptr addrspace(3)".to_string(),
                svod_dtype::DType::Int32,
            );
            let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![as3, val];
            // No manual `s_waitcnt`: the staged load is now COMPILER-TRACKED, so the
            // backend auto-inserts the PRECISE `vmcnt` here (the `<4 x i16> {1}`
            // staged-register input creates the data dependency it waits on); the
            // WAR `lgkmcnt(0)` (prior-gather drain before overwriting the single LDS
            // tile) is already supplied by the cluster-5 barrier's `cbar` drain that
            // precedes this commit. This drops the conservative full `vmcnt(0)
            // lgkmcnt(0)` the opaque-asm load used to force.
            if let Some(p) = prev.take() {
                deps.push(p); // ordering only (not referenced in the template)
            }
            prev = Some(UOp::custom(
                deps,
                "call void asm sideeffect \"ds_write_b64 $0, $1 offset:0\", \"v,v\"\
                 (ptr addrspace(3) {0}, <4 x i16> {1})"
                    .to_string(),
                svod_dtype::DType::Void,
            ));
        }
        let stored = prev.expect("ds_write: at least one sw group").end(smallvec![outer]);
        self.finalize_st(st, stored)
    }

    /// Move a register tile `src` out into `dst` (tinygrad `Group.store`), with the
    /// legal address-space pair resolved at **compile time** via [`StoreInto`](super::StoreInto):
    /// RT→ST (fragment scatter, the layout-transpose hop) and RT→GLOBAL (coalesced
    /// write-back). An illegal pair has no impl, so it is a compile error, not a
    /// runtime panic. `ix` carries the wave/global `block` offset and the REG-side
    /// `frag` offset; `ix.axis` is the global-tile row-stride split.
    ///
    /// # Panics
    /// A REG→GLOBAL store panics if `ix.axis` is out of range for the GLOBAL
    /// destination's rank (the row-stride is the product of the dims after it).
    pub fn store<Dst, Src>(&self, dst: Dst, src: Src, ix: MoveIdx) -> Src::Output
    where
        Src: super::StoreInto<'k, Dst>,
    {
        src.store_into(self, dst, ix)
    }

    /// Cross-wave WAR fence over two just-loaded register reads `a`/`b`: builds ONE
    /// workgroup `Barrier` (passthrough `a`, deps = `b` + `extra` — the cross-iteration
    /// prefetch commits the double-buffer pipeline folds in) and returns BOTH reads
    /// re-threaded `.after([sync])`. The barrier is internal (never returned), so a
    /// read cannot be left un-fenced (you get the fenced tiles back) and nothing can
    /// depend on the barrier as a value (the AMD renderer emits the `s.barrier` fence
    /// but registers no SSA value for it). Emits the identical graph as the hand-built
    /// `a.uop().barrier([b] + extra)` + per-read `.after([sync])`.
    pub fn war_fence2<T: crate::tile::RegTile<'k>>(&self, a: T, b: T, extra: &[Arc<UOp>]) -> (T, T) {
        let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![b.uop().clone()];
        deps.extend(extra.iter().cloned());
        let sync = a.uop().barrier(deps);
        (a.after(smallvec![sync.clone()]), b.after(smallvec![sync]))
    }

    /// The [`LdsGeom`] for filling `st` collaboratively across all group
    /// threads (`elements_per_thread`, pass count, last-pass clamp).
    fn lds_fill_geom(&self, st: &ST) -> LdsGeom {
        let ept = st.base.base.elements_per_thread() as i64;
        let st_cols = st.cols as i64;
        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let total_elems = st.shape()[n - 4] as i64 * st.shape()[n - 3] as i64 * num_elements;
        let slots = self.group_threads() as i64 * ept;
        let total_calls = (total_elems + slots - 1) / slots;
        LdsGeom {
            ept,
            st_cols,
            memcpy_per_row: st_cols / ept,
            base_rows,
            base_cols,
            total_calls,
            num_valid: total_elems / ept,
            clamp: total_calls * slots != total_elems,
        }
    }

    /// The `(height, width, row, col)` LDS fragment coordinate this lane fills at
    /// collaborative pass `(outer, inner)` — the shared per-lane addressing of
    /// the direct fill and the register-staged prefetch (over-subscribed last
    /// pass clamps to the final valid fragment, idempotent).
    fn fill_lane_rc(
        &self,
        geom: &LdsGeom,
        outer: &Arc<UOp>,
        inner: &Arc<UOp>,
    ) -> (Arc<UOp>, Arc<UOp>, Arc<UOp>, Arc<UOp>) {
        let mut load_idx = iadd(&imul(outer, self.group_threads() as i64), &self.laneid());
        if geom.clamp {
            let cond = load_idx.try_cmplt(&cidx(geom.num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(geom.num_valid - 1)).expect("clamp load_idx");
        }
        let row0 = idiv(&load_idx, geom.memcpy_per_row);
        let col0 = iadd(&imod(&imul(&load_idx, geom.ept), geom.st_cols), inner);
        (
            idiv(&row0, geom.base_rows),
            idiv(&col0, geom.base_cols),
            imod(&row0, geom.base_rows),
            imod(&col0, geom.base_cols),
        )
    }

    /// Coalesced GLOBAL→LOCAL fill: every group thread streams
    /// `elements_per_thread` contiguous global elements into the swizzled LDS
    /// tile. When `barrier`, it is closed with a workgroup barrier so the
    /// subsequent gather sees it (the default); a caller wanting to decouple the
    /// fill from its sync passes `false` and inserts the barrier itself.
    pub(super) fn load_global_to_local(&self, st: ST, src: &GL, idxs: &[Idx], axis: usize, barrier: bool) -> ST {
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let idxs_t = scaled_idxs(idxs, axis, st.rows as i64, st.cols as i64);
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let ept = st.base.base.elements_per_thread() as i64;
        let st_cols = st.cols as i64;
        let memcpy_per_row = st_cols / ept;
        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let height_dim = st.shape()[n - 4] as i64;
        let width_dim = st.shape()[n - 3] as i64;
        let total_elems = height_dim * width_dim * num_elements;
        let slots = self.group_threads() as i64 * ept;
        // Round the pass count *up*: a tile smaller than one full group-pass (the
        // multi-wave FA 16×64 K/V block streamed by 512 threads) would otherwise
        // floor to zero passes and load nothing.
        let total_calls = (total_elems + slots - 1) / slots;
        // Over-subscribed last pass (more lane-loads than fragment-loads): clamp
        // the load index to the last valid fragment so the excess lanes redo it
        // (idempotent — same source, same swizzled slot) instead of writing past
        // the tile. A no-op when the tile divides the group evenly (matmul,
        // single-warp FA): `clamp` is false and the index passes through.
        let num_valid = total_elems / ept;
        let clamp = total_calls * slots != total_elems;

        let outer = self.ker.raw_range(total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Upcast);
        let laneid = self.laneid();

        let mut load_idx = iadd(&imul(&outer, self.group_threads() as i64), &laneid);
        if clamp {
            let cond = load_idx.try_cmplt(&cidx(num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(num_valid - 1)).expect("clamp load_idx");
        }
        let row0 = idiv(&load_idx, memcpy_per_row);
        let col0 = iadd(&imod(&imul(&load_idx, ept), st_cols), &inner);
        let height = idiv(&row0, base_rows);
        let width = idiv(&col0, base_cols);
        let row = imod(&row0, base_rows);
        let col = imod(&col0, base_cols);

        let off = iadd(
            &src_i_base,
            &iadd(
                &iadd(&imul(&height, base_rows * row_stride), &imul(&width, base_cols)),
                &iadd(&imul(&row, row_stride), &col),
            ),
        );
        let mut load = load_off(src.uop(), off);
        if src.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        let dst_idx = index_off(st.uop(), swizzled_st_offset(&st, &Idx::Uop(height), &Idx::Uop(width), &row, &col));
        let stored = dst_idx.store(load).end(smallvec![outer, inner]);
        let ended = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, ended)
    }

    /// Vectorized GLOBAL→LOCAL fill: the [`Self::load_global_to_local`]
    /// counterpart that issues **128-bit** (`vec8` bf16) coalesced global loads
    /// (one `global_load_dwordx4`/lane) and commits each into the XOR-swizzled
    /// LDS as `vec8/sw` contiguous `vec_sw` stores. The swizzle's XOR delta is
    /// always a multiple of 8 bytes (`st.cuh:96` `<<3`), so a `sw = 8/itemsize`
    /// element group is never re-ordered (the `vec4` halves stay contiguous);
    /// a single `vec8` LDS store would split on the odd deltas, so we keep the
    /// wide *global* load but narrow the swizzled *LDS* store. Ends in a
    /// workgroup barrier (the matmul fill). bf16-only.
    ///
    /// # Panics
    /// Panics unless the source element itemsize is 2 bytes (bf16), the `src` and
    /// the destination ST element types match (no cast on this path), and the
    /// swizzle period, base cols, ST cols, and source row-stride are all aligned
    /// to the 128-bit vector width.
    pub fn fill_local_vec(&self, dst: ST, src: GL, idxs: &[Idx], axis: usize) -> ST {
        self.load_global_to_local_vec(dst, &src, idxs, axis, true)
    }

    /// [`Self::fill_local_vec`] **without** the trailing workgroup barrier — the
    /// software-pipeline primitive, for the register-staged matmul prefetch (issue
    /// strip k+1's 128-bit global loads at the loop top so the
    /// memory latency overlaps strip k's MFMAs; the caller fences once *after* the
    /// MFMAs so the load and the compute run concurrently).
    pub fn fill_local_vec_nobar(&self, dst: ST, src: GL, idxs: &[Idx], axis: usize) -> ST {
        self.load_global_to_local_vec(dst, &src, idxs, axis, false)
    }

    fn load_global_to_local_vec(&self, st: ST, src: &GL, idxs: &[Idx], axis: usize, barrier: bool) -> ST {
        let itemsize = st.elem().base().bytes() as i64;
        assert_eq!(itemsize, 2, "vec fill: bf16-only (128-bit = vec8)");
        assert_eq!(src.elem(), st.elem(), "vec fill: cast unsupported (use the scalar fill)");
        let vw: i64 = 16 / itemsize; // 8 bf16 — the 128-bit global load width
        let sw: i64 = 8 / itemsize; // 4 bf16 — the swizzle-order-safe LDS store width

        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let st_cols = st.cols as i64;
        // Alignment invariants: the swizzle period and the
        // tile/fragment widths must admit `vw`-aligned 16-byte groups.
        if let Some(period) = st.base.swizzle.period_bytes(st.cols, itemsize) {
            assert_eq!(period % 16, 0, "vec fill: swizzle period {period}B not 16B-aligned");
        }
        assert_eq!(base_cols % vw, 0, "vec fill: base cols {base_cols} not a multiple of vec width {vw}");
        assert_eq!(st_cols % vw, 0, "vec fill: st cols {st_cols} not a multiple of vec width {vw}");

        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        assert_eq!(row_stride % vw, 0, "vec fill: row stride {row_stride} not {vw}-aligned (need N % 8 == 0)");

        let idxs_t = scaled_idxs(idxs, axis, st.rows as i64, st.cols as i64);
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let total_elems = st.shape()[n - 4] as i64 * st.shape()[n - 3] as i64 * num_elements;
        let memcpy_per_row = st_cols / vw;
        let slots = self.group_threads() as i64 * vw;
        let total_calls = (total_elems + slots - 1) / slots;
        let num_valid = total_elems / vw;
        let clamp = total_calls * slots != total_elems;

        let outer = self.ker.raw_range(total_calls, AxisType::Loop);
        let mut load_idx = iadd(&imul(&outer, self.group_threads() as i64), &self.laneid());
        if clamp {
            let cond = load_idx.try_cmplt(&cidx(num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(num_valid - 1)).expect("clamp load_idx");
        }
        // The thread's `vw`-wide run: row `row0`, columns `[col0, col0+vw)` (a
        // `vw`-aligned slice within one base fragment, since `vw | base_cols`).
        let row0 = idiv(&load_idx, memcpy_per_row);
        let col0 = imod(&imul(&load_idx, vw), st_cols);
        let height = idiv(&row0, base_rows);
        let row = imod(&row0, base_rows);
        let width = idiv(&col0, base_cols);

        // One 128-bit coalesced global load of the contiguous `vw`-run.
        let off = iadd(&src_i_base, &iadd(&imul(&row0, row_stride), &col0));
        let loaded = load_vec(src.uop(), off, vw as usize);

        // Commit as `vw/sw` swizzle-safe `vec_sw` LDS stores (delta is constant
        // across the fragment row, so each `sw`-group maps contiguously).
        let stores: Vec<Arc<UOp>> = (0..vw / sw)
            .map(|j| {
                let col = imod(&iadd(&col0, &cidx(j * sw)), base_cols);
                let val = loaded.gep(((j * sw) as usize..(j * sw + sw) as usize).collect());
                let off = swizzled_st_offset(&st, &Idx::Uop(height.clone()), &Idx::Uop(width.clone()), &row, &col);
                index_off(st.uop(), off).store(val)
            })
            .collect();
        let grouped = if stores.len() == 1 { stores.into_iter().next().unwrap() } else { UOp::group(stores) };
        let stored = grouped.end(smallvec![outer]);
        let ended = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, ended)
    }

    /// LOCAL→REG fragment gather: each lane reads its WMMA fragment lanes from
    /// the (swizzled) LDS tile.
    pub(super) fn load_local_to_reg(&self, rt: RT<'k>, st: &ST, dst_idxs: &[Idx], idxs: &[Idx]) -> RT<'k> {
        let laneid = self.ker.laneid();
        let ept = rt.base.base.elements_per_thread() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let n = rt.shape().len();
        let (rt_h, rt_w) = (rt.shape()[n - 3] as i64, rt.shape()[n - 2] as i64);
        // SI-1 off-by-one guard: the wave's RT sub-tile must fit inside the ST.
        let sn = st.shape().len();
        let (st_h, st_w) = (st.shape()[sn - 4] as i64, st.shape()[sn - 3] as i64);
        assert!(rt_h <= st_h && rt_w <= st_w, "load LOCAL→REG: RT {rt_h}×{rt_w} exceeds ST {st_h}×{st_w}");
        let transpose = rt.layout != st.layout;

        // A lane's `ept` fragment elements are a swizzle-safe column-contiguous run —
        // readable in ONE `ds_read_b64` instead of `ept` scalar `ds_read_u16` — when the
        // fragment is a non-transposed, non-interleaved CDNA-MFMA bf16 input: `lane_rc`
        // then maps the element index to consecutive columns, and `ept == 4` is exactly
        // one swizzle group (the `sw`-wide run the vec-fill commits in). `vw` is the read
        // width — the whole run when contiguous, else one element (the proven scalar
        // gather, which also carries the cast / transpose / RDNA-interleave cases).
        let contiguous =
            !transpose && !rt.base.interleave && !rt.base.interleave_t && st.elem() == rt.elem() && ept == 4;
        let vw = if contiguous { ept } else { 1 };

        let height = self.ker.raw_range(rt_h, AxisType::Loop);
        let width = self.ker.raw_range(rt_w, AxisType::Loop);
        let inner = self.ker.raw_range(ept / vw, AxisType::Loop); // groups of `vw` elements
        let elem = imul(&inner, vw); // base element index of this group

        let (row, col) =
            lane_rc(transpose, rt.base.interleave, rt.base.interleave_t, &laneid, base_rows, base_cols, stride, &elem);

        // Wave sub-tile fragment offset (SI-1): the caller passes the wave's
        // `(row_block, col_block)` via `idxs` (already including warp_row/col);
        // empty ⇒ no offset (single-warp). `off` honors the double-buffer parity base.
        let h_idx = wave_offset(idxs.first(), rt_h, &height);
        let w_idx = wave_offset(idxs.get(1), rt_w, &width);
        let off = swizzled_st_offset(st, &h_idx, &w_idx, &row, &col);
        // Compiler-tracked vector / scalar `ds_read`; the inline-`asm` gather is the
        // explicit [`Self::gather_local_asm`] (gfx942 microkernel), not a mode here.
        let mut load = if vw > 1 { load_vec(st.uop(), off, vw as usize) } else { load_off(st.uop(), off) };
        if st.elem() != rt.elem() {
            load = load.cast(rt.elem().clone());
        }
        let mut didx: Vec<Idx> = dst_idxs.to_vec();
        didx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&elem)]);
        let ended = flat_index(rt.uop(), rt.shape(), &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_reg(rt, ended)
    }

    /// Explicit inline-`asm` `ds_read_b64` LOCAL→REG gather — the per-call
    /// counterpart of the old kernel-global `asm_gather` mode (now removed): the
    /// generic [`Self::load_local_to_reg`] always emits the compiler-tracked vector
    /// `load`; the gfx942 asm microkernel ([`crate::kernels::matmul`] `gemm_core_asm`)
    /// calls THIS for its operand gather. Emits one `ds_read` per fragment-row sharing
    /// a single base-address VGPR + per-row `offset:` immediate (the lane-uniform XOR
    /// delta, so no per-read permuted-address spill — see [`Self::gather_asm_unrolled`]).
    /// `ix` carries the wave/frag offset, exactly as [`Self::load`](Group::load).
    ///
    /// # Panics
    /// Panics unless the fragment is the contiguous single-column gemm operand shape
    /// (`rt_w == 1`, `ept == 4`, non-transposed) on a swizzled tile.
    pub fn gather_local_asm(&self, rt: RT<'k>, st: ST, ix: MoveIdx) -> RT<'k> {
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let rt_w = rt.shape()[n - 2] as i64;
        let transpose = rt.layout != st.layout;
        let contiguous =
            !transpose && !rt.base.interleave && !rt.base.interleave_t && st.elem() == rt.elem() && ept == 4;
        let vw = if contiguous { ept } else { 1 };
        assert!(
            vw > 1 && rt_w == 1 && ept / vw == 1,
            "gather_local_asm: gemm operand shape only (contiguous, rt_w==1)"
        );
        let subtile = st.base.swizzle.subtile_cols(st.cols, st.elem().base()).expect("gather_local_asm: tile swizzled");
        self.gather_asm_unrolled(rt, &st, &ix.frag, &ix.block, transpose, vw, subtile)
    }

    /// Rust-unrolled inline-`asm` `ds_read_b64` gather (the gemm operand
    /// shape: `rt_w == 1`, one `vw`-wide column run). Computes ONE base LDS address
    /// (the lane's `height == 0` slot) and emits one `ds_read_b64 ... offset:N` per
    /// fragment-row, `N = h * base_rows * subtile_cols * itemsize` bytes — a
    /// lane-uniform constant (see [`Swizzle::subtile_cols`]). Sharing the base VGPR
    /// across the unrolled MFMA cluster is what keeps VGPR pressure low (no
    /// per-read permuted-address spills); the asm reads are `sideeffect` and chain
    /// through their RT stores so the last store scopes them under one loop `END`.
    #[allow(clippy::too_many_arguments)]
    fn gather_asm_unrolled(
        &self,
        rt: RT<'k>,
        st: &ST,
        dst_idxs: &[Idx],
        idxs: &[Idx],
        transpose: bool,
        vw: i64,
        subtile: i64,
    ) -> RT<'k> {
        let laneid = self.ker.laneid();
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let n = rt.shape().len();
        let rt_h = rt.shape()[n - 3] as i64;
        let itemsize = st.elem().base().bytes() as i64;
        let row_stride_bytes = base_rows * subtile * itemsize; // fragment-row `offset:` step

        // Shared base address: the lane's slot at fragment-row 0 (`elem == 0`), with
        // the wave/parity offset folded in. One `addrspacecast` ⇒ one `$1` VGPR.
        let (row, col) = lane_rc(
            transpose,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &cidx(0),
        );
        let h0 = wave_offset(idxs.first(), rt_h, &cidx(0));
        let w0 = wave_offset(idxs.get(1), 1, &cidx(0));
        let base_off = swizzled_st_offset(st, &h0, &w0, &row, &col);
        let base_as3 = UOp::custom(
            smallvec![index_off(st.uop(), base_off)],
            "addrspacecast ptr {0} to ptr addrspace(3)".to_string(),
            svod_dtype::DType::Int32,
        );

        let i16v = svod_dtype::DType::Int16.vec(vw as usize).expect("ds_read_b64: i16 vec");
        let bf16v = st.elem().vec(vw as usize).expect("ds_read_b64: bf16 vec");
        let mut prev: Option<Arc<UOp>> = None;
        for h in 0..rt_h {
            let nbytes = h * row_stride_bytes;
            assert!(nbytes <= 65535, "ds_read offset {nbytes}B exceeds the 16-bit immediate");
            // `asm sideeffect` so the read can't hoist across the asm MFMAs; the prior
            // fragment's store is carried as an ordering-only operand (not in the
            // template) so the reads stay in program order under one `END`.
            let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![base_as3.clone()];
            if let Some(p) = prev.take() {
                deps.push(p);
            }
            let read = UOp::custom(
                deps,
                format!(
                    "call <4 x i16> asm sideeffect \"ds_read_b64 $0, $1 offset:{nbytes}\", \"=v,v\"\
                     (ptr addrspace(3) {{0}})"
                ),
                i16v.clone(),
            );
            let mut val = read.bitcast(bf16v.clone());
            if st.elem() != rt.elem() {
                val = val.cast(rt.elem().clone());
            }
            let mut didx: Vec<Idx> = dst_idxs.to_vec();
            didx.extend([Idx::Const(h), Idx::Const(0), Idx::Const(0)]);
            prev = Some(flat_index(rt.uop(), rt.shape(), &didx).store(val));
        }
        let terminal = prev.expect("gather_asm_unrolled: at least one fragment-row");
        self.finalize_reg(rt, terminal)
    }

    /// The boundary gate for a GLOBAL↔REG hop: `global_row < shape[axis] &
    /// global_col < shape[last]`, restricted to the axes that are actually ragged
    /// (the extent is not a multiple of the per-block tile span — known at build
    /// time, so an aligned axis adds no gate). `srow`/`scol` are the in-tile
    /// coordinates; the block offset from `idxs` is folded back in to recover the
    /// global position. `None` when both axes divide evenly.
    #[allow(clippy::too_many_arguments)]
    fn boundary_gate(
        &self,
        shape: &[usize],
        idxs: &[Idx],
        axis: usize,
        row_tile: i64,
        col_tile: i64,
        srow: &Arc<UOp>,
        scol: &Arc<UOp>,
    ) -> Option<Arc<UOp>> {
        let mut gate: Option<Arc<UOp>> = None;
        let bound_row = shape[axis] as i64;
        if bound_row % row_tile != 0 {
            let blk = idxs.get(axis).map(|i| i.to_uop()).unwrap_or_else(|| cidx(0));
            let g = iadd(&imul(&blk, row_tile), srow).try_cmplt(&cidx(bound_row)).expect("boundary row gate");
            gate = Some(g);
        }
        let bound_col = shape[shape.len() - 1] as i64;
        if bound_col % col_tile != 0 {
            let blk = idxs.get(3).map(|i| i.to_uop()).unwrap_or_else(|| cidx(0));
            let g = iadd(&imul(&blk, col_tile), scol).try_cmplt(&cidx(bound_col)).expect("boundary col gate");
            gate = Some(match gate {
                Some(r) => r.try_and_op(&g).expect("boundary gate and"),
                None => g,
            });
        }
        gate
    }

    /// GLOBAL→REG fragment gather: each lane reads its register fragment
    /// straight from global memory (the FA Q-tile load). The mirror of
    /// [`Self::store_reg_to_global`]. `masked` gates a tile straddling a ragged
    /// edge (see [`Self::boundary_gate`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_global_to_reg(
        &self,
        rt: RT<'k>,
        src: &GL,
        dst_idxs: &[Idx],
        idxs: &[Idx],
        axis: usize,
        masked: bool,
    ) -> RT<'k> {
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let s3 = rt.shape()[n - 3] as i64;
        let s2 = rt.shape()[n - 2] as i64;

        let idxs_t = scaled_idxs(idxs, axis, s3 * base_rows, s2 * base_cols);
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let laneid = self.ker.laneid();
        let height = self.ker.raw_range(s3, AxisType::Loop);
        let width = self.ker.raw_range(s2, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let base_row = imul(&height, base_rows);
        let base_col = imul(&width, base_cols);
        let (row, col) = lane_rc(
            rt.layout == TileLayout::Col,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &inner,
        );
        let srow = iadd(&base_row, &row);
        let scol = iadd(&base_col, &col);
        let off = iadd(&src_i_base, &iadd(&imul(&srow, row_stride), &scol));

        let gate = masked
            .then(|| self.boundary_gate(src.shape(), idxs, axis, s3 * base_rows, s2 * base_cols, &srow, &scol))
            .flatten();
        let mut load = match gate {
            Some(g) => {
                let zero = if src.elem().is_float() { ConstValue::Float(0.0) } else { ConstValue::Int(0) };
                load_off_gated(src.uop(), off, g, UOp::const_(src.elem().clone(), zero))
            }
            None => load_off(src.uop(), off),
        };
        if src.elem() != rt.elem() {
            load = load.cast(rt.elem().clone());
        }
        let mut didx: Vec<Idx> = dst_idxs.to_vec();
        didx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let ended = flat_index(rt.uop(), rt.shape(), &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_reg(rt, ended)
    }

    /// REG→LOCAL fragment scatter: each lane writes its register fragment into
    /// the (swizzled) LDS tile (the layout-transpose hop before write-back).
    pub(super) fn store_reg_to_local(&self, st: ST, rt: &RT<'k>, idxs: &[Idx], src_idxs: &[Idx]) -> ST {
        let laneid = self.ker.laneid();
        let ept = rt.base.base.elements_per_thread() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let n = rt.shape().len();
        let (rt_h, rt_w) = (rt.shape()[n - 3] as i64, rt.shape()[n - 2] as i64);
        let height = self.ker.raw_range(rt_h, AxisType::Loop);
        let width = self.ker.raw_range(rt_w, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let (row, col) = lane_rc(
            rt.layout != st.layout,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &inner,
        );
        let mut sidx: Vec<Idx> = src_idxs.to_vec();
        sidx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let mut load = load_at(rt.uop(), rt.shape(), &sidx);
        if rt.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        // Wave sub-tile fragment offset (SI-1), symmetric with `load_local_to_reg`.
        let h_idx = wave_offset(idxs.first(), rt_h, &height);
        let w_idx = wave_offset(idxs.get(1), rt_w, &width);
        let off = swizzled_st_offset(&st, &h_idx, &w_idx, &row, &col);
        let ended = index_off(st.uop(), off).store(load).end(smallvec![height, width, inner]);
        self.finalize_st(st, ended)
    }

    /// REG→GLOBAL write-back: each lane writes its register fragment to the
    /// correct global position.
    pub(super) fn store_reg_to_global(
        &self,
        dst: GL,
        rt: &RT<'k>,
        idxs: &[Idx],
        src_idxs: &[Idx],
        axis: usize,
        masked: bool,
    ) -> GL {
        let row_stride: i64 = dst.shape()[axis + 1..].iter().product::<usize>() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let s3 = rt.shape()[n - 3] as i64;
        let s2 = rt.shape()[n - 2] as i64;

        let idxs_t = scaled_idxs(idxs, axis, s3 * base_rows, s2 * base_cols);
        let dst_i_base = flat_offset(dst.shape(), &idxs_t);

        let laneid = self.ker.laneid();
        let height = self.ker.raw_range(s3, AxisType::Loop);
        let width = self.ker.raw_range(s2, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let base_row = imul(&height, base_rows);
        let base_col = imul(&width, base_cols);
        let (row, col) = lane_rc(
            rt.layout == TileLayout::Col,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &inner,
        );
        let srow = iadd(&base_row, &row);
        let scol = iadd(&base_col, &col);
        let off = iadd(&dst_i_base, &iadd(&imul(&srow, row_stride), &scol));

        let mut sidx: Vec<Idx> = src_idxs.to_vec();
        sidx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let mut load = load_at(rt.uop(), rt.shape(), &sidx);
        if rt.elem() != dst.elem() {
            load = load.cast(dst.elem().clone());
        }
        let gate = masked
            .then(|| self.boundary_gate(dst.shape(), idxs, axis, s3 * base_rows, s2 * base_cols, &srow, &scol))
            .flatten();
        let target = match gate {
            Some(g) => index_off_gated(dst.uop(), off, g),
            None => index_off(dst.uop(), off),
        };
        let ended = target.store(load).end(smallvec![height, width, inner]);
        self.finalize_gl(dst, ended)
    }
}

/// ST flat INDEX honoring the optional double-buffer parity [`ST::base_offset`].
/// Identical to [`crate::index::flat_index`] for an ordinary (`base_offset:None`)
/// tile; adds the parity offset for a [`Kernel::st_db`](crate::Kernel) half-view.
/// Flat element offset of in-tile fragment position `(frag_h, frag_w, row, col)`
/// in the (possibly swizzled) LDS tile, plus any double-buffer parity base.
/// `frag_h`/`frag_w` are the fragment-grid indices (height/width, with the wave
/// `block` already folded in for the gather/scatter hops); `row`/`col` are the
/// per-lane position WITHIN the base fragment.
///
/// [`Swizzle::Identity`] keeps the plain fragment-major layout (shape
/// `[H, W, base_rows, base_cols]`); the XOR variants reconstruct the whole-tile
/// `(row, col)` and apply a subtile-structured bank swizzle
/// ([`Swizzle::tile_offset`]) over the full in-tile address — the same bijection
/// on every store and load, so numerics are preserved while the gfx942 LDS banks
/// are spread (the MFMA-gather bank-conflict fix).
fn swizzled_st_offset(st: &ST, frag_h: &Idx, frag_w: &Idx, row: &Arc<UOp>, col: &Arc<UOp>) -> Arc<UOp> {
    let mut off = match st.base.swizzle {
        Swizzle::Identity => {
            flat_offset(st.shape(), &[frag_h.clone(), frag_w.clone(), Idx::Uop(row.clone()), Idx::Uop(col.clone())])
        }
        sw => {
            let mut full_row = iadd(&imul(&frag_h.to_uop(), st.base.base.rows as i64), row);
            let mut full_col = iadd(&imul(&frag_w.to_uop(), st.base.base.cols as i64), col);
            // A `subtile` view's absolute element origin enters the swizzle here,
            // BEFORE the (non-linear) bank remap — see `ST::with_origin`.
            if let Some((orow, ocol)) = st.origin() {
                full_row = iadd(&full_row, orow);
                full_col = iadd(&full_col, ocol);
            }
            sw.tile_offset(full_row, full_col, st.rows, st.cols, st.elem().base())
        }
    };
    if let Some(bo) = st.base_offset() {
        off = off.try_add(bo).expect("swizzled_st_offset: parity base offset add");
    }
    off
}
