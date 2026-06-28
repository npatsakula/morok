//! Single-warp cross-lane shuffle ops built on the [`ds_bpermute`](super::Group)
//! gather primitive `shuffle_lane`: the generic `shuffle`, the butterfly
//! `shuffle_xor`, the `shuffle_down`/`shuffle_up` rotates, and the bitonic
//! `compare_exchange`. One `ds_bpermute` per element — no LDS, no barrier.

use std::sync::Arc;

use super::{Group, SwapDir, iadd, iand, imod, ixor};
use crate::index::{cidx, flat_index, load_at};
use crate::tile::RT;
use svod_ir::UOp;

impl<'k> Group<'k> {
    /// Per-element cross-lane gather (the public face of [`Self::shuffle_lane`]): for
    /// each logical element, `dst` receives `src`'s value at the SAME position but
    /// from lane `src_lane(laneid)`. Single-warp; one `ds_bpermute` per element (no
    /// LDS, no barrier). The shared foundation for `shuffle_xor`/`compare_exchange`
    /// (and, later, scan / arg-reduce). f32 (bitcast) and i32 transports are
    /// supported today; f16/bf16/i64 are a follow-up.
    ///
    /// # Panics
    /// Panics if the group has more than one warp, or if `dst` and `src` have
    /// different shapes.
    pub fn shuffle<F>(&self, dst: RT<'k>, src: &RT<'k>, src_lane: F) -> RT<'k>
    where
        F: Fn(&Arc<UOp>) -> Arc<UOp>,
    {
        assert_eq!(self.warps, 1, "shuffle is a single-warp op");
        assert_eq!(dst.shape(), src.shape(), "shuffle: shape mismatch");
        let sl = src_lane(&self.laneid());
        let (sbuf, sshape) = (self.anchor(src.uop()), src.shape().to_vec());
        let (dbuf, dshape) = (dst.uop().clone(), dst.shape().to_vec());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let v = load_at(&sbuf, &sshape, idxs);
            flat_index(&dbuf, &dshape, idxs).store(self.shuffle_lane(&v, &sl))
        });
        self.finalize_reg(dst, ended)
    }

    /// Butterfly exchange: `dst[pos] = src[pos]` from lane `laneid ^ mask`. Arch-blind
    /// — for any `mask < wave_size` the XOR partner stays in `[0, wave_size)`, so no
    /// modulus is needed (cheaper than [`Self::shuffle_down`]). The sort/reduce primitive.
    ///
    /// # Panics
    /// Panics if `mask` is not in `1..wave_size`, if the group has more than one
    /// warp, or if `dst` and `src` have different shapes.
    pub fn shuffle_xor(&self, dst: RT<'k>, src: &RT<'k>, mask: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(mask > 0 && mask < w, "shuffle_xor mask {mask} must be in 1..{w}");
        self.shuffle(dst, src, |laneid| ixor(laneid, mask))
    }

    /// Shift down: `dst[L] = src[(L + delta) mod wave_size]`.
    ///
    /// # Panics
    /// Panics if `delta` is not in `1..wave_size`, if the group has more than one
    /// warp, or if `dst` and `src` have different shapes.
    pub fn shuffle_down(&self, dst: RT<'k>, src: &RT<'k>, delta: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(delta > 0 && delta < w, "shuffle_down delta {delta} must be in 1..{w}");
        self.shuffle(dst, src, move |laneid| imod(&iadd(laneid, &cidx(delta)), w))
    }

    /// Shift up: `dst[L] = src[(L - delta) mod wave_size]` (the scan primitive).
    ///
    /// # Panics
    /// Panics if `delta` is not in `1..wave_size`, if the group has more than one
    /// warp, or if `dst` and `src` have different shapes.
    pub fn shuffle_up(&self, dst: RT<'k>, src: &RT<'k>, delta: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(delta > 0 && delta < w, "shuffle_up delta {delta} must be in 1..{w}");
        self.shuffle(dst, src, move |laneid| imod(&iadd(laneid, &cidx(w - delta)), w))
    }

    /// The per-lane `keep_min` predicate for a compare-exchange across the
    /// butterfly partner `laneid ^ mask` under `dir`: true when this lane keeps the
    /// *smaller* of the pair (else the larger). The lower-index lane of a pair is
    /// `(laneid & mask) == 0`. Shared by [`Self::compare_exchange`] (value-only) and
    /// [`Self::arg_compare_exchange`] (value + paired index).
    fn keep_min_pred(&self, laneid: &Arc<UOp>, mask: i64, dir: SwapDir) -> Arc<UOp> {
        let is_low = iand(laneid, mask).try_cmpeq(&cidx(0)).expect("ce is_low");
        match dir {
            SwapDir::Ascending => is_low,
            SwapDir::Descending => iand(laneid, mask).try_cmpne(&cidx(0)).expect("ce desc"),
            // Bitonic merge: ascending where `(laneid & bit) == 0`. Keep min iff the
            // low-lane flag equals the ascending flag.
            SwapDir::ByLaneBit(bit) => {
                let asc = iand(laneid, bit).try_cmpeq(&cidx(0)).expect("ce dir bit");
                is_low.try_cmpeq(&asc).expect("ce keep_min")
            }
        }
    }

    /// One bitonic compare-exchange stage across the butterfly partner `laneid ^
    /// mask`: each lane keeps the min or max of its element and the partner's, per
    /// `dir` — the building block of sorting networks. Per element: one `ds_bpermute`
    /// gather + an ALU min/max select (no LDS, no barrier).
    ///
    /// # Panics
    /// Panics if the group has more than one warp, if `dst` and `src` have
    /// different shapes, or if `mask` is not in `1..wave_size`.
    pub fn compare_exchange(&self, dst: RT<'k>, src: &RT<'k>, mask: i64, dir: SwapDir) -> RT<'k> {
        assert_eq!(self.warps, 1, "compare_exchange is a single-warp op");
        assert_eq!(dst.shape(), src.shape(), "compare_exchange: shape mismatch");
        let w = self.ker.caps.wave_size as i64;
        assert!(mask > 0 && mask < w, "compare_exchange mask {mask} must be in 1..{w}");
        let laneid = self.laneid();
        let partner = ixor(&laneid, mask);
        let keep_min = self.keep_min_pred(&laneid, mask, dir);
        let (sbuf, sshape) = (self.anchor(src.uop()), src.shape().to_vec());
        let (dbuf, dshape) = (dst.uop().clone(), dst.shape().to_vec());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let v = load_at(&sbuf, &sshape, idxs);
            let p = self.shuffle_lane(&v, &partner);
            let lt = v.try_cmplt(&p).expect("ce lt");
            let mn = UOp::try_where(lt, v.clone(), p.clone()).expect("ce min");
            let mx = v.try_max(&p).expect("ce max");
            let out = UOp::try_where(keep_min.clone(), mn, mx).expect("ce select");
            flat_index(&dbuf, &dshape, idxs).store(out)
        });
        self.finalize_reg(dst, ended)
    }

    /// The index-carrying [`Self::compare_exchange`]: one bitonic stage over the
    /// butterfly partner `laneid ^ mask` that sorts a `(value, index)` PAIR. The
    /// partner's value AND its index are each gathered with their own `ds_bpermute`
    /// (the index follows its value, never re-derived), and both kept elements are
    /// selected by the SAME predicate so the value stays paired with its index.
    ///
    /// The pair is ordered by the total order `(value, then index)`: the min-keeper
    /// (per `dir`) takes the element that is smaller in value — ties (equal value)
    /// broken toward the **smaller index** (matching `Tensor::argmin`/`topk`); the
    /// max-keeper takes the other. So an ascending sort of equal-valued elements
    /// leaves them ordered by increasing index. `dst_val`/`src_val` are the float
    /// values; `dst_idx`/`src_idx` the `Int32` indices. Returns `(val, idx)`.
    ///
    /// # Panics
    /// Panics if the group has more than one warp, if any of the four tiles disagree
    /// in shape, if `src_idx` is not `Int32`, or if `mask` is not in `1..wave_size`.
    pub fn arg_compare_exchange(
        &self,
        dst_val: RT<'k>,
        dst_idx: RT<'k>,
        src_val: &RT<'k>,
        src_idx: &RT<'k>,
        mask: i64,
        dir: SwapDir,
    ) -> (RT<'k>, RT<'k>) {
        assert_eq!(self.warps, 1, "arg_compare_exchange is a single-warp op");
        assert_eq!(dst_val.shape(), src_val.shape(), "arg_compare_exchange: value shape mismatch");
        assert_eq!(dst_idx.shape(), src_idx.shape(), "arg_compare_exchange: index shape mismatch");
        assert_eq!(src_val.shape(), src_idx.shape(), "arg_compare_exchange: value/index shape mismatch");
        assert_eq!(src_idx.elem(), &svod_dtype::DType::Int32, "arg_compare_exchange: index must be Int32");
        let w = self.ker.caps.wave_size as i64;
        assert!(mask > 0 && mask < w, "arg_compare_exchange mask {mask} must be in 1..{w}");
        let laneid = self.laneid();
        let partner = ixor(&laneid, mask);
        let keep_min = self.keep_min_pred(&laneid, mask, dir);
        let (svbuf, sshape) = (self.anchor(src_val.uop()), src_val.shape().to_vec());
        let sibuf = self.anchor(src_idx.uop());
        let (dvbuf, dibuf) = (dst_val.uop().clone(), dst_idx.uop().clone());
        let ended = self.elementwise(&sshape.clone(), move |idxs| {
            let va = load_at(&svbuf, &sshape, idxs);
            let ia = load_at(&sibuf, &sshape, idxs);
            let vb = self.shuffle_lane(&va, &partner);
            let ib = self.shuffle_lane(&ia, &partner);
            // `self` precedes the partner in the (value, then index) total order.
            let v_lt = va.try_cmplt(&vb).expect("ace v_lt");
            let v_eq = va.try_cmpeq(&vb).expect("ace v_eq");
            let i_lt = ia.try_cmplt(&ib).expect("ace i_lt");
            let self_lt = UOp::try_where(v_eq, i_lt, v_lt).expect("ace self_lt");
            // keep_self = (keep_min == self_lt): the min-keeper keeps the smaller of
            // the pair, the max-keeper the larger — one boolean XNOR, no and/or.
            let keep_self = keep_min.clone().try_cmpeq(&self_lt).expect("ace keep_self");
            let ov = UOp::try_where(keep_self.clone(), va, vb).expect("ace val select");
            let oi = UOp::try_where(keep_self, ia, ib).expect("ace idx select");
            let vs = flat_index(&dvbuf, &sshape, idxs).store(ov);
            let is = flat_index(&dibuf, &sshape, idxs).store(oi);
            UOp::group(vec![vs, is])
        });
        self.finalize_reg_pair(dst_val, dst_idx, ended)
    }
}
