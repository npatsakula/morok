//! Counted bitonic sorting networks over a single 16-wide WMMA fragment, built on
//! the index-carrying [`Group::arg_compare_exchange`] butterfly. The sorted axis is
//! the low-4-bit lane axis (`laneid % 16`); the orthogonal axis (the fragment's
//! other coordinate — carried by `laneid / 16` and the per-lane `inner` registers)
//! is independent, so every such lane/register slot is sorted in parallel by one
//! network. No `loop_dynamic`, no LDS, no barrier — a fixed schedule of
//! `ds_bpermute` butterflies, so there is no GPU-hang risk.
//!
//! The KNN running-top-K uses these to replace the serialized k-step argmin-insert:
//! per corpus tile, [`Self::bitonic_argsort`] sorts the tile's candidates once and
//! [`Self::bitonic_merge_topk`] folds them into the sorted running top-K.

use svod_ir::UOp;

use super::{ArgDir, Group, SwapDir, arg_fold, ixor};
use crate::index::{flat_index, load_at};
use crate::tile::RT;

impl<'k> Group<'k> {
    /// Sort a `(value, index)` pair ASCENDING along the 16-wide sort axis
    /// (`laneid % 16`), per the orthogonal axis (independent). The standard Batcher
    /// bitonic network for N=16: for each merge size `s ∈ {2,4,8,16}`, sweep the
    /// strides `j ∈ {s/2, …, 1}` calling [`Self::arg_compare_exchange`].
    ///
    /// The sort axis lives in the LOW four lane bits, so for `s ∈ {2,4,8}` the
    /// per-stage direction is `ByLaneBit(s)` — `laneid & s` reads the sort
    /// position's `s`-bit directly. The final size-16 merge is ASCENDING for every
    /// position (`pos & 16 == 0` for all `pos < 16`), so it uses [`SwapDir::Ascending`]
    /// rather than `ByLaneBit(16)` (which would read a lane bit BELONGING to the
    /// orthogonal axis). Ties (equal value) settle to the smaller index. Returns the
    /// sorted `(val, idx)` pair.
    pub fn bitonic_argsort(&self, val: RT<'k>, idx: RT<'k>) -> (RT<'k>, RT<'k>) {
        let (mut v, mut i) = (val, idx);
        for s in [2i64, 4, 8] {
            let mut j = s / 2;
            while j >= 1 {
                let (nv, ni) =
                    self.arg_compare_exchange(v.alloc_like(), i.alloc_like(), &v, &i, j, SwapDir::ByLaneBit(s));
                v = nv;
                i = ni;
                j /= 2;
            }
        }
        // Size-16 merge: a bitonic sequence collapsed fully ascending.
        for j in [8i64, 4, 2, 1] {
            let (nv, ni) = self.arg_compare_exchange(v.alloc_like(), i.alloc_like(), &v, &i, j, SwapDir::Ascending);
            v = nv;
            i = ni;
        }
        (v, i)
    }

    /// Merge two ASCENDING sorted `(value, index)` runs over the 16-wide sort axis
    /// and keep the smallest 16 — the running-top-K update. `a` is the running
    /// top-K, `b` the freshly sorted tile candidates. Reverse `b` over the sort axis
    /// (`gather laneid ^ 15`) so `a ++ reverse(b)` is bitonic; the per-position
    /// arg-min of `a` and `reverse(b)` is the bitonic lower half (the 16 smallest,
    /// still bitonic); four ASCENDING [`Self::arg_compare_exchange`] stages
    /// (`j ∈ {8,4,2,1}`) sort that half. Ties settle to the smaller index. Returns
    /// the merged sorted-ascending top-16 `(val, idx)` pair.
    pub fn bitonic_merge_topk(
        &self,
        a_val: &RT<'k>,
        a_idx: &RT<'k>,
        b_val: &RT<'k>,
        b_idx: &RT<'k>,
    ) -> (RT<'k>, RT<'k>) {
        // reverse(b) over the sort axis: lane L gathers lane L ^ 15 (the low-4-bit
        // reflection keeps the orthogonal axis — `laneid / 16` and `inner` — fixed).
        let b_rev_val = self.shuffle(b_val.alloc_like(), b_val, |l| ixor(l, 15));
        let b_rev_idx = self.shuffle(b_idx.alloc_like(), b_idx, |l| ixor(l, 15));

        // Bitonic lower half: per-position arg-min of a and reverse(b) (16 smallest).
        let (mut v, mut i) = self.arg_min_pair(a_val, a_idx, &b_rev_val, &b_rev_idx);

        // Sort the bitonic half ascending.
        for j in [8i64, 4, 2, 1] {
            let (nv, ni) = self.arg_compare_exchange(v.alloc_like(), i.alloc_like(), &v, &i, j, SwapDir::Ascending);
            v = nv;
            i = ni;
        }
        (v, i)
    }

    /// Per-position arg-min of two `(value, index)` pairs (no cross-lane shuffle):
    /// keep the smaller value, ties → the smaller index ([`arg_fold`]). The
    /// elementwise building block of the bitonic merge's lower-half fold.
    fn arg_min_pair(&self, a_val: &RT<'k>, a_idx: &RT<'k>, b_val: &RT<'k>, b_idx: &RT<'k>) -> (RT<'k>, RT<'k>) {
        let (dst_v, dst_i) = (a_val.alloc_like(), a_idx.alloc_like());
        let (avbuf, shape) = (self.anchor(a_val.uop()), a_val.shape().to_vec());
        let aibuf = self.anchor(a_idx.uop());
        let (bvbuf, bibuf) = (self.anchor(b_val.uop()), self.anchor(b_idx.uop()));
        let (dvbuf, dibuf) = (dst_v.uop().clone(), dst_i.uop().clone());
        let ended = self.elementwise(&shape.clone(), move |idxs| {
            let va = load_at(&avbuf, &shape, idxs);
            let ia = load_at(&aibuf, &shape, idxs);
            let vb = load_at(&bvbuf, &shape, idxs);
            let ib = load_at(&bibuf, &shape, idxs);
            let (v, i) = arg_fold(ArgDir::Min, &va, &ia, &vb, &ib);
            UOp::group(vec![flat_index(&dvbuf, &shape, idxs).store(v), flat_index(&dibuf, &shape, idxs).store(i)])
        });
        self.finalize_reg_pair(dst_v, dst_i, ended)
    }
}
