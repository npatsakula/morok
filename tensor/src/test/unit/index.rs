//! Unit tests for the ndarray-style indexing layer (`s!` + `getitem`/`set`).

use crate::s;
use crate::*;
use ndarray::array;
use svod_ir::SInt;

fn get_shape(t: &Tensor) -> Vec<usize> {
    t.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

crate::codegen_tests! {
    // ---- basic getitem ----

    fn test_getitem_range(config) {
        let t = Tensor::from_slice([0f32, 1., 2., 3., 4., 5.]).try_reshape([2, 3]).unwrap();
        let r = t.getitem(s![1..2, ..]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![1, 3]);
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![3., 4., 5.]);
    }

    fn test_getitem_int_collapse(config) {
        let t = Tensor::from_slice([0f32, 1., 2., 3., 4., 5.]).try_reshape([2, 3]).unwrap();
        let r = t.getitem(s![1, ..]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![3]);
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![3., 4., 5.]);
    }

    fn test_getitem_multi_axis(config) {
        let t = Tensor::from_slice((0..12).map(|x| x as f32).collect::<Vec<_>>())
            .try_reshape([3, 4]).unwrap();
        let r = t.getitem(s![1..3, 1..3]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![2, 2]);
        // rows 1,2 cols 1,2 → [[5,6],[9,10]]
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![5., 6., 9., 10.]);
    }

    fn test_getitem_step(config) {
        let t = Tensor::from_slice([0f32, 1., 2., 3., 4., 5.]);
        let r = t.getitem(s![0..6;2]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![3]);
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![0., 2., 4.]);
    }

    fn test_getitem_reverse(config) {
        let t = Tensor::from_slice([0f32, 1., 2., 3.]);
        let r = t.getitem(s![..;-1]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![3., 2., 1., 0.]);
    }

    fn test_getitem_neg_index(config) {
        let t = Tensor::from_slice([0f32, 1., 2., 3.]);
        let r = t.getitem(s![-1]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![3.]);
    }

    fn test_getitem_ellipsis(config) {
        let t = Tensor::from_slice([0f32, 1., 2., 3., 4., 5.]).try_reshape([2, 3]).unwrap();
        let r = t.getitem(s![Ellipsis, 0]).unwrap().contiguous(); // first column
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![2]);
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![0., 3.]);
    }

    fn test_getitem_newaxis(config) {
        let t = Tensor::from_slice([1f32, 2., 3.]);
        let r = t.getitem(s![NewAxis, ..]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![1, 3]);
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![1., 2., 3.]);
    }

    // ---- advanced (fancy) getitem ----

    fn test_getitem_single_axis_fancy(config) {
        // s![.., heads] must equal index_select(1, heads) value-for-value.
        let data = Tensor::from_ndarray(&array![
            [1.0f32, 2., 3., 4.],
            [5., 6., 7., 8.]
        ]); // [2, 4]
        let heads = vec![0usize, 2, 3];

        let got = data.getitem(s![.., heads.clone()]).unwrap().contiguous();
        got.realize_with(&config).unwrap();

        let head_t = Tensor::from_slice([0i64, 2, 3]);
        let want = data.index_select(1, &head_t).unwrap().contiguous();
        want.realize_with(&config).unwrap();

        assert_eq!(get_shape(&got), vec![2, 3]);
        assert_eq!(got.as_vec::<f32>().unwrap(), want.as_vec::<f32>().unwrap());
    }

    fn test_getitem_multi_axis_fancy(config) {
        // Two adjacent fancy axes broadcast together → equivalent to gather.
        let data = Tensor::from_ndarray(&array![
            [10.0f32, 11., 12.],
            [20., 21., 22.],
            [30., 31., 32.]
        ]); // [3, 3]
        let rows = Tensor::from_slice([0i64, 2]);
        let cols = Tensor::from_slice([1i64, 0]);

        let got = data.getitem(s![rows.clone(), cols.clone()]).unwrap().contiguous();
        got.realize_with(&config).unwrap();
        // diagonal-style pick: (0,1)=11, (2,0)=30
        assert_eq!(get_shape(&got), vec![2]);
        assert_eq!(got.as_vec::<f32>().unwrap(), vec![11., 30.]);
    }

    // Regression: two fancy axes at a NON-leading position (d0=1). Previously errored
    // in gather_advanced (try_expand rank mismatch). out[a,k] = t[a, rows[k], cols[k]].
    fn test_getitem_fancy_nonleading_axes(config) {
        let t = Tensor::from_slice((0..24).map(|x| x as f32).collect::<Vec<_>>())
            .try_reshape([2, 3, 4]).unwrap(); // t[a,i,j] = 12a + 4i + j
        let rows = vec![0i64, 2];
        let cols = vec![1i64, 3];
        let r = t.getitem(s![.., rows, cols]).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![2, 2]);
        // a=0: t[0,0,1]=1, t[0,2,3]=11 ; a=1: t[1,0,1]=13, t[1,2,3]=23
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![1.0, 11.0, 13.0, 23.0]);
    }

    // ---- symbolic-bound getitem (the JIT batch path) ----

    fn test_getitem_symbolic_batch(config) {
        // getitem with a symbolic batch bound must be identical to a hand-written
        // try_shrink (it MUST route through try_shrink — slice_with would panic).
        let data = Tensor::from_ndarray(&array![
            [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]]
        ]); // [3, 4, 2]

        let b = Variable::new("gisb", 1, 3).bind(3).unwrap();
        let manual =
            data.try_shrink([Some((SInt::Const(0), b.as_sint())), None, None]).unwrap().contiguous();
        manual.realize_with(&config).unwrap();
        let want: Vec<f32> = manual.array_view::<f32>().unwrap().iter().copied().collect();

        let got = data.getitem(s![Idx::sint(SInt::Const(0), b.as_sint()), .., ..]).unwrap().contiguous();
        got.realize_with(&config).unwrap();
        let got_flat: Vec<f32> = got.array_view::<f32>().unwrap().iter().copied().collect();

        assert_eq!(got_flat, want);
    }

    // ---- set (functional setitem) ----

    fn test_set_basic_region(config) {
        let t = Tensor::from_slice((0..12).map(|x| x as f32).collect::<Vec<_>>())
            .try_reshape([3, 4]).unwrap();
        let block = Tensor::from_slice([100f32, 101., 102., 103.]).try_reshape([1, 4]).unwrap();
        let r = t.set(s![1..2, ..], &block).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(get_shape(&r), vec![3, 4]);
        assert_eq!(
            r.as_vec::<f32>().unwrap(),
            vec![0., 1., 2., 3., 100., 101., 102., 103., 8., 9., 10., 11.]
        );
    }

    fn test_set_int_axis(config) {
        let t = Tensor::from_slice((0..6).map(|x| x as f32).collect::<Vec<_>>())
            .try_reshape([2, 3]).unwrap();
        let row = Tensor::from_slice([7f32, 8., 9.]);
        let r = t.set(s![0, ..], &row).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![7., 8., 9., 3., 4., 5.]);
    }

    // Regression: set a 0-D scalar into an integer-collapsed, non-leading axis.
    // Previously panicked (unsqueeze of a 0-D value at axis 1 → AxisOutOfRange).
    fn test_set_scalar_into_collapsed_column(config) {
        let t = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape([2, 2]).unwrap();
        let scalar = Tensor::const_(9.0, svod_dtype::DType::Float32);
        let r = t.set(s![.., 0], &scalar).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![9.0, 2.0, 9.0, 4.0]);
    }

    fn test_set_scalar_broadcast(config) {
        let t = Tensor::from_slice((0..6).map(|x| x as f32).collect::<Vec<_>>())
            .try_reshape([2, 3]).unwrap();
        let zero = Tensor::from_slice([0f32]); // broadcast into the column
        let r = t.set(s![.., 1..2], &zero).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![0., 0., 2., 3., 0., 5.]);
    }

    fn test_set_advanced_last_writer_wins(config) {
        // Single-axis fancy set with a duplicate index → last value wins (scatter).
        let t = Tensor::from_slice([0f32, 0., 0., 0.]).try_reshape([1, 4]).unwrap();
        let idx = Tensor::from_slice([1i64, 1, 3]); // index 1 written twice
        let vals = Tensor::from_slice([5f32, 9., 7.]).try_reshape([1, 3]).unwrap();
        let r = t.set(s![.., idx.clone()], &vals).unwrap().contiguous();
        r.realize_with(&config).unwrap();
        // position 1 gets the LAST write (9), position 3 gets 7.
        assert_eq!(r.as_vec::<f32>().unwrap(), vec![0., 9., 0., 7.]);
    }

    // ---- proptest: getitem matches try_shrink over random contiguous ranges ----

    fn test_getitem_matches_shrink_proptest(config, a in 0usize..6, len in 0usize..6) {
        let n = 6;
        let b = (a + len).min(n);
        let data: Vec<f32> = (0..n).map(|x| x as f32).collect();
        let t = Tensor::from_slice(data);

        let viafn = t.getitem(s![a as i64 .. b as i64]).unwrap().contiguous();
        viafn.realize_with(&config).unwrap();
        let shrink = t.try_shrink([(a as isize, b as isize)]).unwrap().contiguous();
        shrink.realize_with(&config).unwrap();

        assert_eq!(viafn.as_vec::<f32>().unwrap(), shrink.as_vec::<f32>().unwrap());
    }
}

// =========================================================================
// Shape / error tests (no codegen)
// =========================================================================

#[test]
fn test_getitem_too_many_indices_errors() {
    let t = Tensor::from_slice([1f32, 2., 3., 4.]).try_reshape([2, 2]).unwrap();
    assert!(t.getitem(s![0, 0, 0]).is_err());
}

#[test]
fn test_getitem_double_ellipsis_errors() {
    let t = Tensor::from_slice([1f32, 2., 3., 4.]).try_reshape([2, 2]).unwrap();
    assert!(t.getitem(s![Ellipsis, Ellipsis]).is_err());
}

#[test]
fn test_getitem_oob_index_errors() {
    let t = Tensor::from_slice([1f32, 2., 3.]);
    assert!(t.getitem(s![5]).is_err());
}

#[test]
fn test_set_rejects_fancy_multi_axis() {
    let t = Tensor::from_slice([1f32, 2., 3., 4.]).try_reshape([2, 2]).unwrap();
    let r = Tensor::from_slice([0i64, 1]);
    let c = Tensor::from_slice([0i64, 1]);
    let v = Tensor::from_slice([9f32, 9.]);
    assert!(t.set(s![r, c], &v).is_err());
}

// A range/slice on another axis alongside a fancy set must error (not silently
// drop the region outside the range).
#[test]
fn test_set_rejects_range_plus_fancy() {
    let t = Tensor::from_slice([1f32, 2., 3., 4., 5., 6.]).try_reshape([2, 3]).unwrap();
    let idx = Tensor::from_slice([0i64, 2]);
    let v = Tensor::from_slice([7f32, 8.]);
    assert!(t.set(s![0..1, idx], &v).is_err());
}
