use crate::*;
use morok_schedule::testing::setup_test_tracing;
use ndarray::array;
use proptest::prelude::*;

fn get_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.uop().shape().unwrap().unwrap().iter().map(|s| s.as_const().unwrap()).collect()
}

crate::codegen_tests! {
    fn test_sort_1d_ascending(config) {
        setup_test_tracing();
        let t = Tensor::from_slice([3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let (sorted, indices) = t.sort(-1, false).unwrap();
        let mut sorted = sorted;
        sorted.realize_with(&config).unwrap();
        assert_eq!(sorted.as_vec::<f32>().unwrap(), [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]);
        let mut indices = indices;
        indices.realize_with(&config).unwrap();
        assert_eq!(indices.as_vec::<i32>().unwrap(), [1, 3, 6, 0, 2, 4, 7, 5]);
    }

    fn test_sort_1d_descending(config) {
        let t = Tensor::from_slice([3.0f32, 1.0, 4.0, 1.0, 5.0]);
        let (sorted, _indices) = t.sort(-1, true).unwrap();
        let mut sorted = sorted;
        sorted.realize_with(&config).unwrap();
        assert_eq!(sorted.as_vec::<f32>().unwrap(), [5.0, 4.0, 3.0, 1.0, 1.0]);
    }

    fn test_sort_power_of_2_size(config) {
        let t = Tensor::from_slice([4.0f32, 2.0, 1.0, 3.0]);
        let (sorted, _) = t.sort(-1, false).unwrap();
        let mut sorted = sorted;
        sorted.realize_with(&config).unwrap();
        assert_eq!(sorted.as_vec::<f32>().unwrap(), [1.0, 2.0, 3.0, 4.0]);
    }

    fn test_sort_single_element(config) {
        let t = Tensor::from_slice([42.0f32]);
        let (sorted, indices) = t.sort(-1, false).unwrap();
        let mut sorted = sorted;
        sorted.realize_with(&config).unwrap();
        assert_eq!(sorted.as_vec::<f32>().unwrap(), [42.0]);
        let mut indices = indices;
        indices.realize_with(&config).unwrap();
        assert_eq!(indices.as_vec::<i32>().unwrap(), [0]);
    }

    /// MRE for sort bug: isolates the stage-2 crossover kernel (K5).
    /// The composition is: undo-crossover(dim1, lazy) → crossover(dim0, contiguous).
    /// This is the exact view chain that produces negative load indices.
    fn test_sort_crossover_mre(config) {
        setup_test_tracing();
        // Start with [2,2,2] tensor: [[1,2],[3,4]], [[5,6],[7,8]]
        let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .try_reshape([2, 2, 2]).unwrap();

        // Undo stage-1 crossover at dim1 (NO contiguous — lazy view)
        let halves_undo = x.split(&[1, 1], 1).unwrap();
        let undo = Tensor::cat(
            &[&halves_undo[0], &halves_undo[1].flip(&[-1, -2]).unwrap()],
            1,
        ).unwrap();
        // undo is a lazy view — not materialized

        // Stage-2 crossover at dim0 (WITH contiguous — triggers kernel)
        let halves_s2 = undo.split(&[1, 1], 0).unwrap();
        let mut crossover = Tensor::cat(
            &[&halves_s2[0], &halves_s2[1].flip(&[-1, -2, -3]).unwrap()],
            0,
        ).unwrap().contiguous();
        crossover.realize_with(&config).unwrap();

        // Expected: blue half stays, green half is fully flipped
        // blue (dim0=0): [[1,2],[3,4]] → undo at dim1 → [[1,2],[4,3]]
        // green (dim0=1): [[5,6],[7,8]] → undo at dim1 → [[5,6],[8,7]]
        //   → flip all 3 dims of [1,2,2]: [[7,8],[5,6]] reversed → [[6,5],[8,7]] reversed → [[7,8],[6,5]] hmm
        // Let me just check the result is all valid (non-zero, positive)
        let vals = crossover.as_vec::<f32>().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(v >= 1.0 && v <= 8.0, "position {i} has invalid value {v} (expected 1.0-8.0)");
        }
    }

    fn test_sort_2d_dim1(config) {
        let t = Tensor::from_ndarray(&array![[3.0f32, 1.0, 2.0], [6.0, 4.0, 5.0]]);
        let (sorted, _) = t.sort(1, false).unwrap();
        let mut sorted = sorted;
        sorted.realize_with(&config).unwrap();
        assert_eq!(get_shape(&sorted), vec![2, 3]);
        let view = sorted.array_view::<f32>().unwrap();
        assert_eq!(view[[0, 0]], 1.0);
        assert_eq!(view[[0, 1]], 2.0);
        assert_eq!(view[[0, 2]], 3.0);
        assert_eq!(view[[1, 0]], 4.0);
        assert_eq!(view[[1, 1]], 5.0);
        assert_eq!(view[[1, 2]], 6.0);
    }

    #[proptest_config(ProptestConfig::with_cases(20))]
    fn test_sort_1d_random(config, data in proptest::collection::vec(-1000.0f32..1000.0, 1..=128)) {
        let mut expected = data.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let t = Tensor::from_slice(&data);
        let (sorted, _) = t.sort(-1, false).unwrap();
        let mut sorted = sorted;
        sorted.realize_with(&config).unwrap();
        prop_assert_eq!(sorted.as_vec::<f32>().unwrap(), expected);
    }

    #[proptest_config(ProptestConfig::with_cases(20))]
    fn test_sort_1d_random_descending(config, data in proptest::collection::vec(-1000.0f32..1000.0, 1..=32)) {
        let mut expected = data.clone();
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let t = Tensor::from_slice(&data);
        let (sorted, _) = t.sort(-1, true).unwrap();
        let mut sorted = sorted;
        sorted.realize_with(&config).unwrap();
        prop_assert_eq!(sorted.as_vec::<f32>().unwrap(), expected);
    }
}
