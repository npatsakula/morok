//! Bit-identity tests pinning `Tensor::rand` and `threefry_random_bits`
//! against reference outputs from `jax.extend.random.threefry_2x32`.
//!
//! Reference vectors were generated locally with:
//!
//! ```text
//! uv run --with=numpy --with=jax python3 -c "..."
//! ```

use svod_dtype::{DType, ScalarDType};

use crate::Tensor;
use crate::rand::manual_seed;
use crate::rand::primitive::threefry_random_bits;
use crate::test::helpers::assert_close_f32;

use super::RAND_TEST_LOCK;

fn realize_f32(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<f32> {
    t.realize_with(config).expect("realize");
    t.as_vec::<f32>().expect("read")
}

fn realize_u32(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<u32> {
    t.realize_with(config).expect("realize");
    t.as_vec::<u32>().expect("read")
}

crate::codegen_tests! {
    /// Pins the raw THREEFRY output against `jax.extend.random.threefry_2x32`
    /// for `(key=[0, 1337], counts=arange(20).chunk(2))`.
    fn threefry_random_bits_matches_jax(config) {
        let _g = RAND_TEST_LOCK.lock();
        let u32_dt = DType::Scalar(ScalarDType::UInt32);

        let key = Tensor::from_slice([0u32, 1337u32]);
        let counts = Tensor::arange(0, Some(20), None).unwrap().cast(u32_dt);
        let counts0 = counts.try_shrink([(0usize, 10usize)]).unwrap();
        let counts1 = counts.try_shrink([(10usize, 20usize)]).unwrap();

        let mut r = threefry_random_bits(&key, &counts0, &counts1).unwrap();
        let actual = realize_u32(&mut r, &config);

        // Reference: jax.extend.random.threefry_2x32((np.uint32(1337), np.uint32(0)), np.arange(20, dtype=np.uint32))
        let expected: [u32; 20] = [
            2221762175, 1752107825, 653745012, 1967534793, 1395205442, 3840423848, 2159346757, 603508235, 3319473678,
            3363866483, 3544324138, 1436466838, 2169858556, 2570072943, 2387150698, 3678370550, 2911697663, 403244401,
            2560861638, 1692360114,
        ];
        assert_eq!(actual, expected.to_vec(), "threefry_random_bits diverged from JAX reference");
    }

    /// Pins three consecutive `Tensor::rand` outputs against JAX after
    /// `manual_seed(1337)`. End-to-end bit-identity for the full pipeline
    /// (seed derivation, counter advance, THREEFRY mixing, mantissa-fill).
    #[allow(
        clippy::excessive_precision,
        reason = "JAX-derived f32 literals; extra digits are harmless and aid copy-paste fidelity"
    )]
    fn rand_sequence_matches_jax(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(1337);

        let expected_1: [f32; 20] = [
            0.45735931, 0.6311527, 0.15571284, 0.8149418, 0.7862189, 0.80088085, 0.5685884, 0.985262, 0.4231458,
            0.9811755, 0.38059568, 0.09186363, 0.9497316, 0.5826881, 0.37963307, 0.5610522, 0.16122901, 0.3732344,
            0.9795232, 0.32806563,
        ];
        let mut a = Tensor::rand(&[20]).unwrap();
        assert_close_f32(&realize_f32(&mut a, &config), &expected_1, 1e-5);

        let expected_2: [f32; 20] = [
            0.09199333,
            0.9130762,
            0.7048608,
            0.2225498,
            0.0014830828,
            0.37023449,
            0.7790108,
            0.7484984,
            0.7524605,
            0.19875383,
            0.4853754,
            0.10002851,
            0.53693056,
            0.32947159,
            0.52469575,
            0.7659651,
            0.7949081,
            0.34988296,
            0.97985053,
            0.25995338,
        ];
        let mut b = Tensor::rand(&[20]).unwrap();
        assert_close_f32(&realize_f32(&mut b, &config), &expected_2, 1e-5);

        let expected_3: [f32; 10] = [
            0.31987143, 0.7984923, 0.3208817, 0.47160685, 0.7323365, 0.96638, 0.13873649, 0.16062307, 0.4930085,
            0.10077548,
        ];
        let mut c = Tensor::rand(&[10]).unwrap();
        assert_close_f32(&realize_f32(&mut c, &config), &expected_3, 1e-5);
    }

    /// Confirms that building many lazy `Tensor::rand` calls without
    /// realizing in between, then realizing one, doesn't blow the stack. The
    /// host-side counter makes each call's graph independent of prior calls,
    /// so this should be trivially fine — but the test pins the behavior in
    /// case the design evolves.
    fn rand_chain_833_lazy_calls_then_realize(config) {
        let _g = RAND_TEST_LOCK.lock();
        manual_seed(0);
        let _decoys: Vec<_> = (0..833).map(|_| Tensor::rand(&[1]).unwrap()).collect();
        let mut final_t = Tensor::rand(&[1]).unwrap();
        let v = realize_f32(&mut final_t, &config);
        assert_eq!(v.len(), 1);
        assert!(v[0].is_finite() && (0.0..1.0).contains(&v[0]));
    }
}
