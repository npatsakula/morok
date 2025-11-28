use proptest::prelude::*;

proptest! {
    #[test]
    fn least_upper_dtype(
        lhs in super::generators::scalar_generator().prop_map(crate::DType::Scalar),
        rhs in super::generators::scalar_generator().prop_map(crate::DType::Scalar)
    ) {
        // Void and Index are excluded from type promotion
        prop_assume!(lhs != crate::DType::Void && rhs != crate::DType::Void);
        prop_assume!(lhs != crate::DType::Index && rhs != crate::DType::Index);

        match crate::DType::least_upper_dtype(&[lhs.clone(), rhs.clone()]) {
            Some(_) => (),
            None => prop_assert!([lhs, rhs].contains(&crate::DType::Float64)),
        }
    }
}
