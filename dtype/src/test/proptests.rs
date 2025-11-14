use crate::*;
use proptest::prelude::*;

proptest! {
    #[test]
    fn least_upper_dtype(lhs in DType::scalar_generator(), rhs in DType::scalar_generator()) {
        // Void and Index are excluded from type promotion
        prop_assume!(lhs != DType::Void && rhs != DType::Void);
        prop_assume!(lhs != DType::Index && rhs != DType::Index);

        match DType::least_upper_dtype(&[lhs.clone(), rhs.clone()]) {
            Some(_) => (),
            None => prop_assert!([lhs, rhs].contains(&DType::Float64)),
        }
    }
}
