//! Data layout swizzle/permutation logic for tensor cores.
//!
//! Handles the complex data layout transformations required for optimal
//! tensor core memory access patterns.

use crate::optimizer::renderer::{SwizzleAxis, TcOpt, TensorCore};
use std::collections::HashMap;

/// Generate the base shape from tensor core opts.
///
/// Creates a shape like [U(0), L(0), L(0), L(1), L(1), L(1), U(1), R(0), R(1)]
/// that describes the axes in the kernel after applying tensor core transformations.
pub fn base_shape(tc: &TensorCore) -> Vec<SwizzleAxis> {
    let reduce_count = get_reduce_axes_count(tc);
    let mut ret = Vec::with_capacity(tc.opts.len() + reduce_count);
    let mut u_cnt = 0;
    let mut l_cnt = 0;

    // Process opts to build shape
    for opt in &tc.opts {
        match opt {
            TcOpt::Upcast(_) => {
                ret.push(SwizzleAxis::Upcast(u_cnt));
                u_cnt += 1;
            }
            TcOpt::Local(_) => {
                ret.push(SwizzleAxis::Local(l_cnt));
                l_cnt += 1;
            }
        }
    }

    // Add reduce axes (assumes UNROLL is done after opts)
    for i in 0..reduce_count {
        ret.push(SwizzleAxis::Reduce(i));
    }

    ret
}

/// Get the number of reduce axes for the tensor core.
///
/// Based on log2(K dimension) since each reduce axis splits by 2.
pub fn get_reduce_axes_count(tc: &TensorCore) -> usize {
    let k_dim = tc.dims.2;
    (k_dim as f64).log2().floor() as usize
}

/// Generate remaps for swizzle pattern application.
///
/// Creates mappings from forward shape to swizzle patterns.
fn generate_remaps(tc: &TensorCore) -> Vec<HashMap<SwizzleAxis, SwizzleAxis>> {
    let local_count = tc.opts.iter().filter(|opt| opt.is_local()).count();
    let upcast_count = tc.opts.iter().filter(|opt| opt.is_upcast()).count();
    let reduce_count = get_reduce_axes_count(tc);

    // Build forward shape (canonical order: local, upcast, reduce)
    let total_count = local_count + upcast_count + reduce_count;
    let mut fwd_shape = Vec::with_capacity(total_count);
    for i in 0..local_count {
        fwd_shape.push(SwizzleAxis::Local(i));
    }
    for i in 0..upcast_count {
        fwd_shape.push(SwizzleAxis::Upcast(i));
    }
    for i in 0..reduce_count {
        fwd_shape.push(SwizzleAxis::Reduce(i));
    }

    // Flatten swizzle tuples and create mappings
    let mut remaps = Vec::with_capacity(2); // Always two swizzle parts (A and B)
    for swizzle_part in &[&tc.swizzle.0, &tc.swizzle.1] {
        // Flatten all three parts (local, upcast, reduce)
        let flattened_size = swizzle_part.0.len() + swizzle_part.1.len() + swizzle_part.2.len();
        let mut flattened = Vec::with_capacity(flattened_size);
        flattened.extend_from_slice(&swizzle_part.0);
        flattened.extend_from_slice(&swizzle_part.1);
        flattened.extend_from_slice(&swizzle_part.2);

        // Create mapping from forward shape to swizzle pattern
        let mut remap = HashMap::with_capacity(flattened.len().min(fwd_shape.len()));
        for (i, &key) in fwd_shape.iter().enumerate() {
            if i < flattened.len() {
                remap.insert(key, flattened[i]);
            }
        }
        remaps.push(remap);
    }

    remaps
}

/// Compute permutation indices for the given shape.
///
/// Returns (perm_A, perm_B) where each perm is a list of indices
/// describing how to reorder dimensions.
pub fn permutes_for_shape(tc: &TensorCore, shape: &[SwizzleAxis]) -> (Vec<usize>, Vec<usize>) {
    let remaps = generate_remaps(tc);

    let mut perms = Vec::with_capacity(2); // Always two permutations (A and B)
    for remap in remaps {
        let mut perm = Vec::with_capacity(shape.len());
        for (i, &axis) in shape.iter().enumerate() {
            if let Some(&remapped) = remap.get(&axis) {
                // Find index of remapped value in shape
                if let Some(idx) = shape.iter().position(|&s| s == remapped) {
                    perm.push(idx);
                } else {
                    perm.push(i); // Fallback to identity
                }
            } else {
                perm.push(i); // No remap, use identity
            }
        }
        perms.push(perm);
    }

    (perms[0].clone(), perms[1].clone())
}

/// Build upcast axes configuration for WMMA construction.
///
/// Returns tuples of (axis_id, size) for each dimension's upcast/local/reduce decomposition.
/// Format: (A_axes, B_axes, C_axes)
pub fn build_upcast_axes(
    tc: &TensorCore,
    _new_ranges: &[usize],
) -> (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
    // Simplified implementation - actual implementation would extract from new_ranges
    // and match with tc.elements_per_thread

    // For now, return placeholder based on tensor core configuration
    let a_axes = vec![(0, tc.elements_per_thread.0)];
    let b_axes = vec![(1, tc.elements_per_thread.1)];
    let c_axes = vec![(2, tc.elements_per_thread.2)];

    (a_axes, b_axes, c_axes)
}
