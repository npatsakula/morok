//! Scheduler for kernel optimization.
//!
//! The `Scheduler` manages kernel optimization state and applies transformation primitives (OptOps)
//! to improve performance on specific backends.

use std::cell::OnceCell;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::{Mutex, OnceLock};

use morok_ir::{AxisId, AxisType, Op, UOp, UOpKey};
use smallvec::SmallVec;

use super::error::*;
use super::renderer::Renderer;
use super::types::{Opt, OptOps};

/// Global kernel name counter for deduplication.
///
/// Tracks how many times each kernel name has been generated to avoid collisions.
/// When multiple kernels have the same shape, subsequent ones get suffixed with "n0", "n1", etc.
static KERNEL_NAME_COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

/// Get the kernel name counts map, initializing if needed.
fn kernel_name_counts() -> &'static Mutex<HashMap<String, usize>> {
    KERNEL_NAME_COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Clear kernel name counts (for testing).
#[cfg(test)]
pub fn clear_kernel_name_counts() {
    if let Some(counts) = KERNEL_NAME_COUNTS.get() {
        counts.lock().unwrap().clear();
    }
}

/// Flatten nested ranges in REDUCE and STORE operations.
///
/// Ensures ranges are stored in canonical flat order by collecting all
/// RANGE operations via toposort and replacing them in sorted order.
///
/// Based on Tinygrad's `pm_flatten_range` pattern.
///
/// Note: In Morok's IR, ranges are typically already flat. This function
/// ensures canonical ordering for REDUCE operations and recursively
/// processes STORE operations.
fn flatten_ranges(ast: Arc<UOp>) -> Arc<UOp> {
    match ast.op() {
        Op::Reduce { reduce_op, ranges, src } => {
            // Flatten REDUCE ranges
            let sink = UOp::sink(ranges.iter().cloned().collect());
            let flattened: Vec<_> =
                sink.toposort().into_iter().filter(|node| matches!(node.op(), Op::Range { .. })).collect();

            // Recursively flatten the src
            let flattened_src = flatten_ranges(src.clone());

            // Recreate REDUCE with flattened ranges
            UOp::reduce(flattened_src, flattened.into(), *reduce_op)
        }
        Op::Store { buffer, index, value, ranges } => {
            // Recursively flatten value being stored
            let flattened_value = flatten_ranges(value.clone());

            // Also flatten ranges
            let flattened_ranges: SmallVec<[Arc<UOp>; 4]> = ranges.iter().map(|r| flatten_ranges(r.clone())).collect();

            // Recreate STORE with flattened value and ranges
            UOp::store_with_ranges(buffer.clone(), index.clone(), flattened_value, flattened_ranges)
        }
        _ => {
            // No flattening needed for other operations
            ast
        }
    }
}

/// Scheduler for kernel optimization.
///
/// Manages the optimization state of a kernel, including:
/// - The UOp AST being optimized
/// - Backend capabilities (Renderer)
/// - Applied optimizations history
/// - Cached properties (ranges, shapes, etc.)
///
/// # Architecture
///
/// The Scheduler is the central component of the optimization layer:
/// 1. Created from a kernel AST + backend Renderer
/// 2. Applies OptOps via `apply_opt()` or heuristics
/// 3. Each optimization may create new ranges or modify existing ones
/// 4. Caches are cleared after mutations to ensure consistency
/// 5. Final optimized AST retrieved via `get_optimized_ast()`
///
/// # Example
///
/// ```ignore
/// let renderer = Renderer::cuda();
/// let mut scheduler = Scheduler::new(kernel_ast, renderer);
///
/// // Initial parallelization
/// scheduler.convert_loop_to_global()?;
///
/// // Apply optimizations
/// scheduler.apply_opt(Opt::upcast(0, 4))?;  // Vectorize by 4
/// scheduler.apply_opt(Opt::local(1, 16))?;  // 16 threads per workgroup
///
/// // Get result
/// let optimized = scheduler.get_optimized_ast(None);
/// ```
pub struct Scheduler {
    /// The kernel AST being optimized.
    ///
    /// This is the root UOp representing the computation. Immutable during the lifetime
    /// of a Scheduler instance - transformations create new ASTs.
    ast: Arc<UOp>,

    /// Backend renderer capabilities.
    ///
    /// Describes what optimizations the target backend supports and enforces device limits.
    pub ren: Renderer,

    /// Whether local memory usage is disabled.
    ///
    /// Set to true by NOLOCALS opt or if backend doesn't support local memory.
    pub dont_use_locals: bool,

    /// History of applied optimizations.
    ///
    /// Used for debugging, kernel naming, and potential undo functionality.
    pub applied_opts: Vec<Opt>,

    // Cached properties (computed lazily)
    /// Cached list of all RANGE operations, sorted by (axis_type.priority(), axis_id).
    rngs_cache: OnceCell<Vec<Arc<UOp>>>,

    /// Cached maximum axis_id used in any RANGE.
    maxarg_cache: OnceCell<usize>,
}

impl Scheduler {
    /// Create a new Scheduler for the given kernel AST and backend.
    ///
    /// # Arguments
    ///
    /// * `ast` - The kernel UOp AST to optimize (typically from rangeify phase 5)
    /// * `ren` - Backend renderer describing capabilities and limits
    ///
    /// # Returns
    ///
    /// A new Scheduler with empty optimization history and cleared caches.
    pub fn new(ast: Arc<UOp>, ren: Renderer) -> Self {
        Self {
            ast,
            ren,
            dont_use_locals: false,
            applied_opts: Vec::new(),
            rngs_cache: OnceCell::new(),
            maxarg_cache: OnceCell::new(),
        }
    }

    /// Get a reference to the current AST.
    pub fn ast(&self) -> &Arc<UOp> {
        &self.ast
    }

    /// Set the AST to a new value and clear caches.
    ///
    /// Used by optimization operations that transform the AST.
    pub(crate) fn set_ast(&mut self, ast: Arc<UOp>) {
        self.ast = ast;
        self.clear_caches();
    }

    /// Clear all cached properties.
    ///
    /// Must be called after any mutation to the AST to ensure consistency.
    pub(crate) fn clear_caches(&mut self) {
        self.rngs_cache.take();
        self.maxarg_cache.take();
    }

    /// Get the list of all RANGE operations, sorted by (axis_type.priority(), axis_id).
    ///
    /// Ranges are the fundamental unit of loop structure in the kernel. They are sorted
    /// to determine nesting order: lower priority types are outer loops.
    ///
    /// **Sorting order:**
    /// - Primary: `axis_type.priority()` (Outer=-2, Loop=-1, Global=0, ..., Unroll=5)
    /// - Secondary: `axis_id` (ascending)
    ///
    /// # Returns
    ///
    /// Cached slice of RANGE UOps in canonical order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let rngs = scheduler.rngs();
    /// for (i, rng) in rngs.iter().enumerate() {
    ///     println!("Axis {}: {:?} size={}", i, rng.axis_type(), rng.size());
    /// }
    /// ```
    pub fn rngs(&self) -> &[Arc<UOp>] {
        self.rngs_cache.get_or_init(|| self.compute_rngs())
    }

    /// Compute the list of RANGE operations and sort them.
    ///
    /// This is called lazily the first time `rngs()` is accessed.
    fn compute_rngs(&self) -> Vec<Arc<UOp>> {
        // Collect all RANGE nodes via toposort
        // Filter out size-1 ranges (where vmax == 0) to match Tinygrad's behavior
        // This causes Global(1), Local(1), etc. axes to be excluded from rngs()
        let mut ranges: Vec<Arc<UOp>> = self
            .ast
            .toposort()
            .into_iter()
            .filter(|node| {
                if let Op::Range { .. } = node.op() {
                    // Include only ranges with vmax > 0 (size > 1)
                    // vmax = size - 1, so vmax > 0 means size > 1
                    use morok_ir::ConstValue;
                    match node.vmax() {
                        ConstValue::Int(v) => *v > 0,
                        ConstValue::UInt(v) => *v > 0,
                        _ => false, // Symbolic or unknown sizes are excluded for safety
                    }
                } else {
                    false
                }
            })
            .collect();

        // Sort by (axis_type.priority(), axis_id)
        ranges.sort_by_key(|rng| {
            if let Op::Range { axis_id, axis_type, .. } = rng.op() {
                (axis_type.priority(), *axis_id)
            } else {
                unreachable!("Filtered to only Range ops")
            }
        });

        ranges
    }

    /// Get the number of dimensions (ranges) in the kernel.
    pub fn shape_len(&self) -> usize {
        self.rngs().len()
    }

    /// Get the maximum axis_id used in any RANGE operation.
    ///
    /// This is used when creating new ranges to ensure unique axis_ids.
    ///
    /// # Returns
    ///
    /// The highest axis_id, or 0 if no ranges exist.
    pub fn maxarg(&self) -> usize {
        *self.maxarg_cache.get_or_init(|| {
            self.rngs()
                .iter()
                .filter_map(|rng| if let Op::Range { axis_id, .. } = rng.op() { Some(axis_id.value()) } else { None })
                .max()
                .unwrap_or(0)
        })
    }

    /// Find the first REDUCE operation in the kernel.
    ///
    /// Used to determine if this is a reduction kernel and to extract reduction properties.
    ///
    /// # Returns
    ///
    /// The first REDUCE UOp found via toposort, or None if no reductions exist.
    pub fn reduceop(&self) -> Option<Arc<UOp>> {
        self.ast.toposort().into_iter().find(|node| matches!(node.op(), Op::Reduce { .. }))
    }

    /// Find all REDUCE operations in the kernel.
    ///
    /// Some kernels may have multiple independent reductions.
    ///
    /// # Returns
    ///
    /// Vector of all REDUCE UOps found via toposort.
    pub fn reduceops(&self) -> Vec<Arc<UOp>> {
        self.ast.toposort().into_iter().filter(|node| matches!(node.op(), Op::Reduce { .. })).collect()
    }

    /// Find all buffer access operations (INDEX) in the kernel.
    ///
    /// INDEX operations represent memory loads/stores and are used for:
    /// - Determining which buffers are accessed
    /// - Analyzing memory access patterns
    /// - Applying PADTO optimizations
    ///
    /// # Returns
    ///
    /// Vector of all INDEX UOps found via toposort.
    pub fn bufs(&self) -> Vec<Arc<UOp>> {
        self.ast.toposort().into_iter().filter(|node| matches!(node.op(), Op::Index { .. })).collect()
    }

    /// Get the output shape (dimensions without reduction axes).
    ///
    /// This is the shape of the final result tensor, excluding any REDUCE/UNROLL/GROUP_REDUCE axes.
    ///
    /// # Returns
    ///
    /// Vector of sizes for each non-reduction dimension.
    pub fn output_shape(&self) -> Vec<i64> {
        self.rngs()
            .iter()
            .filter(|rng| if let Op::Range { axis_type, .. } = rng.op() { !axis_type.is_reduce() } else { false })
            .filter_map(|rng| {
                if let Op::Range { end, .. } = rng.op()
                    && let Op::Const(cv) = end.op()
                    && let morok_ir::ConstValue::Int(sz) = cv.0
                {
                    return Some(sz);
                }
                None
            })
            .collect()
    }

    /// Get the full shape including all axes (global, local, reduce, upcast, etc.).
    ///
    /// Returns the sizes of all dimension ranges in order. Returns -1 for symbolic/unknown sizes.
    ///
    /// Used by heuristics to calculate total work and make optimization decisions.
    pub fn full_shape(&self) -> Vec<i64> {
        self.rngs()
            .iter()
            .map(|rng| {
                if let Op::Range { end, .. } = rng.op()
                    && let Op::Const(cv) = end.op()
                    && let morok_ir::ConstValue::Int(sz) = cv.0
                {
                    sz
                } else {
                    -1 // Symbolic or unknown size
                }
            })
            .collect()
    }

    /// Check if any axes have been upcasted.
    ///
    /// Returns true if there are any UPCAST axis types in the kernel.
    /// Used by heuristics to avoid redundant upcasting.
    pub fn upcasted(&self) -> bool {
        !self.axes_of(&[AxisType::Upcast]).is_empty()
    }

    /// Get a reference to the backend renderer.
    ///
    /// Returns the renderer that describes backend capabilities and constraints.
    /// Used by heuristics to check device features and limits.
    pub fn renderer(&self) -> &Renderer {
        &self.ren
    }

    /// Calculate the total upcast size (product of all UPCAST dimensions).
    ///
    /// Upcast size represents vectorization width - how many elements are processed
    /// per loop iteration. Typical values: 1 (no upcast), 2, 4, 8, 16.
    ///
    /// # Returns
    ///
    /// Product of all UPCAST and UNROLL dimension sizes, or 1 if none exist.
    ///
    /// # Note
    ///
    /// This matches Tinygrad's `upcast_size()` which computes:
    /// `prod(self.full_shape[a] for a in self.axes_of(AxisType.UPCAST, AxisType.UNROLL))`
    ///
    /// Used as a generic guard to prevent exponential vector width growth from
    /// multiple unroll sources (K-vectorization, output-upcast, general unrolling).
    pub fn upcast_size(&self) -> usize {
        self.rngs()
            .iter()
            .filter(|rng| {
                if let Op::Range { axis_type, .. } = rng.op() {
                    matches!(axis_type, AxisType::Upcast | AxisType::Unroll)
                } else {
                    false
                }
            })
            .filter_map(|rng| {
                if let Op::Range { end, .. } = rng.op()
                    && let Op::Const(cv) = end.op()
                    && let morok_ir::ConstValue::Int(sz) = cv.0
                {
                    return Some(sz as usize);
                }
                None
            })
            .product()
    }

    /// Count the number of GROUP_REDUCE axes.
    ///
    /// GROUP_REDUCE represents two-stage reductions with shared memory synchronization.
    /// Each GROUP_REDUCE axis adds a synchronization barrier.
    ///
    /// # Returns
    ///
    /// Number of GROUP_REDUCE axes (typically 0 or 1).
    pub fn group_for_reduces(&self) -> usize {
        self.rngs()
            .iter()
            .filter(|rng| {
                if let Op::Range { axis_type, .. } = rng.op() { *axis_type == AxisType::GroupReduce } else { false }
            })
            .count()
    }

    /// Get indices of axes matching any of the given types.
    ///
    /// This is useful for filtering operations that only apply to certain axis types.
    ///
    /// # Arguments
    ///
    /// * `types` - Slice of AxisTypes to match against
    ///
    /// # Returns
    ///
    /// Vector of indices into `rngs()` for matching axes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Get all reduce axes (REDUCE, UNROLL, GROUP_REDUCE)
    /// let reduce_axes = scheduler.axes_of(&[
    ///     AxisType::Reduce,
    ///     AxisType::Unroll,
    ///     AxisType::GroupReduce,
    /// ]);
    /// ```
    pub fn axes_of(&self, types: &[AxisType]) -> Vec<usize> {
        self.rngs()
            .iter()
            .enumerate()
            .filter_map(|(i, rng)| {
                if let Op::Range { axis_type, .. } = rng.op() {
                    if types.contains(axis_type) { Some(i) } else { None }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get Range UOps matching any of the given types.
    ///
    /// Similar to `axes_of()` but returns the actual Range UOps instead of indices.
    ///
    /// # Arguments
    ///
    /// * `types` - Slice of AxisTypes to match against
    ///
    /// # Returns
    ///
    /// Vector of Range UOps with matching axis types.
    pub fn ranges_of(&self, types: &[AxisType]) -> Vec<Arc<UOp>> {
        self.axes_of(types).into_iter().map(|i| self.rngs()[i].clone()).collect()
    }

    /// Get indices of axes that can be upcasted (vectorized).
    ///
    /// Upcastable axes are GLOBAL, LOCAL, or LOOP axes with size > 1.
    /// Note: OUTER is excluded per Tinygrad's design - it's for schedule expansion, not vectorization.
    ///
    /// # Returns
    ///
    /// Vector of indices into `rngs()` for upcastable axes, sorted by position.
    pub fn upcastable_dims(&self) -> Vec<usize> {
        self.rngs()
            .iter()
            .enumerate()
            .filter_map(|(i, rng)| {
                if let Op::Range { axis_type, end, .. } = rng.op() {
                    // Check type: GLOBAL/LOCAL/LOOP are upcastable
                    // (OUTER excluded - it's for schedule expansion, not vectorization)
                    if !matches!(axis_type, AxisType::Global | AxisType::Local | AxisType::Loop) {
                        return None;
                    }

                    // Check size > 1
                    if let Op::Const(cv) = end.op()
                        && let morok_ir::ConstValue::Int(sz) = cv.0
                        && sz > 1
                    {
                        return Some(i);
                    }

                    None
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get indices of axes that can be unrolled.
    ///
    /// Unrollable axes are GROUP_REDUCE or REDUCE axes with size > 1.
    /// These represent reduction loops that can be unrolled for better ILP.
    ///
    /// # Returns
    ///
    /// Vector of indices into `rngs()` for unrollable axes.
    pub fn unrollable_dims(&self) -> Vec<usize> {
        self.rngs()
            .iter()
            .enumerate()
            .filter_map(|(i, rng)| {
                if let Op::Range { axis_type, end, .. } = rng.op() {
                    // Check type
                    if !matches!(axis_type, AxisType::GroupReduce | AxisType::Reduce) {
                        return None;
                    }

                    // Check size > 1
                    if let Op::Const(cv) = end.op()
                        && let morok_ir::ConstValue::Int(sz) = cv.0
                        && sz > 1
                    {
                        return Some(i);
                    }

                    None
                } else {
                    None
                }
            })
            .collect()
    }

    /// Map logical axis index to physical axis index.
    ///
    /// Different OptOps use different logical axis numbering schemes:
    /// - Most ops: Direct index into `rngs()`
    /// - UNROLL: Index into `unrollable_dims()` (only reduction axes)
    /// - GROUP/GROUPTOP: Index into `axes_of([REDUCE])` (only REDUCE axes)
    /// - TC: Returns -1 (no single axis)
    ///
    /// # Arguments
    ///
    /// * `op` - The optimization operation type
    /// * `axis` - The logical axis index (if applicable)
    ///
    /// # Returns
    ///
    /// Physical axis index into `rngs()`, or -1 for TC operations.
    ///
    /// # Errors
    ///
    /// Returns `OptError::AxisOutOfBounds` if the logical axis is out of range.
    pub fn real_axis(&self, op: OptOps, axis: Option<usize>) -> Result<isize, OptError> {
        match op {
            // TC doesn't operate on a single axis
            OptOps::TC => Ok(-1),

            // NOLOCALS doesn't use axis
            OptOps::NOLOCALS => Ok(-1),

            // UNROLL uses logical index into unrollable dims
            OptOps::UNROLL => {
                let axis = axis.ok_or(OptError::MissingAxisParameter)?;

                let unrollable = self.unrollable_dims();
                let real_idx =
                    *unrollable.get(axis).ok_or(OptError::AxisOutOfBounds { axis, max: unrollable.len() })?;

                Ok(real_idx as isize)
            }

            // GROUP/GROUPTOP use logical index into REDUCE axes
            OptOps::GROUP | OptOps::GROUPTOP => {
                let axis = axis.ok_or(OptError::MissingAxisParameter)?;

                let reduce_axes = self.axes_of(&[AxisType::Reduce]);
                let real_idx =
                    *reduce_axes.get(axis).ok_or(OptError::AxisOutOfBounds { axis, max: reduce_axes.len() })?;

                Ok(real_idx as isize)
            }

            // All other ops use direct axis index
            _ => {
                let axis = axis.ok_or(OptError::MissingAxisParameter)?;

                if axis >= self.shape_len() {
                    return Err(OptError::AxisOutOfBounds { axis, max: self.shape_len() });
                }

                Ok(axis as isize)
            }
        }
    }

    /// Get a colored string representation of the kernel shape.
    ///
    /// Each axis is represented by its type letter and size:
    /// - Outer: 'O'
    /// - Loop: 'L'
    /// - Global: 'g'
    /// - Thread: 't'
    /// - Warp: 'w'
    /// - Local: 'l'
    /// - GroupReduce: 'G'
    /// - Upcast: 'u'
    /// - Reduce: 'R'
    /// - Unroll: 'r'
    ///
    /// # Returns
    ///
    /// String like "g16l8R32u4" (Global 16, Local 8, Reduce 32, Upcast 4).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let shape = scheduler.colored_shape();
    /// println!("Kernel: {}", shape); // "g16l16R32u4"
    /// ```
    pub fn colored_shape(&self) -> String {
        self.rngs()
            .iter()
            .filter_map(|rng| {
                if let Op::Range { axis_type, end, .. } = rng.op() {
                    // Get size
                    if let Op::Const(cv) = end.op()
                        && let morok_ir::ConstValue::Int(sz) = cv.0
                    {
                        return Some(format!("{}{}", axis_type.letter(), sz));
                    }
                    // Symbolic size
                    Some(format!("{}?", axis_type.letter()))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get a vector of string representations for each axis.
    ///
    /// Similar to `colored_shape()` but returns individual strings per axis.
    ///
    /// # Returns
    ///
    /// Vector like ["g16", "l8", "R32", "u4"].
    pub fn shape_str(&self) -> Vec<String> {
        self.rngs()
            .iter()
            .filter_map(|rng| {
                if let Op::Range { axis_type, end, .. } = rng.op() {
                    if let Op::Const(cv) = end.op()
                        && let morok_ir::ConstValue::Int(sz) = cv.0
                    {
                        return Some(format!("{}{}", axis_type.letter(), sz));
                    }
                    Some(format!("{}?", axis_type.letter()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the kernel type prefix for naming.
    ///
    /// - "r" for reduction kernels (has REDUCE op)
    /// - "E" for elementwise kernels
    ///
    /// # Returns
    ///
    /// Single character string representing kernel type.
    pub fn kernel_type(&self) -> &'static str {
        if self.reduceop().is_some() { "r" } else { "E" }
    }

    /// Core transformation: split a range into two dimensions.
    ///
    /// This is the fundamental operation used by all OptOps. It splits a single range
    /// of size `old_sz * amount` into two ranges: one of size `old_sz` (reduced original)
    /// and one of size `amount` (new range with `new_type`).
    ///
    /// The `top` parameter controls iteration order and affects memory access patterns.
    ///
    /// # Algorithm
    ///
    /// 1. **Validate divisibility:** Check that `rng.size()` is divisible by `amount`
    /// 2. **Create new range:** Either use `input_new_rng` or create one with `new_type`
    /// 3. **Create reduced old range:** Replace size with `old_sz = size / amount`
    /// 4. **Compute substitution:**
    ///    - If `top=true`: `new_rng * old_sz + replaced_rng` (new varies faster)
    ///    - If `top=false`: `replaced_rng * amount + new_rng` (old varies faster)
    /// 5. **Substitute** in AST and clear caches
    /// 6. **Return** both ranges for further transformations
    ///
    /// # Arguments
    ///
    /// * `rng` - The range to split (must be divisible by `amount`)
    /// * `amount` - The size of the new dimension
    /// * `new_type` - The AxisType for the new range (e.g., Upcast, Local)
    /// * `top` - If true, new range is outer loop; if false, new range is inner loop
    /// * `input_new_rng` - Optional pre-created range (used for specific axis_id control)
    ///
    /// # Returns
    ///
    /// `(replaced_rng, new_rng)` - The reduced old range and the new range
    ///
    /// # Errors
    ///
    /// Returns `OptError::DivisionError` if `amount` doesn't divide `rng.size()` evenly.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Split a Global(16) into Global(4) and Upcast(4)
    /// let global_16 = rngs[0].clone();
    /// let (global_4, upcast_4) = scheduler.shift_to(
    ///     global_16,
    ///     4,              // amount
    ///     AxisType::Upcast,
    ///     false,          // upcast is inner (varies faster)
    ///     None,
    /// )?;
    /// // Result: iteration order is [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
    /// ```
    #[allow(dead_code)] // Will be used in Phase 4 (OptOps implementation)
    pub(crate) fn shift_to(
        &mut self,
        rng: Arc<UOp>,
        amount: usize,
        new_type: AxisType,
        top: bool,
        input_new_rng: Option<Arc<UOp>>,
    ) -> Result<(Arc<UOp>, Arc<UOp>), OptError> {
        use morok_ir::{ConstValue, UOpKey};
        use std::collections::HashMap;

        // 1. Validate divisibility
        let old_sz = rng.divisible_by(amount).ok_or_else(|| {
            // Provide structured error based on size type
            if let Op::Range { end, .. } = rng.op() {
                if let Op::Const(cv) = end.op()
                    && let ConstValue::Int(sz) = cv.0
                {
                    DivisionSnafu { size: sz as usize, amount }.build()
                } else {
                    SymbolicDivisionSnafu { amount }.build()
                }
            } else {
                ExpectedRangeOperationSnafu.build()
            }
        })?;

        // 2. Create new range
        let new_rng = input_new_rng.unwrap_or_else(|| {
            let end = UOp::const_(morok_dtype::DType::Index, ConstValue::Int(amount as i64));
            UOp::range_axis(end, AxisId::Renumbered(self.maxarg() + 1), new_type)
        });

        // 3. Create reduced old range (same axis_id and type, but smaller size)
        let replaced_rng = if let Op::Range { axis_id, axis_type, .. } = rng.op() {
            let new_end = UOp::const_(morok_dtype::DType::Index, ConstValue::Int(old_sz as i64));
            UOp::range_axis(new_end, *axis_id, *axis_type)
        } else {
            return ExpectedRangeOperationSnafu.fail();
        };

        // 4. Compute substitution expression
        let sub_axis = if top {
            // Top order: new varies faster
            // Example: [0,8,16,24, 1,9,17,25, ...]
            let old_sz_uop = UOp::const_(morok_dtype::DType::Index, ConstValue::Int(old_sz as i64));
            new_rng
                .try_mul(&old_sz_uop)
                .expect("Multiplication should not fail for index types")
                .try_add(&replaced_rng)
                .expect("Addition should not fail for index types")
        } else {
            // Bottom order: old varies faster
            // Example: [0,1,2,3, 4,5,6,7, 8,9,10,11, ...]
            let amount_uop = UOp::const_(morok_dtype::DType::Index, ConstValue::Int(amount as i64));
            replaced_rng
                .try_mul(&amount_uop)
                .expect("Multiplication should not fail for index types")
                .try_add(&new_rng)
                .expect("Addition should not fail for index types")
        };

        // 5. Perform substitution
        #[allow(clippy::mutable_key_type)] // UOpKey uses stable ID for Hash/Eq (safe)
        let mut subst_map = HashMap::new();
        subst_map.insert(UOpKey(rng), sub_axis);

        let old_ast_id = self.ast.id;
        self.ast = self.ast.substitute(&subst_map);

        // Record high-level transformation
        if old_ast_id != self.ast.id {
            use morok_ir::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(self.ast.id, old_ast_id, PassName::ShiftTo);
            });
        }

        // Clear caches (maxarg will be recomputed on next access)
        self.clear_caches();

        // 6. Return both ranges
        Ok((replaced_rng, new_rng))
    }

    // ==== Phase 7: Initialization & Finalization ====

    /// Get all ranges from output operations (excluding REDUCE axes).
    ///
    /// Returns ranges that appear in output buffers. These are candidates for
    /// parallelization since they represent independent output elements.
    ///
    /// Based on Tinygrad's `_output_rngs()`.
    fn output_rngs(&self) -> Vec<Arc<UOp>> {
        // Find all STORE operations (outputs)
        let stores: Vec<_> = self
            .ast
            .toposort()
            .into_iter()
            .filter(|node| matches!(node.op(), Op::Store { .. } | Op::Sink { .. }))
            .collect();

        if stores.is_empty() {
            return vec![];
        }

        // Get ranges from all stores, excluding REDUCE axes
        let mut output_ranges = Vec::new();
        for store in stores {
            // Use backward_slice to get all dependencies
            let deps = store.backward_slice();
            for dep in deps {
                if let Op::Range { axis_type, .. } = dep.op() {
                    // Include all non-REDUCE ranges
                    if *axis_type != AxisType::Reduce {
                        // Only add if not already in list (use pointer equality)
                        if !output_ranges.iter().any(|r: &Arc<UOp>| Arc::ptr_eq(r, &dep)) {
                            output_ranges.push(dep);
                        }
                    }
                }
            }
        }

        output_ranges
    }

    /// Get LOOP ranges that can be safely parallelized to GLOBAL.
    ///
    /// A range is globalizable if:
    /// 1. It's currently a LOOP axis
    /// 2. It appears in all output operations (STORE nodes)
    ///
    /// This ensures parallelizing the range won't cause race conditions.
    ///
    /// Based on Tinygrad's `_globalizable_rngs()`.
    fn globalizable_rngs(&self) -> Vec<Arc<UOp>> {
        // Start with LOOP axes from outputs
        let mut candidates: Vec<_> = self
            .output_rngs()
            .into_iter()
            .filter(|r| if let Op::Range { axis_type, .. } = r.op() { *axis_type == AxisType::Loop } else { false })
            .collect();

        // Find all STORE and SINK operations
        let stores: Vec<_> = self
            .ast
            .toposort()
            .into_iter()
            .filter(|node| matches!(node.op(), Op::Store { .. } | Op::Sink { .. }))
            .collect();

        if stores.is_empty() {
            return candidates;
        }

        // Keep only ranges that appear in ALL stores
        for store in &stores {
            let store_deps = store.backward_slice();
            let store_ranges: Vec<_> =
                store_deps.into_iter().filter(|dep| matches!(dep.op(), Op::Range { .. })).collect();

            // Filter candidates to keep only those in this store's ranges
            candidates.retain(|candidate| store_ranges.iter().any(|r| Arc::ptr_eq(r, candidate)));
        }

        candidates
    }

    /// Convert eligible LOOP axes to GLOBAL for parallelization.
    ///
    /// This is the initial transformation that identifies which loops can be
    /// safely parallelized and converts them to GLOBAL (GPU thread) axes.
    ///
    /// Only applicable for GPU backends (has_local=true).
    ///
    /// Based on Tinygrad's `convert_loop_to_global()`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let renderer = Renderer::cuda();
    /// let mut scheduler = Scheduler::new(ast, renderer);
    /// scheduler.convert_loop_to_global()?;
    /// // LOOP axes that appear in all outputs are now GLOBAL
    /// ```
    pub fn convert_loop_to_global(&mut self) -> Result<(), OptError> {
        // Only for GPU backends
        if !self.ren.has_local {
            return Ok(());
        }

        let globalizable = self.globalizable_rngs();
        if globalizable.is_empty() {
            return Ok(());
        }

        // Build substitution map: LOOP → GLOBAL
        #[allow(clippy::mutable_key_type)]
        let mut subst_map = std::collections::HashMap::new();
        for rng in globalizable {
            let new_rng = rng.with_axis_type(AxisType::Global);
            subst_map.insert(UOpKey(rng), new_rng);
        }

        // Apply substitution
        let old_ast_id = self.ast.id;
        self.ast = self.ast.substitute(&subst_map);

        // Record high-level transformation
        if old_ast_id != self.ast.id {
            use morok_ir::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(self.ast.id, old_ast_id, PassName::ConvertLoopToGlobal);
            });
        }

        self.clear_caches();

        Ok(())
    }

    /// Convert OUTER axes to LOOP for CPU vectorization.
    ///
    /// For CPU backends (has_local=false), OUTER axes cannot be parallelized
    /// externally but can be vectorized. Converting them to LOOP allows
    /// the optimizer to apply UPCAST transformations for SIMD operations.
    ///
    /// **Important**: For reduce kernels, OUTER axes represent output dimensions
    /// and should NOT be converted to LOOP. This aligns with Tinygrad's design
    /// where OUTER axes remain non-upcastable in reduce kernels. Converting them
    /// would cause incorrect vectorization across independent reduction lanes.
    ///
    /// This is the CPU counterpart to `convert_loop_to_global()` for GPU.
    pub fn convert_outer_to_loop(&mut self) -> Result<(), OptError> {
        use tracing::debug;

        // Only for CPU backends (no local memory = no GPU parallelization)
        if self.ren.has_local {
            debug!("convert_outer_to_loop: skipping (has_local=true)");
            return Ok(());
        }

        // Don't convert OUTER→LOOP in reduce kernels.
        // In reduce kernels, OUTER axes are output dimensions that should not
        // be vectorized (each output element needs its own independent reduction).
        // This matches Tinygrad's architecture where OUTER is never upcastable.
        if self.reduceop().is_some() {
            debug!("convert_outer_to_loop: skipping (has reduceop)");
            return Ok(());
        }

        let all_rngs = self.rngs();
        debug!(num_rngs = all_rngs.len(), "convert_outer_to_loop: checking rngs");
        for (i, rng) in all_rngs.iter().enumerate() {
            debug!(i, axis_type = ?rng.op(), "convert_outer_to_loop: rng");
        }

        let outer_rngs: Vec<_> = self
            .rngs()
            .iter()
            .filter(|rng| matches!(rng.op(), Op::Range { axis_type: AxisType::Outer, .. }))
            .cloned()
            .collect();

        debug!(num_outer = outer_rngs.len(), "convert_outer_to_loop: found outer ranges");

        if outer_rngs.is_empty() {
            debug!("convert_outer_to_loop: no outer ranges to convert");
            return Ok(());
        }

        // Build substitution map: OUTER → LOOP
        #[allow(clippy::mutable_key_type)]
        let mut subst_map = std::collections::HashMap::new();
        for rng in outer_rngs {
            let new_rng = rng.with_axis_type(AxisType::Loop);
            debug!("convert_outer_to_loop: converting {:?} -> {:?}", rng.op(), new_rng.op());
            subst_map.insert(UOpKey(rng), new_rng);
        }

        // Apply substitution
        let old_ast_id = self.ast.id;
        self.ast = self.ast.substitute(&subst_map);
        debug!(old_id = old_ast_id, new_id = self.ast.id, "convert_outer_to_loop: substitution complete");

        // Record high-level transformation
        if old_ast_id != self.ast.id {
            use morok_ir::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(self.ast.id, old_ast_id, PassName::ConvertOuterToLoop);
            });
        }

        self.clear_caches();

        Ok(())
    }

    /// Get the optimized AST with kernel metadata attached.
    ///
    /// This is the final step of optimization, which:
    /// 1. Generates a kernel name from the shape (e.g., "r_g16l16R32u4")
    /// 2. Flattens nested ranges
    /// 3. Attaches KernelInfo metadata
    ///
    /// # Arguments
    ///
    /// * `name_override` - Optional custom kernel name (otherwise auto-generated)
    ///
    /// # Returns
    ///
    /// UOp with attached KernelInfo metadata containing name, applied_opts, and flags.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let optimized = scheduler.get_optimized_ast(None);
    /// let info = optimized.metadata::<KernelInfo>().unwrap();
    /// println!("Kernel: {}", info.name); // "r_g16l16R32u4"
    /// ```
    pub fn get_optimized_ast(&self, name_override: Option<String>) -> Arc<UOp> {
        use crate::optimizer::KernelInfo;

        // 1. Generate kernel name
        let name = name_override.unwrap_or_else(|| {
            // Prefix: "r" for reduce, "E" for elementwise
            let prefix = if self.reduceop().is_some() { "r" } else { "E" };

            // Encode each range: {letter}{size}
            // Based on Tinygrad's axis_letters mapping
            let shape_parts: Vec<String> = self
                .rngs()
                .iter()
                .filter_map(|rng| {
                    if let Op::Range { end, axis_type, .. } = rng.op() {
                        // Get size if constant
                        let size = if let Op::Const(cv) = end.op()
                            && let morok_ir::ConstValue::Int(sz) = cv.0
                        {
                            sz.to_string()
                        } else {
                            "?".to_string()
                        };

                        // Get letter for axis type
                        let letter = match axis_type {
                            AxisType::Global => "g",
                            AxisType::Local => "l",
                            AxisType::Loop => "L",
                            AxisType::Upcast => "u",
                            AxisType::Reduce => "R",
                            AxisType::GroupReduce => "G",
                            AxisType::Unroll => "r",
                            AxisType::Warp => "w",
                            AxisType::Thread => "t",
                            AxisType::Outer => "O",
                        };

                        Some(format!("{}{}", letter, size))
                    } else {
                        None
                    }
                })
                .collect();

            format!("{}_{}", prefix, shape_parts.join(""))
        });

        // Deduplicate kernel names
        let name = {
            let mut counts = kernel_name_counts().lock().unwrap();
            let count = counts.entry(name.clone()).or_insert(0);
            *count += 1;

            if *count > 1 { format!("{}n{}", name, *count - 1) } else { name }
        };

        // 2. Flatten ranges
        let flattened_ast = flatten_ranges(self.ast.clone());

        // 3. Attach metadata
        let info = KernelInfo { name, applied_opts: self.applied_opts.clone(), dont_use_locals: self.dont_use_locals };
        flattened_ast.with_metadata(info)
    }
}

impl fmt::Display for Scheduler {
    /// Format the scheduler as a kernel descriptor string.
    ///
    /// Format: "{kernel_type}_{colored_shape}"
    ///
    /// Examples:
    /// - "r_g16l16R32u4" - Reduction kernel with Global, Local, Reduce, Upcast
    /// - "E_g256g256" - Elementwise kernel with 2D Global shape
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.kernel_type(), self.colored_shape())
    }
}

impl Clone for Scheduler {
    /// Clone the scheduler state.
    ///
    /// Note: Caches are cleared in the clone to ensure correct behavior.
    fn clone(&self) -> Self {
        Self {
            ast: self.ast.clone(),
            ren: self.ren.clone(),
            dont_use_locals: self.dont_use_locals,
            applied_opts: self.applied_opts.clone(),
            // Clear caches in clone - they'll be recomputed on demand
            rngs_cache: OnceCell::new(),
            maxarg_cache: OnceCell::new(),
        }
    }
}
