//! Core UOp struct and fundamental operations.
//!
//! This module contains the [`UOp`] struct definition and its core methods
//! for accessing operation data, dtype, shape, and graph traversal.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::op::Op;
use crate::shape;
use crate::types::{AxisType, ConstValue};
use morok_dtype::DType;

/// Wrapper for Rc<UOp> that implements Hash and Eq based on stable ID.
///
/// This allows using Rc<UOp> as HashMap keys without implementing
/// Hash/Eq on UOp itself (which would be problematic due to OnceCell fields).
///
/// Note: While UOp contains OnceCell fields, Hash/Eq are based solely on the
/// immutable `id` field, making this safe to use as a HashMap key.
#[allow(clippy::mutable_key_type)]
#[derive(Clone, Debug)]
pub struct UOpKey(pub Rc<UOp>);

impl PartialEq for UOpKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.id == other.0.id
    }
}

impl Eq for UOpKey {}

impl Hash for UOpKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.id.hash(state);
    }
}

/// Micro-operation node in the computation graph.
///
/// UOps form a DAG where operations reference their inputs through the Op enum.
/// Hash consing ensures that structurally identical UOps share the same allocation.
///
/// Shape inference is lazy and cached - computed on first access via `shape()` method.
#[derive(Debug)]
pub struct UOp {
    /// Unique stable ID for this UOp instance.
    /// Used for identity-based caching instead of fragile raw pointers.
    pub id: u64,
    pub(crate) op: Op,
    pub(crate) dtype: DType,
    /// Cached shape - computed lazily on first access.
    /// OnceCell provides thread-safe lazy initialization.
    pub(crate) shape_cache: std::cell::OnceCell<crate::Result<Option<shape::Shape>>>,
    /// Cached list of RANGE operations in this UOp's graph.
    /// Computed lazily via toposort to collect all RANGE ops.
    pub(crate) ranges_cache: std::cell::OnceCell<Vec<Rc<UOp>>>,
    /// Cached set of RANGE operations that are in scope at this UOp.
    /// Unlike ranges_cache which contains ALL ranges in the graph,
    /// this contains only the ranges that are currently "active" (not yet ended).
    /// Computed lazily based on Tinygrad's ranges property.
    /// Uses UOpKey wrapper to enable Hash/Eq based on UOp ID.
    pub(crate) in_scope_ranges_cache: std::cell::OnceCell<HashSet<UOpKey>>,
    /// Cached vmin/vmax range analysis values.
    /// Computed lazily via range propagation through the computation graph.
    /// Returns (vmin, vmax) as ConstValue types.
    pub(crate) vmin_vmax_cache: std::cell::OnceCell<(ConstValue, ConstValue)>,
    /// Optional metadata attached to this UOp.
    ///
    /// Metadata is NOT part of hash consing - attaching metadata creates a new UOp
    /// instance with a different ID. This is used for kernel info (name, opts) after
    /// optimization is complete.
    ///
    /// Uses Arc<dyn Any> to allow attaching any metadata type without circular
    /// dependencies (e.g., schedule::KernelInfo).
    pub(crate) metadata: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
}

impl UOp {
    /// Get the operation.
    pub fn op(&self) -> &Op {
        &self.op
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    /// Get the shape of this UOp.
    ///
    /// Shape is computed lazily on first access and cached.
    /// Returns Ok(None) if shape cannot be determined (e.g., for control flow ops).
    /// Returns Err if there is a shape mismatch error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{UOp, ConstValue};
    /// # use morok_dtype::DType;
    /// let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    /// assert_eq!(scalar.shape().unwrap().as_ref().map(|s| s.len()), Some(0)); // Scalar has empty shape
    /// ```
    pub fn shape(self: &Rc<Self>) -> crate::Result<Option<&shape::Shape>> {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::ShapeProperty;
        match ShapeProperty::get(self) {
            Ok(opt) => Ok(opt.as_ref()),
            Err(e) => Err(e.clone()),
        }
    }

    /// Get the minimum possible value of this UOp.
    ///
    /// Returns the minimum value based on range analysis.
    /// Computed lazily on first access and cached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{UOp, ConstValue};
    /// # use morok_dtype::DType;
    /// let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    /// assert_eq!(five.vmin(), &ConstValue::Int(5));
    /// ```
    pub fn vmin(self: &Rc<Self>) -> &ConstValue {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::VminVmaxProperty;
        &VminVmaxProperty::get(self).0
    }

    /// Get the maximum possible value of this UOp.
    ///
    /// Returns the maximum value based on range analysis.
    /// Computed lazily on first access and cached.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{UOp, ConstValue};
    /// # use morok_dtype::DType;
    /// let five = UOp::const_(DType::Int32, ConstValue::Int(5));
    /// assert_eq!(five.vmax(), &ConstValue::Int(5));
    /// ```
    pub fn vmax(self: &Rc<Self>) -> &ConstValue {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::VminVmaxProperty;
        &VminVmaxProperty::get(self).1
    }

    /// Get the base UOp by walking through movement operations.
    ///
    /// Movement operations (RESHAPE, PERMUTE, EXPAND, etc.) are views that don't
    /// change the underlying data. This method recursively walks through these
    /// operations to find the actual buffer or computation that owns the data.
    ///
    /// Based on Tinygrad's `base` property (ops.py:524-527).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use morok_ir::{UOp, SInt, shape::Shape};
    /// # use morok_dtype::DType;
    /// # use morok_device::DeviceSpec;
    /// let buffer = UOp::new_buffer(DeviceSpec::Cpu, 10, DType::Float32);
    /// let shape = Shape::from_iter([SInt::Const(2), SInt::Const(5)]);
    /// let reshaped = UOp::try_reshape(buffer.clone(), &shape).unwrap();
    ///
    /// // base() walks through RESHAPE to get the original BUFFER
    /// assert!(std::rc::Rc::ptr_eq(&reshaped.base(), &buffer));
    /// ```
    pub fn base(self: &Rc<Self>) -> Rc<Self> {
        match &self.op {
            // Movement operations - recursively get base of source
            Op::Reshape { src, .. }
            | Op::Permute { src, .. }
            | Op::Expand { src, .. }
            | Op::Pad { src, .. }
            | Op::Shrink { src, .. }
            | Op::Flip { src, .. }
            | Op::Multi { src, .. } => src.base(),
            // All other operations are their own base
            _ => self.clone(),
        }
    }

    /// Topological sort of the computation graph.
    ///
    /// Returns nodes in an order where all dependencies come before their dependents.
    pub fn toposort(self: &Rc<Self>) -> Vec<Rc<Self>> {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Rc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else {
                stack.push((node.clone(), true));

                // Use for_each_child for zero-allocation traversal
                let mut children = Vec::new();
                node.op.map_child(|child| {
                    if !visited.contains(&Rc::as_ptr(child)) {
                        children.push(child.clone());
                    }
                });

                // Push in reverse order for proper traversal
                for child in children.into_iter().rev() {
                    stack.push((child, false));
                }
            }
        }

        result
    }

    /// Topological sort with gate function (filtered toposort).
    ///
    /// Only traverses nodes for which `gate(node)` returns true.
    /// Nodes for which gate returns false are excluded from the
    /// traversal entirely (along with their ancestors).
    ///
    /// This is a key optimization for cached property computation,
    /// allowing us to skip nodes that already have a property cached.
    ///
    /// # Performance
    ///
    /// For a graph with 10,000 nodes where 9,900 already have a cached property:
    /// - **Full toposort**: 10,000 nodes visited
    /// - **Filtered toposort**: 100 nodes visited
    /// - **Speedup**: 100x
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Only process nodes that don't have shape cached
    /// let uncached = uop.toposort_filtered(|node| {
    ///     node.shape_cache.get().is_none()
    /// });
    /// ```
    pub fn toposort_filtered<F>(self: &Rc<Self>, gate: F) -> Vec<Rc<Self>>
    where
        F: Fn(&Rc<UOp>) -> bool,
    {
        let mut visited = HashSet::new();
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            let ptr = Rc::as_ptr(&node);

            if visited.contains(&ptr) {
                continue;
            }

            if processed {
                visited.insert(ptr);
                result.push(node);
            } else {
                // Key optimization: only traverse nodes that pass the gate
                if gate(&node) {
                    stack.push((node.clone(), true));

                    let mut children = Vec::new();
                    node.op.map_child(|child| {
                        if !visited.contains(&Rc::as_ptr(child)) {
                            children.push(child.clone());
                        }
                    });

                    // Push in reverse order for proper traversal
                    for child in children.into_iter().rev() {
                        stack.push((child, false));
                    }
                }
            }
        }

        result
    }

    /// Get all RANGE operations in this UOp's computation graph.
    ///
    /// Lazily computed and cached. Useful for rangeify pass to track loop variables.
    pub fn ranges(self: &Rc<Self>) -> &Vec<Rc<Self>> {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::RangesProperty;
        RangesProperty::get(self)
    }

    /// Get the RANGE operations that are in scope at this UOp.
    ///
    /// Returns only the ranges that are currently "active" (not yet ended).
    /// This is computed by:
    /// 1. Merging ranges from all source operations
    /// 2. Removing ranges that are ended by this operation
    /// 3. Adding self if this is a RANGE operation
    ///
    /// Based on Tinygrad's `ranges` property (ops.py:318-320) and
    /// `_ranges` recursive property (ops.py:302-315).
    ///
    /// # Returns
    ///
    /// A HashSet of RANGE UOps that are in scope at this point in the graph.
    /// The result is cached for performance.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use morok_ir::{UOp, AxisType};
    ///
    /// // A simple computation inside a range
    /// let range = UOp::range(end, 0, AxisType::Loop);
    /// let value = UOp::const_(...);
    /// let end_op = UOp::end(value, vec![range.clone()]);
    ///
    /// // Value has range in scope
    /// assert!(value.in_scope_ranges().contains(&range));
    ///
    /// // After END, range is no longer in scope
    /// assert!(!end_op.in_scope_ranges().contains(&range));
    /// ```
    #[allow(clippy::mutable_key_type)]
    pub fn in_scope_ranges(self: &Rc<Self>) -> &HashSet<UOpKey> {
        use crate::uop::cached_property::CachedProperty;
        use crate::uop::properties::InScopeRangesProperty;
        InScopeRangesProperty::get(self)
    }

    /// Internal helper to compute in-scope ranges via toposort.
    ///
    /// Uses toposort to ensure we process nodes in dependency order,
    /// computing each node's scope from its sources' scopes.
    #[allow(clippy::mutable_key_type)]
    pub(crate) fn compute_in_scope_ranges(self: &Rc<Self>) -> HashSet<UOpKey> {
        use crate::Op;

        // Map from UOp ID to its computed in-scope ranges
        let mut scope_map: HashMap<u64, HashSet<UOpKey>> = HashMap::new();

        // Process in topological order (sources before consumers)
        for node in self.toposort() {
            let mut in_scope: HashSet<UOpKey> = HashSet::new();

            // Step 1: Merge ranges from all sources
            node.op.map_child(|src| {
                if let Some(src_ranges) = scope_map.get(&src.id) {
                    in_scope.extend(src_ranges.iter().cloned());
                }
            });

            // Step 2: Remove ended ranges
            for ended in node.op.ended_ranges() {
                match ended.op() {
                    Op::Range { .. } => {
                        // Remove the specific RANGE
                        in_scope.remove(&UOpKey(ended.clone()));
                    }
                    _ => {
                        // Remove all ranges that were in the ended op's scope
                        if let Some(ended_scope) = scope_map.get(&ended.id) {
                            for r in ended_scope.iter() {
                                in_scope.remove(r);
                            }
                        }
                    }
                }
            }

            // Step 3: If this is a RANGE, add it to scope
            if matches!(node.op, Op::Range { .. }) {
                in_scope.insert(UOpKey(node.clone()));
            }

            scope_map.insert(node.id, in_scope);
        }

        // Return the scope for this node
        scope_map.remove(&self.id).unwrap_or_default()
    }

    /// Check if all in-scope ranges at this UOp have the given AxisType.
    ///
    /// Returns true if the in-scope ranges set is empty or all ranges
    /// match the specified axis type.
    ///
    /// # Use Cases
    ///
    /// - `all_in_scope_ranges_are(AxisType::Outer)` - Used in split_store
    ///   to determine if we're at a kernel boundary
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use morok_ir::{UOp, AxisType};
    ///
    /// // At kernel boundary: only OUTER ranges in scope
    /// assert!(uop.all_in_scope_ranges_are(AxisType::Outer));
    ///
    /// // Inside kernel: has non-OUTER ranges
    /// assert!(!uop.all_in_scope_ranges_are(AxisType::Outer));
    /// ```
    #[allow(clippy::mutable_key_type)]
    pub fn all_in_scope_ranges_are(self: &Rc<Self>, axis_type: AxisType) -> bool {
        use crate::Op;

        let ranges = self.in_scope_ranges();

        // Empty scope means we're at the top level (treat as all OUTER)
        if ranges.is_empty() {
            return true;
        }

        ranges.iter().all(|r| match r.0.op() {
            Op::Range { axis_type: at, .. } => *at == axis_type,
            _ => false, // Should never happen
        })
    }

    /// Check if any in-scope range is NOT of the given AxisType.
    ///
    /// Inverse of `all_in_scope_ranges_are`. Useful for Tinygrad-style
    /// filtering: "skip if any range is not OUTER".
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use morok_ir::{UOp, AxisType};
    ///
    /// // Has non-OUTER ranges: should skip in split_store
    /// if uop.has_non_outer_ranges() {
    ///     return None;  // Don't split here
    /// }
    /// ```
    pub fn has_non_outer_ranges(self: &Rc<Self>) -> bool {
        !self.all_in_scope_ranges_are(AxisType::Outer)
    }

    /// Build a consumer map for this UOp's computation graph.
    ///
    /// Returns a HashMap where each UOp maps to the list of UOps that consume it.
    /// Useful for reverse traversal and dependency analysis.
    #[allow(clippy::mutable_key_type)]
    pub fn get_consumer_map(self: &Rc<Self>) -> HashMap<UOpKey, Vec<Rc<Self>>> {
        let mut consumer_map: HashMap<UOpKey, Vec<Rc<Self>>> = HashMap::new();

        for node in self.toposort() {
            node.op.map_child(|child| {
                consumer_map.entry(UOpKey(child.clone())).or_default().push(node.clone());
            });
        }

        consumer_map
    }

    /// Reverse topological sort of the computation graph.
    ///
    /// Returns nodes in bottom-up order (leaves first, root last).
    /// Requires a consumer map to traverse from leaves to roots.
    #[allow(clippy::mutable_key_type)]
    pub fn reverse_toposort(self: &Rc<Self>, consumer_map: &HashMap<UOpKey, Vec<Rc<Self>>>) -> Vec<Rc<Self>> {
        let mut visited = HashMap::new(); // Use HashMap to track visited by ID
        let mut result = Vec::new();
        let mut stack = vec![(self.clone(), false)];

        while let Some((node, processed)) = stack.pop() {
            if visited.contains_key(&node.id) {
                continue;
            }

            if processed {
                visited.insert(node.id, ());
                result.push(node);
            } else {
                stack.push((node.clone(), true));

                // Visit consumers (nodes that depend on this node)
                if let Some(consumers) = consumer_map.get(&UOpKey(node.clone())) {
                    for consumer in consumers {
                        if !visited.contains_key(&consumer.id) {
                            stack.push((consumer.clone(), false));
                        }
                    }
                }
            }
        }

        result
    }

    /// Replace UOps in the computation graph according to a substitution map.
    ///
    /// Recursively traverses the graph and replaces any UOp found in the map.
    /// Returns a new UOp with substitutions applied.
    #[allow(clippy::mutable_key_type)]
    pub fn substitute(self: &Rc<Self>, map: &HashMap<UOpKey, Rc<Self>>) -> Rc<Self> {
        use crate::Op;
        use smallvec::SmallVec;

        // Check if this UOp is in the substitution map
        if let Some(replacement) = map.get(&UOpKey(self.clone())) {
            // Record substitution transformation
            use crate::provenance::{PROVENANCE_TRACKER, PassName};
            PROVENANCE_TRACKER.with(|tracker| {
                tracker.borrow_mut().record_transform(replacement.id, self.id, PassName::Substitute);
            });

            return replacement.clone();
        }

        // Otherwise, recursively substitute in children and reconstruct Op
        let new_op = match &self.op {
            // Nullary operations - no children, just clone
            Op::Const(_)
            | Op::Unique(_)
            | Op::Device(_)
            | Op::Noop
            | Op::Invalid
            | Op::DefineGlobal(_)
            | Op::DefineLocal(_)
            | Op::VConst { .. }
            | Op::DefineVar { .. }
            | Op::DefineReg { .. } => return self.clone(),

            // Unary operations
            Op::Unary(op, src) => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Unary(*op, new_src)
            }
            Op::Cast { src, dtype } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Cast { src: new_src, dtype: dtype.clone() }
            }
            Op::BitCast { src, dtype } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::BitCast { src: new_src, dtype: dtype.clone() }
            }
            Op::Reshape { src, new_shape } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Reshape { src: new_src, new_shape: new_shape.clone() }
            }
            Op::Expand { src, new_shape } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Expand { src: new_src, new_shape: new_shape.clone() }
            }
            Op::Permute { src, axes } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Permute { src: new_src, axes: axes.clone() }
            }
            Op::Flip { src, axes } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Flip { src: new_src, axes: axes.clone() }
            }
            Op::Multi { src, axis } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Multi { src: new_src, axis: *axis }
            }
            Op::ReduceAxis { src, reduce_op, axes } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::ReduceAxis { src: new_src, reduce_op: *reduce_op, axes: axes.clone() }
            }
            Op::MSelect { buffer, device_index } => {
                let new_buffer = buffer.substitute(map);
                if Rc::ptr_eq(&new_buffer, buffer) {
                    return self.clone();
                }
                Op::MSelect { buffer: new_buffer, device_index: *device_index }
            }
            Op::Special { name, end } => {
                let new_end = end.substitute(map);
                if Rc::ptr_eq(&new_end, end) {
                    return self.clone();
                }
                Op::Special { name: name.clone(), end: new_end }
            }
            Op::BufferView { buffer, size, offset } => {
                let new_buffer = buffer.substitute(map);
                if Rc::ptr_eq(&new_buffer, buffer) {
                    return self.clone();
                }
                Op::BufferView { buffer: new_buffer, size: *size, offset: *offset }
            }
            Op::Gep { vector, indices } => {
                let new_vector = vector.substitute(map);
                if Rc::ptr_eq(&new_vector, vector) {
                    return self.clone();
                }
                Op::Gep { vector: new_vector, indices: indices.clone() }
            }
            Op::Range { end, axis_id, axis_type } => {
                let new_end = end.substitute(map);
                if Rc::ptr_eq(&new_end, end) {
                    return self.clone();
                }
                Op::Range { end: new_end, axis_id: *axis_id, axis_type: *axis_type }
            }
            Op::EndIf { if_op } => {
                let new_if_op = if_op.substitute(map);
                if Rc::ptr_eq(&new_if_op, if_op) {
                    return self.clone();
                }
                Op::EndIf { if_op: new_if_op }
            }
            Op::End { computation, ranges } => {
                let new_comp = computation.substitute(map);
                let new_ranges: SmallVec<[Rc<UOp>; 4]> = ranges.iter().map(|r| r.substitute(map)).collect();

                if Rc::ptr_eq(&new_comp, computation)
                    && new_ranges.iter().zip(ranges.iter()).all(|(a, b)| Rc::ptr_eq(a, b))
                {
                    return self.clone();
                }
                Op::End { computation: new_comp, ranges: new_ranges }
            }
            Op::Contract { src, upcast_ranges } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Contract { src: new_src, upcast_ranges: upcast_ranges.clone() }
            }
            Op::Unroll { src, unroll_axes } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Unroll { src: new_src, unroll_axes: unroll_axes.clone() }
            }
            Op::Detach { src } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Detach { src: new_src }
            }
            Op::Contiguous { src } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Contiguous { src: new_src }
            }
            Op::ContiguousBackward { src } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::ContiguousBackward { src: new_src }
            }
            Op::Precast { src } => {
                let new_src = src.substitute(map);
                if Rc::ptr_eq(&new_src, src) {
                    return self.clone();
                }
                Op::Precast { src: new_src }
            }

            // Binary operations
            Op::Binary(op, lhs, rhs) => {
                let new_lhs = lhs.substitute(map);
                let new_rhs = rhs.substitute(map);
                if Rc::ptr_eq(&new_lhs, lhs) && Rc::ptr_eq(&new_rhs, rhs) {
                    return self.clone();
                }
                Op::Binary(*op, new_lhs, new_rhs)
            }
            Op::Buffer { unique, device, size } => {
                let new_unique = unique.substitute(map);
                let new_device = device.substitute(map);
                if Rc::ptr_eq(&new_unique, unique) && Rc::ptr_eq(&new_device, device) {
                    return self.clone();
                }
                Op::Buffer { unique: new_unique, device: new_device, size: *size }
            }
            Op::PointerIndex { ptr, offset } => {
                let new_ptr = ptr.substitute(map);
                let new_offset = offset.substitute(map);
                if Rc::ptr_eq(&new_ptr, ptr) && Rc::ptr_eq(&new_offset, offset) {
                    return self.clone();
                }
                Op::PointerIndex { ptr: new_ptr, offset: new_offset }
            }
            Op::Copy { src, device } => {
                let new_src = src.substitute(map);
                let new_device = device.substitute(map);
                if Rc::ptr_eq(&new_src, src) && Rc::ptr_eq(&new_device, device) {
                    return self.clone();
                }
                Op::Copy { src: new_src, device: new_device }
            }
            Op::AllReduce { src, device, reduce_op } => {
                let new_src = src.substitute(map);
                let new_device = device.substitute(map);
                if Rc::ptr_eq(&new_src, src) && Rc::ptr_eq(&new_device, device) {
                    return self.clone();
                }
                Op::AllReduce { src: new_src, device: new_device, reduce_op: *reduce_op }
            }
            Op::Bind { var, value } => {
                let new_var = var.substitute(map);
                let new_value = value.substitute(map);
                if Rc::ptr_eq(&new_var, var) && Rc::ptr_eq(&new_value, value) {
                    return self.clone();
                }
                Op::Bind { var: new_var, value: new_value }
            }
            Op::Assign { target, value } => {
                let new_target = target.substitute(map);
                let new_value = value.substitute(map);
                if Rc::ptr_eq(&new_target, target) && Rc::ptr_eq(&new_value, value) {
                    return self.clone();
                }
                Op::Assign { target: new_target, value: new_value }
            }
            Op::Load { buffer, index } => {
                let new_buffer = buffer.substitute(map);
                let new_index = index.substitute(map);
                if Rc::ptr_eq(&new_buffer, buffer) && Rc::ptr_eq(&new_index, index) {
                    return self.clone();
                }
                Op::Load { buffer: new_buffer, index: new_index }
            }

            // Ternary operations
            Op::Ternary(op, a, b, c) => {
                let new_a = a.substitute(map);
                let new_b = b.substitute(map);
                let new_c = c.substitute(map);
                if Rc::ptr_eq(&new_a, a) && Rc::ptr_eq(&new_b, b) && Rc::ptr_eq(&new_c, c) {
                    return self.clone();
                }
                Op::Ternary(*op, new_a, new_b, new_c)
            }
            Op::Pad { src, begin_pads, end_pads } => {
                let new_src = src.substitute(map);
                let new_begin = begin_pads.substitute(map);
                let new_end = end_pads.substitute(map);
                if Rc::ptr_eq(&new_src, src) && Rc::ptr_eq(&new_begin, begin_pads) && Rc::ptr_eq(&new_end, end_pads) {
                    return self.clone();
                }
                Op::Pad { src: new_src, begin_pads: new_begin, end_pads: new_end }
            }
            Op::Shrink { src, begins, ends } => {
                let new_src = src.substitute(map);
                let new_begins = begins.substitute(map);
                let new_ends = ends.substitute(map);
                if Rc::ptr_eq(&new_src, src) && Rc::ptr_eq(&new_begins, begins) && Rc::ptr_eq(&new_ends, ends) {
                    return self.clone();
                }
                Op::Shrink { src: new_src, begins: new_begins, ends: new_ends }
            }
            Op::Wmma { a, b, c, metadata } => {
                let new_a = a.substitute(map);
                let new_b = b.substitute(map);
                let new_c = c.substitute(map);
                if Rc::ptr_eq(&new_a, a) && Rc::ptr_eq(&new_b, b) && Rc::ptr_eq(&new_c, c) {
                    return self.clone();
                }
                Op::Wmma { a: new_a, b: new_b, c: new_c, metadata: metadata.clone() }
            }
            Op::LoadGated { buffer, index, gate } => {
                let new_buffer = buffer.substitute(map);
                let new_index = index.substitute(map);
                let new_gate = gate.substitute(map);
                if Rc::ptr_eq(&new_buffer, buffer) && Rc::ptr_eq(&new_index, index) && Rc::ptr_eq(&new_gate, gate) {
                    return self.clone();
                }
                Op::LoadGated { buffer: new_buffer, index: new_index, gate: new_gate }
            }
            Op::Store { buffer, index, value } => {
                let new_buffer = buffer.substitute(map);
                let new_index = index.substitute(map);
                let new_value = value.substitute(map);
                if Rc::ptr_eq(&new_buffer, buffer) && Rc::ptr_eq(&new_index, index) && Rc::ptr_eq(&new_value, value) {
                    return self.clone();
                }
                Op::Store { buffer: new_buffer, index: new_index, value: new_value }
            }
            Op::StoreGated { buffer, index, value, gate } => {
                let new_buffer = buffer.substitute(map);
                let new_index = index.substitute(map);
                let new_value = value.substitute(map);
                let new_gate = gate.substitute(map);
                if Rc::ptr_eq(&new_buffer, buffer)
                    && Rc::ptr_eq(&new_index, index)
                    && Rc::ptr_eq(&new_value, value)
                    && Rc::ptr_eq(&new_gate, gate)
                {
                    return self.clone();
                }
                Op::StoreGated { buffer: new_buffer, index: new_index, value: new_value, gate: new_gate }
            }

            // Variable-arity operations
            Op::Sink { sources } => {
                let new_sources: SmallVec<[Rc<Self>; 4]> = sources.iter().map(|s| s.substitute(map)).collect();
                if sources.iter().zip(&new_sources).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::Sink { sources: new_sources }
            }
            Op::Group { sources } => {
                let new_sources: SmallVec<[Rc<Self>; 4]> = sources.iter().map(|s| s.substitute(map)).collect();
                if sources.iter().zip(&new_sources).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::Group { sources: new_sources }
            }
            Op::Bufferize { compute, ranges, opts } => {
                let new_compute = compute.substitute(map);
                let new_ranges: SmallVec<[Rc<Self>; 4]> = ranges.iter().map(|r| r.substitute(map)).collect();
                if Rc::ptr_eq(&new_compute, compute)
                    && ranges.iter().zip(&new_ranges).all(|(old, new)| Rc::ptr_eq(old, new))
                {
                    return self.clone();
                }
                Op::Bufferize { compute: new_compute, ranges: new_ranges, opts: opts.clone() }
            }
            Op::Index { buffer, indices, gate } => {
                let new_buffer = buffer.substitute(map);
                let new_indices: SmallVec<[Rc<Self>; 4]> = indices.iter().map(|i| i.substitute(map)).collect();
                let new_gate = gate.as_ref().map(|g| g.substitute(map));
                let gate_unchanged = match (gate, &new_gate) {
                    (None, None) => true,
                    (Some(old), Some(new)) => Rc::ptr_eq(old, new),
                    _ => false,
                };
                if Rc::ptr_eq(&new_buffer, buffer)
                    && indices.iter().zip(&new_indices).all(|(old, new)| Rc::ptr_eq(old, new))
                    && gate_unchanged
                {
                    return self.clone();
                }
                Op::Index { buffer: new_buffer, indices: new_indices, gate: new_gate }
            }
            Op::Reduce { src, reduce_op, ranges } => {
                let new_src = src.substitute(map);
                let new_ranges: SmallVec<[Rc<Self>; 4]> = ranges.iter().map(|r| r.substitute(map)).collect();
                if Rc::ptr_eq(&new_src, src) && ranges.iter().zip(&new_ranges).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::Reduce { src: new_src, reduce_op: *reduce_op, ranges: new_ranges }
            }
            Op::If { condition, body } => {
                let new_condition = condition.substitute(map);
                let new_body: SmallVec<[Rc<Self>; 4]> = body.iter().map(|b| b.substitute(map)).collect();
                if Rc::ptr_eq(&new_condition, condition)
                    && body.iter().zip(&new_body).all(|(old, new)| Rc::ptr_eq(old, new))
                {
                    return self.clone();
                }
                Op::If { condition: new_condition, body: new_body }
            }
            Op::Barrier { src, deps } => {
                let new_src = src.substitute(map);
                let new_deps: SmallVec<[Rc<Self>; 4]> = deps.iter().map(|d| d.substitute(map)).collect();
                if Rc::ptr_eq(&new_src, src) && deps.iter().zip(&new_deps).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::Barrier { src: new_src, deps: new_deps }
            }
            Op::Vectorize { elements } => {
                let new_elements: SmallVec<[Rc<Self>; 4]> = elements.iter().map(|e| e.substitute(map)).collect();
                if elements.iter().zip(&new_elements).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::Vectorize { elements: new_elements }
            }
            Op::Cat { sources } => {
                let new_sources: SmallVec<[Rc<Self>; 4]> = sources.iter().map(|s| s.substitute(map)).collect();
                if sources.iter().zip(&new_sources).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::Cat { sources: new_sources }
            }
            Op::PtrCat { sources } => {
                let new_sources: SmallVec<[Rc<Self>; 4]> = sources.iter().map(|s| s.substitute(map)).collect();
                if sources.iter().zip(&new_sources).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::PtrCat { sources: new_sources }
            }
            Op::MStack { buffers } => {
                let new_buffers: SmallVec<[Rc<Self>; 4]> = buffers.iter().map(|b| b.substitute(map)).collect();
                if buffers.iter().zip(&new_buffers).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::MStack { buffers: new_buffers }
            }
            Op::Kernel { sources, ast } => {
                let new_sources: SmallVec<[Rc<Self>; 4]> = sources.iter().map(|s| s.substitute(map)).collect();
                let new_ast = ast.substitute(map);

                if sources.iter().zip(&new_sources).all(|(old, new)| Rc::ptr_eq(old, new)) && Rc::ptr_eq(&new_ast, ast)
                {
                    return self.clone();
                }
                Op::Kernel { sources: new_sources, ast: new_ast }
            }
            Op::After { passthrough, deps } => {
                let new_passthrough = passthrough.substitute(map);
                let new_deps: SmallVec<[Rc<Self>; 4]> = deps.iter().map(|d| d.substitute(map)).collect();
                if Rc::ptr_eq(&new_passthrough, passthrough)
                    && deps.iter().zip(&new_deps).all(|(old, new)| Rc::ptr_eq(old, new))
                {
                    return self.clone();
                }
                Op::After { passthrough: new_passthrough, deps: new_deps }
            }
            Op::Custom { code, deps } => {
                let new_deps: SmallVec<[Rc<Self>; 4]> = deps.iter().map(|d| d.substitute(map)).collect();
                if deps.iter().zip(&new_deps).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::Custom { code: code.clone(), deps: new_deps }
            }
            Op::CustomI { code, deps } => {
                let new_deps: SmallVec<[Rc<Self>; 4]> = deps.iter().map(|d| d.substitute(map)).collect();
                if deps.iter().zip(&new_deps).all(|(old, new)| Rc::ptr_eq(old, new)) {
                    return self.clone();
                }
                Op::CustomI { code: code.clone(), deps: new_deps }
            }
        };

        let new_uop = Self::new(new_op, self.dtype.clone());

        // Record transformation in provenance tracker
        use crate::provenance::{PROVENANCE_TRACKER, PassName};
        PROVENANCE_TRACKER.with(|tracker| {
            tracker.borrow_mut().record_transform(new_uop.id, self.id, PassName::Substitute);
        });

        new_uop
    }

    /// Reconstruct this UOp with new sources.
    ///
    /// Creates a new UOp with the same operation type and dtype, but with the provided
    /// sources replacing the original ones. Hash consing ensures that if an identical
    /// UOp already exists, it will be reused.
    ///
    /// This is used by the graph rewrite engine when sources have been rewritten.
    ///
    /// # Panics
    ///
    /// Panics if the number of sources doesn't match the operation's arity.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Original: a + b
    /// let add = UOp::add(a.clone(), b.clone());
    ///
    /// // Rewrite sources: a' + b'
    /// let new_add = add.with_sources(vec![a_prime, b_prime]);
    /// ```
    pub fn with_sources(self: &Rc<Self>, new_srcs: Vec<Rc<Self>>) -> Rc<Self> {
        use smallvec::SmallVec;

        // Helper to get nth source
        let src = |n: usize| new_srcs[n].clone();

        let new_op = match &self.op {
            // Nullary operations - no sources
            Op::Const(_)
            | Op::Unique(_)
            | Op::Device(_)
            | Op::Noop
            | Op::Invalid
            | Op::DefineGlobal(_)
            | Op::DefineLocal(_)
            | Op::VConst { .. }
            | Op::DefineVar { .. }
            | Op::DefineReg { .. } => {
                assert_eq!(new_srcs.len(), 0, "Nullary op should have no sources");
                return self.clone(); // No sources to replace
            }

            // Unary operations
            Op::Unary(op_type, _) => {
                assert_eq!(new_srcs.len(), 1);
                Op::Unary(*op_type, src(0))
            }

            // Binary operations
            Op::Binary(op_type, _, _) => {
                assert_eq!(new_srcs.len(), 2);
                Op::Binary(*op_type, src(0), src(1))
            }

            // Ternary operations
            Op::Ternary(op_type, _, _, _) => {
                assert_eq!(new_srcs.len(), 3);
                Op::Ternary(*op_type, src(0), src(1), src(2))
            }

            // Type operations
            Op::Cast { dtype, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Cast { src: src(0), dtype: dtype.clone() }
            }
            Op::BitCast { dtype, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::BitCast { src: src(0), dtype: dtype.clone() }
            }

            // Special operations
            Op::MSelect { device_index, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::MSelect { buffer: src(0), device_index: *device_index }
            }
            Op::Special { name, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Special { end: src(0), name: name.clone() }
            }

            // Buffer operations
            Op::Buffer { size, .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::Buffer { unique: src(0), device: src(1), size: *size }
            }
            Op::BufferView { size, offset, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::BufferView { buffer: src(0), size: *size, offset: *offset }
            }
            Op::Bufferize { opts, .. } => {
                assert!(!new_srcs.is_empty());
                Op::Bufferize { compute: src(0), ranges: new_srcs[1..].iter().cloned().collect(), opts: opts.clone() }
            }
            Op::Index { gate, .. } => {
                assert!(!new_srcs.is_empty());
                // First source is buffer, rest are indices, last might be gate
                let buffer = src(0);
                let (indices, gate_new) = if gate.is_some() && new_srcs.len() >= 2 {
                    let gate_src = new_srcs.last().unwrap().clone();
                    let indices: SmallVec<[Rc<Self>; 4]> = new_srcs[1..new_srcs.len() - 1].iter().cloned().collect();
                    (indices, Some(gate_src))
                } else {
                    let indices: SmallVec<[Rc<Self>; 4]> = new_srcs[1..].iter().cloned().collect();
                    (indices, None)
                };
                Op::Index { buffer, indices, gate: gate_new }
            }
            Op::PointerIndex { .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::PointerIndex { ptr: src(0), offset: src(1) }
            }
            Op::Copy { .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::Copy { src: src(0), device: src(1) }
            }
            Op::MStack { .. } => Op::MStack { buffers: new_srcs.iter().cloned().collect() },

            // Movement operations
            Op::Reshape { .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::Reshape { src: src(0), new_shape: src(1) }
            }
            Op::Permute { axes, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Permute { src: src(0), axes: axes.clone() }
            }
            Op::Expand { .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::Expand { src: src(0), new_shape: src(1) }
            }
            Op::Pad { .. } => {
                assert_eq!(new_srcs.len(), 3);
                Op::Pad { src: src(0), begin_pads: src(1), end_pads: src(2) }
            }
            Op::Shrink { .. } => {
                assert_eq!(new_srcs.len(), 3);
                Op::Shrink { src: src(0), begins: src(1), ends: src(2) }
            }
            Op::Flip { axes, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Flip { src: src(0), axes: axes.clone() }
            }
            Op::Multi { axis, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Multi { src: src(0), axis: *axis }
            }

            // Reduction operations
            Op::ReduceAxis { reduce_op, axes, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::ReduceAxis { src: src(0), reduce_op: *reduce_op, axes: axes.clone() }
            }
            Op::Reduce { reduce_op, .. } => {
                assert!(!new_srcs.is_empty());
                Op::Reduce { src: src(0), ranges: new_srcs[1..].iter().cloned().collect(), reduce_op: *reduce_op }
            }
            Op::AllReduce { reduce_op, .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::AllReduce { src: src(0), device: src(1), reduce_op: *reduce_op }
            }

            // Control flow operations
            Op::If { .. } => {
                assert!(!new_srcs.is_empty());
                Op::If { condition: src(0), body: new_srcs[1..].iter().cloned().collect() }
            }
            Op::EndIf { .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::EndIf { if_op: src(0) }
            }
            Op::Range { axis_id, axis_type, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Range { end: src(0), axis_id: *axis_id, axis_type: *axis_type }
            }
            Op::End { .. } => {
                assert!(!new_srcs.is_empty());
                Op::End { computation: src(0), ranges: new_srcs[1..].iter().cloned().collect() }
            }
            Op::Barrier { .. } => {
                assert!(!new_srcs.is_empty());
                Op::Barrier { src: src(0), deps: new_srcs[1..].iter().cloned().collect() }
            }

            // Vector operations
            Op::Vectorize { .. } => Op::Vectorize { elements: new_srcs.iter().cloned().collect() },
            Op::Gep { indices, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Gep { vector: src(0), indices: indices.clone() }
            }
            Op::Cat { .. } => Op::Cat { sources: new_srcs.iter().cloned().collect() },
            Op::PtrCat { .. } => Op::PtrCat { sources: new_srcs.iter().cloned().collect() },

            // Symbolic/Define operations
            Op::Bind { .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::Bind { var: src(0), value: src(1) }
            }

            // Advanced operations
            Op::Wmma { metadata, .. } => {
                assert_eq!(new_srcs.len(), 3);
                Op::Wmma { a: src(0), b: src(1), c: src(2), metadata: metadata.clone() }
            }
            Op::Contract { upcast_ranges, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Contract { src: src(0), upcast_ranges: upcast_ranges.clone() }
            }
            Op::Unroll { unroll_axes, .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Unroll { src: src(0), unroll_axes: unroll_axes.clone() }
            }
            Op::Kernel { .. } => {
                assert!(!new_srcs.is_empty());
                Op::Kernel {
                    sources: new_srcs[..new_srcs.len() - 1].iter().cloned().collect(),
                    ast: src(new_srcs.len() - 1),
                }
            }
            Op::Assign { .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::Assign { target: src(0), value: src(1) }
            }
            Op::Detach { .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Detach { src: src(0) }
            }
            Op::Contiguous { .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Contiguous { src: src(0) }
            }
            Op::ContiguousBackward { .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::ContiguousBackward { src: src(0) }
            }
            Op::After { .. } => {
                assert!(!new_srcs.is_empty());
                Op::After { passthrough: src(0), deps: new_srcs[1..].iter().cloned().collect() }
            }
            Op::Precast { .. } => {
                assert_eq!(new_srcs.len(), 1);
                Op::Precast { src: src(0) }
            }
            Op::Custom { code, .. } => Op::Custom { deps: new_srcs.iter().cloned().collect(), code: code.clone() },
            Op::CustomI { code, .. } => Op::CustomI { deps: new_srcs.iter().cloned().collect(), code: code.clone() },

            // Memory operations
            Op::Load { .. } => {
                assert_eq!(new_srcs.len(), 2);
                Op::Load { buffer: src(0), index: src(1) }
            }
            Op::LoadGated { .. } => {
                assert_eq!(new_srcs.len(), 3);
                Op::LoadGated { buffer: src(0), index: src(1), gate: src(2) }
            }
            Op::Store { .. } => {
                assert_eq!(new_srcs.len(), 3);
                Op::Store { buffer: src(0), index: src(1), value: src(2) }
            }
            Op::StoreGated { .. } => {
                assert_eq!(new_srcs.len(), 4);
                Op::StoreGated { buffer: src(0), index: src(1), value: src(2), gate: src(3) }
            }

            // Graph organization
            Op::Sink { .. } => Op::Sink { sources: new_srcs.iter().cloned().collect() },
            Op::Group { .. } => Op::Group { sources: new_srcs.iter().cloned().collect() },
        };

        Self::new(new_op, self.dtype.clone())
    }
}

impl Clone for UOp {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            op: self.op.clone(),
            dtype: self.dtype.clone(),
            shape_cache: std::cell::OnceCell::new(),
            ranges_cache: std::cell::OnceCell::new(),
            in_scope_ranges_cache: std::cell::OnceCell::new(),
            vmin_vmax_cache: std::cell::OnceCell::new(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Trait for converting scalar values into UOps.
///
/// This allows operator overloading to work with mixed scalar/UOp operands.
/// For example: `uop + 5.0` or `5.0 + uop`.
pub trait IntoUOp {
    fn into_uop(self, dtype: DType) -> Rc<UOp>;
}

impl IntoUOp for ConstValue {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, self)
    }
}

impl IntoUOp for f32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self as f64))
    }
}

impl IntoUOp for f64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Float(self))
    }
}

impl IntoUOp for i32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self as i64))
    }
}

impl IntoUOp for i64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Int(self))
    }
}

impl IntoUOp for u32 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self as u64))
    }
}

impl IntoUOp for u64 {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::UInt(self))
    }
}

impl IntoUOp for bool {
    fn into_uop(self, dtype: DType) -> Rc<UOp> {
        UOp::const_(dtype, ConstValue::Bool(self))
    }
}
