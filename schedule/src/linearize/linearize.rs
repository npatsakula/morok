//! Priority-aware topological sort for linearization.
//!
//! Converts a UOp DAG into a linear instruction sequence suitable for
//! GPU/NPU backends that require sequential instruction streams.

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use morok_ir::AxisType;
use morok_ir::UOp;
use morok_ir::op::Op;
use morok_ir::types::ConstValue;
use morok_ir::uop::core::UOpKey;

/// Priority values for different operation types.
///
/// Lower values = higher priority (scheduled earlier).
/// Based on Tinygrad's linearizer priority assignments.
mod priority {
    pub const DEFINE_GLOBAL: i32 = -20;
    pub const DEFINE_VAR: i32 = -19;
    pub const DEFINE_LOCAL: i32 = -18;
    pub const DEFINE_REG: i32 = -17;
    pub const CONST: i32 = -10;
    pub const END: i32 = -5;
    pub const LOAD: i32 = -1;
    pub const DEFAULT: i32 = 0;
    pub const STORE: i32 = 1;
    pub const RANGE: i32 = 5;
}

/// Ordering key for heap-based scheduling.
///
/// Tuple ordering: (run_count, priority, arg_value, ideal_position, id)
/// - run_count: Higher counts scheduled later (executed in inner loops)
/// - priority: Lower values scheduled earlier
/// - arg_value: For DEFINE_GLOBAL, argument index for consistent ordering
/// - ideal_position: Position in priority-sorted order
/// - id: UOp ID for tie-breaking (ensures stable ordering)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OrderKey {
    run_count: u64,
    priority: i32,
    arg_value: Option<i64>,
    ideal_pos: usize,
    id: u64,
}

/// Convert a UOp DAG into a linear instruction sequence.
///
/// Uses priority-aware topological sorting to produce an optimal
/// instruction order for GPU/NPU execution.
///
/// # Algorithm
///
/// 1. Toposort all nodes from sink
/// 2. Build consumer graph and compute priorities (in REVERSE order!)
/// 3. Create ideal ordering based on priorities
/// 4. Use heap-based linearization respecting data dependencies
/// 5. Reverse result (we build backwards from sink)
///
/// # Priority Assignment
///
/// | Op | Priority | Purpose |
/// |----|----------|---------|
/// | DefineGlobal | -20 | Function arguments first |
/// | DefineVar | -19 | Symbolic variables early |
/// | DefineLocal | -18 | Local memory early |
/// | DefineReg | -17 | Register definitions early |
/// | Const | -10 | Constants before use |
/// | End | -5 | Close loops promptly |
/// | Load | -1 | Loads before compute |
/// | (default) | 0 | Neutral |
/// | Store | 1 | Stores after compute |
/// | Range | 5 | Loop starts late |
///
/// # Example
///
/// ```ignore
/// use morok_schedule::linearize::linearize;
///
/// let kernel_ast = /* ... */;
/// let instructions = linearize(kernel_ast);
///
/// // instructions is now a Vec<Arc<UOp>> in execution order
/// for (i, instr) in instructions.iter().enumerate() {
///     println!("{}: {:?}", i, instr.op());
/// }
/// ```
pub fn linearize(sink: Arc<UOp>) -> Vec<Arc<UOp>> {
    // Step 1: Toposort from sink
    let nodes = sink.toposort();

    if nodes.is_empty() {
        return vec![sink];
    }

    // Step 2: Build consumer graph + priorities
    // CRITICAL: Must iterate in REVERSE order for correct consumer counting
    #[allow(clippy::mutable_key_type)]
    let mut consumers: HashMap<UOpKey, Vec<Arc<UOp>>> = HashMap::new();
    #[allow(clippy::mutable_key_type)]
    let mut out_degree: HashMap<UOpKey, usize> = HashMap::new();
    #[allow(clippy::mutable_key_type)]
    let mut priorities: HashMap<UOpKey, OrderKey> = HashMap::new();
    // Map from UOp ID to Arc<UOp> for lookup
    let mut id_to_uop: HashMap<u64, Arc<UOp>> = HashMap::new();

    for u in nodes.iter().rev() {
        id_to_uop.insert(u.id, u.clone());

        // Build consumer graph
        for src in u.op().sources() {
            consumers.entry(UOpKey(src.clone())).or_default().push(u.clone());
        }

        // Compute run count from ranges
        let run_count = compute_run_count(u);

        // Assign priority based on operation type
        let (base_priority, arg_value) = get_priority(u);

        priorities.insert(
            UOpKey(u.clone()),
            OrderKey { run_count, priority: base_priority, arg_value, ideal_pos: 0, id: u.id },
        );
    }

    // Initialize out_degree (number of consumers)
    for node in &nodes {
        let key = UOpKey(node.clone());
        let degree = consumers.get(&key).map_or(0, |c| c.len());
        out_degree.insert(key, degree);
    }

    // Step 3: Create ideal ordering sorted by priority
    let mut sorted: Vec<_> = nodes.to_vec();
    sorted.sort_by_key(|u| {
        priorities.get(&UOpKey(u.clone())).cloned().unwrap_or(OrderKey {
            run_count: 0,
            priority: priority::DEFAULT,
            arg_value: None,
            ideal_pos: 0,
            id: u.id,
        })
    });

    // Assign ideal positions
    // Use reversed position so that nodes earlier in sorted order have larger ideal_pos.
    // Since BinaryHeap is a max-heap, larger values are popped first,
    // ensuring earlier nodes are processed first (consistent with sorted order).
    #[allow(clippy::mutable_key_type)]
    let nkey: HashMap<UOpKey, usize> =
        sorted.iter().enumerate().map(|(i, u)| (UOpKey(u.clone()), sorted.len() - 1 - i)).collect();

    // Update priorities with ideal positions
    for (key, pos) in &nkey {
        if let Some(order_key) = priorities.get_mut(key) {
            order_key.ideal_pos = *pos;
        }
    }

    // Step 4: Heap-based linearization
    // Use MAX-heap: larger OrderKey (worse priority) popped first.
    // After reversal, better priority nodes appear earlier in output.
    // This matches Tinygrad's use of -nkey in a min-heap.
    let mut heap: BinaryHeap<OrderKey> = BinaryHeap::new();

    let sink_key = priorities.get(&UOpKey(sink.clone())).cloned().unwrap_or(OrderKey {
        run_count: 0,
        priority: priority::DEFAULT,
        arg_value: None,
        ideal_pos: 0,
        id: sink.id,
    });
    heap.push(sink_key);

    let mut result = Vec::with_capacity(nodes.len());
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();

    while let Some(order_key) = heap.pop() {
        let u_id = order_key.id;

        // Skip if already processed (can happen with diamond dependencies)
        if visited.contains(&u_id) {
            continue;
        }
        visited.insert(u_id);

        // Look up the UOp
        let u = match id_to_uop.get(&u_id) {
            Some(uop) => uop.clone(),
            None => continue,
        };

        result.push(u.clone());

        // Decrement out_degree for all sources
        for src in u.op().sources() {
            let src_key = UOpKey(src.clone());
            if let Some(deg) = out_degree.get_mut(&src_key) {
                *deg = deg.saturating_sub(1);
                if *deg == 0 && !visited.contains(&src.id) {
                    // All consumers processed, add to heap
                    if let Some(src_order_key) = priorities.get(&src_key) {
                        heap.push(src_order_key.clone());
                    }
                }
            }
        }
    }

    // Step 5: Reverse result (we built backwards from sink)
    result.reverse();
    result
}

/// Compute the "run count" for a UOp based on its IN-SCOPE ranges.
///
/// The run count estimates how many times this operation executes,
/// based on the loop bounds of enclosing ranges that are CURRENTLY ACTIVE.
///
/// Thread ranges are EXCLUDED because they're pseudo-loops for codegen
/// structure, not actual loops. Instructions that depend on thread_id
/// should still be placed in the entry block.
///
/// CFG predecessors are propagated via the `deps` field on `Op::Range`,
/// which makes `InScopeRangesProperty` accumulate parent loop ranges
/// naturally through `children()`. This matches Tinygrad's
/// `pm_add_control_flow` behavior.
///
/// This matches Tinygrad's linearizer where `run_count = prod([int(r.vmax)+1 for r in u.ranges])`
/// and `u.ranges` returns only ranges that haven't been ended yet at that point.
fn compute_run_count(uop: &Arc<UOp>) -> u64 {
    use morok_ir::uop::cached_property::CachedProperty;
    use morok_ir::uop::properties::InScopeRangesProperty;

    #[allow(clippy::mutable_key_type)]
    let in_scope = InScopeRangesProperty::get(uop);

    if in_scope.is_empty() {
        return 1;
    }

    in_scope
        .iter()
        .filter_map(|key| {
            // Exclude Thread ranges - they're pseudo-loops, not real loops
            if let Op::Range { axis_type, .. } = key.0.op()
                && matches!(axis_type, AxisType::Thread)
            {
                return None;
            }
            // Get the maximum value of the range
            match key.0.vmax() {
                ConstValue::Int(v) => Some((v + 1) as u64),
                ConstValue::UInt(v) => Some(v + 1),
                _ => Some(1),
            }
        })
        .product()
}

/// Get priority and optional argument value for a UOp.
fn get_priority(uop: &Arc<UOp>) -> (i32, Option<i64>) {
    match uop.op() {
        Op::DefineGlobal(id) => (priority::DEFINE_GLOBAL, Some(*id as i64)),
        Op::DefineVar { .. } => (priority::DEFINE_VAR, None),
        Op::DefineLocal(id) => (priority::DEFINE_LOCAL, Some(*id as i64)),
        Op::DefineReg { .. } => (priority::DEFINE_REG, None),
        Op::Const(_) | Op::VConst { .. } => (priority::CONST, None),
        Op::End { .. } => (priority::END, None),
        Op::Load { .. } => (priority::LOAD, None),
        Op::Store { .. } => (priority::STORE, None),
        Op::Range { axis_id, .. } => (priority::RANGE, Some(axis_id.value() as i64)),
        _ => (priority::DEFAULT, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::types::ConstValue;
    use smallvec::smallvec;

    #[test]
    fn test_linearize_single_const() {
        let c = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let sink = UOp::sink(vec![c.clone()]);

        let result = linearize(sink.clone());

        assert_eq!(result.len(), 2); // const + sink
        // Const should come before sink
        assert!(matches!(result[0].op(), Op::Const(_)));
        assert!(matches!(result[1].op(), Op::Sink { .. }));
    }

    #[test]
    fn test_linearize_simple_computation() {
        let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let sum = a.try_add(&b).unwrap();
        let sink = UOp::sink(vec![sum]);

        let result = linearize(sink);

        // Should have: const, const, add, sink
        assert_eq!(result.len(), 4);
        // Constants should come first (priority -10)
        assert!(matches!(result[0].op(), Op::Const(_)));
        assert!(matches!(result[1].op(), Op::Const(_)));
        // Then binary op
        assert!(matches!(result[2].op(), Op::Binary(_, _, _)));
        // Then sink
        assert!(matches!(result[3].op(), Op::Sink { .. }));
    }

    #[test]
    fn test_linearize_with_range() {
        // Create: for i in range(10): end(value)
        let end_val = UOp::index_const(10);
        let range = UOp::range(end_val, 0);
        let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let end = value.end(smallvec![range.clone()]);
        let sink = UOp::sink(vec![end]);

        let result = linearize(sink);

        // Verify RANGE comes before END (RANGE priority 5, END priority -5)
        // But RANGE should come after its sources
        let range_pos = result.iter().position(|u| matches!(u.op(), Op::Range { .. }));
        let end_pos = result.iter().position(|u| matches!(u.op(), Op::End { .. }));

        assert!(range_pos.is_some());
        assert!(end_pos.is_some());
        // END depends on RANGE, so RANGE must come before END
        assert!(range_pos.unwrap() < end_pos.unwrap());
    }

    #[test]
    fn test_linearize_preserves_dependencies() {
        // Create a diamond dependency: a + b, where both depend on c
        let c = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let c2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
        let c3 = UOp::const_(DType::Float32, ConstValue::Float(3.0));
        let a = c.try_add(&c2).unwrap();
        let b = c.try_add(&c3).unwrap();
        let sum = a.try_add(&b).unwrap();
        let sink = UOp::sink(vec![sum.clone()]);

        let result = linearize(sink);

        // c should appear before both a and b
        let c_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &c));
        let a_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &a));
        let b_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &b));
        let sum_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &sum));

        assert!(c_pos.is_some());
        assert!(a_pos.is_some());
        assert!(b_pos.is_some());
        assert!(sum_pos.is_some());

        // Dependencies: c < a, c < b, a < sum, b < sum
        assert!(c_pos.unwrap() < a_pos.unwrap());
        assert!(c_pos.unwrap() < b_pos.unwrap());
        assert!(a_pos.unwrap() < sum_pos.unwrap());
        assert!(b_pos.unwrap() < sum_pos.unwrap());
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_priority_ordering() {
        // Test that priority order is respected: DefineGlobal < Const < default < Range
        assert!(priority::DEFINE_GLOBAL < priority::CONST);
        assert!(priority::CONST < priority::DEFAULT);
        assert!(priority::DEFAULT < priority::RANGE);
        assert!(priority::END < priority::DEFAULT);
        assert!(priority::LOAD < priority::DEFAULT);
        assert!(priority::DEFAULT < priority::STORE);
    }
}
