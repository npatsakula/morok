use super::*;

#[test]
fn test_sync_strategy() {
    assert_eq!(UnifiedExecutor::sync_strategy(&DeviceSpec::Cpu, &DeviceSpec::Cpu), SyncStrategy::None);

    assert_eq!(
        UnifiedExecutor::sync_strategy(&DeviceSpec::Cuda { device_id: 0 }, &DeviceSpec::Cuda { device_id: 1 }),
        SyncStrategy::PeerToPeer
    );

    assert_eq!(
        UnifiedExecutor::sync_strategy(&DeviceSpec::Cpu, &DeviceSpec::Cuda { device_id: 0 }),
        SyncStrategy::CpuMediated
    );
}

#[test]
fn test_executor_creation() {
    let registry = morok_device::registry::registry();
    let executor = UnifiedExecutor::new(registry);
    assert!(executor.contexts.is_empty());
}

#[test]
fn test_execution_graph_empty() {
    let mut graph = ExecutionGraph::new();
    let groups = graph.compute_parallel_groups();
    assert!(groups.is_empty());
    assert!(graph.is_valid());
}

#[test]
fn test_execution_graph_single_node() {
    let mut graph = ExecutionGraph::new();
    graph.add_node(ExecutionNode {
        id: 1,
        device: DeviceSpec::Cpu,
        inputs: vec![],
        outputs: vec![BufferId(100)],
        predecessors: vec![],
        is_transfer: false,
        buffer_access: None,
    });

    let groups = graph.compute_parallel_groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0], vec![1]);
    assert!(graph.is_valid());
}

#[test]
fn test_execution_graph_linear_chain() {
    let mut graph = ExecutionGraph::new();

    // A → B → C (linear dependency)
    graph.add_node(ExecutionNode {
        id: 1,
        device: DeviceSpec::Cpu,
        inputs: vec![],
        outputs: vec![BufferId(100)],
        predecessors: vec![],
        is_transfer: false,
        buffer_access: None,
    });
    graph.add_node(ExecutionNode {
        id: 2,
        device: DeviceSpec::Cpu,
        inputs: vec![BufferId(100)],
        outputs: vec![BufferId(101)],
        predecessors: vec![1],
        is_transfer: false,
        buffer_access: None,
    });
    graph.add_node(ExecutionNode {
        id: 3,
        device: DeviceSpec::Cpu,
        inputs: vec![BufferId(101)],
        outputs: vec![BufferId(102)],
        predecessors: vec![2],
        is_transfer: false,
        buffer_access: None,
    });

    let groups = graph.compute_parallel_groups();
    assert_eq!(groups.len(), 3); // Each node in its own group (no parallelism)
    assert!(graph.is_valid());
}

#[test]
fn test_execution_graph_parallel_nodes() {
    let mut graph = ExecutionGraph::new();

    // A and B are independent, both feed into C
    //   A ──┐
    //       └──→ C
    //   B ──┘
    graph.add_node(ExecutionNode {
        id: 1,
        device: DeviceSpec::Cpu,
        inputs: vec![],
        outputs: vec![BufferId(100)],
        predecessors: vec![],
        is_transfer: false,
        buffer_access: None,
    });
    graph.add_node(ExecutionNode {
        id: 2,
        device: DeviceSpec::Cpu,
        inputs: vec![],
        outputs: vec![BufferId(101)],
        predecessors: vec![],
        is_transfer: false,
        buffer_access: None,
    });
    graph.add_node(ExecutionNode {
        id: 3,
        device: DeviceSpec::Cpu,
        inputs: vec![BufferId(100), BufferId(101)],
        outputs: vec![BufferId(102)],
        predecessors: vec![1, 2],
        is_transfer: false,
        buffer_access: None,
    });

    let groups = graph.compute_parallel_groups();
    assert_eq!(groups.len(), 2); // First group has A,B; second has C
    assert!(groups[0].contains(&1));
    assert!(groups[0].contains(&2));
    assert_eq!(groups[1], vec![3]);
    assert!(graph.is_valid());
}
