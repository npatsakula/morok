//! Unit tests for provenance tracking.

use crate::provenance::{
    OnnxNodeInfo, PROVENANCE_TRACKER, PassName, ProvenanceEvent, ProvenanceTracker, SourceLocation,
    get_relative_location,
};
use crate::uop::UOp;
use std::f32::consts::PI;
use std::panic::Location;

#[test]
fn test_basic_provenance_capture() {
    // Create a UOp - provenance should be captured automatically
    let uop = UOp::native_const(42i32);

    PROVENANCE_TRACKER.with(|tracker| {
        let tracker = tracker.borrow();
        let events = tracker.get_events(uop.id);
        assert!(events.is_some(), "Provenance should be captured for new UOp");

        let events = events.unwrap();
        assert_eq!(events.len(), 1, "Should have one Created event");

        match &events[0] {
            ProvenanceEvent::Created { location } => {
                assert!(!location.file.is_empty());
            }
            _ => panic!("Expected Created event"),
        }
    });
}

#[test]
fn test_transformation_tracking() {
    // Create initial UOps
    let a = UOp::native_const(1i32);
    let b = UOp::native_const(2i32);

    // Perform operation (creates new UOp)
    let c = a.try_add(&b).unwrap();

    // Check provenance
    PROVENANCE_TRACKER.with(|tracker| {
        let tracker = tracker.borrow();

        // Original UOps should have Created events
        assert!(tracker.get_events(a.id).is_some());
        assert!(tracker.get_events(b.id).is_some());

        // Result should also have events
        assert!(tracker.get_events(c.id).is_some(), "Result should have provenance");
    });
}

#[test]
fn test_substitute_transformation() {
    use crate::UOpKey;
    use std::collections::HashMap;

    // Create a simple UOp
    let original = UOp::native_const(10i32);

    // Create replacement
    let replacement = UOp::native_const(20i32);

    // Build substitution map
    #[allow(clippy::mutable_key_type)]
    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(original.clone()), replacement.clone());

    // Apply substitution (should record transformation)
    let result = original.substitute(&subst_map);

    // Result should be the replacement
    assert_eq!(result.id, replacement.id);

    // Check that transformation was recorded
    PROVENANCE_TRACKER.with(|tracker| {
        let binding = tracker.borrow();
        let events = binding.get_events(result.id).unwrap();

        // Should have Created + Transformed events
        let has_transform = events.iter().any(|e| matches!(e, ProvenanceEvent::Transformed { .. }));
        assert!(has_transform, "Should have Transformed event");
    });
}

#[test]
fn test_provenance_chain() {
    use crate::UOpKey;
    use std::collections::HashMap;

    // Create initial UOp
    let uop1 = UOp::native_const(1i32);

    // Transform it multiple times
    let uop2 = UOp::native_const(2i32);
    #[allow(clippy::mutable_key_type)]
    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(uop1.clone()), uop2.clone());
    let result1 = uop1.substitute(&subst_map);

    // Another transformation
    let uop3 = UOp::native_const(3i32);
    #[allow(clippy::mutable_key_type)]
    let mut subst_map2 = HashMap::new();
    subst_map2.insert(UOpKey(result1.clone()), uop3.clone());
    let result2 = result1.substitute(&subst_map2);

    // Check full chain
    PROVENANCE_TRACKER.with(|tracker| {
        let chain = tracker.borrow().get_chain(result2.id);

        // Chain should include events from the transformation history
        assert!(!chain.is_empty(), "Should have provenance chain");

        // Should have at least one Transformed event
        let has_transforms = chain.iter().any(|e| matches!(e, ProvenanceEvent::Transformed { .. }));
        assert!(has_transforms, "Chain should include transformations");
    });
}

#[test]
fn test_onnx_node_attachment() {
    // Create a UOp
    let uop = UOp::native_const(PI);

    // Attach ONNX node information
    let onnx_node = OnnxNodeInfo {
        name: Some("conv1".to_string()),
        op_type: "Conv".to_string(),
        domain: "ai.onnx".to_string(),
        version: 11,
    };

    PROVENANCE_TRACKER.with(|tracker| {
        tracker.borrow_mut().attach_onnx_node(uop.id, onnx_node.clone());
    });

    // Verify attachment
    PROVENANCE_TRACKER.with(|tracker| {
        let binding = tracker.borrow();
        let events = binding.get_events(uop.id).unwrap();

        // Should have Created + FromOnnx events
        let has_onnx = events.iter().any(|e| match e {
            ProvenanceEvent::FromOnnx { node, .. } => node.op_type == "Conv" && node.name == Some("conv1".to_string()),
            _ => false,
        });
        assert!(has_onnx, "Should have ONNX event");
    });
}

#[test]
fn test_source_location_display() {
    // Test SourceLocation display
    let loc = SourceLocation::new("tensor/src/ops.rs", 42, 10);
    let display = loc.to_string();
    assert!(display.contains("tensor/src/ops.rs"));
    assert!(display.contains("42"));
    assert!(display.contains("10"));
}

#[test]
fn test_onnx_node_info_display() {
    // Test OnnxNodeInfo display with name
    let node = OnnxNodeInfo {
        name: Some("layer1".to_string()),
        op_type: "Add".to_string(),
        domain: "ai.onnx".to_string(),
        version: 13,
    };
    let display = node.to_string();
    assert!(display.contains("Add"));
    assert!(display.contains("layer1"));
    assert!(display.contains("13"));

    // Test OnnxNodeInfo display without name
    let node_no_name =
        OnnxNodeInfo { name: None, op_type: "Mul".to_string(), domain: "ai.onnx".to_string(), version: 13 };
    let display = node_no_name.to_string();
    assert!(display.contains("Mul"));
    assert!(!display.contains("layer"));
}

#[test]
fn test_provenance_event_display() {
    let loc = SourceLocation::new("test.rs", 10, 5);
    let created = ProvenanceEvent::Created { location: loc };
    let display = created.to_string();
    assert!(display.contains("Created"));
    assert!(display.contains("test.rs"));

    let transformed = ProvenanceEvent::Transformed { from_id: 1, pass_name: PassName::Substitute };
    let display = transformed.to_string();
    assert!(display.contains("substitute"));
    assert!(display.contains("UOp 1")); // from_id
}

#[test]
fn test_tracker_cleanup() {
    use std::collections::HashSet;

    // Create tracker and add many entries
    let mut tracker = ProvenanceTracker::default();

    let loc = Location::caller();

    // Add 100 entries
    for i in 0..100 {
        tracker.capture(i, loc);
    }

    assert_eq!(tracker.len(), 100);

    // Keep only even IDs alive (0, 2, 4, ..., 98)
    let live_set: HashSet<u64> = (0..100).filter(|i| i % 2 == 0).collect();
    assert_eq!(live_set.len(), 50);

    // Cleanup with live set
    tracker.cleanup_with_live_set(&live_set);

    // Should have only 50 entries remaining
    assert_eq!(tracker.len(), 50);

    // Verify that only live IDs remain
    for i in 0..100 {
        if i % 2 == 0 {
            assert!(tracker.get_events(i).is_some(), "Even ID {} should still exist", i);
        } else {
            assert!(tracker.get_events(i).is_none(), "Odd ID {} should be removed", i);
        }
    }

    // Test clear() as well
    tracker.clear();
    assert_eq!(tracker.len(), 0);
    assert!(tracker.is_empty());
}

#[test]
fn test_multiple_parents() {
    use crate::UOpKey;
    use std::collections::HashMap;

    // Create two UOps
    let a = UOp::native_const(1i32);
    let b = UOp::native_const(2i32);

    // Create a UOp that depends on both
    let c = a.try_add(&b).unwrap();

    // Now substitute 'a' in 'c'
    let a_new = UOp::native_const(10i32);
    #[allow(clippy::mutable_key_type)]
    let mut subst_map = HashMap::new();
    subst_map.insert(UOpKey(a.clone()), a_new.clone());
    let c_new = c.substitute(&subst_map);

    // Verify provenance tracking
    PROVENANCE_TRACKER.with(|tracker| {
        let binding = tracker.borrow();
        let events = binding.get_events(c_new.id);
        assert!(events.is_some(), "Result should have provenance");

        let events = events.unwrap();
        let has_transforms = events.iter().any(|e| matches!(e, ProvenanceEvent::Transformed { .. }));
        assert!(has_transforms, "Should have transformations");
    });
}

#[test]
fn test_format_chain() {
    use crate::provenance::format_chain;

    // Create a simple chain
    let mut tracker = ProvenanceTracker::default();
    let loc = Location::caller();

    tracker.capture(1, loc);
    tracker.record_transform(2, 1, PassName::Substitute);

    let chain = tracker.get_chain(2);

    // Format the chain
    let formatted = format_chain(&chain);

    // Should contain information about both UOps
    assert!(!formatted.is_empty());
    assert!(formatted.contains("substitute"));
}

#[test]
#[cfg(feature = "serde")]
fn test_provenance_serialization() {
    // Test that ProvenanceEvent can be serialized
    let loc = SourceLocation::new("test.rs", 100, 20);
    let event = ProvenanceEvent::Created { location: loc };

    // Serialize
    let serialized = serde_json::to_string(&event).expect("Serialization should succeed");

    // Verify serialized output contains expected data
    assert!(serialized.contains("test.rs"));
    assert!(serialized.contains("100"));

    // Deserialize
    let deserialized: ProvenanceEvent = serde_json::from_str(&serialized).expect("Deserialization should succeed");

    // Verify
    assert_eq!(event, deserialized);
}

#[test]
#[cfg(feature = "serde")]
fn test_onnx_node_serialization() {
    let node = OnnxNodeInfo {
        name: Some("test_node".to_string()),
        op_type: "Conv".to_string(),
        domain: "ai.onnx".to_string(),
        version: 11,
    };

    let event = ProvenanceEvent::FromOnnx { node };

    // Serialize and deserialize
    let serialized = serde_json::to_string(&event).expect("Serialization should succeed");
    let deserialized: ProvenanceEvent = serde_json::from_str(&serialized).expect("Deserialization should succeed");

    assert_eq!(event, deserialized);
}

#[test]
#[cfg(feature = "serde")]
fn test_transformed_event_serialization() {
    let event = ProvenanceEvent::Transformed { from_id: 1, pass_name: PassName::Substitute };

    // Serialize and deserialize
    let serialized = serde_json::to_string(&event).expect("Serialization should succeed");
    let deserialized: ProvenanceEvent = serde_json::from_str(&serialized).expect("Deserialization should succeed");

    assert_eq!(event, deserialized);
}

#[test]
fn test_error_provenance_logging() {
    use crate::error::{Error, log_provenance};

    // Create a UOp
    let uop = UOp::native_const(42i32);

    // Create an error
    let error = Error::DivisionByZero;

    // Log provenance (this should not panic)
    log_provenance(uop.id, &error);

    // Also test with non-existent UOp ID
    log_provenance(99999, &error);
}

#[test]
fn test_get_relative_location() {
    // Test that get_relative_location produces a workspace-relative path
    let loc = Location::caller();
    let relative = get_relative_location(loc);

    // Path should be relative to workspace root (determined from CARGO_MANIFEST_DIR)
    // and start with the crate name
    assert!(relative.starts_with("ir/"), "Expected relative path starting with 'ir/', got: {}", relative);
    assert!(relative.contains("provenance.rs"), "Expected path to contain 'provenance.rs', got: {}", relative);
}
