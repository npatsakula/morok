//! Provenance tracking for UOps.
//!
//! This module provides infrastructure for tracking the origin and transformation
//! history of UOps through the compilation pipeline using an event-based approach.
//!
//! Each UOp has a chain of provenance events that describe:
//! - Where it was created (source location)
//! - What ONNX node it came from (if applicable)
//! - What transformations created it from parent UOps
//!
//! Provenance data is:
//! - Fully serializable/deserializable (unlike standard Location types)
//! - Stored separately from UOps (doesn't affect hash consing)
//! - Thread-local for zero-overhead access
//! - Workspace-portable (uses relative paths)

use derive_more::Display;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    cell::RefCell,
    collections::{HashMap, HashSet},
    panic::Location,
    path::{Path, PathBuf},
    sync::OnceLock,
};

/// Deserializable source code location with workspace-relative paths.
///
/// Unlike `std::panic::Location`, this type:
/// - Uses owned Strings (fully serializable)
/// - Stores workspace-relative paths (portable across machines)
/// - Can be round-tripped through JSON/binary formats
#[derive(Debug, Clone, PartialEq, Eq, Hash, Display)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[display("{file}:{line}:{column}")]
pub struct SourceLocation<'i> {
    /// Path relative to workspace root (e.g., "tensor/src/ops.rs")
    pub file: Cow<'i, str>,
    pub line: u32,
    pub column: u32,
}

impl SourceLocation<'static> {
    #[cfg(test)]
    pub(crate) fn new<F: Into<String>>(file: F, line: u32, column: u32) -> Self {
        Self { file: Cow::Owned(file.into()), line, column }
    }

    /// Create a SourceLocation from a panic::Location, converting to workspace-relative path.
    pub fn from_caller(loc: &'static Location<'static>) -> Self {
        Self { file: Cow::Borrowed(get_relative_location(loc)), line: loc.line(), column: loc.column() }
    }
}

/// ONNX node information.
///
/// Stores the origin of a UOp that was created during ONNX model import.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OnnxNodeInfo {
    /// Optional node name from the ONNX graph
    pub name: Option<String>,
    /// Operation type (e.g., "Conv", "Add", "MatMul")
    pub op_type: String,
    /// Domain (usually "ai.onnx")
    pub domain: String,
    /// Opset version
    pub version: i64,
}

impl std::fmt::Display for OnnxNodeInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ONNX:{}/{} v{}", self.domain, self.op_type, self.version)?;
        if let Some(ref name) = self.name {
            write!(f, " ({})", name)?;
        }

        Ok(())
    }
}

/// Transformation pass name.
///
/// All passes in the compilation pipeline that transform UOps.
/// Using an enum provides type safety and eliminates string allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PassName {
    /// UOp graph substitution (substitute operation)
    #[display("substitute")]
    Substitute,
    /// Range splitting transformation (shift_to)
    #[display("shift_to")]
    ShiftTo,
    /// Loop to global parallelization conversion
    #[display("convert_loop_to_global")]
    ConvertLoopToGlobal,
    /// Outer to loop conversion for CPU vectorization
    #[display("convert_outer_to_loop")]
    ConvertOuterToLoop,
    /// Pattern-based graph rewriting
    #[display("rewrite_pattern")]
    RewritePattern,
}

/// Individual provenance event in a UOp's history.
///
/// Each UOp has a chain of events describing its creation and transformations.
/// Events form a directed acyclic graph through parent/child relationships.
///
/// Note: UOp IDs are stored in the ProvenanceTracker's HashMap keys, not in the events themselves.
#[derive(Debug, Clone, PartialEq, Display)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ProvenanceEvent {
    /// UOp was created at this source location
    #[display("Created at {location}")]
    Created { location: SourceLocation<'static> },

    /// UOp originated from an ONNX node during model import
    #[display("From {node}")]
    FromOnnx { node: OnnxNodeInfo },

    /// UOp was created by transforming a parent UOp
    #[display("Transformed from UOp {from_id} by {pass_name}")]
    Transformed {
        from_id: u64,        // Parent UOp ID
        pass_name: PassName, // Transformation pass name
    },
}

/// A chain of provenance events representing a UOp's complete history.
pub type ProvenanceChain = Vec<ProvenanceEvent>;

/// Global provenance tracker for all UOps.
///
/// Maintains a mapping from UOp IDs to their event chains.
/// Events are appended as UOps are created and transformed.
///
/// Cleanup is explicit via `cleanup_with_live_set()` - no automatic cleanup.
#[derive(Default)]
pub struct ProvenanceTracker {
    /// Map from UOp ID to its provenance events
    events: HashMap<u64, Vec<ProvenanceEvent>>,
}

impl ProvenanceTracker {
    /// Capture provenance for a UOp at creation.
    ///
    /// Automatically creates a `Created` event with the caller's source location.
    pub fn capture(&mut self, uop_id: u64, location: &'static Location<'static>) {
        let event = ProvenanceEvent::Created { location: SourceLocation::from_caller(location) };

        self.events.entry(uop_id).or_default().push(event);
    }

    /// Record a transformation from one UOp to another.
    ///
    /// Creates a `Transformed` event linking the new UOp to its parent.
    pub fn record_transform(&mut self, new_id: u64, old_id: u64, pass_name: PassName) {
        self.events.entry(new_id).or_default().push(ProvenanceEvent::Transformed { from_id: old_id, pass_name });
    }

    /// Attach ONNX node information to a UOp.
    ///
    /// Creates a `FromOnnx` event for the given UOp.
    pub fn attach_onnx_node(&mut self, uop_id: u64, node: OnnxNodeInfo) {
        self.events.entry(uop_id).or_default().push(ProvenanceEvent::FromOnnx { node });
    }

    /// Get all events for a specific UOp.
    pub fn get_events(&self, uop_id: u64) -> Option<&[ProvenanceEvent]> {
        self.events.get(&uop_id).map(|v| v.as_slice())
    }

    /// Get the full provenance chain for a UOp by recursively following parent relationships.
    ///
    /// Returns a flattened chain of all events from the root UOps to the given UOp.
    pub fn get_chain(&self, uop_id: u64) -> ProvenanceChain {
        let mut chain = Vec::new();
        self.collect_chain_recursive(uop_id, &mut chain, &mut HashSet::new());
        chain
    }

    fn collect_chain_recursive(&self, uop_id: u64, chain: &mut ProvenanceChain, visited: &mut HashSet<u64>) {
        if visited.contains(&uop_id) {
            return;
        }
        visited.insert(uop_id);

        if let Some(events) = self.events.get(&uop_id) {
            // First, recursively collect parent chains
            for event in events {
                if let ProvenanceEvent::Transformed { from_id, .. } = event {
                    self.collect_chain_recursive(*from_id, chain, visited);
                }
            }

            // Then add this UOp's events
            chain.extend(events.iter().cloned());
        }
    }

    /// Remove provenance entries for UOps that are no longer alive.
    ///
    /// Users must provide the set of live UOp IDs. This is typically obtained
    /// by traversing the current AST and collecting all UOp IDs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use std::collections::HashSet;
    ///
    /// let live_ids: HashSet<u64> = ast.toposort().iter().map(|u| u.id).collect();
    /// PROVENANCE_TRACKER.with(|t| t.borrow_mut().cleanup_with_live_set(&live_ids));
    /// ```
    pub fn cleanup_with_live_set(&mut self, live_uops: &HashSet<u64>) {
        self.events.retain(|&id, _| live_uops.contains(&id));
    }

    /// Clear all provenance data.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Get the number of tracked UOps.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

thread_local! {
    /// Global thread-local provenance tracker.
    pub static PROVENANCE_TRACKER: RefCell<ProvenanceTracker> = RefCell::default();
}

/// Get the workspace root path, computed from CARGO_MANIFEST_DIR at compile time.
///
/// For workspace member crates (like "ir"), CARGO_MANIFEST_DIR points to the crate directory
/// (e.g., `/path/to/morok/ir`). The workspace root is the parent directory (`/path/to/morok`).
///
/// This is computed once on first use and cached forever.
fn workspace_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir);
        // For workspace members, parent is the workspace root
        path.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(manifest_dir))
    })
    .as_path()
}

/// Get a location string relative to the workspace root.
///
/// The workspace root is determined at compile time from CARGO_MANIFEST_DIR,
/// making this approach more reliable than runtime CWD-based stripping.
///
/// Returns a workspace-relative path (e.g., "ir/src/provenance.rs") when possible.
/// Falls back to the full absolute path if prefix stripping fails.
///
/// This provides readable, concise paths for error messages and debugging while
/// maintaining full context when relative pathing isn't possible.
///
/// Returns a `&'static str` slice from the binary without allocating.
pub(crate) fn get_relative_location(loc: &'static Location<'static>) -> &'static str {
    let file = loc.file();
    let root = workspace_root().to_str().expect("workspace root must be valid UTF-8");

    if let Some(stripped) = file.strip_prefix(root) {
        // Remove leading path separator
        stripped.strip_prefix('/').or_else(|| stripped.strip_prefix('\\')).unwrap_or(stripped)
    } else {
        file
    }
}

/// Format a provenance chain for display.
///
/// Creates a human-readable multi-line representation of the event chain.
pub fn format_chain(chain: &ProvenanceChain) -> String {
    let mut output = String::new();

    for (i, event) in chain.iter().enumerate() {
        output.push_str(&format!("\n  [{}] {}", i, event));
    }

    output
}
