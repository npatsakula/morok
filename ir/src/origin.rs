//! Hierarchical origin tracking for IR nodes.
//!
//! An [`Origin`] is one frame of a parent-linked tree interned in a process-global
//! arena, so a node carries a 4-byte [`OriginId`] and the hierarchy is reconstructed
//! by walking parents. Frames are pushed by the RAII [`OriginScope`], which keeps a
//! thread-local leaf that [`UOp::new`](crate::UOp::new) reads with a single TLS load.
//!
//! Capture is off unless `SVOD_ORIGIN` is set (or [`set_enabled`] is called): with
//! [`current`] pinned at `None` the node content hash is byte-identical to a build
//! without this module.

use std::cell::Cell;
use std::fmt;
use std::num::NonZeroU32;
use std::panic::Location;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use papaya::HashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;

// ============================================================================
// Source locations
// ============================================================================

/// Deserializable source location with a workspace-relative path.
///
/// Unlike [`std::panic::Location`] this owns (or borrows `'static`) its path and
/// round-trips through serde, and the path is workspace-relative so it is portable
/// across machines.
#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display, Serialize)]
#[display("{file}:{line}:{column}")]
pub struct SourceLocation {
    /// Path relative to the workspace root (e.g. `tensor/src/ops.rs`).
    pub file: std::borrow::Cow<'static, str>,
    pub line: u32,
    pub column: u32,
}

/// Manual deserialization: a derive would see the `'static` in the field type and
/// emit a borrowing impl (`'de: 'static`) that nothing can satisfy. The encoded
/// shape is exactly the derived one.
impl<'de> Deserialize<'de> for SourceLocation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Repr {
            file: String,
            line: u32,
            column: u32,
        }
        let Repr { file, line, column } = Repr::deserialize(deserializer)?;
        Ok(Self { file: std::borrow::Cow::Owned(file), line, column })
    }
}

impl SourceLocation {
    pub fn new<F: Into<String>>(file: F, line: u32, column: u32) -> Self {
        Self { file: std::borrow::Cow::Owned(file.into()), line, column }
    }

    /// Build a location from a caller site, rewriting the path to be workspace-relative.
    pub fn from_caller(loc: &'static Location<'static>) -> Self {
        Self { file: std::borrow::Cow::Borrowed(relative_location(loc)), line: loc.line(), column: loc.column() }
    }
}

/// Workspace root, derived once from `CARGO_MANIFEST_DIR` (this crate's parent).
fn workspace_root() -> &'static Path {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let path = Path::new(manifest_dir);
        path.parent().map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(manifest_dir))
    })
    .as_path()
}

/// A caller's file path relative to the workspace root, without allocating.
/// Falls back to the absolute path when the prefix does not match.
pub(crate) fn relative_location(loc: &'static Location<'static>) -> &'static str {
    let file = loc.file();
    let root = workspace_root().to_str().expect("workspace root must be valid UTF-8");
    match file.strip_prefix(root) {
        Some(stripped) => stripped.strip_prefix('/').or_else(|| stripped.strip_prefix('\\')).unwrap_or(stripped),
        None => file,
    }
}

// ============================================================================
// Frames
// ============================================================================

/// `Arc<str>` serde without serde's `rc` feature: the string is written by value,
/// so sharing is a runtime property of the arena rather than of the encoding.
mod arc_str {
    use super::*;

    pub fn serialize<S: Serializer>(value: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(value)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Arc<str>, D::Error> {
        String::deserialize(deserializer).map(Arc::from)
    }
}

mod opt_arc_str {
    use super::*;

    pub fn serialize<S: Serializer>(value: &Option<Arc<str>>, serializer: S) -> Result<S::Ok, S::Error> {
        match value {
            Some(value) => serializer.serialize_some(&**value),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Arc<str>>, D::Error> {
        Ok(Option::<String>::deserialize(deserializer)?.map(Arc::from))
    }
}

/// Name of a public entry point. Spelled as an alias because serde's derive scans
/// field types syntactically and emits a borrowing impl for any lifetime it finds.
pub type OpName = &'static str;

/// [`OpName`] serde: written by value, and read back through a tiny process-wide
/// interner so a decoded stream leaks at most one allocation per distinct op name.
mod static_str {
    use super::*;

    pub fn serialize<S: Serializer>(value: &OpName, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(value)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<OpName, D::Error> {
        Ok(intern_static(&String::deserialize(deserializer)?))
    }

    fn intern_static(text: &str) -> OpName {
        static NAMES: OnceLock<RwLock<std::collections::HashSet<&'static str>>> = OnceLock::new();
        let names = NAMES.get_or_init(RwLock::default);
        if let Some(found) = names.read().expect("op-name interner is poisoned").get(text) {
            return found;
        }
        let mut names = names.write().expect("op-name interner is poisoned");
        if let Some(found) = names.get(text) {
            return found;
        }
        let leaked: OpName = Box::leak(text.to_owned().into_boxed_str());
        names.insert(leaked);
        leaked
    }
}

/// One segment of an origin path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OriginFrame {
    /// A named scope: one state-dict segment such as `encoder`, `layers.3`, `ffn1`.
    Module {
        #[serde(with = "arc_str")]
        name: Arc<str>,
    },
    /// A public entry point, located through the `#[track_caller]` chain.
    Call {
        #[serde(with = "static_str")]
        op: OpName,
        at: SourceLocation,
    },
    /// A node of an imported ONNX graph.
    Onnx {
        index: u32,
        #[serde(with = "opt_arc_str")]
        name: Option<Arc<str>>,
        #[serde(with = "arc_str")]
        op_type: Arc<str>,
        #[serde(with = "arc_str")]
        domain: Arc<str>,
        version: i64,
    },
    /// Free-form segment for embedders and pipeline stages (`mel`, `initializer`).
    Label {
        #[serde(with = "arc_str")]
        text: Arc<str>,
    },
}

/// Renders one path segment: named frames render as their name (an ONNX node as its
/// graph name when it has one, else `#<index>:<op_type>`), a call frame as
/// `@ <op> <file>:<line>`.
impl fmt::Display for OriginFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Module { name } | Self::Label { text: name } => f.write_str(name),
            Self::Call { op, at } => write!(f, "@ {op} {}:{}", at.file, at.line),
            Self::Onnx { index, name, op_type, .. } => match name {
                Some(name) => f.write_str(name),
                None => write!(f, "#{index}:{op_type}"),
            },
        }
    }
}

/// One node of the origin tree: a frame plus the scope it was pushed inside.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Origin {
    pub parent: Option<OriginId>,
    pub frame: OriginFrame,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.frame, f)
    }
}

// ============================================================================
// Identifiers and sets
// ============================================================================

/// Interned handle into the process-global origin arena.
///
/// One-based, so `Option<OriginId>` is four bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OriginId(NonZeroU32);

impl OriginId {
    /// The raw arena index, always non-zero.
    #[inline]
    pub fn get(self) -> u32 {
        self.0.get()
    }

    /// Rebuild an id from its raw form (wire decoding). `0` is not an id.
    #[inline]
    pub fn from_raw(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }
}

impl fmt::Display for OriginId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&path(*self))
    }
}

/// Sorted, deduplicated set of origins — what a kernel, plan item or profile row
/// carries once many nodes have been folded into one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OriginSet(SmallVec<[OriginId; 4]>);

impl OriginSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one id, keeping the set sorted and deduplicated. Returns whether it was new.
    pub fn insert(&mut self, id: OriginId) -> bool {
        match self.0.binary_search(&id) {
            Ok(_) => false,
            Err(at) => {
                self.0.insert(at, id);
                true
            }
        }
    }

    /// Merge `other` into `self`.
    pub fn union(&mut self, other: &Self) {
        for &id in &other.0 {
            self.insert(id);
        }
    }

    pub fn iter(&self) -> std::slice::Iter<'_, OriginId> {
        self.0.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl std::ops::Deref for OriginSet {
    type Target = [OriginId];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromIterator<OriginId> for OriginSet {
    fn from_iter<I: IntoIterator<Item = OriginId>>(iter: I) -> Self {
        let mut ids: SmallVec<[OriginId; 4]> = iter.into_iter().collect();
        ids.sort_unstable();
        ids.dedup();
        Self(ids)
    }
}

impl Extend<OriginId> for OriginSet {
    fn extend<I: IntoIterator<Item = OriginId>>(&mut self, iter: I) {
        for id in iter {
            self.insert(id);
        }
    }
}

impl<'a> IntoIterator for &'a OriginSet {
    type Item = &'a OriginId;
    type IntoIter = std::slice::Iter<'a, OriginId>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl fmt::Display for OriginSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, id) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            f.write_str(&path(*id))?;
        }
        Ok(())
    }
}

// ============================================================================
// Arena
// ============================================================================

/// Interning map plus the append-only reverse table. Interning happens on scope
/// entry (hundreds of times per forward pass), never inside `UOp::new`, so the
/// write lock is uncontended and reads of the table are cold (rendering only).
struct Arena {
    ids: HashMap<Origin, OriginId>,
    table: RwLock<Vec<Origin>>,
}

fn arena() -> &'static Arena {
    static ARENA: OnceLock<Arena> = OnceLock::new();
    ARENA.get_or_init(|| Arena { ids: HashMap::new(), table: RwLock::new(Vec::new()) })
}

/// Intern one origin, returning its stable id. Idempotent: equal origins (same
/// parent and frame) always map to the same id for the life of the process.
pub fn intern(origin: Origin) -> OriginId {
    let arena = arena();
    let guard = arena.ids.guard();
    if let Some(&id) = arena.ids.get(&origin, &guard) {
        return id;
    }
    // Allocating under the write lock keeps `table[id - 1] == origin` for every id.
    let mut table = arena.table.write().expect("origin arena is poisoned");
    if let Some(&id) = arena.ids.get(&origin, &guard) {
        return id;
    }
    let raw = u32::try_from(table.len() + 1).expect("origin arena exceeded u32::MAX entries");
    let id = OriginId(NonZeroU32::new(raw).expect("arena indices are one-based"));
    table.push(origin.clone());
    arena.ids.insert(origin, id, &guard);
    id
}

/// Resolve an id. `None` for an id minted by another process (a decoded wire graph
/// carries raw ids whose arena entries are local to the encoding process).
pub fn get(id: OriginId) -> Option<Origin> {
    arena().table.read().expect("origin arena is poisoned").get(id.get() as usize - 1).cloned()
}

/// Every origin interned so far, in id order (`snapshot()[i]` is id `i + 1`).
/// The arena is append-only, so the length only grows.
pub fn snapshot() -> Vec<Origin> {
    arena().table.read().expect("origin arena is poisoned").clone()
}

// ============================================================================
// Rendering
// ============================================================================

/// The chain from the root scope down to `id`, root first.
pub fn chain(id: OriginId) -> Vec<OriginId> {
    let mut chain = vec![id];
    let mut cursor = id;
    while let Some(parent) = get(cursor).and_then(|origin| origin.parent) {
        chain.push(parent);
        cursor = parent;
    }
    chain.reverse();
    chain
}

/// The ancestor `depth` frames from the root (`depth == 1` is the root itself).
/// `None` for `depth == 0`; a depth past the leaf yields the leaf, so profiler
/// rollups can ask for a fixed depth without special-casing shallow paths.
pub fn truncate(id: OriginId, depth: usize) -> Option<OriginId> {
    if depth == 0 {
        return None;
    }
    let chain = chain(id);
    Some(*chain.get(depth - 1).unwrap_or_else(|| chain.last().expect("chain always contains the leaf")))
}

/// Render the full path root-to-leaf, e.g. `encoder.layers.3.ffn1.linear2` with a
/// trailing ` @ mul tensor/src/arithmetic.rs:31` when the leaf is a call frame.
pub fn path(id: OriginId) -> String {
    let mut rendered = String::new();
    let mut previous_was_name = false;
    for frame_id in chain(id) {
        let frame = get(frame_id).map(|origin| origin.frame);
        let is_call = matches!(frame, Some(OriginFrame::Call { .. }));
        if !rendered.is_empty() && (is_call || !previous_was_name) {
            rendered.push(' ');
        } else if previous_was_name {
            rendered.push('.');
        }
        match frame {
            Some(frame) => rendered.push_str(&frame.to_string()),
            None => rendered.push_str(&format!("<origin {}>", frame_id.get())),
        }
        previous_was_name = !is_call;
    }
    rendered
}

// ============================================================================
// Capture
// ============================================================================

fn flag() -> &'static AtomicBool {
    static FLAG: OnceLock<AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| {
        AtomicBool::new(std::env::var("SVOD_ORIGIN").is_ok_and(|value| !matches!(value.as_str(), "" | "0")))
    })
}

/// Whether scope constructors capture. Process-wide, seeded once from `SVOD_ORIGIN`
/// (unset, empty or `0` means off).
#[inline]
pub fn enabled() -> bool {
    flag().load(Ordering::Relaxed)
}

/// Turn capture on or off, returning the previous setting.
pub fn set_enabled(value: bool) -> bool {
    flag().swap(value, Ordering::Relaxed)
}

thread_local! {
    /// Leaf of the scope stack for this thread. Constant `None` while capture is off.
    static CURRENT: Cell<Option<OriginId>> = const { Cell::new(None) };
}

/// The innermost scope on this thread — one TLS read, on `UOp::new`'s hot path.
#[inline]
pub fn current() -> Option<OriginId> {
    CURRENT.with(Cell::get)
}

/// RAII scope: installs an origin for the current thread and restores the previous
/// one on drop, including while a panic unwinds through it.
#[must_use = "an origin scope only applies while its guard is alive"]
pub struct OriginScope {
    previous: Option<OriginId>,
}

impl OriginScope {
    /// Push a frame under the current scope. While [`enabled`] is false the frame is
    /// never built, so a disabled scope costs one TLS read and one atomic load.
    fn push(frame: impl FnOnce() -> OriginFrame) -> Self {
        let previous = current();
        if enabled() {
            CURRENT.with(|cell| cell.set(Some(intern(Origin { parent: previous, frame: frame() }))));
        }
        Self { previous }
    }

    /// A named scope, one segment of a module path.
    pub fn module(name: impl Into<Arc<str>>) -> Self {
        Self::push(|| OriginFrame::Module { name: name.into() })
    }

    /// A free-form scope for pipeline stages and embedders.
    pub fn label(text: impl Into<Arc<str>>) -> Self {
        Self::push(|| OriginFrame::Label { text: text.into() })
    }

    /// An ONNX graph node.
    pub fn onnx(index: u32, name: Option<&str>, op_type: &str, domain: &str, version: i64) -> Self {
        Self::push(|| OriginFrame::Onnx {
            index,
            name: name.map(Arc::from),
            op_type: Arc::from(op_type),
            domain: Arc::from(domain),
            version,
        })
    }

    /// A public entry point plus the caller location the `#[track_caller]` chain resolved.
    pub fn call(op: OpName, at: &'static Location<'static>) -> Self {
        Self::push(|| OriginFrame::Call { op, at: SourceLocation::from_caller(at) })
    }

    /// Detach from the enclosing scope: nodes built inside carry no origin.
    pub fn suspend() -> Self {
        install(None)
    }
}

impl Drop for OriginScope {
    fn drop(&mut self) {
        CURRENT.with(|cell| cell.set(self.previous));
    }
}

/// Adopt an already-interned scope, typically a captured [`current`] re-installed on
/// a worker thread. Unlike the frame constructors this always applies: the id it
/// carries was interned while capture was on.
pub fn install(id: Option<OriginId>) -> OriginScope {
    let previous = CURRENT.with(|cell| cell.replace(id));
    OriginScope { previous }
}
