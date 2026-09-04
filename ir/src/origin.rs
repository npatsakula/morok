//! Hierarchical origin tracking for IR nodes.
//!
//! An [`Origin`] is one frame of a parent-linked tree interned in a process-global
//! arena, so a node carries a 4-byte [`OriginId`] and the hierarchy is reconstructed
//! by walking parents. Frames are pushed by the RAII [`OriginScope`], which keeps a
//! thread-local leaf that [`UOp::new`](crate::UOp::new) reads with a single TLS load.
//!
//! Capture is off unless `SVOD_ORIGIN` is set (or [`capture_for_thread`] is used):
//! with [`current`] pinned at `None` the node content hash is byte-identical to a
//! build without this module.

use std::cell::Cell;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::num::NonZeroU32;
use std::panic::Location;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use serde::{Deserialize, Serialize};

// ============================================================================
// Source locations
// ============================================================================

/// Source location with a workspace-relative path, serializable unlike
/// [`std::panic::Location`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Display, Serialize)]
#[display("{file}:{line}:{column}")]
pub struct SourceLocation {
    /// Path relative to the workspace root (e.g. `tensor/src/ops.rs`).
    pub file: std::borrow::Cow<'static, str>,
    pub line: u32,
    pub column: u32,
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

/// One segment of an origin path.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub enum OriginFrame {
    /// A named scope: one state-dict segment such as `encoder`, `layers.3`, `ffn1`.
    Module { name: String },
    /// A public entry point, located through the `#[track_caller]` chain.
    Call { op: &'static str, at: SourceLocation },
    /// A node of an imported ONNX graph.
    Onnx { index: u32, name: Option<String>, op_type: String, domain: String, version: i64 },
    /// Free-form segment for embedders and pipeline stages (`mel`, `initializer`).
    Label { text: String },
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
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Origin {
    pub parent: Option<OriginId>,
    pub frame: OriginFrame,
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.frame, f)
    }
}

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

/// The origins a kernel, plan item or profile row carries once many nodes have
/// been folded into one.
pub type OriginSet = BTreeSet<OriginId>;

// ============================================================================
// Arena
// ============================================================================

/// Interning map plus the append-only reverse table (`table[id - 1]`). Interning
/// happens on scope entry, never inside `UOp::new`, so one mutex is plenty.
#[derive(Default)]
struct Arena {
    ids: HashMap<Origin, OriginId>,
    table: Vec<Origin>,
}

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    static ARENA: OnceLock<Mutex<Arena>> = OnceLock::new();
    ARENA.get_or_init(Mutex::default).lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Intern one origin, returning its stable id. Idempotent: equal origins (same
/// parent and frame) always map to the same id for the life of the process.
pub fn intern(origin: Origin) -> OriginId {
    let mut arena = arena();
    if let Some(&id) = arena.ids.get(&origin) {
        return id;
    }
    let raw = u32::try_from(arena.table.len() + 1).expect("origin arena exceeded u32::MAX entries");
    let id = OriginId(NonZeroU32::new(raw).expect("arena indices are one-based"));
    arena.table.push(origin.clone());
    arena.ids.insert(origin, id);
    id
}

/// Resolve an id. `None` for an id minted by another process (a decoded wire graph
/// carries raw ids whose arena entries are local to the encoding process).
pub fn get(id: OriginId) -> Option<Origin> {
    arena().table.get(id.get() as usize - 1).cloned()
}

/// Every origin interned so far, in id order (`snapshot()[i]` is id `i + 1`).
pub fn snapshot() -> Vec<Origin> {
    arena().table.clone()
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

/// This thread's capture state: whether scopes capture at all, the innermost
/// scope, and whether that scope is a call frame (so nested public ops keep only
/// the outermost call).
#[derive(Clone, Copy)]
struct State {
    enabled: bool,
    current: Option<OriginId>,
    in_call: bool,
}

/// Process-wide default, read once from `SVOD_ORIGIN` (unset, empty or `0` = off).
fn default_enabled() -> bool {
    static DEFAULT: OnceLock<bool> = OnceLock::new();
    *DEFAULT.get_or_init(|| std::env::var("SVOD_ORIGIN").is_ok_and(|value| !matches!(value.as_str(), "" | "0")))
}

thread_local! {
    static STATE: Cell<State> = Cell::new(State { enabled: default_enabled(), current: None, in_call: false });
}

/// Whether scope constructors capture on this thread.
#[inline]
pub fn enabled() -> bool {
    STATE.with(|state| state.get().enabled)
}

/// The innermost scope on this thread — one TLS read, on `UOp::new`'s hot path.
#[inline]
pub fn current() -> Option<OriginId> {
    STATE.with(|state| state.get().current)
}

/// RAII guard over this thread's capture state: every constructor saves the state
/// and `Drop` restores it, including while a panic unwinds through it.
#[must_use = "an origin scope only applies while its guard is alive"]
pub struct OriginScope {
    previous: State,
}

impl OriginScope {
    fn replace(next: impl FnOnce(State) -> State) -> Self {
        let previous = STATE.with(Cell::get);
        STATE.with(|state| state.set(next(previous)));
        Self { previous }
    }

    /// Push a frame under the current scope. While capture is off the frame is
    /// never built, so a disabled scope costs one TLS read and write.
    fn push(frame: impl FnOnce() -> OriginFrame, in_call: bool) -> Self {
        Self::replace(|previous| {
            if !previous.enabled {
                return previous;
            }
            let current = Some(intern(Origin { parent: previous.current, frame: frame() }));
            State { current, in_call, ..previous }
        })
    }

    /// A named scope, one segment of a module path.
    pub fn module(name: impl Into<String>) -> Self {
        Self::push(|| OriginFrame::Module { name: name.into() }, false)
    }

    /// A free-form scope for pipeline stages and embedders.
    pub fn label(text: impl Into<String>) -> Self {
        Self::push(|| OriginFrame::Label { text: text.into() }, false)
    }

    /// An ONNX graph node.
    pub fn onnx(index: u32, name: Option<&str>, op_type: &str, domain: &str, version: i64) -> Self {
        Self::push(
            || OriginFrame::Onnx {
                index,
                name: name.map(String::from),
                op_type: op_type.to_owned(),
                domain: domain.to_owned(),
                version,
            },
            false,
        )
    }

    /// A public entry point plus the caller location the `#[track_caller]` chain resolved.
    pub fn call(op: &'static str, at: &'static Location<'static>) -> Self {
        Self::push(|| OriginFrame::Call { op, at: SourceLocation::from_caller(at) }, true)
    }

    /// A public entry point that yields to an outer one: the frame is pushed only
    /// when the current leaf is not already a call, so an op implemented on top of
    /// other public ops keeps exactly one call frame — the outermost, which is the
    /// one that names user code.
    pub fn outer_call(op: &'static str, at: &'static Location<'static>) -> Self {
        if STATE.with(|state| state.get().in_call) { Self::replace(|previous| previous) } else { Self::call(op, at) }
    }

    /// Detach from the enclosing scope: nodes built inside carry no origin.
    pub fn suspend() -> Self {
        install(None)
    }
}

impl Drop for OriginScope {
    fn drop(&mut self) {
        STATE.with(|state| state.set(self.previous));
    }
}

/// Adopt an already-interned scope, typically a captured [`current`] re-installed on
/// a worker thread. Unlike the frame constructors this always applies: the id it
/// carries was interned while capture was on.
pub fn install(id: Option<OriginId>) -> OriginScope {
    let in_call = id.and_then(get).is_some_and(|origin| matches!(origin.frame, OriginFrame::Call { .. }));
    OriginScope::replace(|previous| State { current: id, in_call, ..previous })
}

/// Force capture on or off for the current thread until the guard drops. Tests use
/// this: capture changes node identity, so a process-wide switch would reshape
/// graphs that other threads build in parallel.
pub fn capture_for_thread(enabled: bool) -> OriginScope {
    OriginScope::replace(|previous| State { enabled, ..previous })
}
