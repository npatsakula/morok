//! Tree visualization for UOp graphs.
//!
//! Provides pretty-printing of UOp computation graphs as ASCII trees.

use std::cell::RefCell;
use std::collections::HashSet;
use std::io;
use std::sync::Arc;
use std::{borrow::Cow, rc::Rc};

use ptree::{Style, TreeItem};

use crate::{Op, UOp};

/// Wrapper for compact tree rendering with back-references for shared nodes.
///
/// Since UOp is a DAG (hash-consed), nodes can appear multiple times in the graph.
/// This renderer shows `[id] → (see above)` for already-visited nodes.
#[derive(Clone)]
pub struct UOpTreeCompact {
    uop: Arc<UOp>,
    visited: Rc<RefCell<HashSet<u64>>>,
    /// True if this node was already visited when write_self was called
    is_backref: RefCell<bool>,
}

impl UOpTreeCompact {
    /// Create a new compact tree renderer.
    pub fn new(uop: &Arc<UOp>) -> Self {
        Self { uop: uop.clone(), visited: Rc::new(RefCell::new(HashSet::new())), is_backref: RefCell::new(false) }
    }

    fn from_child(uop: Arc<UOp>, visited: Rc<RefCell<HashSet<u64>>>) -> Self {
        Self { uop, visited, is_backref: RefCell::new(false) }
    }
}

impl TreeItem for UOpTreeCompact {
    type Child = UOpTreeCompact;

    fn write_self<W: io::Write>(&self, f: &mut W, _style: &Style) -> io::Result<()> {
        let mut visited = self.visited.borrow_mut();
        if visited.contains(&self.uop.id) {
            // Already visited - show back-reference
            *self.is_backref.borrow_mut() = true;
            write!(f, "[{}] → (see above)", self.uop.id)
        } else {
            visited.insert(self.uop.id);
            write!(f, "{}", format_node(&self.uop))
        }
    }

    fn children(&self) -> Cow<'_, [Self::Child]> {
        // Don't show children for back-references
        if *self.is_backref.borrow() {
            return Cow::Borrowed(&[]);
        }

        let sources = self.uop.op().sources();
        let children: Vec<_> =
            sources.iter().map(|src| UOpTreeCompact::from_child(src.clone(), self.visited.clone())).collect();
        Cow::Owned(children)
    }
}

/// Wrapper for full tree rendering that expands shared nodes every time.
///
/// This is more verbose but shows the complete subtree for every occurrence.
#[derive(Clone)]
pub struct UOpTreeFull {
    uop: Arc<UOp>,
}

impl UOpTreeFull {
    /// Create a new full tree renderer.
    pub fn new(uop: &Arc<UOp>) -> Self {
        Self { uop: uop.clone() }
    }
}

impl TreeItem for UOpTreeFull {
    type Child = UOpTreeFull;

    fn write_self<W: io::Write>(&self, f: &mut W, _style: &Style) -> io::Result<()> {
        write!(f, "{}", format_node(&self.uop))
    }

    fn children(&self) -> Cow<'_, [Self::Child]> {
        let sources = self.uop.op().sources();
        let children: Vec<_> = sources.iter().map(|src| UOpTreeFull { uop: src.clone() }).collect();
        Cow::Owned(children)
    }
}

/// Format a single UOp node for display.
///
/// Output format: `[id] OP_NAME : dtype shape=[...]`
fn format_node(uop: &Arc<UOp>) -> String {
    let op_str = match uop.op() {
        Op::Const(val) => format!("CONST({:?})", val.0),
        Op::DefineGlobal(id) => format!("DEFINE_GLOBAL({})", id),
        Op::DefineLocal(id) => format!("DEFINE_LOCAL({})", id),
        Op::DefineVar { name, min_val, max_val } => format!("DEFINE_VAR('{}', min={}, max={})", name, min_val, max_val),
        Op::Buffer { size, .. } => format!("BUFFER(size={})", size),
        Op::Bufferize { .. } => "BUFFERIZE".to_string(),
        Op::Load { .. } => "LOAD".to_string(),
        Op::LoadGated { .. } => "LOAD_GATED".to_string(),
        Op::Store { .. } => "STORE".to_string(),
        Op::StoreGated { .. } => "STORE_GATED".to_string(),
        Op::Index { gate: Some(_), .. } => "INDEX_GATED".to_string(),
        Op::Index { .. } => "INDEX".to_string(),
        Op::PointerIndex { .. } => "PTR_INDEX".to_string(),
        Op::Binary(bop, ..) => format!("{:?}", bop),
        Op::Unary(uop_kind, ..) => format!("{:?}", uop_kind),
        Op::Ternary(top, ..) => format!("{:?}", top),
        Op::Cast { .. } => "CAST".to_string(),
        Op::BitCast { .. } => "BITCAST".to_string(),
        Op::Reduce { reduce_op, ranges, .. } => {
            let range_ids: Vec<u64> = ranges.iter().map(|r| r.id).collect();
            format!("REDUCE({:?}, ranges={:?})", reduce_op, range_ids)
        }
        Op::ReduceAxis { reduce_op, axes, .. } => format!("REDUCE_AXIS({:?}, axes={:?})", reduce_op, axes),
        Op::AllReduce { reduce_op, .. } => format!("ALL_REDUCE({:?})", reduce_op),
        Op::Bind { .. } => "BIND".to_string(),
        Op::Range { axis_id, axis_type, .. } => format!("RANGE({}, {:?})", axis_id, axis_type),
        Op::End { .. } => "END".to_string(),
        Op::Sink { .. } => "SINK".to_string(),
        Op::Group { .. } => "GROUP".to_string(),
        Op::Kernel { .. } => "KERNEL".to_string(),
        Op::Vectorize { elements } => format!("VECTORIZE(len={})", elements.len()),
        Op::Gep { indices, .. } => format!("GEP(indices={:?})", indices),
        Op::VConst { values } => format!("VCONST(len={})", values.len()),
        Op::Cat { .. } => "CAT".to_string(),
        Op::PtrCat { .. } => "PTR_CAT".to_string(),
        Op::Reshape { .. } => "RESHAPE".to_string(),
        Op::Permute { axes, .. } => format!("PERMUTE(axes={:?})", axes),
        Op::Expand { .. } => "EXPAND".to_string(),
        Op::Pad { .. } => "PAD".to_string(),
        Op::Shrink { .. } => "SHRINK".to_string(),
        Op::Flip { axes, .. } => format!("FLIP(axes={:?})", axes),
        Op::Multi { axis, .. } => format!("MULTI(axis={})", axis),
        Op::Contiguous { .. } => "CONTIGUOUS".to_string(),
        Op::ContiguousBackward { .. } => "CONTIGUOUS_BACKWARD".to_string(),
        Op::Copy { .. } => "COPY".to_string(),
        Op::Assign { .. } => "ASSIGN".to_string(),
        Op::Custom { code, .. } => format!("CUSTOM('{}')", code.chars().take(20).collect::<String>()),
        Op::CustomI { code, .. } => format!("CUSTOM_I('{}')", code.chars().take(20).collect::<String>()),
        Op::Unique(id) => format!("UNIQUE({})", id),
        Op::Device(spec) => format!("DEVICE({:?})", spec),
        Op::Noop => "NOOP".to_string(),
        Op::Invalid => "INVALID".to_string(),
        Op::BufferView { size, offset, .. } => format!("BUFFER_VIEW(size={}, offset={})", size, offset),
        Op::MStack { .. } => "MSTACK".to_string(),
        Op::MSelect { device_index, .. } => format!("MSELECT(idx={})", device_index),
        Op::Special { name, .. } => format!("SPECIAL('{}')", name),
        Op::If { .. } => "IF".to_string(),
        Op::EndIf { .. } => "END_IF".to_string(),
        Op::Barrier { .. } => "BARRIER".to_string(),
        Op::DefineReg { size } => format!("DEFINE_REG(size={})", size),
        Op::Wmma { .. } => "WMMA".to_string(),
        Op::Contract { .. } => "CONTRACT".to_string(),
        Op::Unroll { .. } => "UNROLL".to_string(),
        Op::Detach { .. } => "DETACH".to_string(),
        Op::After { .. } => "AFTER".to_string(),
        Op::Precast { .. } => "PRECAST".to_string(),
        #[allow(unreachable_patterns)]
        _ => format!("{:?}", std::mem::discriminant(uop.op())),
    };

    // Get shape if available
    let shape_str = match uop.shape() {
        Ok(Some(shape)) => format!(" shape={:?}", shape.as_slice()),
        Ok(None) => String::new(),
        Err(_) => " shape=?".to_string(),
    };

    format!("[{}] {} : {:?}{}", uop.id, op_str, uop.dtype(), shape_str)
}

/// Render a UOp graph as a compact ASCII tree string.
///
/// Shared nodes (appearing multiple times due to hash-consing) are shown
/// as back-references: `[id] → (see above)`
pub fn render_tree_compact(uop: &Arc<UOp>) -> String {
    let tree = UOpTreeCompact::new(uop);
    let mut buf = Vec::new();
    ptree::write_tree(&tree, &mut buf).expect("tree rendering failed");
    String::from_utf8(buf).expect("invalid utf8 in tree")
}

/// Render a UOp graph as a full ASCII tree string.
///
/// Shared nodes are expanded every time they appear (verbose but complete).
pub fn render_tree_full(uop: &Arc<UOp>) -> String {
    let tree = UOpTreeFull::new(uop);
    let mut buf = Vec::new();
    ptree::write_tree(&tree, &mut buf).expect("tree rendering failed");
    String::from_utf8(buf).expect("invalid utf8 in tree")
}
