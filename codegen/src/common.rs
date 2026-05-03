//! Common utilities shared between codegen backends.

use std::sync::Arc;

use morok_ir::{Op, UOp};

use crate::{Error, Result};

/// Check whether a buffer (PARAM/DefineGlobal) is used as a STORE target in the graph.
pub fn is_output_buffer(def_global: &Arc<UOp>, nodes: &[Arc<UOp>]) -> bool {
    let buffer_id = def_global.id;

    for node in nodes {
        if let Some(buffer) = node.store_buffer() {
            if buffer.id == buffer_id {
                return true;
            }
            if let Op::Index { buffer: idx_buf, .. } = buffer.op()
                && idx_buf.id == buffer_id
            {
                return true;
            }
        }
    }
    false
}

/// Collect buffer and variable parameters from a UOp graph.
///
/// Collects:
/// - Buffers: PARAM, DEFINE_LOCAL operations
/// - Variables: DEFINE_VAR operations (passed as i64 kernel params)
///
/// Returns (buffers, variables) sorted for deterministic function signatures.
pub fn collect_buffers_and_vars(root: &Arc<UOp>) -> (Vec<Arc<UOp>>, Vec<Arc<UOp>>) {
    let nodes = root.toposort();

    // Collect buffers
    let mut buffers = Vec::new();
    for node in &nodes {
        match node.op() {
            Op::Buffer { .. } | Op::Param { device: None, .. } | Op::DefineLocal(_) => {
                buffers.push(node.clone());
            }
            _ => {}
        }
    }

    // Sort buffers by internal ID (matches split_kernel.rs ordering)
    buffers.sort_by_key(|b| match b.op() {
        Op::Param { slot, device: None, .. } => *slot as u64,
        Op::DefineLocal(id) => (*id as u64) + (1u64 << 32),
        Op::Buffer { .. } => b.id + (1u64 << 48),
        _ => b.id,
    });

    // Collect DefineVar nodes - these become i64 kernel parameters
    let mut variables = Vec::new();
    for node in &nodes {
        if matches!(node.op(), Op::DefineVar { .. }) {
            variables.push(node.clone());
        }
    }

    // Sort variables by name for deterministic function signatures
    variables.sort_by_key(|v| if let Op::DefineVar { name, .. } = v.op() { name.clone() } else { String::new() });

    (buffers, variables)
}

pub fn validate_custom_template_strict(template: &str, arg_count: usize) -> Result<()> {
    let mut chars = template.chars().peekable();
    let mut auto_idx = 0usize;
    let mut saw_auto = false;
    let mut saw_manual = false;

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if matches!(chars.peek(), Some('{')) {
                chars.next();
                continue;
            }

            let mut token = String::new();
            let mut found_close = false;
            for next in chars.by_ref() {
                if next == '}' {
                    found_close = true;
                    break;
                }
                token.push(next);
            }

            if !found_close {
                return Err(Error::InvalidGraph {
                    reason: format!("custom template has unmatched '{{': {template:?}"),
                });
            }

            let idx = if token.is_empty() {
                saw_auto = true;
                let i = auto_idx;
                auto_idx += 1;
                i
            } else {
                saw_manual = true;
                token.parse::<usize>().map_err(|_| Error::InvalidGraph {
                    reason: format!(
                        "custom template placeholder must be empty or numeric, got {{{token}}} in {template:?}"
                    ),
                })?
            };

            if saw_auto && saw_manual {
                return Err(Error::InvalidGraph {
                    reason: format!("custom template mixes automatic {{}} and manual {{N}} placeholders: {template:?}"),
                });
            }

            if idx >= arg_count {
                return Err(Error::InvalidGraph {
                    reason: format!(
                        "custom template placeholder index {idx} out of bounds (args={arg_count}) in {template:?}"
                    ),
                });
            }
        } else if ch == '}' {
            if matches!(chars.peek(), Some('}')) {
                chars.next();
            } else {
                return Err(Error::InvalidGraph {
                    reason: format!("custom template has unmatched '}}': {template:?}"),
                });
            }
        }
    }

    Ok(())
}

pub fn format_custom_template_strict(template: &str, args: &[String]) -> Result<String> {
    validate_custom_template_strict(template, args.len())?;

    let mut out = String::new();
    let mut chars = template.chars().peekable();
    let mut auto_idx = 0usize;

    while let Some(ch) = chars.next() {
        if ch == '{' {
            if matches!(chars.peek(), Some('{')) {
                chars.next();
                out.push('{');
                continue;
            }

            let mut token = String::new();
            for next in chars.by_ref() {
                if next == '}' {
                    break;
                }
                token.push(next);
            }

            let idx = if token.is_empty() {
                let i = auto_idx;
                auto_idx += 1;
                i
            } else {
                token.parse::<usize>().expect("placeholder token validated")
            };

            out.push_str(&args[idx]);
        } else if ch == '}' {
            chars.next();
            out.push('}');
        } else {
            out.push(ch);
        }
    }

    Ok(out)
}
