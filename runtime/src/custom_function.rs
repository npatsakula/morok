use std::collections::HashMap;
use std::sync::Arc;

use morok_device::Buffer;
use morok_ir::{CustomFunctionKind, UOp};

use crate::{Error, Result};

fn unsupported(kind: &str, attrs: &[Arc<UOp>], buffers: &[Buffer], vars: &HashMap<String, i64>) -> Error {
    Error::Unsupported {
        kind: kind.to_string(),
        reason: format!(
            "runtime is reserved but not implemented (attrs={}, buffers={}, vars={})",
            attrs.len(),
            buffers.len(),
            vars.len()
        ),
    }
}

pub fn run_custom_function(
    kind: &CustomFunctionKind,
    attrs: &[Arc<UOp>],
    buffers: &mut [Buffer],
    vars: &HashMap<String, i64>,
) -> Result<()> {
    let label = match kind {
        CustomFunctionKind::EncDec => "EncDec",
        CustomFunctionKind::Graph => "Graph",
    };
    Err(unsupported(label, attrs, buffers, vars))
}

#[cfg(test)]
#[path = "test/unit/custom_function.rs"]
mod tests;
