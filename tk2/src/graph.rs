//! Wrap a tk2 [`Program`] as an opaque graph-node [`Tensor`] (`custom_kernel` /
//! `Op::Call`) so it composes into the tensor graph and is measured through the
//! shipped `Tensor::prepare_with → ExecutionPlan::profile` path — real on-device
//! time plus the in-process gfx942 PMC table — instead of the single-kernel
//! direct-dispatch [`crate::launch`].
//!
//! The scheduler treats the hand-lowered body as **opaque**: [`crate::lower`] stamps
//! `KernelInfo.opts_to_apply = Some(vec![])`, so `prepare()`'s optimizer never
//! rewrites tk2's own schedule — it only places, orders (against the rest of the
//! graph), and profiles the kernel. This is the layer the perf saga measures at,
//! because it (a) reports true device time, (b) arms the PMC counters every lever
//! gates on, and (c) composes multiple kernels into one plan (a single tk2 kernel is
//! just the one-node case).

use snafu::ResultExt;
use svod_tensor::Tensor;

use crate::error::{self, Result};
use crate::kernels::Program;
use crate::lower;

/// Wrap `program` as a lazy output [`Tensor`]. The kernel body binds the
/// `[out, ins...]` PARAM placeholders (outputs-first — the builder's `global()`
/// declaration order) as its ABI globals, and the scheduler realizes/profiles it
/// like any tensor op. `out` is the output template ([`Tensor::empty`] of the result
/// shape+dtype); `ins` are the (realized) inputs. Input/output *shapes* only size the
/// flat buffers — tk2 addresses them flat, so passing shaped tensors is fine (the
/// placeholder's `RESHAPE(PARAM)` view is unwrapped to its flat PARAM in lowering).
pub fn graph_kernel(program: Program, out: Tensor, ins: &[&Tensor]) -> Result<Tensor> {
    let name = program.name.clone();
    Tensor::graph_kernel(&name, out, ins, move |ph| lower::lower_as_graph_node(&program, &ph))
        .context(error::GraphKernelSnafu { name })
}
