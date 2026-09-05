//! Compute-only float demotion for renderers without a wide float type.
//!
//! Metal (and WebGPU) have no `double`, yet the frontend uses `Float64` as
//! precision scaffolding inside kernels (`linspace`, requantization, resize
//! coordinate transforms). When the renderer profile does not support
//! `Float64`, every *internal* f64 value — constants, casts, ALU, reductions,
//! register/threadgroup scratch — is computed in `Float32` instead. External
//! f64 storage (PARAMs and their loads/stores) is left untouched: its layout
//! belongs to the host, and the renderer reports it as unsupported, exactly as
//! tinygrad's Metal renderer does.

use std::sync::Arc;

use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::pattern::{Matcher, RewriteResult};
use svod_ir::{Op, UOp, ops};

use crate::graph_rewrite;
use crate::optimizer::Renderer;

/// Demote internal `Float64` to `Float32` when `renderer` cannot compute in f64.
pub fn demote_unsupported_floats(sink: Arc<UOp>, renderer: &Renderer) -> Arc<UOp> {
    if renderer.supports_alu_dtype(ScalarDType::Float64) {
        return sink;
    }
    graph_rewrite(&DemoteFloat { from: ScalarDType::Float64, to: ScalarDType::Float32 }, sink, &mut ())
}

/// Rewrites values of scalar dtype `from` to compute in `to`. Applied with
/// [`graph_rewrite`], so every node sees already-demoted children.
pub struct DemoteFloat {
    pub from: ScalarDType,
    pub to: ScalarDType,
}

impl DemoteFloat {
    fn target(&self, dtype: &DType) -> Option<DType> {
        match dtype {
            DType::Scalar(scalar) if *scalar == self.from => Some(DType::Scalar(self.to)),
            DType::Vector { scalar, count } if *scalar == self.from => DType::Scalar(self.to).vec(*count),
            _ => None,
        }
    }

    fn demote(&self, node: &Arc<UOp>) -> Option<Arc<UOp>> {
        let target = self.target(&node.dtype())?;
        // External storage keeps its layout: PARAMs, their addresses, and
        // loads through them stay in the wide dtype.
        if node.addrspace() == Some(AddrSpace::Global) {
            return None;
        }
        match node.op() {
            Op::Param(..) | Op::BitCast(..) => None,
            Op::Load(ops::Load { index, .. }) if index.addrspace() == Some(AddrSpace::Global) => None,
            Op::Buffer(ops::Buffer { shape, arg }) => {
                let mut arg = (**arg).clone();
                arg.dtype = target.clone();
                Some(UOp::new(Op::Buffer(ops::Buffer { shape: shape.clone(), arg: arg.into() }), target))
            }
            Op::Cast(ops::Cast { src, .. }) => Some(src.cast(target)),
            Op::Const(value) => UOp::try_const_(target, value.0).ok(),
            Op::VConst(ops::VConst { values }) => Some(UOp::vconst(values.clone(), DType::Scalar(self.to))),
            _ => {
                // Children were demoted first; the only wide operands left are
                // loads from external storage, which convert at the use site.
                let sources = node
                    .op()
                    .sources()
                    .iter()
                    .map(|source| match self.target(&source.dtype()) {
                        Some(narrow) => source.cast(narrow),
                        None => source.clone(),
                    })
                    .collect::<Vec<_>>();
                Some(node.replace().dtype(target).src(sources).call())
            }
        }
    }
}

impl Matcher<()> for DemoteFloat {
    fn rewrite(&self, node: &Arc<UOp>, _ctx: &mut ()) -> RewriteResult {
        match self.demote(node) {
            Some(rewritten) if !Arc::ptr_eq(&rewritten, node) => RewriteResult::Rewritten(rewritten),
            _ => RewriteResult::NoMatch,
        }
    }
}
