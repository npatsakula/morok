use svod_tensor::Tensor;
use svod_tensor::nn::Module;

use super::basic_block::{BasicBlock, BlockKind};
use super::bottleneck::Bottleneck;
use super::error::Result;

#[derive(Clone, Module)]
pub enum Block {
    Basic(BasicBlock),
    Bottleneck(Bottleneck),
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Block::Basic(b) => b.forward(x),
            Block::Bottleneck(b) => b.forward(x),
        }
    }
}

#[derive(Clone, Module)]
pub struct ResidualStage {
    /// Flattened: the blocks are keyed `{i}` directly under the stage prefix.
    #[module(key = "")]
    pub blocks: Vec<Block>,
}

impl ResidualStage {
    /// Construct a fresh stage. The first block may downsample (`stride`);
    /// remaining blocks always have stride 1. Channel width follows the
    /// canonical schedule: every block in the stage emits `planes * expansion`
    /// channels, and the next block sees that as its `in_planes`.
    pub fn empty(kind: BlockKind, in_planes: usize, planes: usize, num_blocks: usize, stride: usize) -> Self {
        let mut current_in = in_planes;
        let blocks = (0..num_blocks)
            .map(|i| {
                let s = if i == 0 { stride } else { 1 };
                let block = match kind {
                    BlockKind::Basic => Block::Basic(BasicBlock::empty(current_in, planes, s)),
                    BlockKind::Bottleneck => Block::Bottleneck(Bottleneck::empty(current_in, planes, s)),
                };
                current_in = planes * kind.expansion();
                block
            })
            .collect();
        Self { blocks }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.blocks.iter().try_fold(x.clone(), |x, b| b.forward(&x))
    }
}
