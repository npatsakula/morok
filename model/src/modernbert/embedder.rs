//! [`ModernBertEmbedder`] — finished embeddings from the ModernBERT backbone:
//! `input_ids` + `attention_mask` → mean-pooled, L2-normalized `(B, D)` vectors.
//!
//! Implements [`EncoderHead`] (with [`Embed`] fixing the output kinds) so it drops
//! straight into an
//! [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The model
//! owns the forward + fused pooling/normalization (via
//! [`ModernBertEmbedderJit`]); the pipeline owns chunking and profile assembly.
//!
//! For the one-call hub loader that builds this embedder alongside a matching
//! [`HfTokenizer`](svod_arch::pipelines::text::HfTokenizer), see
//! [`from_hub`](crate::modernbert::from_hub).
//!
//! The JIT plan is sized once at construction (from `max_batch` + `max_seq`) and
//! runs at that size every call: inputs are padded to `[max_batch, max_seq]`,
//! `execute()` runs the full batch, and the live rows are sliced out of the
//! `[max_batch, D]` output — the same pad-and-slice shape gigaam's transcriber
//! uses.

use snafu::ResultExt;
use svod_arch::pipelines::text::{Embed, Embedding, EncoderHead, Encoding, RunProfile};
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;
use crate::modernbert::ModernBert;
use crate::modernbert::embedder_jit::ModernBertEmbedderJit;
use crate::modernbert::head_jit::{HeadError, JitSnafu, execute_head};

/// Finished-embeddings model over a [`ModernBert`] backbone. Build once (eager
/// JIT prepare) and reuse across calls.
pub struct ModernBertEmbedder {
    jit: ModernBertEmbedderJit,
    max_batch: usize,
    max_seq: usize,
    hidden_size: usize,
}

impl ModernBertEmbedder {
    /// Prepare the embedder JIT at `[max_batch, max_seq]`. `max_batch`/`max_seq`
    /// are caller-chosen and typically flow in from the pipeline (the chunker's
    /// `max_seq` at assembly). The model's config must already reflect the
    /// checkpoint (e.g. via [`ModernBert::from_hub`]).
    pub fn new(model: ModernBert, max_batch: usize, max_seq: usize) -> Result<Self, HeadError> {
        let hidden_size = model.config.hidden_size;
        // `b` is declared `vars { b: (1, model.config.max_batch_size) }` in the JIT
        // wrapper, but the caller-chosen `max_batch` (which sizes the input buffers
        // below) is what the plan must bake in. Override the upper bound so prepare
        // binds `b` to `max_batch` and the output buffer is sized `max_batch × D` —
        // rebinding `b` to a smaller live batch at execute only shrinks the live
        // region, never the allocation (the JIT batch-rebind contract).
        let mut jit = ModernBertEmbedderJit::new(model).with_b_bound(max_batch);
        // `b` binds to max_batch at prepare (see the jit_wrapper codegen), so the
        // plan runs at max_batch every execute. ids are i64 (the embedding-gather
        // convention); the mask is i64 0/1 here and cast to bool inside the build
        // closure (InputSpec has no bool constructor).
        let ids_spec = InputSpec::i64(&[max_batch, max_seq]);
        let mask_spec = InputSpec::i64(&[max_batch, max_seq]);
        jit.prepare_with_config(ids_spec, mask_spec, &PrepareConfig::from_env()).context(JitSnafu)?;
        Ok(Self { jit, max_batch, max_seq, hidden_size })
    }
}

impl EncoderHead for ModernBertEmbedder {
    type Output = Embedding;
    type Error = HeadError;

    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.max_seq)
    }

    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Embedding>, Option<RunProfile>), HeadError> {
        let (b, flat, prof) = execute_head(&mut self.jit, batch, self.max_batch, self.max_seq, profile, "embed")?;
        let d = self.hidden_size;
        // The output buffer is always max_batch-sized (the plan bakes the upper
        // bound); only the first `b` rows are live.
        let embeddings: Vec<Embedding> =
            (0..b).map(|i| Embedding { values: flat[i * d..i * d + d].to_vec() }).collect();
        Ok((embeddings, prof))
    }
}

impl Embed for ModernBertEmbedder {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
