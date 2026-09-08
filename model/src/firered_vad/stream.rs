//! Streaming FireRedVAD: the causal `Stream-VAD` checkpoint (`N2 = 0`, no
//! lookahead) with per-layer conv caches carried **on-device** between chunks.
//!
//! Each FSMN layer's state is the last `ORDER - 1` columns of its pre-conv
//! projected features. A chunk forward concatenates the cache in front of the
//! chunk on the time axis and runs the lookback conv unpadded — exactly N
//! outputs, aligned to the chunk, mathematically identical to a forward that
//! saw the real left context (zero-init cache ≡ conv zero-padding at a cold
//! start). The new cache is the tail of that concatenation; the JIT declares
//! the caches as one `state { .. }` array slot, so `execute()` recycles all
//! [`LAYERS`] of them in the JIT's own input buffers with no host round-trip.
//!
//! [`FireRedVadStreamer`] is the host driver: feed arbitrary sample slices
//! with [`push`](FireRedVadStreamer::push), receive speech start/end
//! [`VadEvent`]s as they fire, and [`flush`](FireRedVadStreamer::flush) at end
//! of stream for the segment timestamps. The model is causal, so the
//! zero-filled tail of the final partial chunk cannot leak backward into real
//! frames — flush is exact, but the cache it leaves behind is poisoned by the
//! padding, so the streamer is terminal after flush until
//! [`reset`](FireRedVadStreamer::reset).

use std::collections::VecDeque;

use bon::bon;
use snafu::{ResultExt, Snafu};
use svod_dtype::DType;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;

use super::{FRAME_SHIFT, FireRedFbank, LAYERS, N_MELS, ORDER, PROJ, Result, hub_file, time_filter};

use crate::init::fan_in_uniform;
use crate::jit::InputSpec;
use crate::state;

/// Per-layer cache length: the lookback conv consumes `ORDER - 1` frames of
/// left context (`(N1 - 1) * S1` in the reference).
pub(crate) const STREAM_CACHE: usize = ORDER - 1;
/// 10 ms fbank hop at 16 kHz.
const FRAMES_PER_SEC: f32 = 16_000.0 / FRAME_SHIFT as f32;

/// Causal FSMN memory layer: depthwise lookback conv only (`N2 = 0`), state
/// threaded explicitly as `[1, P, STREAM_CACHE]`.
#[derive(Clone)]
struct FsmnStream {
    lookback: Tensor,
}

impl FsmnStream {
    /// `x [1, T, P]`, `cache [1, P, STREAM_CACHE]` ->
    /// `(memory [1, T, P], new_cache)`. The cache is prepended on the time
    /// axis and the conv runs unpadded: kernel `ORDER` over
    /// `STREAM_CACHE + T` columns yields exactly `T` outputs, each summing
    /// frames `[t - ORDER + 1, t]` of the extended sequence — the reference's
    /// padded conv with the cache standing in for the left zero-pad.
    fn forward(&self, x: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let len = x.dim_const(1)?;
        // [1,T,P] -> [1,P,T]: conv1d wants (N, C, L).
        let xt = x.try_transpose(-1, -2)?;
        let ext = Tensor::cat(&[cache, &xt], -1)?;
        let lookback = ext.conv1d().weight(&self.lookback).groups(PROJ).call()?;
        let memory = xt.try_add(&lookback)?;
        let new_cache = ext.narrow(-1, len, STREAM_CACHE)?;
        Ok((memory.try_transpose(-1, -2)?, new_cache))
    }
}

/// `fc1 (P->H) + ReLU -> fc2 (H->P, bias-free) -> FSMN -> + skip`.
#[derive(Clone)]
struct DfsmnBlockStream {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fsmn: FsmnStream,
}

impl DfsmnBlockStream {
    fn forward(&self, x: &Tensor, cache: &Tensor) -> Result<(Tensor, Tensor)> {
        let h = x.linear().weight(&self.fc1_weight).bias(&self.fc1_bias).call()?.relu()?;
        let p = h.linear().weight(&self.fc2_weight).call()?;
        let (memory, new_cache) = self.fsmn.forward(&p, cache)?;
        Ok((memory.try_add(x)?, new_cache))
    }
}

/// Streaming DFSMN weights (the `Stream-VAD` checkpoint — a different model
/// from [`super::FireRedVad`], not a reconfiguration: causal-only, no
/// lookahead filters).
#[derive(Clone)]
pub struct FireRedVadStream {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
    fsmn1: FsmnStream,
    blocks: Vec<DfsmnBlockStream>,
    dnn_weight: Tensor,
    dnn_bias: Tensor,
    out_weight: Tensor,
    out_bias: Tensor,
    cmvn_means: Tensor,
    cmvn_istd: Tensor,
}

impl FireRedVadStream {
    pub fn from_hub() -> Result<Self> {
        Self::from_safetensors(&hub_file("firered_vad_stream.safetensors")?)
    }

    /// Load from a safetensors produced by
    /// `scripts/convert_firered_vad.py --stream` (same layout as the
    /// non-streaming file minus the lookahead filters).
    pub fn from_safetensors(path: &std::path::Path) -> Result<Self> {
        let sd = state::load_safetensors(path)?;
        let get = |key: &str| -> Result<Tensor> { Ok(state::get_tensor(&sd, key)?) };
        let blocks = (0..super::BLOCKS)
            .map(|i| {
                Ok(DfsmnBlockStream {
                    fc1_weight: get(&format!("blocks.{i}.fc1.weight"))?,
                    fc1_bias: get(&format!("blocks.{i}.fc1.bias"))?,
                    fc2_weight: get(&format!("blocks.{i}.fc2.weight"))?,
                    fsmn: FsmnStream { lookback: time_filter(&get(&format!("blocks.{i}.lookback.weight"))?)? },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            fc1_weight: get("fc1.weight")?,
            fc1_bias: get("fc1.bias")?,
            fc2_weight: get("fc2.weight")?,
            fc2_bias: get("fc2.bias")?,
            fsmn1: FsmnStream { lookback: time_filter(&get("fsmn1.lookback.weight")?)? },
            blocks,
            dnn_weight: get("dnn.weight")?,
            dnn_bias: get("dnn.bias")?,
            out_weight: get("out.weight")?,
            out_bias: get("out.bias")?,
            cmvn_means: get("cmvn_means")?,
            cmvn_istd: get("cmvn_istd")?,
        })
    }

    /// Random weights with the streaming checkpoint's layout (see
    /// [`super::FireRedVad::with_random_weights`]).
    pub fn with_random_weights() -> Self {
        let dt = DType::Float32;
        let lin = |out: usize, inp: usize| fan_in_uniform(&[out, inp], inp, dt.clone());
        let bias = |out: usize, fan_in: usize| fan_in_uniform(&[out], fan_in, dt.clone());
        let fsmn = || FsmnStream { lookback: fan_in_uniform(&[PROJ, 1, ORDER], ORDER, DType::Float32) };
        Self {
            fc1_weight: lin(super::HIDDEN, N_MELS),
            fc1_bias: bias(super::HIDDEN, N_MELS),
            fc2_weight: lin(PROJ, super::HIDDEN),
            fc2_bias: bias(PROJ, super::HIDDEN),
            fsmn1: fsmn(),
            blocks: (0..super::BLOCKS)
                .map(|_| DfsmnBlockStream {
                    fc1_weight: lin(super::HIDDEN, PROJ),
                    fc1_bias: bias(super::HIDDEN, PROJ),
                    fc2_weight: lin(PROJ, super::HIDDEN),
                    fsmn: fsmn(),
                })
                .collect(),
            dnn_weight: lin(super::HIDDEN, PROJ),
            dnn_bias: bias(super::HIDDEN, PROJ),
            out_weight: lin(1, super::HIDDEN),
            out_bias: bias(1, super::HIDDEN),
            cmvn_means: fan_in_uniform(&[N_MELS], N_MELS, dt.clone()),
            cmvn_istd: fan_in_uniform(&[N_MELS], N_MELS, dt),
        }
    }

    /// Zero caches (`LAYERS x [1, P, STREAM_CACHE]`) — the cold-start
    /// state, equivalent to conv zero-padding at a sequence start. With these,
    /// `forward_stream` over a whole sequence IS the full causal forward.
    pub fn zero_caches() -> Result<Vec<Tensor>> {
        (0..LAYERS).map(|_| Ok(Tensor::zeros(&[1, PROJ, STREAM_CACHE], DType::Float32))).collect()
    }

    /// Causal DFSMN forward over one chunk: pre-CMVN fbank `[1, T, N_MELS]` +
    /// per-layer caches -> (`[1, T]` speech probs, updated caches). CMVN runs
    /// in-graph as the first op.
    pub fn forward_stream(&self, feat: &Tensor, caches: &[Tensor]) -> Result<(Tensor, Vec<Tensor>)> {
        assert_eq!(caches.len(), LAYERS, "one cache per FSMN layer");
        let x = feat.try_sub(&self.cmvn_means)?.try_mul(&self.cmvn_istd)?;
        let h = x.linear().weight(&self.fc1_weight).bias(&self.fc1_bias).call()?.relu()?;
        let mut m = h.linear().weight(&self.fc2_weight).bias(&self.fc2_bias).call()?.relu()?;
        let mut new_caches = Vec::with_capacity(LAYERS);
        let (m1, nc) = self.fsmn1.forward(&m, &caches[0])?;
        m = m1;
        new_caches.push(nc);
        for (block, cache) in self.blocks.iter().zip(&caches[1..]) {
            let (mb, nc) = block.forward(&m, cache)?;
            m = mb;
            new_caches.push(nc);
        }
        let d = m.linear().weight(&self.dnn_weight).bias(&self.dnn_bias).call()?.relu()?;
        let logits = d.linear().weight(&self.out_weight).bias(&self.out_bias).call()?;
        Ok((logits.sigmoid()?.try_squeeze(Some(-1))?, new_caches))
    }
}

/// `state { caches: [Tensor; LAYERS] }` needs a literal length.
const _: () = assert!(LAYERS == 8, "the stream JIT's cache slot length must track LAYERS");

/// One chunk of [`FireRedVadStream::forward_stream`] shaped for the JIT's
/// build tuple: probs plus the new caches as the `state` slot's array.
fn stream_chunk(
    model: &FireRedVadStream,
    feat: &Tensor,
    caches: [&Tensor; LAYERS],
) -> Result<(Tensor, [Tensor; LAYERS])> {
    let owned: Vec<Tensor> = caches.iter().map(|&c| c.clone()).collect();
    let (probs, new) = model.forward_stream(feat, &owned)?;
    Ok((probs, std::array::from_fn(|i| new[i].clone())))
}

jit_wrapper! {
    FireRedVadStreamJit(FireRedVadStream) {
        inputs { feat: Tensor }
        // The caches recycle in place: each new cache is stored into its own
        // input buffer, so `execute()` leaves the state where the next chunk
        // reads it. Read-before-write is safe — every cache is read exactly
        // once (the cat) before its store.
        state { caches: [Tensor; 8] }
        outputs { probs }

        build(feat, caches) {
            stream_chunk(model, feat, caches)
        }
    }
}

/// Speech boundary events from the [`StreamVadPostprocessor`]. Frame indices
/// are 1-based fbank frames (10 ms each), the reference's convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStart { frame: usize },
    SpeechEnd { start_frame: usize, end_frame: usize },
}

/// Knobs for [`StreamVadPostprocessor`], defaulting to the reference's
/// `FireRedStreamVadConfig` (frame counts at 10 ms per frame). `threshold`
/// consults `SVOD_VAD_THRESHOLD`.
#[derive(Debug, Clone)]
pub struct StreamVadConfig {
    pub smooth_window: usize,
    pub threshold: f32,
    /// Frames of leading context folded into an emitted segment start
    /// (clamped to at least `smooth_window` by the postprocessor).
    pub pad_start_frames: usize,
    pub min_speech_frames: usize,
    /// Force-split segments longer than this (the reference's 20 s cap).
    pub max_speech_frames: usize,
    pub min_silence_frames: usize,
}

impl Default for StreamVadConfig {
    fn default() -> Self {
        Self {
            smooth_window: 5,
            threshold: std::env::var("SVOD_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5),
            pad_start_frames: 5,
            min_speech_frames: 8,
            max_speech_frames: 2000,
            min_silence_frames: 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VadState {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

/// Online 4-state speech FSM — a verbatim port of the reference
/// `StreamVadPostprocessor` (smoothed-prob thresholding, min-speech onset
/// with back-padded start, min-silence close, max-speech force-split that
/// re-opens on the next frame). Pure host state, one prob per 10 ms frame.
pub struct StreamVadPostprocessor {
    cfg: StreamVadConfig,
    state: VadState,
    frame_cnt: usize,
    speech_cnt: usize,
    silence_cnt: usize,
    /// f64 running sum: the incremental add/sub never resets, so f32 would
    /// accumulate drift over long streams.
    smooth_sum: f64,
    smooth_window: VecDeque<f32>,
    hit_max_speech: bool,
    segment_open: bool,
    last_speech_start: usize,
    last_speech_end: usize,
}

impl StreamVadPostprocessor {
    pub fn new(mut cfg: StreamVadConfig) -> Self {
        cfg.smooth_window = cfg.smooth_window.max(1);
        cfg.pad_start_frames = cfg.pad_start_frames.max(cfg.smooth_window);
        Self {
            cfg,
            state: VadState::Silence,
            frame_cnt: 0,
            speech_cnt: 0,
            silence_cnt: 0,
            smooth_sum: 0.0,
            smooth_window: VecDeque::new(),
            hit_max_speech: false,
            segment_open: false,
            last_speech_start: 0,
            last_speech_end: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.cfg.clone());
    }

    /// Running mean of the last `smooth_window` raw probs (numerically equal
    /// to the batch `smooth_trailing`, including its cumulative-mean ramp).
    pub(crate) fn smooth(&mut self, prob: f32) -> f32 {
        if self.cfg.smooth_window <= 1 {
            return prob;
        }
        self.smooth_window.push_back(prob);
        self.smooth_sum += prob as f64;
        if self.smooth_window.len() > self.cfg.smooth_window {
            self.smooth_sum -= self.smooth_window.pop_front().expect("non-empty window") as f64;
        }
        (self.smooth_sum / self.smooth_window.len() as f64) as f32
    }

    /// Advance one frame; speech boundary events (0..=2 per frame) are
    /// appended to `events`.
    pub fn process_one_frame(&mut self, raw_prob: f32, events: &mut Vec<VadEvent>) {
        self.frame_cnt += 1;
        let is_speech = self.smooth(raw_prob) >= self.cfg.threshold;

        // A max-speech split closed the segment on the previous frame; the
        // follow-up segment opens here, regardless of this frame's decision.
        if self.hit_max_speech {
            self.open(self.frame_cnt, events);
            self.hit_max_speech = false;
        }

        match self.state {
            VadState::Silence => {
                if is_speech {
                    self.state = VadState::PossibleSpeech;
                    self.speech_cnt += 1;
                } else {
                    self.silence_cnt += 1;
                    self.speech_cnt = 0;
                }
            }
            VadState::PossibleSpeech => {
                if is_speech {
                    self.speech_cnt += 1;
                    if self.speech_cnt >= self.cfg.min_speech_frames {
                        self.state = VadState::Speech;
                        // Back-pad the start, but never before frame 1 or
                        // into the previous segment.
                        let padded =
                            self.frame_cnt as i64 - self.speech_cnt as i64 + 1 - self.cfg.pad_start_frames as i64;
                        let start = padded.max(1).max(self.last_speech_end as i64 + 1) as usize;
                        self.open(start, events);
                        self.silence_cnt = 0;
                    }
                } else {
                    self.state = VadState::Silence;
                    self.silence_cnt = 1;
                    self.speech_cnt = 0;
                }
            }
            VadState::Speech => {
                self.speech_cnt += 1;
                if is_speech {
                    self.silence_cnt = 0;
                    self.split_if_max(events);
                } else {
                    self.state = VadState::PossibleSilence;
                    self.silence_cnt += 1;
                }
            }
            VadState::PossibleSilence => {
                self.speech_cnt += 1;
                if is_speech {
                    self.state = VadState::Speech;
                    self.silence_cnt = 0;
                    self.split_if_max(events);
                } else {
                    self.silence_cnt += 1;
                    if self.silence_cnt >= self.cfg.min_silence_frames {
                        self.state = VadState::Silence;
                        self.close(events);
                        self.speech_cnt = 0;
                    }
                }
            }
        }
    }

    /// End of stream: close a still-open segment at the last frame.
    pub fn finalize(&mut self, events: &mut Vec<VadEvent>) {
        if self.segment_open {
            self.state = VadState::Silence;
            self.close(events);
            self.speech_cnt = 0;
        }
    }

    fn open(&mut self, frame: usize, events: &mut Vec<VadEvent>) {
        events.push(VadEvent::SpeechStart { frame });
        self.last_speech_start = frame;
        self.segment_open = true;
    }

    fn close(&mut self, events: &mut Vec<VadEvent>) {
        events.push(VadEvent::SpeechEnd { start_frame: self.last_speech_start, end_frame: self.frame_cnt });
        self.last_speech_end = self.frame_cnt;
        self.segment_open = false;
    }

    /// The max-speech cap: emit the end now, schedule the follow-up start for
    /// the next frame.
    fn split_if_max(&mut self, events: &mut Vec<VadEvent>) {
        if self.speech_cnt >= self.cfg.max_speech_frames {
            self.hit_max_speech = true;
            self.speech_cnt = 0;
            self.close(events);
        }
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum FireRedVadStreamError {
    #[snafu(display("loading FireRedVAD stream model: {source}"))]
    StreamLoad {
        #[snafu(source(from(super::Error, Box::new)))]
        source: Box<super::Error>,
    },
    #[snafu(display("building FireRedVAD stream JIT: {source}"))]
    StreamPrepare {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("running FireRedVAD stream JIT: {source}"))]
    StreamStep {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("streamer flushed; call reset() before feeding more audio"))]
    StreamFlushed,
}

/// [`FireRedVadStreamer::flush`] result: the events that fired during the
/// final frames plus every closed segment of the stream as
/// `(start_sec, end_sec)`.
#[derive(Debug, Clone)]
pub struct StreamFlush {
    pub events: Vec<VadEvent>,
    pub timestamps: Vec<(f32, f32)>,
}

/// Stateful streaming VAD driver: 16 kHz samples in, [`VadEvent`]s out.
///
/// Samples buffer host-side until a full fbank frame (400 samples, 160 hop)
/// and then a full `chunk_frames` feature chunk is available; each chunk is
/// one JIT dispatch with the conv caches recycling on-device. Latency is
/// bounded by `chunk_frames` (default 16 -> 160 ms) plus the 240-sample
/// fbank window overlap.
pub struct FireRedVadStreamer {
    fbank: FireRedFbank,
    jit: FireRedVadStreamJit,
    chunk_frames: usize,
    /// Samples not yet covered by a complete fbank frame (< 400).
    remainder: Vec<f32>,
    /// Feature rows awaiting a full chunk (< `chunk_frames * N_MELS`).
    pending: Vec<f32>,
    /// Per-dispatch prob read-back scratch.
    scratch: Vec<f32>,
    /// All raw probs since the last reset (10 ms per entry; ~400 B/s — kept
    /// for parity tests and debugging).
    probs: Vec<f32>,
    post: StreamVadPostprocessor,
    /// Closed segments `(start_frame, end_frame)`, 1-based.
    segments: Vec<(usize, usize)>,
    flushed: bool,
}

#[bon]
impl FireRedVadStreamer {
    /// Prepare the streaming JIT for `model` at a fixed `chunk_frames` chunk
    /// size (the JIT graph shape; smaller = lower latency, more dispatches).
    #[builder]
    pub fn builder(
        model: FireRedVadStream,
        #[builder(default = 16)] chunk_frames: usize,
        #[builder(default)] vad: StreamVadConfig,
    ) -> std::result::Result<Self, FireRedVadStreamError> {
        assert!(chunk_frames >= 1, "chunk_frames must be >= 1");
        let mut jit = FireRedVadStreamJit::new(model);
        jit.prepare_with_config(
            InputSpec::f32(&[1, chunk_frames, N_MELS]),
            std::array::from_fn(|_| InputSpec::f32(&[1, PROJ, STREAM_CACHE])),
            &svod_tensor::PrepareConfig::device_local(),
        )
        .context(StreamPrepareSnafu)?;
        Ok(Self {
            fbank: FireRedFbank::new(),
            jit,
            chunk_frames,
            remainder: Vec::new(),
            pending: Vec::new(),
            scratch: vec![0.0; chunk_frames],
            probs: Vec::new(),
            post: StreamVadPostprocessor::new(vad),
            segments: Vec::new(),
            flushed: false,
        })
    }

    /// Download the converted `Stream-VAD` weights from HF Hub and prepare
    /// the JIT with default knobs.
    pub fn from_hub() -> std::result::Result<Self, FireRedVadStreamError> {
        Self::builder().model(FireRedVadStream::from_hub().context(StreamLoadSnafu)?).build()
    }

    pub fn from_safetensors(path: &std::path::Path) -> std::result::Result<Self, FireRedVadStreamError> {
        Self::builder().model(FireRedVadStream::from_safetensors(path).context(StreamLoadSnafu)?).build()
    }

    /// Feed any number of samples (16 kHz, `[-1, 1]`); returns the speech
    /// boundary events that fired. Dispatches only complete chunks — trailing
    /// samples wait for more audio (or [`flush`](Self::flush)).
    pub fn push(&mut self, samples: &[f32]) -> std::result::Result<Vec<VadEvent>, FireRedVadStreamError> {
        snafu::ensure!(!self.flushed, StreamFlushedSnafu);
        self.remainder.extend_from_slice(samples);
        let n_frames = self.fbank.num_frames(self.remainder.len());
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        let rows = self.fbank.forward(&self.remainder);
        // Frame n_frames starts at n_frames * FRAME_SHIFT; keep the
        // (< FRAME_LENGTH) overlap so framing matches a single whole-waveform
        // pass exactly.
        self.remainder.drain(..n_frames * FRAME_SHIFT);
        self.push_feat(&rows)
    }

    /// Feed pre-computed fbank rows (`[n * N_MELS]`, pre-CMVN), bypassing the
    /// sample buffer — the feature-level entry behind [`push`](Self::push),
    /// exposed for parity tests that isolate the model from the fbank.
    pub(crate) fn push_feat(&mut self, rows: &[f32]) -> std::result::Result<Vec<VadEvent>, FireRedVadStreamError> {
        snafu::ensure!(!self.flushed, StreamFlushedSnafu);
        debug_assert_eq!(rows.len() % N_MELS, 0, "whole feature rows");
        self.pending.extend_from_slice(rows);
        let mut events = Vec::new();
        while self.pending.len() >= self.chunk_frames * N_MELS {
            self.dispatch(self.chunk_frames, &mut events).context(StreamStepSnafu)?;
        }
        Ok(events)
    }

    /// End of stream: run the remaining frames (the final partial chunk is
    /// zero-filled — exact for the real frames, since the causal model never
    /// reads forward), close any open segment, and return all segment
    /// timestamps in seconds. Terminal: feed more audio only after
    /// [`reset`](Self::reset).
    pub fn flush(&mut self) -> std::result::Result<StreamFlush, FireRedVadStreamError> {
        snafu::ensure!(!self.flushed, StreamFlushedSnafu);
        let mut events = Vec::new();
        while self.pending.len() >= self.chunk_frames * N_MELS {
            self.dispatch(self.chunk_frames, &mut events).context(StreamStepSnafu)?;
        }
        let tail = self.pending.len() / N_MELS;
        if tail > 0 {
            self.dispatch(tail, &mut events).context(StreamStepSnafu)?;
        }
        let before = events.len();
        self.post.finalize(&mut events);
        let final_events: Vec<VadEvent> = events[before..].to_vec();
        self.record(&final_events);
        self.flushed = true;
        let timestamps = self
            .segments
            .iter()
            .map(|&(s, e)| ((s - 1) as f32 / FRAMES_PER_SEC, (e - 1) as f32 / FRAMES_PER_SEC))
            .collect();
        Ok(StreamFlush { events, timestamps })
    }

    /// Zero the on-device conv caches and all host state for a new stream.
    pub fn reset(&mut self) -> std::result::Result<(), FireRedVadStreamError> {
        self.jit.reset().context(StreamStepSnafu)?;
        self.remainder.clear();
        self.pending.clear();
        self.probs.clear();
        self.segments.clear();
        self.post.reset();
        self.flushed = false;
        Ok(())
    }

    /// All raw (pre-smoothing) per-frame probs since the last reset.
    pub fn raw_probs(&self) -> &[f32] {
        &self.probs
    }

    /// One JIT dispatch over the first `n_real` pending rows (zero-filling
    /// the chunk tail), feeding the FSM with the real frames only.
    fn dispatch(&mut self, n_real: usize, events: &mut Vec<VadEvent>) -> crate::jit::Result<()> {
        {
            let mut view = self.jit.feat_view_mut::<f32>()?;
            let slice = view.as_slice_mut().expect("contiguous feat");
            slice[..n_real * N_MELS].copy_from_slice(&self.pending[..n_real * N_MELS]);
            slice[n_real * N_MELS..].fill(0.0);
        }
        self.jit.execute()?;
        self.jit.probs()?.copyout_prefix(bytemuck::cast_slice_mut(&mut self.scratch[..n_real]))?;
        self.pending.drain(..n_real * N_MELS);

        self.probs.extend_from_slice(&self.scratch[..n_real]);
        let before = events.len();
        for i in 0..n_real {
            self.post.process_one_frame(self.scratch[i], events);
        }
        let new_events: Vec<VadEvent> = events[before..].to_vec();
        self.record(&new_events);
        Ok(())
    }

    fn record(&mut self, events: &[VadEvent]) {
        for e in events {
            if let VadEvent::SpeechEnd { start_frame, end_frame } = *e {
                self.segments.push((start_frame, end_frame));
            }
        }
    }
}
