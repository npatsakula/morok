//! Streaming FireRedVAD tests.
//!
//! Cheap default tier: streaming forward graph shape on random weights (no
//! compile), the 4-state FSM against hand-traced reference sequences, and the
//! streaming deque smoothing against the batch `smooth_trailing`. The
//! `#[ignore]` tier compiles and executes the streaming JIT on the local CPU
//! device: chunked-with-caches vs full causal forward, flush-tail exactness
//! through the sample path, and — when the converted `Stream-VAD` checkpoint
//! is present (`scripts/convert_firered_vad.py --stream`) — parity against
//! the PyTorch streaming golden.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::firered_vad::{
    FireRedFbank, FireRedVadStream, FireRedVadStreamer, N_MELS, StreamVadConfig, StreamVadPostprocessor, VadEvent,
    smooth_trailing,
};

use super::firered_vad::{lcg, load_golden_vec, real_file, synthetic_waveform};

// ---------------------------------------------------------------------------
// Cheap default tests.
// ---------------------------------------------------------------------------

/// Streaming forward on a 16-frame chunk: probs `[1, 16]`, one updated
/// `[1, P, ORDER-1]` cache per FSMN layer. Catches axis/cat/shrink bugs
/// without any compile.
#[test]
fn stream_forward_shape() {
    let model = FireRedVadStream::with_random_weights();
    let feat = Tensor::zeros(&[1, 16, N_MELS], DType::Float32);
    let caches = FireRedVadStream::zero_caches().unwrap();
    let (probs, new_caches) = model.forward_stream(&feat, &caches).unwrap();

    let shape = |t: &Tensor| -> Vec<usize> { t.dims().unwrap() };
    assert_eq!(shape(&probs), vec![1, 16]);
    assert_eq!(new_caches.len(), caches.len());
    for nc in &new_caches {
        assert_eq!(shape(nc), vec![1, 128, 19]);
    }
}

/// FSM config with smoothing disabled (raw-prob thresholding) so traces are
/// exact; the constructor's `pad >= smooth_window` clamp stays inert.
fn fsm_cfg(min_speech: usize, max_speech: usize, min_silence: usize, pad: usize) -> StreamVadConfig {
    StreamVadConfig {
        smooth_window: 1,
        threshold: 0.5,
        pad_start_frames: pad,
        min_speech_frames: min_speech,
        max_speech_frames: max_speech,
        min_silence_frames: min_silence,
    }
}

fn run_fsm(cfg: StreamVadConfig, probs: &[f32], finalize: bool) -> Vec<VadEvent> {
    let mut post = StreamVadPostprocessor::new(cfg);
    let mut events = Vec::new();
    for &p in probs {
        post.process_one_frame(p, &mut events);
    }
    if finalize {
        post.finalize(&mut events);
    }
    events
}

/// Onset back-padding clamps to frame 1 at the stream start, and min-silence
/// closes the segment at the frame where the count is reached.
#[test]
fn fsm_onset_clamps_to_frame_one() {
    let probs = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let events = run_fsm(fsm_cfg(3, 100, 2, 3), &probs, false);
    // Speech confirmed at frame 4 (3rd consecutive); start back-pads to
    // max(1, 4 - 3 + 1 - 3) = 1. Silence frames 6-7 reach min_silence at 7.
    assert_eq!(events, vec![VadEvent::SpeechStart { frame: 1 }, VadEvent::SpeechEnd { start_frame: 1, end_frame: 7 }]);
}

/// Mid-stream onset back-pads by `pad_start_frames`; a follow-up segment's
/// start is floored at the previous segment's end + 1; the trailing open
/// segment closes on finalize.
#[test]
fn fsm_pad_backfill_and_previous_segment_floor() {
    let probs = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let events = run_fsm(fsm_cfg(3, 100, 2, 3), &probs, true);
    assert_eq!(
        events,
        vec![
            // Confirmed at frame 9: start = max(1, 9 - 3 + 1 - 3) = 4.
            VadEvent::SpeechStart { frame: 4 },
            VadEvent::SpeechEnd { start_frame: 4, end_frame: 12 },
            // Confirmed at frame 15: back-pad would reach 10, floored at 13.
            VadEvent::SpeechStart { frame: 13 },
            VadEvent::SpeechEnd { start_frame: 13, end_frame: 15 },
        ]
    );
}

/// The max-speech cap force-splits: the end fires on the capping frame, the
/// follow-up start on the NEXT frame (the reference's `hit_max_speech`
/// hand-off), repeatedly under sustained speech.
#[test]
fn fsm_max_speech_force_split() {
    let probs = [1.0; 12];
    let events = run_fsm(fsm_cfg(2, 5, 2, 3), &probs, true);
    assert_eq!(
        events,
        vec![
            VadEvent::SpeechStart { frame: 1 },
            VadEvent::SpeechEnd { start_frame: 1, end_frame: 5 },
            VadEvent::SpeechStart { frame: 6 },
            VadEvent::SpeechEnd { start_frame: 6, end_frame: 10 },
            VadEvent::SpeechStart { frame: 11 },
            VadEvent::SpeechEnd { start_frame: 11, end_frame: 12 },
        ]
    );
}

/// Speech shorter than `min_speech_frames` never opens a segment.
#[test]
fn fsm_short_speech_is_discarded() {
    let events = run_fsm(fsm_cfg(3, 100, 2, 3), &[1.0, 1.0, 0.0, 0.0, 0.0], true);
    assert_eq!(events, vec![]);
}

/// The streaming running-mean deque must agree with the batch
/// `smooth_trailing` (same window, same cumulative-mean ramp) so the two
/// formulations can't silently drift apart.
#[test]
fn fsm_smoothing_matches_smooth_trailing() {
    let probs = [0.1, 0.9, 0.2, 0.8, 0.5, 0.3, 0.95, 0.05, 0.6, 0.4];
    let want = smooth_trailing(&probs, 5);
    let mut post = StreamVadPostprocessor::new(StreamVadConfig { smooth_window: 5, ..Default::default() });
    for (&p, w) in probs.iter().zip(&want) {
        let got = post.smooth(p);
        assert!((got - w).abs() < 1e-6, "deque smoothing drifted: {got} vs {w}");
    }
}

// ---------------------------------------------------------------------------
// Heavy tests (compile + execute on the local CPU device).
// ---------------------------------------------------------------------------

fn max_abs_delta(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch");
    got.iter().zip(want).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max)
}

/// THE cache-correctness test: chunked streaming (caches threaded on-device
/// across dispatches, zero-filled flush tail) == one full causal forward with
/// zero caches. Two full chunks + a 7-frame tail exercise recycle and flush.
#[test]
#[ignore = "heavy: streaming JIT compile + execute on random weights"]
fn streaming_chunks_match_full_causal() {
    let model = FireRedVadStream::with_random_weights();
    let n_frames = 2 * 16 + 7;
    let mut seed = 0x5eed;
    let feat: Vec<f32> = (0..n_frames * N_MELS).map(|_| lcg(&mut seed)).collect();

    let feat_t =
        Tensor::from_slice(feat.clone()).try_reshape([1isize, n_frames as isize, N_MELS as isize]).expect("reshape");
    let caches = FireRedVadStream::zero_caches().expect("caches");
    let (full, _) = model.forward_stream(&feat_t, &caches).expect("forward");
    full.realize().expect("realize");
    let want = full.as_vec::<f32>().expect("readout");

    let mut streamer = FireRedVadStreamer::builder().model(model).build().expect("prepare");
    streamer.push_feat(&feat).expect("push");
    streamer.flush().expect("flush");

    let max_abs = max_abs_delta(streamer.raw_probs(), &want);
    assert!(max_abs < 1e-4, "streamed probs drifted from full causal forward: max |delta| = {max_abs}");
}

/// `reset()` zeros the JIT's `state { caches: [..] }` slots, so a second pass
/// over the same features must reproduce the first bit for bit — the cold
/// start is the zero cache, not a fresh `prepare`.
#[test]
#[ignore = "heavy: streaming JIT compile + execute on random weights"]
fn reset_restores_the_cold_start_caches() {
    let model = FireRedVadStream::with_random_weights();
    let n_frames = 2 * 16 + 7;
    let mut seed = 0xd15ea5e;
    let feat: Vec<f32> = (0..n_frames * N_MELS).map(|_| lcg(&mut seed)).collect();

    let mut streamer = FireRedVadStreamer::builder().model(model).build().expect("prepare");
    streamer.push_feat(&feat).expect("push");
    streamer.flush().expect("flush");
    let first = streamer.raw_probs().to_vec();

    // Flush poisons the caches with the zero-padded tail; reset must undo it.
    streamer.reset().expect("reset");
    assert!(streamer.raw_probs().is_empty(), "reset must clear the prob history");
    streamer.push_feat(&feat).expect("push after reset");
    streamer.flush().expect("flush after reset");

    assert_eq!(streamer.raw_probs(), first.as_slice(), "reset run diverged from the cold start");
}

/// Sample-path exactness: pushing audio in awkward block sizes (framing
/// remainder + pending-row buffering + zero-filled flush tail all in play)
/// must reproduce the whole-waveform fbank -> full causal forward.
#[test]
#[ignore = "heavy: streaming JIT compile + execute on random weights"]
fn flush_tail_exactness() {
    let model = FireRedVadStream::with_random_weights();
    let waveform = synthetic_waveform(16_000);
    let fbank = FireRedFbank::new();
    let n_frames = fbank.num_frames(waveform.len());
    assert_eq!(n_frames % 16, 2, "want a partial flush chunk");
    let feat = fbank.forward(&waveform);

    let feat_t = Tensor::from_slice(feat).try_reshape([1isize, n_frames as isize, N_MELS as isize]).expect("reshape");
    let caches = FireRedVadStream::zero_caches().expect("caches");
    let (full, _) = model.forward_stream(&feat_t, &caches).expect("forward");
    full.realize().expect("realize");
    let want = full.as_vec::<f32>().expect("readout");

    let mut streamer = FireRedVadStreamer::builder().model(model).build().expect("prepare");
    for block in waveform.chunks(333) {
        streamer.push(block).expect("push");
    }
    streamer.flush().expect("flush");

    let max_abs = max_abs_delta(streamer.raw_probs(), &want);
    assert!(max_abs < 1e-4, "sample-path probs drifted from full causal forward: max |delta| = {max_abs}");
}

/// Parity against the PyTorch streaming reference: the chunked-with-caches
/// path vs the golden's chunkwise probs, and the full causal forward vs the
/// golden's whole-sequence probs (golden feat in — isolates the model).
#[test]
#[ignore = "heavy: real-weights streaming JIT vs PyTorch golden (local or HF Hub download)"]
fn stream_real_weights_match_golden() {
    let weights = real_file("firered_vad_stream.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden_stream.safetensors")).expect("golden");
    let feat = load_golden_vec(&golden, "feat");
    let probs_chunked = load_golden_vec(&golden, "probs");
    let probs_full = load_golden_vec(&golden, "probs_full");
    let chunk_frames = load_golden_vec(&golden, "chunk_frames")[0] as usize;
    let n_frames = probs_full.len();

    let model = FireRedVadStream::from_safetensors(&weights).expect("load weights");

    let feat_t =
        Tensor::from_slice(feat.clone()).try_reshape([1isize, n_frames as isize, N_MELS as isize]).expect("reshape");
    let caches = FireRedVadStream::zero_caches().expect("caches");
    let (full, _) = model.forward_stream(&feat_t, &caches).expect("forward");
    full.realize().expect("realize");
    let full_delta = max_abs_delta(&full.as_vec::<f32>().expect("readout"), &probs_full);
    assert!(full_delta < 1e-3, "full causal forward drifted from golden: max |delta| = {full_delta}");

    let mut streamer = FireRedVadStreamer::builder().model(model).chunk_frames(chunk_frames).build().expect("prepare");
    streamer.push_feat(&feat).expect("push");
    streamer.flush().expect("flush");
    let chunked_delta = max_abs_delta(streamer.raw_probs(), &probs_chunked);
    assert!(chunked_delta < 1e-3, "chunked probs drifted from golden: max |delta| = {chunked_delta}");

    eprintln!("stream parity: full {full_delta:.2e}, chunked {chunked_delta:.2e}");
}

/// Hub round-trip: `from_hub` must download the published streaming weights
/// and reproduce the PyTorch golden's chunked probs — verifies the uploaded
/// artifact, not just the local conversion output.
#[test]
#[ignore = "heavy: downloads weights from HF Hub + JIT compile"]
fn stream_from_hub_matches_golden() {
    let model = FireRedVadStream::from_hub().expect("hub weights");
    let golden_path = crate::firered_vad::hub_file("golden_stream.safetensors").expect("hub golden");
    let golden = crate::state::load_safetensors(&golden_path).expect("golden");
    let feat = load_golden_vec(&golden, "feat");
    let probs_chunked = load_golden_vec(&golden, "probs");
    let chunk_frames = load_golden_vec(&golden, "chunk_frames")[0] as usize;

    let mut streamer = FireRedVadStreamer::builder().model(model).chunk_frames(chunk_frames).build().expect("prepare");
    streamer.push_feat(&feat).expect("push");
    streamer.flush().expect("flush");
    let max_abs = max_abs_delta(streamer.raw_probs(), &probs_chunked);
    assert!(max_abs < 1e-3, "hub weights drifted from PyTorch golden: max |delta| = {max_abs}");
    eprintln!("stream hub parity: max |delta| = {max_abs:.2e}");
}

/// End-to-end on the golden speech sample: events arrive incrementally and
/// well-formed (alternating, monotonic), timestamps cover the utterance's
/// speech core, the streamer is terminal after flush, and reset() restores it.
#[test]
#[ignore = "heavy: real-weights streaming end-to-end (local or HF Hub download)"]
fn stream_end_to_end_segments() {
    let golden = crate::state::load_safetensors(&real_file("golden_stream.safetensors")).expect("golden");
    let samples = load_golden_vec(&golden, "samples");

    let model = FireRedVadStream::from_safetensors(&real_file("firered_vad_stream.safetensors")).expect("load weights");
    let vad = StreamVadConfig { threshold: 0.4, ..Default::default() };
    let mut streamer = FireRedVadStreamer::builder().model(model).vad(vad).build().expect("prepare");

    let mut events = Vec::new();
    for block in samples.chunks(4_000) {
        events.extend(streamer.push(block).expect("push"));
    }
    let flush = streamer.flush().expect("flush");
    events.extend(flush.events);
    let timestamps = flush.timestamps;

    let mut open = false;
    let mut last_end = 0usize;
    for e in &events {
        match *e {
            VadEvent::SpeechStart { frame } => {
                assert!(!open, "nested SpeechStart: {events:?}");
                assert!(frame > last_end, "start not after previous end: {events:?}");
                open = true;
            }
            VadEvent::SpeechEnd { start_frame, end_frame } => {
                assert!(open, "SpeechEnd without open segment: {events:?}");
                assert!(start_frame <= end_frame, "inverted segment: {events:?}");
                last_end = end_frame;
                open = false;
            }
        }
    }
    assert!(!open, "unclosed segment after flush: {events:?}");

    // The reference places speech at ~0.44-1.82 s in hello_zh.wav; require
    // coverage of the conservative core.
    assert!(timestamps.iter().any(|&(s, e)| s <= 0.6 && e >= 1.6), "no segment covers the speech core: {timestamps:?}",);

    // Terminal after flush; reset() restores a fresh stream.
    assert!(streamer.push(&samples[..1_600]).is_err(), "push after flush must fail");
    streamer.reset().expect("reset");
    for block in samples.chunks(4_000) {
        streamer.push(block).expect("push after reset");
    }
    let ts2 = streamer.flush().expect("flush after reset").timestamps;
    assert_eq!(ts2.len(), timestamps.len(), "reset run diverged: {ts2:?} vs {timestamps:?}");
}
