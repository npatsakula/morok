use svod_tensor::Tensor;

use crate::audio::mel::hann_window;
use crate::audio::{MelConfig, MelScale, MelSpectrogram};
use crate::whisper::WhisperMel;

struct MelOutput {
    data: ndarray::Array3<f32>,
}

impl MelOutput {
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2])
    }

    fn as_slice(&self) -> &[f32] {
        self.data.as_slice().expect("contiguous mel buffer")
    }
}

fn run_mel(config: &MelConfig, waveform: &[f32]) -> MelOutput {
    let mel = MelSpectrogram::new(config).unwrap();
    let n_mels = mel.n_mels();
    let n_frames = mel.num_frames(waveform.len());
    let mut data = ndarray::Array3::<f32>::zeros((1, n_mels, n_frames));
    let mut view = data.view_mut().into_dyn();
    mel.forward_into(waveform, &mut view);
    MelOutput { data }
}

#[test]
fn test_mel_spectrogram_shape_center_true() {
    let config = MelConfig {
        sample_rate: 16000,
        n_fft: 400,
        hop_length: 160,
        win_length: 400,
        n_mels: 64,
        center: true,
        mel_scale: MelScale::Htk,
    };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    assert_eq!(output.shape(), (1, 64, 101));
}

#[test]
fn test_mel_spectrogram_shape_center_false() {
    let config = MelConfig {
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        n_mels: 64,
        center: false,
        mel_scale: MelScale::Htk,
    };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    assert_eq!(output.shape(), (1, 64, 99));
}

#[test]
fn test_mel_spectrogram_values_finite() {
    let config = MelConfig {
        sample_rate: 16000,
        n_fft: 400,
        hop_length: 160,
        win_length: 400,
        n_mels: 64,
        center: true,
        mel_scale: MelScale::Htk,
    };

    let waveform: Vec<f32> = vec![0.0; 1600];
    let output = run_mel(&config, &waveform);

    for v in output.as_slice() {
        assert!(v.is_finite(), "mel output contains non-finite value: {v}");
    }
}

#[test]
fn test_mel_spectrogram_sine_wave() {
    let config = MelConfig {
        sample_rate: 16000,
        n_fft: 400,
        hop_length: 160,
        win_length: 400,
        n_mels: 64,
        center: true,
        mel_scale: MelScale::Htk,
    };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    let vals = output.as_slice();
    let (_, n_mels, n_frames) = output.shape();

    let mut avg_energy: Vec<f32> = vec![0.0; n_mels];
    for mel_idx in 0..n_mels {
        for frame in 0..n_frames {
            avg_energy[mel_idx] += vals[mel_idx * n_frames + frame];
        }
        avg_energy[mel_idx] /= n_frames as f32;
    }

    let lower_avg: f32 = avg_energy[..20].iter().sum::<f32>() / 20.0;
    let upper_avg: f32 = avg_energy[40..].iter().sum::<f32>() / 24.0;
    assert!(
        lower_avg > upper_avg,
        "Expected lower mel bins to have more energy for 440Hz sine: lower={lower_avg:.2}, upper={upper_avg:.2}"
    );
}

#[test]
fn test_hann_window_matches_torch_periodic_default() {
    let window = hann_window(8, 8);
    let expected = [0.0, 0.14644662, 0.5, 0.8535534, 1.0, 0.8535533, 0.5, 0.1464465];

    for (got, want) in window.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
}

// =========================================================================
// Graph path against the realfft host path
// =========================================================================

fn gigaam_config() -> MelConfig {
    MelConfig {
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        n_mels: 64,
        center: true,
        mel_scale: MelScale::Htk,
    }
}

/// Speech-like synthetic signal: a few tones with drifting amplitude plus
/// deterministic noise, so every mel band carries energy across the clip.
fn synthetic(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
            let t = i as f32 / 16000.0;
            let env = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 0.7 * t).sin();
            env * (0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.3 * (2.0 * std::f32::consts::PI * 1375.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 4310.0 * t).sin())
                + 0.05 * noise
        })
        .collect()
}

/// `ru_clip_0.wav` from the repository root (16 kHz mono int16), if present.
fn real_clip() -> Option<Vec<f32>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../ru_clip_0.wav");
    let mut reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    assert_eq!((spec.channels, spec.sample_rate), (1, 16000), "real clip must be 16 kHz mono");
    Some(reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect())
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Host `forward_into` of each window (its own length) against one graph
/// batch of host-framed rows; the graph's columns past a window's frame count
/// must be zero, as `pack_mel_buffer` leaves them.
fn assert_gigaam_parity(windows: &[&[f32]], label: &str) -> f32 {
    let mel = MelSpectrogram::new(&gigaam_config()).unwrap();
    let max_frames = windows.iter().map(|w| mel.num_frames(w.len())).max().unwrap();
    let framed_len = (max_frames - 1) * 160 + 320;
    let mut framed = vec![0.0f32; windows.len() * framed_len];
    for (row, window) in framed.chunks_mut(framed_len).zip(windows) {
        mel.frame_into(window, row);
    }
    let frames: Vec<i32> = windows.iter().map(|w| mel.num_frames(w.len()) as i32).collect();
    let graph = mel
        .forward_tensor(
            &Tensor::from_slice(framed).try_reshape([windows.len() as isize, framed_len as isize]).unwrap(),
            &Tensor::from_slice(frames.clone()),
        )
        .unwrap();
    assert_eq!(graph.dims().unwrap(), vec![windows.len(), 64, max_frames]);
    let graph = graph.to_vec::<f32>().unwrap();

    let mut worst = 0.0f32;
    for (bi, window) in windows.iter().enumerate() {
        let valid = frames[bi] as usize;
        let mut host = ndarray::Array3::<f32>::zeros((1, 64, valid));
        mel.forward_into(window, &mut host.view_mut().into_dyn());
        let host = host.as_slice().unwrap();
        let row = &graph[bi * 64 * max_frames..(bi + 1) * 64 * max_frames];
        for m in 0..64 {
            let got = &row[m * max_frames..m * max_frames + valid];
            worst = worst.max(max_abs_diff(got, &host[m * valid..(m + 1) * valid]));
            assert!(row[m * max_frames + valid..(m + 1) * max_frames].iter().all(|&v| v == 0.0), "unmasked tail");
        }
    }
    eprintln!("gigaam graph-vs-host log-mel max abs diff ({label}): {worst:.3e}");
    assert!(worst <= 1e-3, "{label}: max abs diff {worst}");
    worst
}

fn assert_whisper_parity(windows: &[&[f32]], label: &str) -> f32 {
    let mel = WhisperMel::new(80).unwrap();
    let framed_len = mel.framed_len();
    let mut framed = vec![0.0f32; windows.len() * framed_len];
    for (row, window) in framed.chunks_mut(framed_len).zip(windows) {
        mel.frame_into(window, row);
    }
    let graph = mel
        .forward_tensor(&Tensor::from_slice(framed).try_reshape([windows.len() as isize, framed_len as isize]).unwrap())
        .unwrap();
    assert_eq!(graph.dims().unwrap(), vec![windows.len(), 80, crate::whisper::N_FRAMES]);
    let graph = graph.to_vec::<f32>().unwrap();
    let per_row = 80 * crate::whisper::N_FRAMES;
    let worst = windows
        .iter()
        .enumerate()
        .map(|(bi, window)| max_abs_diff(&graph[bi * per_row..(bi + 1) * per_row], &mel.compute(window)))
        .fold(0.0, f32::max);
    eprintln!("whisper graph-vs-host log-mel max abs diff ({label}): {worst:.3e}");
    assert!(worst <= 1e-3, "{label}: max abs diff {worst}");
    worst
}

#[test]
fn graph_mel_matches_host_on_synthetic_windows() {
    // Two VAD-style windows of different lengths share one framed batch.
    let (long, short) = (synthetic(16000 * 3 + 77, 1), synthetic(16000 + 5, 2));
    assert_gigaam_parity(&[&long, &short], "synthetic");
}

#[test]
fn graph_whisper_mel_matches_host_on_synthetic_windows() {
    // A full 30 s window and one that pad_or_trim zero-extends.
    let (full, short) = (synthetic(crate::whisper::N_SAMPLES, 3), synthetic(16000 * 7, 4));
    assert_whisper_parity(&[&full, &short], "synthetic");
}

#[test]
fn graph_mel_matches_host_on_real_clip() {
    let Some(clip) = real_clip() else {
        eprintln!("ru_clip_0.wav not found; skipping the real-clip parity check");
        return;
    };
    // The whole clip as one window plus a VAD-sized cut, as the pipelines see them.
    assert_gigaam_parity(&[&clip, &clip[16000 * 5..16000 * 12 + 321]], "ru_clip_0");
    assert_whisper_parity(&[&clip, &clip[..16000 * 11]], "ru_clip_0");
}

#[test]
fn use_graph_mel_follows_the_default_device_unless_overridden() {
    // The env override is process-global, so only the unset default is checked here.
    if std::env::var_os("SVOD_GRAPH_MEL").is_none() {
        let on_cpu = matches!(svod_dtype::default_device::default_device(), svod_dtype::DeviceSpec::Cpu);
        assert_eq!(crate::audio::use_graph_mel(), !on_cpu);
    }
}
