use morok_dtype::DType;
use morok_tensor::Tensor;

use crate::audio::{MelConfig, MelSpectrogram};

fn run_mel(config: &MelConfig, waveform: &[f32]) -> Tensor {
    let mel = MelSpectrogram::new(config);
    let n_mels = mel.n_mels();
    let n_frames = mel.num_frames(waveform.len());
    let mut t = Tensor::full(&[1, n_mels, n_frames], 0.0f32, DType::Float32).unwrap();
    t.realize().unwrap();
    let mut view = t.array_view_mut::<f32>().unwrap();
    mel.forward_into(waveform, &mut view);
    t
}

#[test]
fn test_mel_spectrogram_shape_center_true() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 400, hop_length: 160, win_length: 400, n_mels: 64, center: true };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    let shape = output.shape().unwrap();
    assert_eq!(shape[0].as_const().unwrap(), 1);
    assert_eq!(shape[1].as_const().unwrap(), 64);
    assert_eq!(shape[2].as_const().unwrap(), 101);
}

#[test]
fn test_mel_spectrogram_shape_center_false() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 320, hop_length: 160, win_length: 320, n_mels: 64, center: false };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    let shape = output.shape().unwrap();
    assert_eq!(shape[0].as_const().unwrap(), 1);
    assert_eq!(shape[1].as_const().unwrap(), 64);
    assert_eq!(shape[2].as_const().unwrap(), 99);
}

#[test]
fn test_mel_spectrogram_values_finite() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 400, hop_length: 160, win_length: 400, n_mels: 64, center: true };

    let waveform: Vec<f32> = vec![0.0; 1600];
    let output = run_mel(&config, &waveform);

    let vals = output.as_vec::<f32>().unwrap();
    for v in &vals {
        assert!(v.is_finite(), "mel output contains non-finite value: {v}");
    }
}

#[test]
fn test_mel_spectrogram_sine_wave() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 400, hop_length: 160, win_length: 400, n_mels: 64, center: true };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    let vals = output.as_vec::<f32>().unwrap();
    let shape = output.shape().unwrap();
    let n_mels = shape[1].as_const().unwrap();
    let n_frames = shape[2].as_const().unwrap();

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
