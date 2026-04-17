//! GigaAM JIT inference example with VAD-based chunking for long audio.
//!
//! Usage:
//!   cargo run -p morok-model --example gigaam_infer -- audio_1.wav
//!
//! For audio >25s, uses Silero VAD to segment speech and processes each
//! segment through the JIT-compiled ASR model. Short audio is processed
//! in one pass.

use std::env;

use morok_dtype::DType;
use morok_model::audio::MelSpectrogram;
use morok_model::gigaam::{GigaAm, GigaAmJit};
use morok_model::vad::{SileroVad, VadInference};
use morok_tensor::Tensor;

const VOCAB: &[&str] = &[
    " ", "а", "б", "в", "г", "д", "е", "ж", "з", "и", "й", "к", "л", "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х",
    "ц", "ч", "ш", "щ", "ъ", "ы", "ь", "э", "ю", "я",
];
const BLANK_ID: usize = VOCAB.len();
const LONGFORM_THRESHOLD: usize = 25 * 16000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wav_path = env::args().nth(1).ok_or("usage: gigaam_infer <audio.wav>")?;

    println!("Loading audio: {wav_path}");
    let waveform = load_wav(&wav_path)?;
    let duration_s = waveform.len() as f32 / 16000.0;
    println!("Samples: {} ({:.1}s)", waveform.len(), duration_s);

    println!("\nLoading GigaAM...");
    let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;
    let sample_rate = model.config.sample_rate;
    let subs_kernel_size = model.config.subs_kernel_size;
    println!("Loaded: {} layers, d_model={}", model.config.n_layers, model.config.d_model);

    let mel = MelSpectrogram::new(&morok_model::audio::MelConfig {
        sample_rate,
        n_fft: model.config.n_fft,
        hop_length: model.config.hop_length,
        win_length: model.config.win_length,
        n_mels: model.config.n_mels,
        center: model.config.mel_center,
    });
    let n_mels = mel.n_mels();

    if waveform.len() <= LONGFORM_THRESHOLD {
        let text = transcribe_full(&waveform, model, &mel, n_mels, subs_kernel_size)?;
        println!("\nTranscription: {text}");
        return Ok(());
    }

    // --- VAD segmentation for long audio ---
    println!("\nLoading Silero VAD...");
    let vad = SileroVad::from_hub()?;
    let mut vad_inf = VadInference::new(vad)?;
    let segments = vad_inf.segment(&waveform, 0.5);
    println!("Found {} speech segments", segments.len());

    if segments.is_empty() {
        println!("No speech detected.");
        return Ok(());
    }

    // Find the longest segment for JIT plan shape
    let max_samples = segments.iter().map(|(s, e)| e - s).max().unwrap();
    let max_mel_frames = mel.num_frames(max_samples);

    let mut mel_tensor = Tensor::full(&[1, n_mels, max_mel_frames], 0.0f32, DType::Float32)?;
    mel_tensor.realize().unwrap();

    let mut jit = GigaAmJit::new(model);
    println!("Preparing JIT plan... [1, {n_mels}, {max_mel_frames}]");
    jit.prepare(&mel_tensor)?;
    println!("Plan captured.");

    let mut full_text = String::new();
    for (i, (start, end)) in segments.iter().enumerate() {
        let segment_wave = &waveform[*start..*end];
        let actual_mel = mel.num_frames(segment_wave.len());
        let actual_sub = subs_output_length(subs_kernel_size, actual_mel);

        {
            let mut view = jit.mel_mut()?.as_array_mut::<f32>()?;
            let slice = view.as_slice_mut().expect("contiguous");
            slice.fill(0.0);
        }
        {
            let mut view = jit.mel_mut()?.as_array_mut::<f32>()?;
            mel.forward_into(segment_wave, &mut view);
        }
        jit.execute()?;

        let text = ctc_greedy_decode(jit.output()?, BLANK_ID, actual_sub);
        let start_s = *start as f32 / sample_rate as f32;
        let end_s = *end as f32 / sample_rate as f32;
        if !text.is_empty() {
            println!("[{:.1}s - {:.1}s] {}", start_s, end_s, text);
            full_text.push_str(&text);
            full_text.push(' ');
        }
    }

    println!("\n--- Full transcription ---");
    println!("{}", full_text.trim());

    Ok(())
}

fn transcribe_full(
    waveform: &[f32],
    model: GigaAm,
    mel: &MelSpectrogram,
    n_mels: usize,
    subs_kernel_size: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let n_frames = mel.num_frames(waveform.len());

    let mut mel_tensor = Tensor::full(&[1, n_mels, n_frames], 0.0f32, DType::Float32)?;
    mel_tensor.realize().unwrap();

    let mut jit = GigaAmJit::new(model);
    println!("Preparing JIT plan... [1, {n_mels}, {n_frames}]");
    jit.prepare(&mel_tensor)?;
    println!("Plan captured.");

    {
        let mut view = jit.mel_mut()?.as_array_mut::<f32>()?;
        mel.forward_into(waveform, &mut view);
    }
    jit.execute()?;

    let actual_sub = subs_output_length(subs_kernel_size, n_frames);
    Ok(ctc_greedy_decode(jit.output()?, BLANK_ID, actual_sub))
}

fn load_wav(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?
        }
    };
    Ok(samples)
}

fn subs_output_length(kernel_size: usize, mel_frames: usize) -> usize {
    let pad = (kernel_size - 1) / 2;
    let mut len = mel_frames;
    for _ in 0..2 {
        len = (len + 2 * pad - kernel_size) / 2 + 1;
    }
    len
}

fn ctc_greedy_decode(output_buf: &morok_device::Buffer, blank_id: usize, max_frames: usize) -> String {
    let logits = output_buf.as_array::<f32>().expect("failed to read output buffer");
    let total_vocab = blank_id + 1;
    let n_frames = (logits.len() / total_vocab).min(max_frames);

    let mut prev = blank_id;
    let mut text = String::new();
    for t in 0..n_frames {
        let base = t * total_vocab;
        let best = (0..total_vocab).max_by(|&a, &b| logits[base + a].partial_cmp(&logits[base + b]).unwrap()).unwrap();
        if best != blank_id && best != prev {
            text.push_str(VOCAB[best]);
        }
        prev = best;
    }
    text
}
