use morok_model::vad::{SileroVad, VadInference};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading Silero VAD...");
    let vad = SileroVad::from_hub()?;
    println!("Loaded. Creating inference engine...");
    let mut inf = VadInference::new(vad)?;
    println!("Engine ready.");

    let mut reader = hound::WavReader::open("/home/mrpink/projects/morok/audio_1.wav")?;
    let waveform: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    println!("Audio: {} samples ({:.1}s)", waveform.len(), waveform.len() as f32 / 16000.0);

    let segments = inf.segment(&waveform, 0.5);
    println!("\nSegments (threshold=0.5):");
    for (i, (start, end)) in segments.iter().enumerate() {
        let start_s = *start as f32 / 16000.0;
        let end_s = *end as f32 / 16000.0;
        println!("  {}: {:.1}s - {:.1}s ({:.1}s)", i, start_s, end_s, end_s - start_s);
    }

    Ok(())
}
