"""Compare GigaAM CTC inference output between Morok and PyTorch.

Usage:
    cd submodules/GigaAM
    uv run --with torch --with torchaudio --with omegaconf --with hydra-core \
        python3 ../../model/tests/gigaam/compare_pytorch.py [audio.wav]

If an audio file is given, prints CTC transcription. Otherwise runs a sine wave.
"""

import math
import sys
import torch
import torchaudio


def sine_wave(duration_secs: float, freq: float, sample_rate: int) -> torch.Tensor:
    n = int(duration_secs * sample_rate)
    t = torch.arange(n, dtype=torch.float32)
    return (t * freq * 2.0 * math.pi / sample_rate).sin()


def main():
    from gigaam import load_model

    print("Loading model...")
    model = load_model("v3_ctc", device="cpu")
    model.eval().float()

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Loading audio: {audio_path}")
        from gigaam.preprocess import load_audio
        wav = load_audio(audio_path)
        wav = wav.unsqueeze(0)
        with torch.no_grad():
            result = model.transcribe(audio_path)
        if result:
            print(f"\nTranscription: {result.text}")
    else:
        wav = sine_wave(1.0, 440.0, 16000).unsqueeze(0)
        mel, mel_len = model.preprocessor(wav, torch.tensor([wav.shape[1]]))
        with torch.no_grad():
            encoded, _ = model.encoder(mel, mel_len)
            log_probs = model.head(encoder_output=encoded)
        vocab = list(model.cfg.decoding.vocabulary)
        blank_id = len(vocab)
        n_frames = log_probs.shape[1]
        print(f"\nTop-5 tokens per frame ({n_frames} total):")
        for t in range(min(n_frames, 5)):
            probs = log_probs[0, t]
            topk = torch.topk(probs, 5)
            tokens_str = ", ".join(
                f"{'<blank>' if v.item() == blank_id else repr(vocab[v.item()])}={p.item():.3f}"
                for p, v in zip(topk.values, topk.indices)
            )
            print(f"  frame {t}: {tokens_str}")


if __name__ == "__main__":
    main()
