"""Generate the ModernBERT golden fixture for the parity tests.

Produces `golden.safetensors` under the ModernBERT data dir
(`SVOD_MODERNBERT` env var or `../data/modernbert` relative to this crate) with:

  input_ids          (L,)    int64   tokenized sample (batch = 1)
  attention_mask     (L,)    int64   1 = real token, 0 = pad
  input_ids_shape    (2,)    int64   (batch, seq_len)
  last_hidden_state  (L, D)  float32 backbone output (real tokens only matter)
  mlm_logits         (L, V)  float32 MLM head output (vocab logits per token)
  expected_embedding (D,)    float32 masked mean-pool + L2-normalize

The first four feed `last_hidden_state_matches_pytorch` (backbone parity);
`mlm_logits` feeds `mlm_logits_match_pytorch` (MLM-head parity);
`expected_embedding` feeds `embeddings_match_pytorch` (embed-pipeline parity).
All are produced by HuggingFace `transformers` so Svod's f32 output is compared
against an independent reference.

Usage:
    uv run scripts/generate_modernbert_golden.py
    SVOD_MODERNBERT=/path uv run scripts/generate_modernbert_golden.py
    uv run scripts/generate_modernbert_golden.py --repo answerdotai/ModernBERT-base --max-seq 32
"""

import argparse
import os
from pathlib import Path

import torch
import safetensors.torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

# Pure masked mean-pool + L2-normalize (the canonical sentence-embedding recipe,
# matching Svod's pool_embed). Svod guards the denominator/norm with EPS=1e-12;
# that is negligible against the 1e-3 parity bound, so we omit it here on
# purpose so the reference stays independent of Svod's exact numerics.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="answerdotai/ModernBERT-base")
    parser.add_argument("--text", default="The quick brown fox jumps over the lazy dog near the quiet riverbank.")
    parser.add_argument("--max-seq", type=int, default=32, help="pad/truncate target length (keeps the mask load-bearing)")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or os.environ.get("SVOD_MODERNBERT", Path(__file__).resolve().parent.parent / "../data/modernbert"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    # Force f32: the parity tests run Svod in f32, so the golden must too.
    # Without this a checkpoint whose config defaults to bfloat16 (e.g. a
    # fine-tune saved in bf16) would load/run in bf16 and produce bf16-rounded
    # logits that Svod's f32 output can never match (divergence ~0.1).
    model = AutoModel.from_pretrained(args.repo, torch_dtype=torch.float32).to(device).eval()
    # Same f32 rationale as the backbone: the MLM head's matmul/layernorm/GELU
    # accumulate in f32, and Svod's parity test runs in f32 too.
    mlm = AutoModelForMaskedLM.from_pretrained(args.repo, torch_dtype=torch.float32).to(device).eval()

    enc = tokenizer(args.text, padding="max_length", max_length=args.max_seq, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state  # (1, L, D) float32
        mlm_logits = mlm(input_ids=input_ids, attention_mask=attention_mask).logits  # (1, L, V) float32

    mask_f = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)        # (1, L, 1)
    summed = (last_hidden_state * mask_f).sum(dim=1)                         # (1, D)
    denom = mask_f.sum(dim=1)                                                # (1, 1)
    mean = summed / denom                                                    # (1, D)
    norm = torch.linalg.vector_norm(mean, ord=2, dim=-1, keepdim=True)       # (1, 1)
    embedding = mean / norm                                                  # (1, D)

    golden = {
        "input_ids": input_ids.squeeze(0).to(torch.int64).cpu(),
        "attention_mask": attention_mask.squeeze(0).to(torch.int64).cpu(),
        "input_ids_shape": torch.tensor([1, args.max_seq], dtype=torch.int64),
        "last_hidden_state": last_hidden_state.squeeze(0).to(torch.float32).cpu(),
        "mlm_logits": mlm_logits.squeeze(0).to(torch.float32).cpu(),
        "expected_embedding": embedding.squeeze(0).to(torch.float32).cpu(),
    }
    golden_path = out_dir / "golden.safetensors"
    safetensors.torch.save_file({k: v.contiguous() for k, v in golden.items()}, str(golden_path))
    real = int(attention_mask.sum())
    print(f"Wrote {golden_path} ({len(golden)} tensors)")
    print(f"  seq_len={args.max_seq} hidden={last_hidden_state.shape[-1]} real_tokens={real} pad_tokens={args.max_seq - real}")


if __name__ == "__main__":
    main()
