"""Convert a PyTorch checkpoint (.bin/.pt/.ckpt) to safetensors format.

Usage:
    python scripts/pt_to_safetensors.py <input.bin> <output.safetensors>

The output file uses the original PyTorch key names unchanged.
"""

import sys
import torch
import safetensors.torch


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.safetensors>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    sd = {k: v.contiguous().to(torch.float32) for k, v in state_dict.items()}
    safetensors.torch.save_file(sd, output_path)

    print(f"Converted {len(sd)} tensors -> {output_path}")


if __name__ == "__main__":
    main()
