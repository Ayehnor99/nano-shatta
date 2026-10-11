#!/usr/bin/env python3
"""Generate Shatta-style bars from a trained LoRA adapter.

Loads the base model (4-bit on GPU) + your adapter, then writes lyrics.

Usage:
    python generate.py --adapter shatta-lora
    python generate.py --adapter shatta-lora --prompt "[Verse 1] Mi a di realest" --max-new-tokens 120
    python generate.py --adapter shatta-lora --no-4bit   # bf16 full precision (more VRAM)
"""
from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="shatta-lora",
                   help="Path to the trained adapter dir (default: shatta-lora)")
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B",
                   help="Base model HF id (default: Qwen/Qwen2.5-0.5B)")
    p.add_argument("--prompt", default="[Verse 1]",
                   help="Prompt to start generation from")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-4bit", action="store_true",
                   help="Load base in bf16 instead of 4-bit (needs more VRAM)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.adapter)

    kwargs: dict = {"device_map": "auto"}
    if torch.cuda.is_available() and not args.no_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    print(f"[generate] loading base {args.base} ...")
    base = AutoModelForCausalLM.from_pretrained(args.base, **kwargs)
    print(f"[generate] loading adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    print("\n" + "-" * 50)
    print(tok.decode(out[0], skip_special_tokens=True))
    print("-" * 50)


if __name__ == "__main__":
    main()
