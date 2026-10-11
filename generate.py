#!/usr/bin/env python3
"""Generate Shatta-style bars from a trained LoRA adapter.

The base model is read FROM the adapter config (base_model_name_or_path) —
never passed by hand. This makes the vocab-size mismatch bug structurally
impossible: you cannot accidentally generate with a different base than
you trained on.

Usage:
    python generate.py --adapter shatta-lora
    python generate.py --adapter shatta-lora --prompt "[Verse 1] Mi a di realest"
    python generate.py --adapter shatta-lora --no-4bit   # bf16 full precision
"""
from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="shatta-lora",
                   help="Path to the trained adapter dir (default: shatta-lora)")
    p.add_argument("--prompt", default="[Verse 1]",
                   help="Prompt to start generation from")
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-4bit", action="store_true",
                   help="Load base in bf16 instead of 4-bit (needs more VRAM)")
    p.add_argument("--force-base", default=None,
                   help="OVERRIDE the adapter's recorded base model (dangerous — "
                        "only if you know the vocab sizes match)")
    return p.parse_args()


def resolve_base(adapter_dir: str, override: str | None) -> str:
    """Read the base model id from the adapter's own config."""
    from peft import PeftConfig

    cfg = PeftConfig.from_pretrained(adapter_dir)
    recorded = cfg.base_model_name_or_path
    if override:
        print(f"[warn] --force-base overrides adapter's recorded base!\n"
              f"       recorded: {recorded}\n"
              f"       using:    {override} — expect vocab mismatch if wrong")
        return override
    if not recorded:
        raise RuntimeError(
            f"adapter at {adapter_dir} has no base_model_name_or_path in its config; "
            "pass --force-base to override (at your own risk)"
        )
    return recorded


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_name = resolve_base(args.adapter, args.force_base)
    print(f"[generate] base (from adapter config): {base_name}")

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

    print(f"[generate] loading base {base_name} ...")
    base = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
    print(f"[generate] loading adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt")
    ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(model.device)
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
