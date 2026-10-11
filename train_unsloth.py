#!/usr/bin/env python3
"""LoRA fine-tune with Unsloth — 2x faster, ~70% less VRAM than HF+PEFT.

Same data pipeline, flags, and live monitoring as train.py, but the model
loading and training loop run through Unsloth's optimized kernels
(FastLanguageModel + SFTTrainer). Recommended base:

    unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit   (pre-quantized, fast download)

Unsloth handles embedding sizing itself — no manual resize_token_embeddings,
which is what caused the vocab-size mismatch bug in the HF path.

Usage (Colab T4):
    python train_unsloth.py \
        --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit \
        --data shatta_clean_v3.txt --out shatta-lora \
        --epochs 3 --batch-size 2 --grad-accum 8

Smoke test (CPU, tiny model, 2 steps):
    python train_unsloth.py --smoke
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset as HFDataset  # SFTTrainer requires HF Dataset
from transformers import set_seed

# Reuse everything from the HF trainer: data pipeline, packing,
# collate, live monitor (sparkline/ETA/val/samples). SFTTrainer
# subclasses Trainer, so callbacks carry over unchanged.
from train import (
    LiveMonitor,
    build_blocks,
    collate,
    load_documents,
    make_monitor_callback,
    _BLOCK_SIZE,
)


def to_hf_dataset(blocks: list[list[int]]) -> HFDataset:
    """Wrap packed blocks as an HF Dataset (SFTTrainer requirement).

    Each row: {"input_ids": [...], "labels": [...]}. The custom collate
    still runs at batch time; this only satisfies the type check.
    """
    return HFDataset.from_dict({
        "input_ids": blocks,
        "labels": [list(b) for b in blocks],
    })

# Tiny random model for --smoke. Unsloth needs a supported arch; the tiny
# Llama checkpoint is the smallest that works. On CPU Unsloth kernels are
# unavailable, so smoke mode falls back to the plain HF path internally
# (same code path as train.py's smoke) to stay testable without GPU.
SMOKE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Model loading (Unsloth)
# ---------------------------------------------------------------------------

def load_model_unsloth(model_name: str, max_seq_length: int, load_in_4bit: bool):
    """Load via Unsloth's FastLanguageModel. Returns (model, tokenizer)."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,            # auto: fp16 on T4, bf16 on Ampere+
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def apply_lora_unsloth(model, r: int, alpha: int, dropout: float):
    """Attach LoRA via Unsloth's get_peft_model.

    Standard LoRA targeting the attention + MLP projections (q/k/v/o +
    gate/up/down). We deliberately do NOT target embed_tokens/lm_head:
    Unsloth promotes those to full-matrix training (modules_to_save),
    which costs VRAM and offloads embeddings to disk. For style
    fine-tuning (not continued pretraining) that's pure overhead.
    """
    model = FastLanguageModel_get_peft(
        model,
        r=r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=alpha,
        lora_dropout=0 if dropout == 0 else dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    return model


def FastLanguageModel_get_peft(*args, **kwargs):
    """Indirection so the import stays at call time (smoke mode may not
    have unsloth installed at all)."""
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                   help="Base model (HF id). Default: Unsloth pre-quantized Qwen2.5-1.5B-Instruct")
    p.add_argument("--data", default="shatta_clean_v3.txt",
                   help="Training corpus (PAD-separated songs)")
    p.add_argument("--out", default="shatta-lora",
                   help="Output dir for adapter + tokenizer")
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="Unsloth-optimized at 0")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true",
                   help="16-bit LoRA instead of 4-bit (needs more VRAM)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in --out/checkpoints")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--sample-every", type=int, default=50)
    p.add_argument("--sample-prompt", default="[Verse 1]")
    p.add_argument("--sample-tokens", type=int, default=48)
    p.add_argument("--smoke", action="store_true",
                   help="CPU smoke test: tiny model, 2 steps (falls back to HF path)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.smoke:
        # Smoke mode runs the train.py HF path with the tiny model —
        # Unsloth's Triton kernels need a GPU and can't do 2-step CPU runs.
        print("[smoke] delegating to train.py --smoke (Unsloth kernels need GPU)")
        import subprocess

        cmd = [sys.executable, "train.py", "--smoke", "--out", args.out]
        r = subprocess.run(cmd, cwd=Path(__file__).parent)
        sys.exit(r.returncode)

    cuda = torch.cuda.is_available()
    if not cuda:
        print("ERROR: train_unsloth.py needs a GPU (Unsloth Triton kernels).\n"
              "For CPU testing use: python train.py --smoke", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    load_in_4bit = not args.no_4bit

    # ---- data (identical pipeline to train.py) ----
    docs = load_documents(args.data)
    print(f"[data] {len(docs)} documents from {args.data}")
    # Tokenizer comes from the model load (Unsloth pairs them).
    model, tokenizer = load_model_unsloth(args.model, args.block_size, load_in_4bit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks = build_blocks(tokenizer, docs, args.block_size)
    print(f"[data] {len(blocks)} blocks x {args.block_size}")
    if not blocks:
        print("ERROR: no training blocks produced", file=sys.stderr)
        sys.exit(1)

    # Train/val split — same logic as train.py.
    import random as _random
    _rng = _random.Random(args.seed)
    idxs = list(range(len(blocks)))
    _rng.shuffle(idxs)
    n_val = int(len(blocks) * args.val_frac) if args.val_frac > 0 else 0
    val_idx = set(idxs[:n_val])
    train_blocks = [b for i, b in enumerate(blocks) if i not in val_idx]
    val_blocks = [b for i, b in enumerate(blocks) if i in val_idx]
    train_ds = to_hf_dataset(train_blocks)
    val_ds = to_hf_dataset(val_blocks) if val_blocks else None
    print(f"[data] train={len(train_blocks)} | val={len(val_blocks)} blocks")
    _BLOCK_SIZE[0] = args.block_size

    # ---- LoRA ----
    model = apply_lora_unsloth(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # ---- train ----
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported

    # for_training is mandatory with Unsloth (gradients won't flow otherwise).
    FastLanguageModel.for_training(model)

    sft_args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",   # unused: we pass pre-tokenized dicts
        max_length=args.block_size,
        packing=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        processing_class=tokenizer,
    )

    # ---- live monitor (same as train.py) ----
    if args.sample_every > 0 or val_ds is not None:
        monitor = LiveMonitor(
            tokenizer=tokenizer,
            val_ds=val_ds,
            sample_prompt=args.sample_prompt,
            sample_every=args.sample_every,
            max_new_tokens=args.sample_tokens,
            out_dir=out,
        )
        trainer.add_callback(make_monitor_callback(monitor))
        print(f"[monitor] sample/val every {args.sample_every} steps | "
              f"log -> {out}/samples.log")

    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # ---- save ----
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)      # adapter only (~50MB)
    tokenizer.save_pretrained(out)
    print(f"[done] adapter + tokenizer saved to {out.resolve()}")
    print("[tip] to merge into a standalone GGUF later:")
    print("      model.save_pretrained_gguf(out, tokenizer, quantization_method='q4_k_m')")


if __name__ == "__main__":
    main()
