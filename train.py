#!/usr/bin/env python3
"""LoRA fine-tune a small base model on the nano-shatta lyrics corpus.

Pipeline: corpus txt -> split on  song separators -> tokenize
-> pack into fixed-length blocks -> LoRA fine-tune (4-bit on GPU)
-> save adapter + tokenizer to --out.

Corpus format: songs separated by a `` line; blank lines
separate sections inside a song. Twi orthography (ɛ, ɔ, ŋ, Ọ, ọ) is
preserved as-is -- the tokenizer handles them as regular characters.

Usage (Colab T4):
    python train.py --model Qwen/Qwen2.5-0.5B \
        --data shatta_clean_v3.txt --out shatta-lora \
        --epochs 3 --batch-size 2 --grad-accum 8

Smoke test (CPU, tiny random model, 2 steps, ~1 min):
    python train.py --smoke
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Song separator in the corpus files written by clean.py / merge.py.
PAD_MARKER = "<|endoftext|>"

# Tiny random model for --smoke (no real weights downloaded).
# Llama arch -- PEFT's all-linear targeting works cleanly on it.
SMOKE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# Smoke mode uses the Llama tokenizer to match the tiny model's 32k vocab.
# (The Qwen tokenizer emits ids up to 151k, which overflow the tiny model.)
SMOKE_TOKENIZER = "hf-internal-testing/tiny-random-LlamaForCausalLM"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_documents(path: str | Path) -> list[str]:
    """Split the corpus on  markers into song documents.

    Drops empty/tiny fragments (<50 chars after strip).
    """
    text = Path(path).read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split(PAD_MARKER)]
    docs = [d for d in docs if len(d) >= 50]
    return docs


def build_blocks(
    tokenizer,
    docs: list[str],
    block_size: int,
    min_doc_tokens: int = 16,
) -> list[list[int]]:
    """Tokenize each doc, append EOS, concatenate into one stream, then
    slice into fixed-length blocks (nanoGPT-style packing).

    The tail partial block is dropped -- we never train on padding.
    """
    eos = tokenizer.eos_token_id
    stream: list[int] = []
    vocab = len(tokenizer)  # real size incl. special tokens; >= vocab_size
    for d in docs:
        ids = tokenizer(d, add_special_tokens=False)["input_ids"]
        if len(ids) < min_doc_tokens:
            continue
        # Clamp any id the tokenizer emitted beyond the embedding table
        # (special tokens like `` sit at index == vocab_size, which
        # overflows a model loaded with vocab_size rows -- resize at load).
        ids = [min(i, vocab - 1) for i in ids[:block_size]]
        stream.extend(ids)
        stream.append(eos)
    blocks = [
        stream[i : i + block_size]
        for i in range(0, len(stream) - block_size + 1, block_size)
    ]
    return blocks


class PackedDataset(Dataset):
    """Fixed-length blocks; labels == input_ids (causal LM)."""

    def __init__(self, blocks: list[list[int]]):
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> dict:
        ids = self.blocks[i]
        return {"input_ids": ids, "labels": list(ids)}


def collate(features: list[dict]) -> dict:
    ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
    return {"input_ids": ids, "labels": ids.clone()}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_base_model(model_name: str, use_4bit: bool, tokenizer=None):
    """Load base model. 4-bit NF4 quantization when on CUDA, else fp32/bf16."""
    kwargs: dict = {}
    if use_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    # The tokenizer can emit ids up to len(tokenizer)-1 (special tokens),
    # but the base model's embedding table is often sized to vocab_size
    # (smaller). Resize so EOS/`` never overflows the embedding.
    if tokenizer is not None:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False  # required for gradient checkpointing
    return model


def apply_lora(model, r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                   help="Base model (HF id). Default: Qwen2.5-0.5B")
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
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Hard cap on optimizer steps (-1 = use epochs)")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in --out/checkpoints")
    p.add_argument("--smoke", action="store_true",
                   help="CPU smoke test: tiny random model, 2 steps, block=128")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cuda = torch.cuda.is_available()

    if args.smoke:
        model_name = SMOKE_MODEL
        tok_name = SMOKE_TOKENIZER
        block_size = 128
        epochs, max_steps = 1.0, 2
        batch, accum, save_steps = 2, 1, 2
        out = Path(args.out)
        if out.exists():
            shutil.rmtree(out)
        # Smoke mode always trains on the v2 corpus (v3 may not exist yet).
        data_path = "shatta_clean_v2.txt"
        print("[smoke] tiny random model, real tokenizer, 2 steps, CPU")
    else:
        model_name = args.model
        tok_name = args.model
        block_size = args.block_size
        epochs, max_steps = args.epochs, args.max_steps
        batch, accum, save_steps = args.batch_size, args.grad_accum, args.save_steps
        out = Path(args.out)
        use_4bit = True
        data_path = args.data

    # ---- data ----
    docs = load_documents(data_path)
    print(f"[data] {len(docs)} documents from {data_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    blocks = build_blocks(tokenizer, docs, block_size)
    n_tokens = len(blocks) * block_size
    print(f"[data] {len(blocks)} blocks x {block_size} = {n_tokens:,} tokens")
    if not blocks:
        print("ERROR: no training blocks produced", file=sys.stderr)
        sys.exit(1)
    train_ds = PackedDataset(blocks)

    # ---- model ----
    if args.smoke:
        model = load_base_model(model_name, use_4bit=False, tokenizer=tokenizer)
    else:
        model = load_base_model(model_name, use_4bit=use_4bit, tokenizer=tokenizer)
    model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # ---- train ----
    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=args.lr,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        bf16=cuda,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        use_cpu=not cuda,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collate,
    )
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # ---- save adapter ----
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[done] adapter + tokenizer saved to {out.resolve()}")


if __name__ == "__main__":
    main()
