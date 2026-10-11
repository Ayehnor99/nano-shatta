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
import time
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

# Default base for real training. Instruct variant: better prompt-following
# for lyric generation from a "[Verse 1]" prompt.
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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
    # Resize embeddings to the model's OWN vocab_size (from its config),
    # never len(tokenizer). len(tokenizer) can be smaller than the model's
    # embedding table (Qwen2.5: len(tok)=151665 < vocab_size=151936), which
    # caused a size-mismatch bug when the adapter was later loaded onto a
    # base with the config-sized table.
    if tokenizer is not None:
        target = model.config.vocab_size
        if model.get_input_embeddings().num_embeddings != target:
            model.resize_token_embeddings(target)
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
# Live tracking (Karpathy-style: watch the model learn in real time)
# ---------------------------------------------------------------------------

# Sparkline chars for the loss curve (low -> high).
_SPARK = "▁▂▃▄▅▆▇█"

def sparkline(values: list[float], width: int = 40) -> str:
    """Render a loss history as a one-line sparkline."""
    if not values:
        return ""
    if len(values) > width:
        # Downsample by averaging buckets so the line stays `width` chars.
        bucket = len(values) / width
        values = [
            sum(values[int(i * bucket):int((i + 1) * bucket) or int(i * bucket) + 1])
            / max(1, len(values[int(i * bucket):int((i + 1) * bucket) or int(i * bucket) + 1]))
            for i in range(width)
        ]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(_SPARK[min(7, int((v - lo) / span * 7))] for v in values)


def fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # negative or NaN
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class LiveMonitor:
    """Prints a one-line training status + sparkline every logging step,
    plus val loss + a sample generation every `sample_every` steps.

    Designed to be called from a TrainerCallback so it works on any
    backend (Colab, local GPU, CPU smoke).
    """

    def __init__(self, tokenizer, val_ds=None, sample_prompt="[Verse 1]",
                 sample_every=200, max_new_tokens=48, out_dir=None):
        self.tok = tokenizer
        self.val_ds = val_ds
        self.sample_prompt = sample_prompt
        self.sample_every = sample_every
        self.max_new_tokens = max_new_tokens
        self.out_dir = Path(out_dir) if out_dir else None
        self.losses: list[float] = []
        self.val_losses: list[float] = []
        self.t0 = time.time()
        self.last_sample_step = -10**9

    # -- helpers ---------------------------------------------------------
    def _sample(self, model, step: int) -> str:
        """Generate a short bar from the current (mid-training) model."""
        was_training = model.training
        model.eval()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            enc = self.tok(self.sample_prompt, return_tensors="pt")
            ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
            ids = ids.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=self.max_new_tokens,
                    do_sample=True, temperature=0.9, top_p=0.95,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.eos_token_id,
                )
            text = self.tok.decode(out[0], skip_special_tokens=True)
            # Drop the prompt prefix so we only see what the model wrote.
            if text.startswith(self.sample_prompt):
                text = text[len(self.sample_prompt):]
            return text.strip()
        except Exception as exc:  # sampling must never kill training
            return f"(sample failed: {exc})"
        finally:
            if was_training:
                model.train()

    def _val_loss(self, model, trainer) -> float | None:
        """Quick val loss: average over up to 32 val batches."""
        if self.val_ds is None or len(self.val_ds) == 0:
            return None
        was_training = model.training
        model.eval()
        losses, n = [], min(32, len(self.val_ds))
        try:
            loader = torch.utils.data.DataLoader(
                self.val_ds, batch_size=trainer.args.per_device_train_batch_size,
                collate_fn=collate,
            )
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= n:
                        break
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    losses.append(model(**batch).loss.item())
        except Exception as exc:
            return None
        finally:
            if was_training:
                model.train()
        return sum(losses) / len(losses) if losses else None

    # -- callback --------------------------------------------------------
    def on_log(self, model, trainer, state, logs: dict) -> None:
        loss = logs.get("loss")
        if loss is not None:
            self.losses.append(loss)
        step = state.global_step
        max_steps = trainer.args.max_steps if trainer.args.max_steps > 0 else None
        elapsed = time.time() - self.t0
        # tokens/sec based on optimizer steps and effective batch
        eff_batch = trainer.args.per_device_train_batch_size * trainer.args.gradient_accumulation_steps
        tok_per_step = eff_batch * _BLOCK_SIZE[0]
        tok_s = (step * tok_per_step / elapsed) if elapsed > 0 else 0.0
        eta_s = (elapsed / step * (max_steps - step)) if (step > 0 and max_steps) else float("nan")

        if loss is not None:
            line = (f"step {step:>5}"
                    + (f"/{max_steps}" if max_steps else "")
                    + f" | loss {loss:.4f}")
        else:
            line = f"step {step:>5}" + (f"/{max_steps}" if max_steps else "")
        if "learning_rate" in logs:
            line += f" | lr {logs['learning_rate']:.2e}"
        line += f" | {tok_s:,.0f} tok/s | elapsed {fmt_eta(elapsed)}"
        if max_steps:
            line += f" | ETA {fmt_eta(eta_s)}"
        line += f" | {sparkline(self.losses)}"
        print(line, flush=True)

        # Periodic val loss.
        if self.val_ds is not None and step > 0 and step % self.sample_every < trainer.args.logging_steps:
            vl = self._val_loss(model, trainer)
            if vl is not None:
                self.val_losses.append(vl)
                print(f"  >> val_loss {vl:.4f} | val curve {sparkline(self.val_losses)}", flush=True)

        # Periodic sample generation (the fun part — watch it learn to rap).
        if step > 0 and step - self.last_sample_step >= self.sample_every:
            self.last_sample_step = step
            text = self._sample(model, step)
            print(f"  >> sample @ step {step}:\n{'-'*50}\n{text}\n{'-'*50}", flush=True)
            if self.out_dir is not None:
                self.out_dir.mkdir(parents=True, exist_ok=True)
                with (self.out_dir / "samples.log").open("a", encoding="utf-8") as fh:
                    fh.write(f"\n===== step {step} (loss {loss}) =====\n{text}\n")


# Module-level so the callback can read the real block size without plumbing it through.
_BLOCK_SIZE = [512]


def make_monitor_callback(monitor: "LiveMonitor"):
    """Wrap LiveMonitor as a transformers TrainerCallback."""
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def on_log(self, args, state, control, model=None, logs=None, **kw):
            monitor.on_log(model, _TrainerShim(args), state, logs or {})

    return _CB()


class _TrainerShim:
    """Minimal adapter so LiveMonitor can read args off TrainingArguments."""
    def __init__(self, args):
        self.args = args


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
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of blocks held out for val loss (0 to disable)")
    p.add_argument("--sample-every", type=int, default=200,
                   help="Generate a sample bar every N steps (0 to disable)")
    p.add_argument("--sample-prompt", default="[Verse 1]",
                   help="Prompt used for mid-training sample generations")
    p.add_argument("--sample-tokens", type=int, default=48,
                   help="Max new tokens per mid-training sample")
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

    # Train/val split (deterministic shuffle on blocks, not docs, so a
    # block never straddles the boundary).
    import random as _random
    _rng = _random.Random(args.seed)
    idxs = list(range(len(blocks)))
    _rng.shuffle(idxs)
    n_val = int(len(blocks) * args.val_frac) if args.val_frac > 0 else 0
    val_idx = set(idxs[:n_val])
    train_blocks = [b for i, b in enumerate(blocks) if i not in val_idx]
    val_blocks = [b for i, b in enumerate(blocks) if i in val_idx]
    train_ds = PackedDataset(train_blocks)
    val_ds = PackedDataset(val_blocks) if val_blocks else None
    print(f"[data] train={len(train_blocks)} blocks | val={len(val_blocks)} blocks")
    _BLOCK_SIZE[0] = block_size  # publish for the live monitor

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
        eval_dataset=val_ds,
        data_collator=collate,
    )

    # ---- live monitor (Karpathy-style: watch it learn) ----
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
        print(f"[monitor] sample every {args.sample_every} steps | "
              f"val every {args.sample_every} steps | log -> {out}/samples.log")

    trainer.train(resume_from_checkpoint=True if args.resume else None)

    # ---- save adapter ----
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"[done] adapter + tokenizer saved to {out.resolve()}")


if __name__ == "__main__":
    main()
