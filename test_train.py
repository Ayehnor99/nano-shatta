"""Offline tests for train.py data pipeline + a full CPU smoke run.

No GPU, no big model downloads. The smoke test uses a tiny random model
and the real Qwen2.5-0.5B tokenizer + the real corpus.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/btclord/nano-shatta")

from train import PAD_MARKER, PackedDataset, build_blocks, collate, load_documents

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")

print("=== corpus / data pipeline ===")
docs = load_documents("shatta_clean_v2.txt")
check("loads >150 docs", len(docs) > 150)
check("PAD marker actually splits docs", len(docs) > 1)
check("no doc contains the PAD marker", all(PAD_MARKER not in d for d in docs))
check("Twi ɛ preserved in docs", any("ɛ" in d for d in docs))
check("Twi ɔ preserved in docs", any("ɔ" in d for d in docs))

# Use a real tokenizer for the block tests (small download, cached after first run).
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
eos = tok.eos_token_id

BS = 128
blocks = build_blocks(tok, docs, BS)
check("blocks produced", len(blocks) > 50)
check("all blocks are exactly block_size", all(len(b) == BS for b in blocks))
check("blocks contain EOS", any(eos in b for b in blocks))
flat = [t for b in blocks for t in b]
# Qwen's pad_token == eos_token (both ``), and we deliberately append
# EOS between docs -- so "pad in blocks" is EXPECTED. The real check is:
# no token id outside the model's vocab range (overflow guard), and EOS
# IS present (proves doc-boundary packing worked).
vocab = len(tok)  # real vocab size incl. special tokens (151665 > vocab_size 151643)
check("all token ids within vocab", all(0 <= t < vocab for t in flat))
check("blocks contain EOS", any(eos in b for b in blocks))

ds = PackedDataset(blocks)
check("dataset length == block count", len(ds) == len(blocks))
item = ds[0]
check("item has input_ids + labels", "input_ids" in item and "labels" in item)
check("labels == input_ids", item["labels"] == item["input_ids"])

batch = collate([ds[0], ds[1], ds[2]])
import torch
check("collate shape (3, block_size)", tuple(batch["input_ids"].shape) == (3, BS))
check("collate labels shape matches", tuple(batch["labels"].shape) == (3, BS))
check("collate labels == input_ids", torch.equal(batch["labels"], batch["input_ids"]))

print("=== smoke training run (tiny random model, 2 steps, CPU) ===")
r = subprocess.run(
    [sys.executable, "train.py", "--smoke", "--out", "/tmp/nano_smoke_out"],
    capture_output=True, text=True, cwd="/home/btclord/nano-shatta",
    timeout=600,
)
tail = "\n".join((r.stdout + r.stderr).splitlines()[-12:])
print(tail)
check("smoke run exit 0", r.returncode == 0)
out = Path("/tmp/nano_smoke_out")
check("adapter_config.json saved", (out / "adapter_config.json").exists())
check("adapter weights saved",
      (out / "adapter_model.safetensors").exists() or (out / "adapter_model.bin").exists())
check("tokenizer saved", (out / "tokenizer_config.json").exists())
if (out / "adapter_config.json").exists():
    cfg = json.loads((out / "adapter_config.json").read_text())
    check("LoRA r=16 in config", cfg.get("r") == 16)
    check("task_type CAUSAL_LM", cfg.get("task_type") == "CAUSAL_LM")

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
