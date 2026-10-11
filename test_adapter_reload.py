"""Regression test for the vocab-size mismatch bug.

Trains a smoke adapter end to end (CPU, tiny model), then reloads it with
PeftModel.from_pretrained and asserts the load succeeds with NO size
mismatch — the exact failure that hit Colab with
'RuntimeError: size mismatch for base_model.model.model.embed_tokens.weight'.

Also verifies generate.py's resolve_base() reads the right base id from
the adapter config.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/btclord/nano-shatta")

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")

SMOKE_OUT = Path("/tmp/nano_smoke_out")
SMOKE_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"

print("=== step 1: train smoke adapter (CPU, tiny model) ===")
r = subprocess.run(
    [sys.executable, "train.py", "--smoke", "--out", str(SMOKE_OUT)],
    capture_output=True, text=True, cwd="/home/btclord/nano-shatta", timeout=600,
)
check("smoke train exit 0", r.returncode == 0)
if r.returncode != 0:
    print("\n".join((r.stdout + r.stderr).splitlines()[-15:]))
    sys.exit(1)
check("adapter saved", (SMOKE_OUT / "adapter_config.json").exists())

print("=== step 2: reload adapter with PeftModel.from_pretrained ===")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

cfg = PeftConfig.from_pretrained(str(SMOKE_OUT))
check("adapter config records base", cfg.base_model_name_or_path is not None)
check("recorded base == trained base", cfg.base_model_name_or_path == SMOKE_MODEL)

tok = AutoTokenizer.from_pretrained(str(SMOKE_OUT))
base = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, torch_dtype=torch.float32)

# The exact operation that failed on Colab:
try:
    model = PeftModel.from_pretrained(base, str(SMOKE_OUT))
    check("PeftModel.from_pretrained loads WITHOUT size mismatch", True)
except RuntimeError as exc:
    check("PeftModel.from_pretrained loads WITHOUT size mismatch", False)
    print(f"  FAILURE DETAIL: {exc}")
    sys.exit(1)

# embedding row counts must match
adapter_emb = None
sd = model.state_dict()
for k, v in sd.items():
    if "embed_tokens" in k:
        adapter_emb = v.shape[0]
        break
base_emb = base.get_input_embeddings().num_embeddings
if adapter_emb is not None:
    check(f"embedding rows match ({adapter_emb} == {base_emb})", adapter_emb == base_emb)

# quick generation sanity (tiny model = gibberish, but must not crash)
ids = tok("[Verse 1]", return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=8, do_sample=False)
check("generation runs", out.shape[1] > ids.shape[1])

print("=== step 3: generate.py resolve_base ===")
import importlib
gen = importlib.import_module("generate")
resolved = gen.resolve_base(str(SMOKE_OUT), None)
check("resolve_base returns adapter's recorded base", resolved == SMOKE_MODEL)

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
