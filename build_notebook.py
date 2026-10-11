#!/usr/bin/env python3
"""Build nano_shatta_train.ipynb from cell definitions (avoids hand-writing JSON)."""
import json
import sys

OWNER = sys.argv[1] if len(sys.argv) > 1 else "<YOUR_GH_USER>"
REPO = "nano-shatta"
REPO_URL = f"https://github.com/{OWNER}/{REPO}.git"


def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = [
    md(r"""# nano-shatta — train a Shatta Wale-style lyrics model

This notebook fine-tunes **Qwen2.5-0.5B** with **LoRA** on 938 dancehall songs
(2MB corpus, already in this repo as `shatta_clean_v3.txt`).

- **Cost:** free (Colab T4 GPU)
- **Time:** ~15–25 min for 3 epochs
- **Output:** `shatta-lora.zip` (~50MB adapter) you download at the end

**What you get:** a model that writes Shatta-Wale-style bars.

Run the cells **top to bottom**. Each section says what it does and what to expect."""),

    md(r"""## 1 · Setup

Clones this repo into the Colab VM and installs dependencies.
**Edit `REPO_URL` below only if you forked this repo** — otherwise leave it."""),

    code(r"""REPO_URL = """ + repr(REPO_URL) + r"""  # edit only if you forked this repo
!git clone {REPO_URL}
%cd nano-shatta
!pip install -q -r requirements-train.txt bitsandbytes
print("setup done")"""),

    md(r"""## 2 · Keep-alive (prevents idle disconnect)

Colab kills free sessions after ~90 min idle. Run this cell and **leave it
running in a background tab** — it pings every 60s so training never stops."""),

    code(r"""import time
import IPython.display as ipd
while True:
    ipd.clear_output(wait=True)
    print("keep-alive", time.strftime("%H:%M:%S"))
    time.sleep(60)"""),

    md(r"""## 3 · Check the GPU

You must see a GPU name (T4 / L4 / A100). If it says `NO GPU`:
**Runtime → Change runtime type → T4 GPU**, then re-run this cell."""),

    code(r"""import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    print("NO GPU — go to Runtime -> Change runtime type -> T4 GPU")"""),

    md(r"""## 4 · Train (LoRA, 4-bit)

- 3 epochs over ~900 blocks of 512 tokens (effective batch 16)
- Loss should drop from ~4.0 toward ~2.0 or lower
- Checkpoints save every 100 steps to `shatta-lora/checkpoints/`

If the session dies mid-training, re-run this cell with `--resume` appended."""),

    code(r"""!python train.py \
    --model Qwen/Qwen2.5-0.5B \
    --data shatta_clean_v3.txt \
    --out shatta-lora \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --block-size 512 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --save-steps 100"""),

    md(r"""## 5 · Generate a test bar

Loads the freshly trained adapter and writes a verse.
Edit the `--prompt` to try your own opening line."""),

    code(r"""!python generate.py --adapter shatta-lora \
    --prompt "[Verse 1] Mi a di realest" \
    --max-new-tokens 100 --seed 7"""),

    md(r"""## 6 · Download the adapter

Zips `shatta-lora/` and downloads it. It contains `adapter_config.json`,
`adapter_model.safetensors`, and the tokenizer — the whole trained model,
~50MB. Load it on top of the base Qwen2.5-0.5B anywhere (see README)."""),

    code(r"""!zip -q -r shatta-lora.zip shatta-lora
from google.colab import files
files.download("shatta-lora.zip")"""),

    md(r"""## 7 · (Optional) Back up to Google Drive

If the session dies mid-training, Colab's VM is wiped. Mount Drive and copy
the output so a `--resume` can pick up where you left off."""),

    code(r"""from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/nano-shatta-backup
!cp -r shatta-lora /content/drive/MyDrive/nano-shatta-backup/
print("backed up to Drive")"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"name": "nano-shatta train", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = "nano_shatta_train.ipynb"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
    fh.write("\n")

# validate round-trip
with open(out, encoding="utf-8") as fh:
    loaded = json.load(fh)
print(f"wrote {out}: {len(loaded['cells'])} cells, "
      f"{sum(1 for c in loaded['cells'] if c['cell_type']=='code')} code, "
      f"{sum(1 for c in loaded['cells'] if c['cell_type']=='markdown')} markdown")
