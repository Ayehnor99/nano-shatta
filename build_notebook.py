#!/usr/bin/env python3
"""Build nano_shatta_train.ipynb from cell definitions (avoids hand-writing JSON)."""
import json

OWNER = "Ayehnor99"
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
(If you forked this repo, edit `REPO_URL` below to point at your fork.)"""),

    code(r"""REPO_URL = """ + repr(REPO_URL) + r"""  # edit only if you forked this repo
!git clone {REPO_URL}
%cd nano-shatta
!pip install -q -r requirements-train.txt bitsandbytes
print("setup done")"""),

    md(r"""## 2 · Keep-alive (prevents idle disconnect)

Colab kills free sessions after ~90 min idle. This cell starts a keep-alive
in a **background thread** and finishes immediately — run it and move on.
(During active training the GPU keeps the session alive; keep-alive only
matters if you step away mid-run.)"""),

    code(r"""import threading, time
    import IPython.display as ipd

    def _keep_alive():
        while True:
            ipd.clear_output(wait=True)
            print("keep-alive", time.strftime("%H:%M:%S"))
            time.sleep(60)

    threading.Thread(target=_keep_alive, daemon=True).start()
    print("keep-alive running in background — this cell finishes immediately")"""),

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

    md(r"""## 4 · Train (LoRA, 4-bit) — with LIVE monitoring

You'll see a one-line status every 5 steps with loss, lr, tok/s, elapsed,
ETA, and a live **sparkline** of the loss curve:

```
step   100/170 | loss 2.8431 | lr 1.90e-04 | 8,192 tok/s | elapsed 03:12 | ETA 02:41 | ▁▂▂▃▄▄▅▅▆▇
```

Every `--sample-every` steps (default 200) it also prints a **val loss**
and generates a **sample bar** from the mid-training model — you can
literally watch it learn to write Shatta-style lyrics. Samples are also
appended to `shatta-lora/samples.log`.

- 3 epochs over ~900 blocks of 512 tokens (effective batch 16)
- 5% of blocks held out for val loss
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
    --save-steps 100 \
    --val-frac 0.05 \
    --sample-every 50 \
    --sample-prompt "[Verse 1]" \
    --sample-tokens 48"""),

    md(r"""## 5 · Generate a test bar

Loads the freshly trained adapter and writes a verse.
Edit the `--prompt` to try your own opening line."""),

    code(r"""!python generate.py --adapter shatta-lora \
    --prompt "[Verse 1] Mi a di realest" \
    --max-new-tokens 100 --seed 7"""),

    md(r"""## 6 · Download the adapter

Zips `shatta-lora/` and downloads it. It contains `adapter_config.json`,
`adapter_model.safetensors`, and the tokenizer files — the whole trained model,
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

# validate round-trip + no placeholder left
with open(out, encoding="utf-8") as fh:
    loaded = json.load(fh)
raw = json.dumps(loaded)
assert "<YOUR_GH_USER>" not in raw, "placeholder still in notebook!"
assert "Ayehnor99" in raw, "username not in notebook!"
print(f"wrote {out}: {len(loaded['cells'])} cells, "
      f"{sum(1 for c in loaded['cells'] if c['cell_type']=='code')} code, "
      f"{sum(1 for c in loaded['cells'] if c['cell_type']=='markdown')} markdown")
print("badge URL points at: Ayehnor99/nano-shatta")
