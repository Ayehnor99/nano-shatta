# nano-shatta — train a Shatta Wale-style lyrics model on Colab (free T4)

Colab notebook cells — run top to bottom. Free tier T4, ~10–20 min for 3 epochs.

---

## Cell 1 — Setup: clone the repo + install deps

```python
!git clone https://github.com/<YOUR_USERNAME>/nano-shatta.git
%cd nano-shatta
!pip install -q -r requirements-train.txt bitsandbytes
```

> If you don't have a GitHub repo yet: upload the whole `nano-shatta/` folder
> to your Google Drive instead, then:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> %cd /content/drive/MyDrive/nano-shatta
> !pip install -q -r requirements-train.txt bitsandbytes
> ```

---

## Cell 2 — Keep the session alive (prevents idle disconnect)

```python
import threading, time
import IPython.display as ipd

def _keep_alive():
    while True:
        ipd.clear_output(wait=True)
        print("keep-alive", time.strftime("%H:%M:%S"))
        time.sleep(60)

threading.Thread(target=_keep_alive, daemon=True).start()
print("keep-alive running in background — this cell finishes immediately")
```

> This cell **finishes immediately** — the keep-alive runs in a background thread.
> Free tier kicks after ~90 min idle. (During active training the GPU keeps
> the session alive, so keep-alive only matters if you step away mid-run.)

---

## Cell 3 — Verify GPU

```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
!nvidia-smi --query-gpu=memory.total --format=csv
```

> Runtime → Change runtime type → **T4 GPU** (or L4/A100 on Pro).

---

## Cell 4 — Train (LoRA, 4-bit) with live monitoring

Watch it learn in real time. Every 5 steps you get a status line:

```
step   100/170 | loss 2.8431 | lr 1.90e-04 | 8,192 tok/s | elapsed 03:12 | ETA 02:41 | ▁▂▂▃▄▄▅▅▆▇
```

Every `--sample-every` steps (set to 50 below) it prints a **val loss**
and generates a **sample bar** from the mid-training model. Samples are
also saved to `shatta-lora/samples.log`.

```python
!python train.py --model Qwen/Qwen2.5-0.5B \
    --data shatta_clean_v3.txt --out shatta-lora \
    --epochs 3 --batch-size 2 --grad-accum 8 \
    --block-size 512 --lr 2e-4 \
    --lora-r 16 --lora-alpha 32 --save-steps 100 \
    --val-frac 0.05 \
    --sample-every 50 \
    --sample-prompt "[Verse 1]" \
    --sample-tokens 48
```

> Resume after a disconnect: add `--resume` to the command above.
> Tune the cadence: `--sample-every 0` disables sampling, `--val-frac 0` disables val.

---

## Cell 5 — Generate a test bar (sanity check)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    load_in_4bit=True,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained("shatta-lora")
model = PeftModel.from_pretrained(base, "shatta-lora")
model.eval()

prompt = "[Verse 1]\nShatta Wale pon di microphone, "
ids = tok(prompt, return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=80, temperature=0.9, top_p=0.95,
                     do_sample=True, eos_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## Cell 6 — Generate a WHOLE song (structure-aware)

The model writes each section in turn, carrying the song so far into every
prompt so it stays on theme:

```python
!python generate.py --adapter shatta-lora --song \
    --title "Street Crown" \
    --seed-line "Street crown pon mi head, dem cyan tek it from mi" \
    --structure "Intro,Verse 1,Chorus,Verse 2,Chorus,Bridge,Outro" \
    --section-tokens 90 --temperature 0.9 \
    --repetition-penalty 1.25 --no-repeat-ngram 4 \
    --out song.txt
print(open("song.txt").read())
```

- **`--seed-line` steers the theme** — a title alone won't (the model reads
  it as a header, not a topic). The seed opens section 1 and propagates.
- **`--no-repeat-ngram 4`** blocks any repeated 4-word phrase (kills loops)
- **`--repetition-penalty 1.25`** backs it up
- `--section-tokens` controls section length; `--seed 3` reproduces a run
- Colon-style markers the model invents (`Verse 1:`) are trimmed automatically

---

## Cell 7 — Download the adapter

```python
!zip -q -r shatta-lora.zip shatta-lora
files.download("shatta-lora.zip")
```

> The zip contains `adapter_config.json`, `adapter_model.safetensors`,
> and the tokenizer files. ~30–60MB. That's the whole trained model —
> you load it ON TOP of the base Qwen2.5-0.5B anywhere.

---

## Optional — Push checkpoints to Drive (survive session death)

```python
from google.colab import drive
 drive.mount('/content/drive')
!cp -r shatta-lora /content/drive/MyDrive/nano-shatta-backup/
```

Run this after training finishes (and every ~30 min if you want belt-and-braces).
