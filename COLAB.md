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
import time, IPython.display as ipd
while True:
    ipd.clear_output(wait=True)
    print("keep-alive", time.strftime("%H:%M:%S"))
    time.sleep(60)
```

> Run this in a cell, then **leave it running** in a background tab.
> Or use Colab Pro's background execution. Free tier kicks after ~90 min idle.

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

## Cell 4 — Train (LoRA, 4-bit, Qwen2.5-0.5B)

```python
!python train.py \
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
    --save-steps 100
```

Expected: ~226 blocks/epoch on v2, ~6,400 blocks/epoch on v3 (2MB corpus).
Effective batch = 2 × 8 = 16. T4 should do ~3–5 sec/step → **3 epochs ≈ 15–25 min**.

> Resume after a disconnect:
> ```python
> !python train.py --model Qwen/Qwen2.5-0.5B --data shatta_clean_v3.txt \
>     --out shatta-lora --epochs 3 --batch-size 2 --grad-accum 8 --resume
> ```

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

## Cell 6 — Download the adapter (zip)

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
