[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GH_USER>/nano-shatta/blob/main/nano_shatta_train.ipynb)

# nano-shatta

Train a small **Shatta Wale-style dancehall lyrics model** with LoRA on a free Colab T4 GPU.

**One click to train:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GH_USER>/nano-shatta/blob/main/nano_shatta_train.ipynb)

> Replace `<YOUR_GH_USER>` in the badge URLs above with your GitHub username.

## What it does

Fine-tunes **Qwen2.5-0.5B** (4-bit) with **LoRA** on **938 dancehall songs** (2MB):
Shatta Wale, Stonebwoy, Vybz Kartel, Patoranking, Alkaline, Popcaan, Skillibeng.
Output: a ~50MB adapter that turns the base model into a bar-writer.

## Pipeline

```
Genius API → scrape (per-song, resumable) → clean (homoglyphs, dedup, drop interviews)
         → merge (938 songs,  PAD-separated) → LoRA fine-tune → generate bars
```

## Corpus (`shatta_clean_v3.txt`, 2MB / 938 songs)

| Artist | Songs |
|---|---|
| Shatta Wale | 194 |
| Patoranking | 149 |
| Stonebwoy | 149 |
| Vybz Kartel | 147 |
| Alkaline | 100 |
| Popcaan | 100 |
| Skillibeng | 99 |

Cleaned: 0 Genius junk, 0 Cyrillic homoglyphs, 0 CRLF. Twi orthography
(ɛ ɔ ŋ Ọ ọ) preserved.

> **Rebuild the corpus yourself:** `python scrape_fleet.py && python merge_fleet.py`
> (needs a free Genius token in `.env`). Intermediate corpora (v1/v2, raw scrape)
> are intentionally not committed — they're reproducible from the pipeline.

## Files

| File | Purpose |
|---|---|
| `nano_shatta_train.ipynb` | Colab notebook — the whole training pipeline in 7 cells |
| `train.py` | LoRA fine-tune (4-bit GPU / CPU smoke mode) |
| `generate.py` | Load adapter + base, write bars |
| `lyrics_scraper.py` | Genius scraper (lyricsgenius 3.15.0, 27/27 tests) |
| `scraper_ckpt.py` | Crash-safe scraper (checkpoint per song, resumable) |
| `scrape_fleet.py` | Scrape the 6 neighbor artists |
| `run_scrape.py` | Scrape Shatta Wale |
| `clean.py` | Polish a raw corpus (drop interviews, fix homoglyphs, dedup) |
| `merge.py` / `merge_fleet.py` | Merge scraped songs → v2 / v3 corpus |
| `build_notebook.py` | Regenerate the .ipynb from cell definitions |
| `test_scraper.py` | 27 offline scraper tests |
| `test_train.py` | 22 tests: data pipeline + full CPU smoke training run |
| `test_generate.py` | End-to-end generate test (tiny model + smoke adapter) |
| `probe_token.py` | Pass/fail Genius token check (never prints the value) |
| `check_token_format.py` | Format-only token check (never prints the value) |
| `COLAB.md` | Markdown version of the notebook cells |
| `LICENSE` | MIT |

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt bitsandbytes
cp .env.example .env   # paste your Genius token (https://genius.com/api-clients)
```

## Run everything locally

```bash
python run_scrape.py        # Shatta Wale -> data/raw/lyrics/
python scrape_fleet.py      # 6 neighbors -> data/raw/<artist>/
python merge_fleet.py       # -> shatta_clean_v3.txt
python test_train.py        # 22 tests, no GPU needed
python test_generate.py     # end-to-end generate test
python train.py --smoke     # tiny CPU smoke training run
```

## Train on Colab (free)

Click the badge at the top, or see `COLAB.md`. ~15–25 min on a T4.

## Generate bars (after training)

```bash
python generate.py --adapter shatta-lora --prompt "[Verse 1] Mi a di realest"
```

## License

MIT — see `LICENSE`. Do what you want, just keep the copyright notice.

## Secrets

`.env` is gitignored. Never commit or paste a real API token anywhere.
