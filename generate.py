#!/usr/bin/env python3
"""Generate Shatta-style bars or a full structured song from a LoRA adapter.

The base model is read FROM the adapter config (base_model_name_or_path) —
never passed by hand. This makes the vocab-size mismatch bug structurally
impossible: you cannot accidentally generate with a different base than
you trained on.

Single-shot (a verse):
    python generate.py --adapter shatta-lora
    python generate.py --adapter shatta-lora --prompt "[Verse 1] Mi a di realest"

Full structured song (intro/verse/chorus/bridge/outro):
    python generate.py --adapter shatta-lora --song --title "Street Crown"

Steer the theme with an opening line (the model continues from it, and the
song-so-far carries it into every later section):
    python generate.py --adapter shatta-lora --song --title "Street Crown" \\
        --seed-line "Street crown pon mi head, dem cyan tek it from mi"
"""
from __future__ import annotations

import argparse
import re

import torch

# Default song skeleton. The model knows these markers.
DEFAULT_SONG_STRUCTURE = [
    "Intro", "Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Outro",
]

# Matches a whole line that is a section marker, in EITHER style the model
# emits: "[Chorus]" (bracket) or "Verse 1:" (colon). Group 1 = bracket name,
# group 2 = colon name.
MARKER_RE = re.compile(
    r"^\s*(?:\[\s*([^\]]{1,30}?)\s*\]|([A-Za-z][A-Za-z0-9 '\-]{0,25}?)\s*:)\s*$"
)


# ---------------------------------------------------------------------------
# Pure helpers (testable without a model)
# ---------------------------------------------------------------------------

def parse_structure(spec: str | None) -> list[str]:
    """Turn a comma-separated spec into a section list.

    'Verse 1,Chorus,Verse 2,Chorus'  ->  ['Verse 1', 'Chorus', 'Verse 2', 'Chorus']
    None/empty -> DEFAULT_SONG_STRUCTURE
    """
    if not spec:
        return list(DEFAULT_SONG_STRUCTURE)
    parts = [s.strip() for s in spec.split(",")]
    return [p for p in parts if p]


def marker_name(line: str) -> str | None:
    """Return the section name if `line` is a marker line, else None.

    Handles both '[Chorus]' and 'Verse 1:' styles.
    """
    m = MARKER_RE.match(line)
    if not m:
        return None
    return (m.group(1) or m.group(2) or "").strip()


def _same_section(a: str | None, b: str) -> bool:
    if a is None:
        return False
    return a.strip().lower().replace("-", " ") == b.strip().lower().replace("-", " ")


def build_prompt(title: str | None, sections: list[tuple[str, str]],
                 next_section: str, seed_line: str | None = None) -> str:
    """Compose the prompt for the next section.

    title:        optional song title (printed as a header line)
    sections:     already-generated [(name, text), ...]
    next_section: the marker to open, e.g. 'Chorus'
    seed_line:    optional opening line for the FIRST section — the strongest
                  way to steer the theme (the model continues from it)
    """
    blocks: list[str] = []
    if title:
        blocks.append(title)
    for name, text in sections:
        text = text.strip()
        blocks.append(f"[{name}]\n{text}" if text else f"[{name}]")
    opening = f"[{next_section}]"
    if not sections and seed_line:
        opening += f"\n{seed_line.strip()}"
    blocks.append(opening)
    return "\n\n".join(blocks)


def trim_section(text: str, current: str) -> str:
    """Cut a section at the next section marker so it can't bleed onward.

    Catches BOTH marker styles ('[Chorus]' and 'Verse 1:'), so a colon-style
    header invented by the model doesn't leak into the current section.
    Also drops an echoed copy of the current section's own marker.
    """
    out: list[str] = []
    for ln in text.split("\n"):
        name = marker_name(ln)
        if name is not None:
            if _same_section(name, current):
                continue  # echoed our own marker — skip the marker line
            break         # the NEXT section started — stop here
        out.append(ln.rstrip())
    trimmed = "\n".join(out).strip()
    while "\n\n\n" in trimmed:
        trimmed = trimmed.replace("\n\n\n", "\n\n")
    return trimmed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="shatta-lora",
                   help="Path to the trained adapter dir (default: shatta-lora)")
    p.add_argument("--prompt", default="[Verse 1]",
                   help="Prompt for single-shot mode")
    p.add_argument("--song", action="store_true",
                   help="Generate a full structured song (sections)")
    p.add_argument("--title", default=None,
                   help="Song title (song mode; printed as a header)")
    p.add_argument("--seed-line", default=None,
                   help="Opening line for the first section — steers the theme")
    p.add_argument("--structure", default=None,
                   help="Comma-separated sections, e.g. 'Intro,Verse 1,Chorus,Verse 2,Chorus,Outro'")
    p.add_argument("--section-tokens", type=int, default=90,
                   help="Max new tokens per section in song mode")
    p.add_argument("--max-new-tokens", type=int, default=100,
                   help="Max new tokens in single-shot mode")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.25,
                   help="Penalize repeated tokens (stops loops). 1.0 = off")
    p.add_argument("--no-repeat-ngram", type=int, default=4,
                   help="Block repeating any N-gram (0 = off). Kills loops dead.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None,
                   help="Also write the generated text to this file")
    p.add_argument("--no-4bit", action="store_true",
                   help="Load base in bf16 instead of 4-bit (needs more VRAM)")
    p.add_argument("--force-base", default=None,
                   help="OVERRIDE the adapter's recorded base model (dangerous)")
    return p.parse_args()


def resolve_base(adapter_dir: str, override: str | None) -> str:
    """Read the base model id from the adapter's own config."""
    from peft import PeftConfig

    cfg = PeftConfig.from_pretrained(adapter_dir)
    recorded = cfg.base_model_name_or_path
    if override:
        print(f"[warn] --force-base overrides adapter's recorded base!\n"
              f"       recorded: {recorded}\n"
              f"       using:    {override} — expect vocab mismatch if wrong")
        return override
    if not recorded:
        raise RuntimeError(
            f"adapter at {adapter_dir} has no base_model_name_or_path in its config; "
            "pass --force-base to override (at your own risk)"
        )
    return recorded


def load_model(adapter: str, force_base: str | None, no_4bit: bool):
    """Load base (from adapter config) + adapter. Returns (model, tokenizer)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_name = resolve_base(adapter, force_base)
    print(f"[generate] base (from adapter config): {base_name}")
    tok = AutoTokenizer.from_pretrained(adapter)

    kwargs: dict = {"device_map": "auto"}
    if torch.cuda.is_available() and not no_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    print(f"[generate] loading base {base_name} ...")
    base = AutoModelForCausalLM.from_pretrained(base_name, **kwargs)
    print(f"[generate] loading adapter {adapter} ...")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tok


def generate_text(model, tok, prompt: str, max_new_tokens: int,
                  temperature: float, top_p: float,
                  repetition_penalty: float,
                  no_repeat_ngram_size: int = 0) -> str:
    """Generate continuation for `prompt`, returning ONLY the new text."""
    enc = tok(prompt, return_tensors="pt")
    ids = (enc["input_ids"] if isinstance(enc, dict) else enc.input_ids).to(model.device)
    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    out = model.generate(ids, **gen_kwargs)
    full = tok.decode(out[0], skip_special_tokens=True)
    # Strip the prompt so we only return what the model wrote.
    return full[len(prompt):] if full.startswith(prompt) else full


def compose_song(model, tok, args) -> str:
    """Generate a full song section by section."""
    structure = parse_structure(args.structure)
    done: list[tuple[str, str]] = []
    for i, section in enumerate(structure, 1):
        seed = args.seed_line if not done else None
        prompt = build_prompt(args.title, done, section, seed_line=seed)
        raw = generate_text(model, tok, prompt, args.section_tokens,
                            args.temperature, args.top_p,
                            args.repetition_penalty, args.no_repeat_ngram)
        text = trim_section(raw, section)
        if seed and not text.strip().startswith(seed.strip()):
            # keep the user's seed line as the section opener
            text = (seed.strip() + "\n" + text).strip()
        done.append((section, text))
        print(f"  [{i}/{len(structure)}] {section}: {len(text)} chars", flush=True)

    blocks: list[str] = []
    if args.title:
        blocks.append(args.title)
    for name, text in done:
        blocks.append(f"[{name}]\n{text}" if text else f"[{name}]")
    return "\n\n".join(blocks)


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tok = load_model(args.adapter, args.force_base, args.no_4bit)

    if args.song:
        print(f"[song] structure: {' -> '.join(parse_structure(args.structure))}")
        if args.seed_line:
            print(f"[song] seed line: {args.seed_line}")
        text = compose_song(model, tok, args)
    else:
        text = args.prompt + generate_text(
            model, tok, args.prompt, args.max_new_tokens,
            args.temperature, args.top_p,
            args.repetition_penalty, args.no_repeat_ngram)

    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
