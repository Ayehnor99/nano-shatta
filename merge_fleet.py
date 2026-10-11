#!/usr/bin/env python3
"""Merge ALL scraped artist corpora into one training file.

Reads every data/raw/<artist>/*.txt, applies the same normalization as
merge.py (homoglyph fix, whitespace cleanup), dedups globally, and
emits shatta_clean_v3.txt with  song separators.

Also writes a per-artist breakdown so you can see the mix.
"""
import os
import re
from collections import Counter
from pathlib import Path

PAD = '<|' + 'endoftext' + '|>'
RAW = Path('data/raw')
OUT = 'shatta_clean_v3.txt'

HOMOGLYPHS = {
    'е': 'e',   # Cyrillic ie -> latin e
    'ε': 'ɛ',   # Greek epsilon -> Twi epsilon
    'ᴈ': 'I',
    'Ↄ': '',
}

def normalize(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = []
    for l in text.split('\n'):
        s = l.strip()
        s = re.sub(r'\s{2,}', ' ', s)
        for bad, good in HOMOGLYPHS.items():
            if bad in s:
                s = s.replace(bad, good)
        lines.append(s)
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

def fingerprint(text: str) -> str:
    low = re.sub(r'[^a-z0-9]+', '', text.lower())
    return low[:200]

songs = []
seen = set()
per_artist = Counter()
skipped = Counter()

artist_dirs = sorted([d for d in RAW.iterdir() if d.is_dir()])
print(f"found {len(artist_dirs)} artist dirs: {[d.name for d in artist_dirs]}")

for adir in artist_dirs:
    for f in sorted(adir.glob('*.txt')):
        norm = normalize(f.read_text(encoding='utf-8'))
        if len(norm) < 50:
            skipped['tiny'] += 1
            continue
        fp = fingerprint(norm)
        if fp in seen:
            skipped['dup'] += 1
            continue
        seen.add(fp)
        songs.append((adir.name, norm))
        per_artist[adir.name] += 1

# Build output, grouped by artist for readability
out_parts = []
for artist in sorted(per_artist):
    artist_songs = [s for a, s in songs if a == artist]
    out_parts.append((PAD + '\n\n').join(artist_songs))

out = (PAD + '\n\n').join(out_parts) + '\n'
Path(OUT).write_text(out, encoding='utf-8', newline='\n')

print("\n=== fleet merge report ===")
print(f"total songs: {len(songs)}  ->  {OUT}")
print(f"size: {len(out):,} chars ({len(out)/1024:.0f} KB)")
print(f"PAD separators: {out.count(PAD)}")
print(f"skipped: {dict(skipped)}")
print("\nper-artist song counts:")
for a, n in per_artist.most_common():
    print(f"  {a:15s} {n}")

# verify
cyr = re.compile(r'[\u0400-\u04FF]')
check = out
print("\nCyrillic remaining:", sum(1 for l in check.split('\n') if cyr.search(l)))
print("CRLF:", check.count('\r'))
print("ends with newline:", check.endswith('\n'))
for ch in 'ɛɔŋỌọ':
    print(f"  Twi {ch!r}: {check.count(ch)}")
