#!/usr/bin/env python3
"""Merge scraped Genius lyrics + existing corpus into shatta_clean_v2.txt.

Pipeline:
  1. Load each scraped song file, apply homoglyph fix (Cyrillic е->e,
     Greek ε->ɛ) + whitespace cleanup (same rules as clean.py).
  2. Global near-dedup: drop songs whose normalized text is an exact
     duplicate of one already seen (across ALL sources).
  3. Load shatta_clean.txt (already polished), split on  song
     separators, dedup against scraped songs too.
  4. Emit shatta_clean_v2.txt: every song separated by  PAD,
     blank line between songs, LF endings, trailing newline.
"""
import os
import re
from collections import Counter

PAD = '<|' + 'endoftext' + '|>'
LYRICS_DIR = 'data/raw/lyrics'
OLD_CLEAN = 'shatta_clean.txt'
OUT = 'shatta_clean_v2.txt'

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
    # collapse 3+ blank lines
    text = '\n'.join(lines)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text

def fingerprint(text: str) -> str:
    """For dedup: lowercase, strip non-alphanumerics, first 200 chars."""
    low = re.sub(r'[^a-z0-9]+', '', text.lower())
    return low[:200]

songs = []          # list of normalized song texts
seen_fps = set()
stats = Counter()

# 1. Scraped songs
for f in sorted(os.listdir(LYRICS_DIR)):
    if not f.endswith('.txt'):
        continue
    raw = open(os.path.join(LYRICS_DIR, f), encoding='utf-8').read()
    norm = normalize(raw)
    if len(norm) < 50:
        stats['skipped_tiny'] += 1
        continue
    fp = fingerprint(norm)
    if fp in seen_fps:
        stats['skipped_dup_scrape'] += 1
        continue
    seen_fps.add(fp)
    songs.append(norm)
    stats['from_scrape'] += 1

# 2. Existing polished corpus
old = open(OLD_CLEAN, encoding='utf-8').read()
for seg in old.split(PAD):
    norm = normalize(seg)
    if len(norm) < 50:
        continue
    fp = fingerprint(norm)
    if fp in seen_fps:
        stats['skipped_dup_old'] += 1
        continue
    seen_fps.add(fp)
    songs.append(norm)
    stats['from_old'] += 1

# 3. Write
out = (PAD + '\n\n').join(songs) + '\n'
open(OUT, 'w', encoding='utf-8', newline='\n').write(out)

print("=== merge report ===")
print(f"songs in v2: {len(songs)} (scraped {stats['from_scrape']} + old {stats['from_old']})")
print(f"skipped: tiny={stats['skipped_tiny']} dup_scrape={stats['skipped_dup_scrape']} dup_old={stats['skipped_dup_old']}")
print(f"size: {len(out):,} chars ({len(out)/1024:.0f} KB)")
print(f"PAD separators: {out.count(PAD)}")

# verify
check = open(OUT, encoding='utf-8').read()
cyr = re.compile(r'[\u0400-\u04FF]')
print("Cyrillic remaining:", sum(1 for l in check.split('\n') if cyr.search(l)))
print("CRLF:", check.count('\r'))
print("ends with newline:", check.endswith('\n'))
for ch in 'ɛɔŋỌọ':
    print(f"  Twi {ch!r}: {check.count(ch)}")
