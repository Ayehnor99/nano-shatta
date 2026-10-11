#!/usr/bin/env python3
"""Polish shatta_pretrain_refined.txt -> shatta_clean.txt

1. Drop 3 podcast/interview segments (seg 192, 193, 194 of 196).
2. CRLF -> LF.
3. Homoglyph normalization: Cyrillic е->e, Greek ε->ɛ, ᴈ->I, Ↄ->
   (keep legit Twi ɛ ɔ ŋ Ọ ọ Ɛ Ɔ Ŋ).
4. Whitespace: strip line edges, collapse double spaces.
5. Collapse consecutive identical lines: keep at most 2 in a row.
6. Preserve  song separators, structure markers,
   blank-line song framing, trailing newline.
"""
import re
from collections import Counter

SRC = 'shatta_pretrain_refined.txt'
DST = 'shatta_clean.txt'
PAD = '<|' + 'endoftext' + '|>'
DROP_SEGMENTS = {192, 193, 194}  # podcast/interview transcripts

# --- homoglyph map: only known-bad chars ---
HOMOGLYPHS = {
    'е': 'e',   # Cyrillic ie -> latin e (111 lines)
    'ε': 'ɛ',   # Greek epsilon -> Twi epsilon
    'ᴈ': 'I',   # small capital I junk
    'Ↄ': '',    # turned c junk
}

raw = open(SRC, encoding='utf-8').read().replace('\r\n', '\n')
segments = raw.split(PAD)
# 196 segments: text before 1st PAD + one per PAD (last may be trailing junk)
assert len(segments) == 196, len(segments)

stats = Counter()
kept_segments = []
for si, seg in enumerate(segments):
    if si in DROP_SEGMENTS:
        stats['segments_dropped'] += 1
        stats['chars_dropped'] += len(seg)
        continue
    lines = seg.split('\n')
    out = []
    prev = None
    run = 0
    for l in lines:
        s = l.strip()
        # whitespace cleanup
        s = re.sub(r'\s{2,}', ' ', s)
        # homoglyphs
        for bad, good in HOMOGLYPHS.items():
            if bad in s:
                stats['homoglyph_' + repr(bad)] += s.count(bad)
                s = s.replace(bad, good)
        # blank lines: keep as framing, no dedup
        if not s:
            out.append('')
            prev, run = None, 0
            continue
        # collapse consecutive identical lines, keep max 2
        if s == prev:
            run += 1
            if run > 2:
                stats['consecutive_dups_dropped'] += 1
                continue
        else:
            prev, run = s, 1
        out.append(s)
    kept_segments.append('\n'.join(out))

# reassemble: PAD between segments, blank line around PAD (as original framing)
clean = (PAD + '\n\n').join(kept_segments)
if not clean.endswith('\n'):
    clean += '\n'

open(DST, 'w', encoding='utf-8', newline='\n').write(clean)

# ---- verify ----
orig_lines = [l for l in raw.split('\n') if l.strip() and l.strip() != PAD]
clean_raw = open(DST, encoding='utf-8').read()
clean_lines = [l for l in clean_raw.split('\n') if l.strip() and l.strip() != PAD]
clean_segs = [s for s in clean_raw.split(PAD) if s.strip()]

print("=== polish report ===")
print(f"segments: 196 -> {len(clean_segs)} (dropped {stats['segments_dropped']})")
print(f"size: {len(raw)} -> {len(clean_raw)} chars ({len(raw)-len(clean_raw)} dropped, interview segs: {stats['chars_dropped']})")
print(f"non-pad lines: {len(orig_lines)} -> {len(clean_lines)}")
print(f"homoglyphs fixed: {dict((k,v) for k,v in stats.items() if k.startswith('homoglyph'))}")
print(f"consecutive dups collapsed: {stats['consecutive_dups_dropped']}")
print("CRLF remaining:", clean_raw.count('\r'))
cyr = re.compile(r'[\u0400-\u04FF]')
print("Cyrillic remaining:", sum(1 for l in clean_lines if cyr.search(l)))
print("Greek epsilon remaining:", sum(1 for l in clean_lines if 'ε' in l))
print("lines w/ double space:", sum(1 for l in clean_lines if '  ' in l))
print("PAD count:", clean_raw.count(PAD), "| ends with newline:", clean_raw.endswith('\n'))
print("interview markers remaining:", sum(clean_raw.lower().count(m) for m in ['podcast', 'welcome back to the show', 'thank you for having me']))
# Twi orthography preserved?
for ch in 'ɛɔŋỌọ':
    print(f"  Twi {ch!r} count: {clean_raw.count(ch)}")
# consecutive runs now?
worst = 0
prev, run = None, 0
for l in clean_lines:
    s = l.strip()
    if s == prev:
        run += 1
        worst = max(worst, run)
    else:
        prev, run = s, 1
print("worst consecutive run now:", worst)
print("first 3 clean lines:", clean_lines[:3])
print("sample verse:", [l for l in clean_lines if l.startswith('A trick')][0])
