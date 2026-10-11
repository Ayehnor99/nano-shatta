"""Filter data/raw/lyrics to true Shatta Wale primary-artist songs only.

Genius search_artist(include_features=True) returns features too, but
labels primary_artist as the queried artist. The reliable signal is the
song URL slug: genius.com/<Primary-Artist>-<song>-lyrics. If the slug's
artist part doesn't contain "shatta-wale", it's a feature/mislabel.
"""
import json
import re
import os
from urllib.parse import urlparse

MANIFEST = 'data/raw/lyrics/manifest.jsonl'

def slug_artist(url: str) -> str:
    path = urlparse(url).path.strip('/')
    # /Sarkodie-mgye-wo-girl-lyrics -> sarkodie
    path = re.sub(r'-lyrics$', '', path)
    return path.split('-')[0].lower()

def is_shatta(url: str) -> bool:
    """True if the Genius URL slug belongs to Shatta Wale.

    lyricsgenius slugifies artist names by replacing spaces with hyphens,
    so 'Shatta Wale' -> 'shatta-wale'. The slug's artist part is everything
    before the FIRST hyphen that starts the song title -- but titles also
    contain hyphens. Robust approach: the artist part is the longest
    leading run of tokens that, joined, equals a known Shatta Wale alias.
    """
    path = urlparse(url).path.strip('/')
    path = re.sub(r'-lyrics$', '', path)
    aliases = {'shatta-wale', 'shattawale', 'shatta_wale', 'charles-neshie'}
    return any(path == a or path.startswith(a + '-') for a in aliases)

kept, dropped = [], []
for line in open(MANIFEST, encoding='utf-8'):
    rec = json.loads(line)
    if is_shatta(rec['url']):
        kept.append(rec)
    else:
        dropped.append((rec['title'], slug_artist(rec['url']), rec['path']))

print(f"kept (Shatta primary): {len(kept)}")
print(f"dropped (features/mislabels): {len(dropped)}")
for title, artist, path in dropped:
    if os.path.exists(path):
        os.remove(path)
    print(f"  removed: {title!r} (slug artist: {artist})")

# Rewrite manifest
with open(MANIFEST, 'w', encoding='utf-8') as fh:
    for rec in kept:
        fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
print("manifest rewritten")
