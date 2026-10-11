"""Scrape a list of dancehall/reggae artists into data/raw/<artist_slug>/.

Artists (edit ARTISTS below to add/remove):
  Shatta Wale (already done), Stonebwoy, Patoranking, Alkaline,
  Popcaan, Skillibeng, Vybz Kartel.

Per-artist output dir + manifest => fully resumable per artist.
Runs sequentially to stay under the Genius rate limiter.
"""
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/btclord/nano-shatta")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from scraper_ckpt import fetch_artist_lyrics_ckpt

ARTISTS = [
    ("Stonebwoy", 150),
    ("Patoranking", 150),
    ("Alkaline", 100),
    ("Popcaan", 100),
    ("Skillibeng", 100),
    ("Vybz Kartel", 150),
]

def slug(name: str) -> str:
    return name.lower().replace(" ", "_")

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, max_songs in ARTISTS:
        if only and only.lower() not in name.lower():
            continue
        out = Path("data/raw") / slug(name)
        print(f"\n=== {name} -> {out} (max {max_songs}) ===")
        try:
            res = fetch_artist_lyrics_ckpt(
                artist_name=name,
                max_songs=max_songs,
                output_dir=out,
                include_features=True,
            )
            print(f"[{name}] saved {len(res)} songs")
        except Exception as exc:
            # One artist failing must not kill the rest of the fleet.
            logging.error("[%s] FAILED: %s", name, exc)
        time.sleep(3)  # breather between artists
    print("\nALL ARTISTS DONE")
