"""Run the crash-safe Shatta Wale scrape into data/raw/lyrics/."""
import logging
import sys

sys.path.insert(0, "/home/btclord/nano-shatta")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from scraper_ckpt import fetch_artist_lyrics_ckpt

results = fetch_artist_lyrics_ckpt(
    artist_name="Shatta Wale",
    max_songs=200,
    output_dir="data/raw/lyrics",
    include_features=True,
)
print(f"\nDONE: {len(results)} songs saved")
