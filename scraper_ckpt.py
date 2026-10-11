"""Add per-song checkpointing to fetch_artist_lyrics so a crash never
loses more than one song. Mirrors the existing manifest logic but
saves each song immediately after fetching (before the next song).

This is a standalone patched version; the original module keeps its
API. Use fetch_artist_lyrics_ckpt for resumable, crash-safe scraping.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from lyrics_scraper import (
    clean_lyrics_text,
    get_genius_client,
    sanitize_filename,
    with_exponential_backoff,
)

logger = logging.getLogger(__name__)


def fetch_artist_lyrics_ckpt(
    artist_name: str = "Shatta Wale",
    max_songs: int = 100,
    output_dir: str | Path = "data/raw/lyrics",
    token: Optional[str] = None,
    include_features: bool = True,
) -> List[Dict[str, Any]]:
    """Crash-safe variant: each song is written to disk + manifest before
    the next song is fetched. If the process dies, a rerun resumes from
    the manifest and refetches at most the in-flight song.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Load completed titles (resume support).
    done_titles: set[str] = set()
    done_paths: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done_titles.add(rec.get("title", ""))
                    done_paths.add(rec.get("path", ""))
                except json.JSONDecodeError:
                    continue
        logger.info("Resuming: %d song(s) already saved.", len(done_titles))

    client = get_genius_client(token=token)

    def _search() -> Any:
        return client.search_artist(
            artist_name,
            max_songs=max_songs,
            sort="popularity",
            include_features=include_features,
            get_full_info=False,
            per_page=50,
        )

    artist = with_exponential_backoff(
        _search, max_retries=8, base_delay=5.0, max_delay=120.0,
        operation="search-artist",
    )
    if artist is None:
        logger.error("Artist search failed; nothing to fetch.")
        return []

    songs = list(getattr(artist, "songs", []) or [])[:max_songs]
    logger.info("Found %d song(s) for '%s'.", len(songs), artist_name)

    results: List[Dict[str, Any]] = []
    pending = [s for s in songs if getattr(s, "title", "untitled") not in done_titles]
    logger.info("Pending after resume: %d song(s).", len(pending))

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for song in tqdm(pending, desc="Scraping lyrics", unit="song"):
            title = getattr(song, "title", "untitled")
            raw = song.lyrics or None
            if not raw:
                logger.warning("Skipping '%s': no lyrics.", title)
                continue
            cleaned = clean_lyrics_text(raw)
            if len(cleaned) < 50:
                logger.warning("Skipping '%s': too short after cleaning.", title)
                continue

            fname = sanitize_filename(title) + ".txt"
            fpath = out_dir / fname
            counter = 1
            while fpath.exists() or str(fpath) in done_paths:
                fpath = out_dir / f"{sanitize_filename(title)}_{counter}.txt"
                counter += 1

            # --- CHECKPOINT: write file, flush, append manifest, flush ---
            fpath.write_text(cleaned + "\n", encoding="utf-8")
            record = {
                "title": title,
                "url": getattr(song, "url", ""),
                "path": str(fpath),
                "chars": len(cleaned),
                "primary_artist": getattr(
                    getattr(song, "primary_artist", None), "name", artist_name
                ),
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            os.fsync(manifest.fileno())
            # --------------------------------------------------------

            done_titles.add(title)
            done_paths.add(str(fpath))
            results.append(record)
            time.sleep(0.5)

    logger.info("Saved %d new song(s) to %s", len(results), out_dir)
    return results
