"""
Module 1: Genius Lyrics Scraper for Shatta Wale.

Uses the `lyricsgenius` library to fetch song lyrics for a given artist,
with fault tolerance for rate limits (HTTP 429) and transient network errors.

Typical usage:
    from lyrics_scraper import fetch_artist_lyrics
    fetch_artist_lyrics(max_songs=100, output_dir="data/raw/lyrics")

Environment:
    # Genius API token (get one free at https://genius.com/api-clients)
    # Set GENIUS_ACCESS_TOKEN in your shell or in a .env file (gitignored).
    # NEVER hardcode a real token in source.

"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from tqdm import tqdm

# Type variable for the generic retry wrapper.
T = TypeVar("T")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise patterns observed in raw Genius lyrics dumps.
# ---------------------------------------------------------------------------

# Section headers like [Verse 1], [Chorus], [Intro], [Bridge], [Hook],
# [Pre-Chorus], [Post-Chorus], [Outro], [Interlude], [Skit], [Refrain], etc.
SECTION_HEADER_RE = re.compile(
    r"^\s*\[(?:intro|outro|verse|chorus|bridge|hook|pre-chorus|post-chorus|"
    r"interlude|refrain|skit|break|drop|riff|instrumental|spoken|solo|"
    r"dialogue)[^\]]*\]\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Any leftover bracketed annotation, e.g. "[Repeat x2]".
ANY_BRACKET_RE = re.compile(r"\[[^\]\n]{0,80}\]")

# "You might also like" interstitials injected by Genius web embeds.
YOU_MIGHT_ALSO_LIKE_RE = re.compile(
    r"^.*You might also like.*$", re.IGNORECASE | re.MULTILINE
)

# Trailing "Embed" widgets / counters, e.g. "123Embed" or a lone "Embed" line.
EMBED_RE = re.compile(r"^\s*\d*\s*Embed\s*$", re.IGNORECASE | re.MULTILINE)

# Contributor / translation headers, e.g. "12 ContributorsShatta Wale - ... Lyrics"
# or "Shatta Wale Lyrics".
CONTRIBUTORS_RE = re.compile(
    r"^\s*\d*\s*Contributors.*Lyrics\s*$", re.IGNORECASE | re.MULTILINE
)

# Repeated blank lines collapsed later.
MULTI_BLANK_RE = re.compile(r"\n{3,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_filename(name: str, max_len: int = 120) -> str:
    """Convert a song title into a safe filename stem.

    Args:
        name: Original song title.
        max_len: Maximum length of the returned stem.

    Returns:
        Filesystem-safe string (no <>:\"/\\|?*), spaces -> underscores.
    """
    # Replace illegal characters with underscore.
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()
    safe = re.sub(r"\s+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_.")
    if not safe:
        safe = "untitled"
    return safe[:max_len]


def clean_lyrics_text(raw: str) -> str:
    """Strip Genius web noise while preserving lyrical content.

    Removes section headers ([Verse 1], [Chorus], ...), "Embed" markers,
    "You might also like" lines, and contributor headers. Collapses
    whitespace but does NOT lowercase or spell-correct (Pidgin preserved).

    Args:
        raw: Raw lyrics string from the Genius API.

    Returns:
        Cleaned lyrics string.
    """
    if not raw:
        return ""

    text = raw

    # Drop contributor header lines first (they sit at the very top).
    text = CONTRIBUTORS_RE.sub("", text)

    # Drop the Genius promo block anchored at "You might also like":
    # the marker line plus everything up to the next section header or
    # first real lyric line is web chrome ("See X Live", "Get tickets...",
    # "More on Genius", "Translations", etc.). Run this BEFORE the
    # single-line regex below, otherwise the regex eats the marker
    # before we can use it as an anchor.
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if "you might also like" in line.lower():
            # Keep the lines before the marker.
            kept = lines[:idx]
            # Skip the marker plus up to 6 promo lines, or stop early at
            # a section header / long lyric line (signals real content).
            j = idx + 1
            skipped = 0
            while j < len(lines) and skipped < 12:
                nxt = lines[j].strip()
                if not nxt:
                    j += 1
                    continue  # blank lines don't count against the promo budget
                if SECTION_HEADER_RE.match(nxt):
                    break
                # Known promo chrome phrases are cut regardless of length.
                low = nxt.lower()
                if low.startswith(("see ", "get tickets", "more on genius",
                                   "buy tickets", "translations", "embed",
                                   "share", "about", "lyrics provided")):
                    j += 1
                    skipped += 1
                    continue
                # A line longer than 40 chars that isn't known chrome is
                # real lyrical content.
                if len(nxt) > 40:
                    break
                # Unknown short line: treat as start of real content.
                break
            kept.extend(lines[j:])
            text = "\n".join(kept)
            break

    # Belt-and-braces: drop any stray "You might also like" line that
    # survived (e.g. mid-song annotations).
    text = YOU_MIGHT_ALSO_LIKE_RE.sub("", text)

    # Drop Embed widget lines.
    text = EMBED_RE.sub("", text)

    # Remove full-line section headers. We replace with a blank line so
    # verses stay separated instead of being glued together.
    text = SECTION_HEADER_RE.sub("\n", text)

    # Remove any remaining short bracketed annotations inline.
    text = ANY_BRACKET_RE.sub("", text)

    # Normalize line endings and strip trailing spaces per line.
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]

    # Drop lines that became empty noise (e.g. lone "Embed" without digits).
    cleaned_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if stripped.lower() == "embed":
            continue
        if "you might also like" in stripped.lower():
            continue
        cleaned_lines.append(line.strip())

    text = "\n".join(cleaned_lines)

    # Collapse 3+ newlines into 2, strip leading/trailing blank lines.
    text = MULTI_BLANK_RE.sub("\n\n", text).strip()

    return text


def with_exponential_backoff(
    fn: Callable[[], T],
    max_retries: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    operation: str = "genius-request",
) -> Optional[T]:
    """Run `fn` with exponential backoff on rate limits / transient errors.

    Retries on HTTP 429, 5xx, timeouts, and connection errors. Gives up
    after `max_retries` attempts and returns None (caller decides to skip).

    Args:
        fn: Zero-argument callable performing the request.
        max_retries: Maximum retry attempts (excluding the first try).
        base_delay: Initial sleep in seconds; doubled each retry.
        max_delay: Upper bound for sleep in seconds.
        operation: Human-readable label for log messages.

    Returns:
        Return value of `fn`, or None if all retries are exhausted.
    """
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - must catch transport errors broadly
            msg = str(exc)
            # Resolve the HTTP status from whichever error style we got:
            #  - requests HTTPError carries .response.status_code
            #  - lyricsgenius raises HTTPError(status, description) positionally
            #    (status lands in exc.args[0], .response is None)
            #  - lyricsgenius raises AssertionError("Unexpected response
            #    status code: NNN. ...") for non-200 API responses
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is None and exc.args and isinstance(exc.args[0], int):
                status = exc.args[0]
            if status is None:
                m = re.search(r"status code: (\d{3})", msg)
                if m:
                    status = int(m.group(1))

            if status is not None:
                # Known status: retry only transient ones. 4xx (except 429)
                # are permanent -- never retry a 401/403/404.
                retryable = status in (429, 500, 502, 503, 504)
            else:
                # Transport-level error (no HTTP status). Match specific
                # transient phrases, case-insensitively. Do NOT match bare
                # "connection": lyricsgenius AssertionError messages embed
                # response headers containing "Connection: keep-alive",
                # which made permanent 401s look retryable (live bug).
                low = msg.lower()
                retryable = any(
                    t in low
                    for t in (
                        "429", "rate limit", "503", "502",
                        "timeout", "timed out",
                        "connection refused", "connection reset",
                        "connection aborted", "connection error",
                        "temporary failure",
                    )
                )

            if not retryable or attempt >= max_retries:
                logger.error(
                    "[%s] failed after %d attempt(s): %s",
                    operation, attempt + 1, exc,
                )
                return None

            # Respect Retry-After header when present.
            resp = getattr(exc, "response", None)
            retry_after = getattr(resp, "headers", {}).get("Retry-After", None) \
                if resp is not None else None
            try:
                sleep_for = float(retry_after) if retry_after else delay
            except (TypeError, ValueError):
                sleep_for = delay
            sleep_for = min(sleep_for, max_delay)

            logger.warning(
                "[%s] attempt %d/%d failed (%s). Retrying in %.1fs...",
                operation, attempt + 1, max_retries + 1, exc, sleep_for,
            )
            time.sleep(sleep_for)
            delay = min(delay * 2, max_delay)
    return None


def _load_dotenv() -> None:
    """Load .env from cwd if python-dotenv is installed (best-effort).

    Never fatal: if dotenv isn't installed or no .env exists, we just
    fall back to whatever is already in os.environ.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def get_genius_client(
    token: Optional[str] = None,
    timeout: int = 15,
    retries: int = 3,
    sleep_time: float = 1.5,
) -> Any:
    """Build an authenticated `lyricsgenius.Genius` client.

    Args:
        token: Genius Client Access Token. Defaults to GENIUS_ACCESS_TOKEN env.
        timeout: HTTP timeout in seconds per request.
        retries: Internal retries performed by lyricsgenius itself.
        sleep_time: Pause between API calls to stay under rate limits.

    Returns:
        Configured Genius client instance.

    Raises:
        RuntimeError: If no token is available or lyricsgenius is missing.
        ImportError: If `lyricsgenius` is not installed.
    """
    try:
        from lyricsgenius import Genius
    except ImportError as exc:
        raise ImportError(
            "The 'lyricsgenius' package is required. "
            "Install with: pip install -r requirements.txt"
        ) from exc

    _load_dotenv()
    token = token or os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "GENIUS_ACCESS_TOKEN is not set. "
            "Export it or place it in a .env file (see .env.example)."
        )

    client = Genius(
        token,
        timeout=timeout,
        retries=retries,
        sleep_time=sleep_time,
        remove_section_headers=False,  # we clean headers ourselves
        skip_non_songs=True,
        excluded_terms=["(Remix)", "(Live)"],
        per_page=5,  # lyricsgenius 3.x caps constructor per_page at 5
    )
    return client


def fetch_artist_lyrics(
    artist_name: str = "Shatta Wale",
    max_songs: int = 100,
    output_dir: str | Path = "data/raw/lyrics",
    token: Optional[str] = None,
    include_features: bool = True,
) -> List[Dict[str, Any]]:
    """Fetch up to `max_songs` lyrics for `artist_name` and persist raw output.

    Fault tolerance:
      * Artist search is wrapped in exponential backoff (handles 429).
      * Per-song lyric fetches are individually guarded so one failure
        does not abort the whole run.
      * Progress is checkpointed to `manifest.jsonl` so reruns skip songs
        that were already saved.

    Args:
        artist_name: Artist to query on Genius (default "Shatta Wale").
        max_songs: Upper bound on songs to fetch.
        output_dir: Directory for per-song .txt files + manifest.jsonl.
        token: Optional Genius token (falls back to env var).
        include_features: If True (default), include features. The
            Artist.add_song check matches the artist against the song's
            featured_artists list when this is True.

    Returns:
        List of dicts with keys: title, url, path, chars, primary_artist.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    # Load already-fetched titles so reruns are resumable.
    seen_titles: set[str] = set()
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    seen_titles.add(json.loads(line).get("title", ""))
                except json.JSONDecodeError:
                    continue
        logger.info("Resuming: %d song(s) already in manifest.", len(seen_titles))

    client = get_genius_client(token=token)

    logger.info("Searching Genius for artist '%s' ...", artist_name)

    def _search() -> Any:
        # lyricsgenius 3.x: search_artist fetches each song's lyrics eagerly
        # (one extra web request per song inside the call). We pass
        # get_full_info=False to skip a redundant per-song API call and
        # raise the library's internal retry count for transient errors.
        return client.search_artist(
            artist_name,
            max_songs=max_songs,
            sort="popularity",
            include_features=include_features,
            get_full_info=False,
            per_page=50,  # artist_songs endpoint allows up to 50/page
        )

    artist = with_exponential_backoff(
        _search, max_retries=8, base_delay=5.0, max_delay=120.0,
        operation="search-artist",
    )
    if artist is None:
        logger.error("Artist search failed; nothing to fetch.")
        return []

    # search_artist returns an Artist with .songs already populated
    # (lyrics fetched eagerly inside the call in lyricsgenius 3.x).
    songs = list(getattr(artist, "songs", []) or [])[:max_songs]
    logger.info("Found %d song(s) for '%s'.", len(songs), artist_name)

    results: List[Dict[str, Any]] = []

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for song in tqdm(songs, desc="Scraping lyrics", unit="song"):
            title = getattr(song, "title", "untitled")
            if title in seen_titles:
                continue

            # lyricsgenius 3.x fetches lyrics eagerly during search_artist;
            # song.lyrics is a plain str attribute by the time we get here.
            raw_lyrics = song.lyrics or None
            if not raw_lyrics:
                logger.warning("Skipping '%s': no lyrics retrieved.", title)
                continue

            cleaned = clean_lyrics_text(raw_lyrics)
            if len(cleaned) < 50:
                logger.warning("Skipping '%s': lyrics too short after cleaning.", title)
                continue

            fname = sanitize_filename(title) + ".txt"
            fpath = out_dir / fname
            # Avoid collisions from sanitization (e.g. duplicate stems).
            counter = 1
            while fpath.exists():
                fpath = out_dir / f"{sanitize_filename(title)}_{counter}.txt"
                counter += 1

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
            seen_titles.add(title)
            results.append(record)

            # Gentle pacing. The library itself sleeps `sleep_time` (1.5s)
            # per request; this extra beat is for the per-song page scrape
            # and keeps us comfortably under the Genius rate limiter.
            time.sleep(0.5)

    logger.info("Saved %d new song(s) to %s", len(results), out_dir)
    return results
