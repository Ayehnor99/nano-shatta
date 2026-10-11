"""Pass/fail probe: is GENIUS_ACCESS_TOKEN set and valid?

Never prints the token. Only prints its length (allowed) and whether a
tiny live API call succeeds.
"""
import os
import sys

sys.path.insert(0, "/home/btclord/nano-shatta")

from lyrics_scraper import (
    _load_dotenv,
    get_genius_client,
    with_exponential_backoff,
)

# Load .env before reading the env var (get_genius_client also loads it,
# but the probe reads the var directly first).
_load_dotenv()

# 1. Token present?
token = os.getenv("GENIUS_ACCESS_TOKEN")
if not token:
    print("PROBE FAIL: GENIUS_ACCESS_TOKEN not set (check .env or export)")
    sys.exit(1)
print(f"token present: yes (length {len(token)})")

# 2. Client builds?
try:
    client = get_genius_client()
except Exception as exc:
    print(f"PROBE FAIL: client build error: {type(exc).__name__}: {str(exc)[:150]}")
    sys.exit(1)
print("client built: OK")

# 3. Tiny live call - max_songs=1, cheapest possible probe.
def _probe() -> object:
    return client.search_artist("Shatta Wale", max_songs=1, get_full_info=False)

try:
    artist = _probe()
except Exception as exc:
    msg = str(exc)
    if "401" in msg or "invalid_token" in msg:
        print("PROBE FAIL: Genius rejected the token (401 invalid_token).")
        print("Fix: go to https://genius.com/api-clients, create a NEW client,")
        print("click 'GENERATE ACCESS TOKEN', copy the ACCESS TOKEN (not the")
        print("Client Secret or Client ID), paste it into nano-shatta/.env as:")
        print("  GENIUS_ACCESS_TOKEN=<token>")
        print("Do NOT paste the token into chat.")
        sys.exit(1)
    print(f"PROBE transient error ({type(exc).__name__}), retrying with backoff...")
    artist = with_exponential_backoff(_probe, operation="probe-search")

if artist is None:
    print("PROBE FAIL: search_artist returned None (network or rate limit)")
    sys.exit(1)
print(f"live API OK: artist resolved as '{artist.name}', songs returned: {len(artist.songs)}")
if artist.songs:
    s = artist.songs[0]
    print(f"sample song: {s.title!r} | lyrics chars: {len(s.lyrics or '')}")
print("PROBE PASS")
