"""Format-only check of the .env Genius token. NEVER prints the value.

Answers: is it present, and is it locally well-formed? A clean format
with a server-side 401 means the token itself is wrong/revoked.
"""
import os
import re
import sys

sys.path.insert(0, "/home/btclord/nano-shatta")
from lyrics_scraper import _load_dotenv

_load_dotenv()
t = os.getenv("GENIUS_ACCESS_TOKEN") or ""

print("present:", bool(t))
print("length:", len(t))
print("leading/trailing whitespace:", t != t.strip())
print("internal whitespace/newline:", bool(re.search(r"\s", t)))
print("wrapped in quotes:", len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0])
print("charset A-Za-z0-9_- only:", bool(re.fullmatch(r"[A-Za-z0-9_\-]+", t)))

ok = (
    t
    and t == t.strip()
    and not re.search(r"\s", t)
    and re.fullmatch(r"[A-Za-z0-9_\-]+", t)
)
print("format:", "CLEAN" if ok else "MALFORMED (fix .env: no quotes, no spaces)")
