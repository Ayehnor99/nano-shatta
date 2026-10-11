"""Full test suite for lyrics_scraper.py - no network, no token needed."""
import sys
sys.path.insert(0, "/home/btclord/nano-shatta")

import requests.exceptions as rex
import lyrics_scraper as ls

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")

print("=== clean_lyrics_text ===")
raw = (
    "7 ContributorsShatta Wale - Amount Lyrics\n"
    "[Verse 1]\n"
    "Oh yeah Amount\n"
    "How much be your amount?\n"
    "You might also like\n"
    "See Shatta Wale Live\n"
    "Get tickets for your next Shatta Wale show\n"
    "More on Genius\n"
    "[Chorus]\n"
    "Yeah.\n"
    "Yeah.\n"
    "123Embed\n"
)
out = ls.clean_lyrics_text(raw)
print("  output:", repr(out))
check("contributors header gone", "Contributors" not in out)
check("section headers gone", "[Verse 1]" not in out and "[Chorus]" not in out)
check("Embed gone", "Embed" not in out)
check("'You might also like' gone", "You might also like" not in out)
check("promo 'See X Live' gone", "See Shatta Wale Live" not in out)
check("promo 'Get tickets' gone", "Get tickets" not in out)
check("promo 'More on Genius' gone", "More on Genius" not in out)
check("lyrics kept", "Oh yeah Amount" in out and "How much be your amount?" in out)
check("chorus kept", "Yeah." in out)
check("empty input -> empty", ls.clean_lyrics_text("") == "")

# CRLF + trailing spaces
out2 = ls.clean_lyrics_text("[Intro]\r\nWake up  \r\nDem fraid\r\n")
check("CRLF normalized", "\r" not in out2)
check("trailing spaces stripped", "Wake up\n" in out2 or "Wake up" in out2)

# inline [Repeat x2] annotation removed
out3 = ls.clean_lyrics_text("[Chorus]\nBig lie dat [Repeat x2]\n")
check("inline bracket annotation removed", "[Repeat x2]" not in out3)
check("lyric text around annotation kept", "Big lie dat" in out3)

print("=== sanitize_filename ===")
check("illegal chars replaced", ls.sanitize_filename('a/b:c*d?e"f<g>h|i') == "a_b_c_d_e_f_g_h_i")
check("spaces -> underscores", ls.sanitize_filename("My Song Title") == "My_Song_Title")
check("blank -> untitled", ls.sanitize_filename("   ") == "untitled")
check("truncated to max_len", len(ls.sanitize_filename("x" * 200)) == 120)
check("underscore collapse", ls.sanitize_filename("a   b") == "a_b")

print("=== with_exponential_backoff ===")
import time
calls = {"n": 0}
def flaky():
    calls["n"] += 1
    if calls["n"] < 3:
        raise rex.HTTPError(429, "Rate limited")  # library-style positional
    return "ok"
ls_time = time.time
time.sleep = lambda s: None  # don't actually sleep in tests
try:
    r = ls.with_exponential_backoff(flaky, max_retries=5, base_delay=0)
finally:
    time.sleep = ls_time
check("retries through positional 429 then succeeds", r == "ok" and calls["n"] == 3)

calls["n"] = 0
def always_429():
    calls["n"] += 1
    raise rex.HTTPError(429, "Rate limited")
time.sleep = lambda s: None
try:
    r = ls.with_exponential_backoff(always_429, max_retries=2, base_delay=0)
finally:
    time.sleep = ls_time
check("gives up after max_retries -> None", r is None and calls["n"] == 3)

calls["n"] = 0
def timeout_err():
    calls["n"] += 1
    raise rex.Timeout("Request timed out")
time.sleep = lambda s: None
try:
    r = ls.with_exponential_backoff(timeout_err, max_retries=1, base_delay=0)
finally:
    time.sleep = ls_time
check("Timeout retried then None", r is None and calls["n"] == 2)

calls["n"] = 0
def value_err():
    calls["n"] += 1
    raise ValueError("bad input")
r = ls.with_exponential_backoff(value_err, max_retries=5, base_delay=0)
check("ValueError NOT retried (fails fast)", r is None and calls["n"] == 1)

def conn_err():
    raise rex.ConnectionError("Connection refused")
time.sleep = lambda s: None
try:
    r = ls.with_exponential_backoff(conn_err, max_retries=1, base_delay=0)
finally:
    time.sleep = ls_time
check("ConnectionError retried", r is None)

calls["n"] = 0
def assert_401():
    calls["n"] += 1
    # lyricsgenius AssertionError for a non-200 API response. Note the
    # embedded response headers containing "Connection: keep-alive" --
    # this must NOT make a permanent 401 look retryable (live bug fix).
    raise AssertionError(
        "Unexpected response status code: 401. Expected 200 or 204. "
        "Response headers: {'Connection': 'keep-alive'}."
    )
time.sleep = lambda s: None
try:
    r = ls.with_exponential_backoff(assert_401, max_retries=5, base_delay=0)
finally:
    time.sleep = ls_time
check("401 AssertionError NOT retried (fails fast)", r is None and calls["n"] == 1)

calls["n"] = 0
def assert_503():
    calls["n"] += 1
    raise AssertionError("Unexpected response status code: 503. Expected 200 or 204.")
time.sleep = lambda s: None
try:
    r = ls.with_exponential_backoff(assert_503, max_retries=2, base_delay=0)
finally:
    time.sleep = ls_time
check("503 AssertionError retried then None", r is None and calls["n"] == 3)

calls["n"] = 0
def assert_404():
    calls["n"] += 1
    raise AssertionError("Unexpected response status code: 404. Expected 200 or 204.")
r = ls.with_exponential_backoff(assert_404, max_retries=5, base_delay=0)
check("404 AssertionError NOT retried", r is None and calls["n"] == 1)

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
