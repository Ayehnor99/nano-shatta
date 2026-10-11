# INCIDENT: Genius API token committed in .env.example

**Status: token revoked at Genius (user action). Repo history rewritten. Force-push done.**

## What happened

`.env.example` — a file whose entire purpose is to be a *template* — was
committed with a **real, working Genius API token** in it
(`GENIUS_ACCESS_TOKEN=<live value>`, length 86). The `.gitignore` had
`.env` but not `.env.example`, so git happily tracked it.

The token leaked in **8 of 9 commits** (every commit from `19c9a78`
onward), and the repo is **public** on GitHub, so anyone could have
scraped it.

## Blast radius

- **Only file affected:** `.env.example` (86-char value)
- **`.env` itself:** never committed (verified across all history)
- **`lyrics_scraper.py`:** never had a hardcoded token (verified)
- **`probe_token.py`:** only contains the literal string `<token>` as a
  print message, never a real value (verified)
- **Other secrets:** none in the repo

## Root cause

`.gitignore` listed `.env` but not `.env.example`. A template file with
a real value slipped through `git add -A`.

## Fix applied

1. **User revoked the token at Genius** (genius.com/api-clients) and
   generated a fresh one. Old token is dead.
2. **Local history rewritten** with `git filter-repo` to blank the value
   in `.env.example` across all 8 affected commits.
3. **`.gitignore` updated** to also ignore `.env.example` (it's a
   template, but a typo could re-leak it) — actually we keep tracking a
   *clean* `.env.example` (no value) because it documents the env var.
   The rewritten version has `GENIUS_ACCESS_TOKEN=` (empty).
4. **Force-pushed** rewritten history to GitHub.

## Prevention

- `scan_history.py` — run it after any commit to catch token values in
  tracked files. It scans every commit, never prints the value.
- `.env.example` now contains only `GENIUS_ACCESS_TOKEN=` (empty).
- Pre-commit hook (optional, not yet added): block any commit where a
  tracked file contains `GENIUS_ACCESS_TOKEN=<non-empty>`.

## Lesson

Never put a real secret in a file named `*.example`, `*.template`,
`*.sample`, or `*.dist` — those are exactly the files people `git add`
without thinking. The only safe pattern: the example file ships with an
**empty** value, and the real value lives in `.env` (gitignored).
