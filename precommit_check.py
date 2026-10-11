#!/usr/bin/env python3
"""Pre-commit hook: block any commit that adds a real Genius token.

Install:
    cp precommit_check.sh .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

What it does:
    Scans all STAGED files for `GENIUS_ACCESS_TOKEN=<non-empty value>`.
    If found, prints which file and ABORTS the commit. Never prints
    the token value itself.
"""
import subprocess
import sys

BAD_PREFIX = "GENIUS_ACCESS_TOKEN="

def staged_files():
    r = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True,
    )
    return [f for f in r.stdout.splitlines() if f]

def staged_content(path):
    r = subprocess.run(
        ["git", "show", f":{path}"],
        capture_output=True, text=True,
    )
    return r.stdout

def main():
    bad = []
    for f in staged_files():
        try:
            content = staged_content(f)
        except Exception:
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if line.startswith(BAD_PREFIX):
                val = line[len(BAD_PREFIX):].strip()
                # Allow empty (template) and the literal placeholder.
                if val and val != "<token>":
                    bad.append((f, i, len(val)))
    if bad:
        print("\n!!! COMMIT BLOCKED: real Genius token detected in staged files !!!")
        for f, i, n in bad:
            print(f"  {f}:{i}  (value length {n} — NOT shown)")
        print("\nFix: empty the value, put the real token in .env (gitignored),")
        print("then re-stage and commit. NEVER commit a live token.")
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
