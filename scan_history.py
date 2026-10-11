"""Scan ALL commits for any file containing a real GENIUS_ACCESS_TOKEN value.

Never prints the token — only reports which commit/file has a value and its length.
"""
import subprocess

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd="/home/btclord/nano-shatta").stdout

commits = git("rev-list", "--all").split()
print(f"scanning {len(commits)} commits...")

hits = []
for c in commits:
    files = git("ls-tree", "-r", "--name-only", c).splitlines()
    for f in files:
        try:
            content = git("show", f"{c}:{f}")
        except Exception:
            continue
        for line in content.splitlines():
            if line.startswith("GENIUS_ACCESS_TOKEN="):
                val = line.split("=", 1)[1]
                if val and val not in ("<token>", ""):
                    hits.append((c[:8], f, len(val)))

if not hits:
    print("CLEAN: no commit has a real token value")
else:
    print(f"LEAK: {len(hits)} commit/file locations with a real value:")
    seen_files = set()
    for c, f, n in hits:
        key = (f, n)
        if key not in seen_files:
            seen_files.add(key)
            print(f"  file={f}  value_length={n}  first_seen_in_commit={c}")
    commits_affected = sorted(set(h[0] for h in hits))
    print(f"commits affected: {commits_affected}")
    print(f"earliest affected commit: {commits_affected[0]}")
