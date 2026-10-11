# FINAL PUSH — rewritten history (token scrubbed)

The local history has been rewritten to remove the leaked Genius token
from `.env.example` (it was in 8 commits). The remote still has the old
history with the token. **This push must be a force-push.**

Run this yourself in a terminal — it needs your token:

```bash
cd ~/nano-shatta
git push --force-with-lease origin main
```

When git asks:

```
Username for 'https://github.com': Ayehnor99
Password for 'https://Ayehnor99@github.com': <paste your classic PAT>
```

Expected output:

```
Writing objects: 100% ..., done.
To https://github.com/Ayehnor99/nano-shatta.git
 + 765ee24...3026628 main -> main (forced update)
```

The `forced update` line is what we want — it means the remote's
token-containing history was replaced with the clean one.

---

## Verify the fix landed on GitHub

```bash
# From any fresh clone, the token must be GONE:
git clone https://github.com/Ayehnor99/nano-shatta.git /tmp/verify-ns
cd /tmp/verify-ns
python3 scan_history.py
# Expected: "CLEAN: no commit has a real token value"
```

Or check on GitHub directly:
https://github.com/Ayehnor99/nano-shatta/blob/main/.env.example

The line should read `GENIUS_ACCESS_TOKEN=` (empty). If you see a value,
the force-push didn't land — check the output above.

---

## If `--force-with-lease` is rejected

Someone (or something) pushed to the remote since our last fetch.
Fetch first, then force-push:

```bash
git fetch origin
git push --force-with-lease origin main
```

If it still complains, use `--force` (we own this repo, nobody else
is pushing to it):

```bash
git push --force origin main
```

---

## After the push — you're done

1. https://github.com/Ayehnor99/nano-shatta → 29 files, Colab badge, clean LICENSE
2. Click the badge → Colab → Runtime → T4 GPU → Run all
3. ~15–25 min → download `shatta-lora.zip`
4. Train your Shatta Wale bar-writer

See `INCIDENT.md` for the full story of what leaked and how it was fixed.
