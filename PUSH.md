# Push nano-shatta to GitHub — step by step

The push needs your GitHub login, which I can't type for you (and shouldn't).
Run these **yourself** in a terminal. Three steps total.

---

## Step 1 — Get a Personal Access Token (PAT)

GitHub no longer accepts your account password for `git push` over HTTPS.
You need a token:

1. Go to https://github.com/settings/tokens
2. Click **Generate new token → Generate new token (classic)**
3. Give it a name: `nano-shatta push`
4. Tick the **`repo`** scope (that's all it needs)
5. Click **Generate token**
6. **Copy it immediately** — GitHub only shows it once

Keep it somewhere safe. Treat it like a password.

---

## Step 2 — Run the push

Open a terminal and run:

```bash
cd ~/nano-shatta
git push -u origin main
```

When git asks:

```
Username for 'https://github.com': Ayehnor99
Password for 'https://Ayehnor99@github.com': <paste your PAT here>
```

- **Username:** `Ayehnor99`
- **Password:** paste the PAT (it won't show on screen — that's normal)

> **Never paste the token into this chat.** It goes straight into the git prompt only.

Expected output:

```
Enumerating objects: 60, done.
Writing objects: 100% (60/60), 2.06 MiB | ...
To https://github.com/Ayehnor99/nano-shatta.git
 * [new branch]      main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## Step 3 — Verify it landed

Open https://github.com/Ayehnor99/nano-shatta in your browser.
You should see:

- 24 files at the root
- `README.md` with the **Open in Colab** badge at the top
- `shatta_clean_v3.txt` (2MB)
- `nano_shatta_train.ipynb`
- `LICENSE` (MIT)
- Commits: 5 total

---

## Step 4 — Test the Colab badge

Click the badge in the README on GitHub. It should open Colab with the
notebook loaded and ready to run. If it 404s, the badge URL has a typo —
check that it says `Ayehnor99` (not `<YOUR_GH_USER>`) and `main` (not `master`).

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Authentication failed` | The PAT is wrong/expired. Generate a new one with the `repo` scope. |
| `remote: Repository not found` | The repo name isn't exactly `nano-shatta`, or it's private. Check the URL. |
| `! [rejected] main -> main (fetch first)` | Someone pushed since. Run `git pull --rebase origin main` then push again. |
| `could not read Username` | Your terminal isn't prompting. Use a real terminal (not this chat). |

---

## After the push — train on Colab

1. Go to https://github.com/Ayehnor99/nano-shatta
2. Click the **Open in Colab** badge
3. Runtime → Change runtime type → **T4 GPU**
4. Run all cells (the keep-alive cell runs forever — leave it in a tab)
5. ~15–25 min → download `shatta-lora.zip`

Done. You trained a Shatta Wale bar-writer on a free GPU.
