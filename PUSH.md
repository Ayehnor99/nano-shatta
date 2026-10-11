# Push nano-shatta to GitHub — final step

Everything is merged and ready. The push needs your token again because this
shell session doesn't remember it. Run this yourself in a terminal.

---

## Step 1 — Run the push

```bash
cd ~/nano-shatta
git push -u origin main
```

When git asks:

```
Username for 'https://github.com': Ayehnor99
Password for 'https://Ayehnor99@github.com': <paste your classic PAT here>
```

- **Username:** `Ayehnor99`
- **Password:** paste your **classic** Personal Access Token (the one with `repo` ticked)
- It won't show on screen — that's normal
- **Never paste the token into chat** — only into the git prompt

---

## Step 2 — What success looks like

```
Enumerating objects: 70, done.
Writing objects: 100% (70/70), 2.06 MiB | ...
To https://github.com/Ayehnor99/nano-shatta.git
   765ee24..ec13a3f  main -> main
Branch 'main' set up to track remote branch 'main' from 'origin'.
```

---

## Step 3 — Verify on GitHub

Open https://github.com/Ayehnor99/nano-shatta

You should see:
- **25 files** at the root
- `README.md` with the **Open in Colab** badge at the top
- `shatta_clean_v3.txt` (2MB)
- `nano_shatta_train.ipynb`
- `LICENSE` (MIT, copyright `BTC_Lord$`)
- **7 commits** (6 ours + 1 merged from GitHub)

---

## Step 4 — Test the Colab badge

Click the badge in the README. It should open Colab with the notebook loaded.
If it 404s, check the badge URL says `Ayehnor99` and `main`.

---

## Step 5 — Train on Colab

1. Badge → Colab opens
2. **Runtime → Change runtime type → T4 GPU**
3. **Run all cells** (the keep-alive cell runs forever — leave it in a tab)
4. ~15–25 min → download `shatta-lora.zip`

---

## If it still 403s

Your token is fine-grained without this repo, or a classic token without
`repo`. Make a fresh **classic** token:
https://github.com/settings/tokens → Generate new token (classic) → tick `repo` → copy → push again.

After it works, **revoke the old token** so it doesn't sit there as a loose end.
