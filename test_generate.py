"""Offline test for generate.py — no GPU, no big model.

Uses the tiny random model + the smoke-test adapter to verify the
full load → generate → decode path works end to end on CPU.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/home/btclord/nano-shatta")

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")

SMOKE = Path("/tmp/nano_smoke_out")

print("=== generate.py smoke (tiny model + smoke adapter, CPU) ===")
if not (SMOKE / "adapter_config.json").exists():
    print("  SKIP: no smoke adapter at /tmp/nano_smoke_out — run test_train.py first")
    sys.exit(0)

r = subprocess.run(
    [sys.executable, "generate.py",
     "--adapter", str(SMOKE),
     "--base", "hf-internal-testing/tiny-random-LlamaForCausalLM",
     "--prompt", "[Verse 1]",
     "--max-new-tokens", "16",
     "--seed", "7",
     "--no-4bit"],
    capture_output=True, text=True,
    cwd="/home/btclord/nano-shatta", timeout=300,
)
print("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
check("generate exit 0", r.returncode == 0)
check("output has separator", "-" * 50 in r.stdout)
check("prompt echoed in output", "[Verse 1]" in r.stdout)
# random tiny model -> just check it produced *some* decoded text
check("produced decoded text", len(r.stdout.split("-" * 50)[1].strip()) > 0)

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
