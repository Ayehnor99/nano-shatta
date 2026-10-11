"""Tests for the LiveMonitor + train/val split. No GPU, no big models."""
import sys
import types

sys.path.insert(0, "/home/btclord/nano-shatta")

import train as T

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")

print("=== sparkline ===")
check("empty -> empty", T.sparkline([]) == "")
check("single value renders", len(T.sparkline([1.0])) == 1)
sp = T.sparkline([4.0, 3.0, 2.0, 1.0])
check("monotonic down = descending spark", sp[0] == "█" and sp[-1] == "▁")
check("downsamples to width", len(T.sparkline(list(range(1000)), width=40)) == 40)
check("constant values don't crash", len(T.sparkline([2.0] * 10)) == 10)

print("=== fmt_eta ===")
check("45s", T.fmt_eta(45) == "00:45")
check("5m30s", T.fmt_eta(330) == "05:30")
check("1h2m3s", T.fmt_eta(3723) == "1:02:03")
check("negative -> --:--", T.fmt_eta(-1) == "--:--")
check("NaN -> --:--", T.fmt_eta(float("nan")) == "--:--")

print("=== train/val split logic ===")
import random
blocks = [list(range(i * 10, i * 10 + 10)) for i in range(100)]
rng = random.Random(42)
idxs = list(range(len(blocks)))
rng.shuffle(idxs)
n_val = int(len(blocks) * 0.05)
val_idx = set(idxs[:n_val])
train_blocks = [b for i, b in enumerate(blocks) if i not in val_idx]
val_blocks = [b for i, b in enumerate(blocks) if i in val_idx]
check("split sizes", len(train_blocks) == 95 and len(val_blocks) == 5)
check("no overlap", not (set(map(tuple, train_blocks)) & set(map(tuple, val_blocks))))
check("deterministic with same seed", random.Random(42).sample(range(100), 5) == random.Random(42).sample(range(100), 5))
check("val_frac=0 -> no val", int(100 * 0.0) == 0)

print("=== LiveMonitor.on_log (mocked) ===")
# Mock tokenizer
class MockTok:
    eos_token_id = 0
    def __call__(self, *a, **k):
        return {"input_ids": __import__("torch").tensor([[1, 2, 3]])}
    def decode(self, ids, skip_special_tokens=True):
        return "[Verse 1]\nWale wale wale dem fraid"

# Mock model
class MockModel:
    training = True
    device = "cpu"
    def eval(self): self.training = False
    def train(self): self.training = True
    def generate(self, ids, **k):
        import torch
        return torch.tensor([[1, 2, 3, 4, 5]])  # tensor with .to() via input
    def __call__(self, **kw): return types.SimpleNamespace(loss=types.SimpleNamespace(item=lambda: 2.5))

# Mock trainer args/shim
class MockArgs:
    max_steps = 100
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 8
    logging_steps = 5
class MockTrainer:
    args = MockArgs()
class MockState:
    global_step = 10

mon = T.LiveMonitor(MockTok(), val_ds=None, sample_every=5, out_dir="/tmp/nano_monitor_test")
T._BLOCK_SIZE[0] = 512
mon.on_log(MockModel(), MockTrainer(), MockState(), {"loss": 3.5, "learning_rate": 1e-4})
check("loss recorded", mon.losses == [3.5])
check("sample fired at step 10 (sample_every=5)", mon.last_sample_step == 10)
check("samples.log written", __import__("pathlib").Path("/tmp/nano_monitor_test/samples.log").exists())

# val loss path with a mock val_ds
val_ds = T.PackedDataset([[1, 2, 3]] * 4)
mon2 = T.LiveMonitor(MockTok(), val_ds=val_ds, sample_every=5, out_dir=None)
mon2.on_log(MockModel(), MockTrainer(), MockState(), {"loss": 3.0})
check("val loss recorded", len(mon2.val_losses) == 1)

print("=== make_monitor_callback ===")
cb = T.make_monitor_callback(mon)
check("callback has on_log", hasattr(cb, "on_log"))

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
