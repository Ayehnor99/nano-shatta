"""Tests for generate.py — single-shot + full-song mode. No GPU needed.

Pure helpers (structure parsing, prompt building, section trimming) are
tested directly. compose_song() is tested with a mock model so the
section loop and assembly are verified without weights.
"""
import sys
import types

sys.path.insert(0, "/home/btclord/nano-shatta")

import generate as G

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")

print("=== parse_structure ===")
check("default structure when None", G.parse_structure(None) == G.DEFAULT_SONG_STRUCTURE)
check("custom spec parses", G.parse_structure("Intro,Verse 1,Chorus") == ["Intro", "Verse 1", "Chorus"])
check("whitespace stripped", G.parse_structure(" Intro , Chorus ") == ["Intro", "Chorus"])
check("empty spec -> default", G.parse_structure("") == G.DEFAULT_SONG_STRUCTURE)
check("trailing commas dropped", G.parse_structure("Intro,,Chorus,") == ["Intro", "Chorus"])

print("=== build_prompt ===")
p0 = G.build_prompt(None, [], "Intro")
check("first section prompt has no done-sections", p0 == "[Intro]")
p1 = G.build_prompt("Street Crown", [], "Verse 1")
check("title included", p1.startswith("Street Crown\n\n[Verse 1]"))
p2 = G.build_prompt(None, [("Intro", "Yeah yeah"), ("Verse 1", "Mi a di realest")], "Chorus")
check("prior sections included in order",
      p2.index("[Intro]") < p2.index("[Verse 1]") < p2.index("[Chorus]"))
check("section text carried", "Mi a di realest" in p2)

print("=== trim_section ===")
raw = "Mi a di realest\nDem nuh know\n[Chorus]\nBig lie dat"
check("cuts at next section marker", G.trim_section(raw, "Verse 1") == "Mi a di realest\nDem nuh know")
raw2 = "[Verse 1]\nMi a di realest"
check("drops echoed own marker", G.trim_section(raw2, "Verse 1") == "Mi a di realest")
raw3 = "a\n\n\n\nb"
check("collapses 3+ blank lines", "\n\n\n" not in G.trim_section(raw3, "Intro"))
check("empty stays empty", G.trim_section("", "Intro") == "")
raw4 = "line one\n[Bridge]\nignored"
check("stops at any marker", "ignored" not in G.trim_section(raw4, "Verse 2"))

print("=== compose_song (mock model) ===")
class MockTok:
    eos_token_id = 0
    def __call__(self, text, return_tensors="pt"):
        import torch
        # encode length as token count so the model can echo it
        self._last = text
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    def decode(self, ids, skip_special_tokens=True):
        # model "writes" a bar then starts the NEXT section (Bridge) —
        # that trailing part must be trimmed off.
        return self._last + "Mi a di realest general\n[Bridge]\nbleed"

class MockModel:
    device = "cpu"
    def generate(self, ids, **kw):
        import torch
        return torch.tensor([[0]])
    def eval(self): pass

class Args:
    title = "Street Crown"
    structure = "Intro,Verse 1,Chorus"
    section_tokens = 50
    temperature = 0.9
    top_p = 0.95
    repetition_penalty = 1.15

song = G.compose_song(MockModel(), MockTok(), Args())
print(song)
check("song has 3 sections", song.count("[") >= 3)
check("title present", song.startswith("Street Crown"))
check("sections in order",
      song.index("[Intro]") < song.index("[Verse 1]") < song.index("[Chorus]"))
check("bleed trimmed at next marker (no 'bleed')", "bleed" not in song)
check("generated content present", "realest" in song)

print(f"\n=== {passed} passed, {failed} failed ===")
sys.exit(1 if failed else 0)
