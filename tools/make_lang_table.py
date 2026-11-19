"""Verified-language table from the G-2995 confirm run (CONFIRM.json), never typed by hand.
usage: python make_lang_table.py CONFIRM.json label_names.json > LANGUAGES_TABLE.md"""
import json, sys
C, N = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
SRC = {"cv": "Common Voice 17 test", "vxte": "VoxLingua107 videos", "iv": "IndicVoices valid", "vp": "VoxPopuli test"}
def src(s): p = s.split("_")[0]; return SRC.get(p, p)
rows = []
for k in C["verified_list"]:
    per = C["per_lang"][k]; worst = min(100 * r / n for _, r, n, *_ in per if n >= 10)
    cells = "; ".join(f"{src(s)} {round(100 * r / n)}% [{lo}-{hi}] (n={n})" for s, r, n, lo, hi in per)
    rows.append((N.get(k, k), k, round(worst), cells))
rows.sort()
assert len(rows) == C["verified"] == 52, len(rows)
print(f"**{C['verified']} of {C['tested']} tested languages verified** (≥ 80% right on every never-trained source with n ≥ 10; full clips, first 10 s; 95% Wilson intervals).\n")
print("| language | code | worst source | each source: % [95% CI] (n) |\n|---|---|---|---|")
for name, k, w, cells in rows: print(f"| {name} | `{k}` | {w}% | {cells} |")
