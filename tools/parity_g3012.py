"""Release check: the shipped loader + int8 file must reproduce (1) the in-memory quantize_dynamic model exactly, and
(2) G-3012's ours-int8 per-clip answers on its 130 clips, at 1.5 s and full length. Control: one scale changed must break (1)."""
import json, os, sys, copy, numpy as np, soundfile as sf, torch
torch.set_num_threads(8)
HF, EXP = sys.argv[1], os.path.expanduser("~/artifacts/eg2-reader/g2908/g2991/export"); G = os.path.expanduser("~/artifacts/eg2-reader/g3012")
sys.path.insert(0, HF); from langid import LangID, build_tower
sys.path.insert(0, EXP); from lid_cut07_forward import LID
m = LangID(HF); ref = LID(EXP); ref.tower = torch.ao.quantization.quantize_dynamic(ref.tower, {torch.nn.Linear}, dtype=torch.qint8)
CL = [r for r in json.load(open(f"{G}/clips.json")) if r["slice"] in ("shared", "unverified")]
def mid(x, n=24000): s = max(0, (len(x) - n) // 2); return x[s:s + n]
old = {}
for f, crop in (("ours-int8", 0), ("ours-int8_unv", 0), ("ours-int8_crop1.5", 1), ("ours-int8_crop1.5_unv", 1)):
    for r in json.load(open(f"{G}/out/{f}.json"))["rows"]: old[(r["file"], crop)] = r["pred"]
res = {"exact_vs_inmemory": 0, "checked_inmemory": 0}; agree = {0: 0, 1: 0}; right = {0: 0, 1: 0}; n = {0: 0, 1: 0}
for i, r in enumerate(CL):
    x0 = sf.read(r["file"], dtype="float32")[0]
    for crop in (0, 1):
        x = mid(x0) if crop else x0; s = m.scores(x); top = m.sh[int(s.argmax())]
        n[crop] += 1; agree[crop] += top == old[(r["file"], crop)]; right[crop] += top == r["key"]
        if i < 10 and crop == 1:
            res["checked_inmemory"] += 1; res["exact_vs_inmemory"] += bool(torch.equal(s, ref(x)["combined"]))
# control: change one stored scale by 1% and the scores must change
from safetensors.torch import load_file
sd = load_file(f"{HF}/langid_cut07_int8.safetensors"); k = sorted(x for x in sd if x.endswith(".scale"))[5]; sd[k] = sd[k] * 1.01
m2 = copy.copy(m); m2.tower = build_tower(f"{HF}/cut07_config.json", sd)
x = mid(sf.read(CL[0]["file"], dtype="float32")[0]); res["control_changed_scale_still_exact"] = bool(torch.equal(m2.scores(x), m.scores(x)))
res.update({"n": n, "agree_with_g3012": agree, "right": right})
print(json.dumps(res)); json.dump(res, open(f"{HF}/../parity_g3012.json", "w"), indent=1)
