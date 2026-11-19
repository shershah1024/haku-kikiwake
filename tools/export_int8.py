"""Write langid_cut07_int8.safetensors from the frozen fp32 two-head cut (G-2995, md5 7c4bc88b1487a7171d52a6d6a4ef97b3).
Recipe = the one G-3012 timed and scored: torch dynamic quantization of every nn.Linear, int8 weights (PyTorch default dynamic qconfig: one scale per weight tensor).
The int8 values, scales and zero points are stored as plain tensors, so the file needs no pickle.
usage: python export_int8.py SRC_DIR OUT_DIR"""
import json, os, sys, hashlib, torch
from safetensors.torch import load_file, save_file
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioConfig, Gemma4AudioModel
SRC, OUT = sys.argv[1:3]; F = f"{SRC}/lid_cut07_2heads_fp32.safetensors"
assert hashlib.md5(open(F, "rb").read()).hexdigest() == "7c4bc88b1487a7171d52a6d6a4ef97b3", "not the frozen G-2995 file"
sd = load_file(F)
tower = Gemma4AudioModel(Gemma4AudioConfig.from_json_file(f"{SRC}/cut07_config.json")); tower.output_proj = torch.nn.Identity()
tower.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith("tower.")}, strict=True); tower.eval()
lin = [n for n, m in tower.named_modules() if isinstance(m, torch.nn.Linear)]
q = torch.ao.quantization.quantize_dynamic(tower, {torch.nn.Linear}, dtype=torch.qint8); qm = dict(q.named_modules())
out = {}; schemes = {}
for n in lin:
    w, b = qm[n].weight(), qm[n].bias()
    p = f"tower.q8.{n}"; out[f"{p}.int8"] = w.int_repr().contiguous()
    if w.qscheme() == torch.per_tensor_affine:
        out[f"{p}.scale"] = torch.tensor([w.q_scale()], dtype=torch.float64); out[f"{p}.zero_point"] = torch.tensor([w.q_zero_point()], dtype=torch.int64)
    else:
        assert w.qscheme() == torch.per_channel_affine and w.q_per_channel_axis() == 0, (n, w.qscheme())
        out[f"{p}.scale"] = w.q_per_channel_scales().contiguous(); out[f"{p}.zero_point"] = w.q_per_channel_zero_points().contiguous()
    schemes[str(w.qscheme())] = schemes.get(str(w.qscheme()), 0) + 1
    if b is not None: out[f"{p}.bias"] = b.detach().contiguous()
skip = {f"{n}.{s}" for n in lin for s in ("weight", "bias")}
for k, v in sd.items():
    if k.startswith("tower."):
        if k[6:] not in skip: out[k] = v.contiguous()
    else: out[k] = v.contiguous()   # head.*, head2.* (fp32)
os.makedirs(OUT, exist_ok=True)
save_file(out, f"{OUT}/langid_cut07_int8.safetensors", metadata={"source": "G-2995 lid_cut07_2heads_fp32 md5 7c4bc88b", "recipe": "quantize_dynamic qint8 per-channel, nn.Linear only", "base": "google/embeddinggemma-2@f9972ac (modified: audio layers 0-7)"})
print("linears", len(lin), schemes, "MB", round(os.path.getsize(f"{OUT}/langid_cut07_int8.safetensors") / 1e6, 1))
