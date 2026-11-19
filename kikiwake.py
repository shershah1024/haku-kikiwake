"""Haku Kikiwake: spoken language ID from 1.5 s+ of speech.
A cut of EmbeddingGemma 2's audio encoder (layers 0-7, int8) + two heads. Everything it needs is in this folder:
kikiwake_int8.safetensors, encoder_config.json, feature_extractor_config.json and the label files. Nothing is downloaded.

usage:  python kikiwake.py clip.wav [clip2.wav ...]          (any sample rate; mono or stereo)
        from kikiwake import Kikiwake; m = Kikiwake(); m("clip.wav") -> [(code, name, score), ...]"""
import json, os, sys
import numpy as np, torch
from safetensors.torch import load_file
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioConfig, Gemma4AudioModel

HERE = os.path.dirname(os.path.abspath(__file__))
SR, MAX_S = 16000, 10.0

def build_encoder(cfg_path, sd):
    """fp32 skeleton -> load the non-Linear tensors -> dynamic int8 -> load the stored int8 Linear weights."""
    enc = Gemma4AudioModel(Gemma4AudioConfig.from_json_file(cfg_path)); enc.output_proj = torch.nn.Identity()
    mods = dict(enc.named_modules()); lin = [n for n, m in mods.items() if isinstance(m, torch.nn.Linear)]
    plain = {k[6:]: v for k, v in sd.items() if k.startswith("tower.") and not k.startswith("tower.q8.")}
    missing, unexpected = enc.load_state_dict(plain, strict=False)
    want = {f"{n}.{p}" for n in lin for p in ("weight", "bias") if getattr(mods[n], p) is not None}
    assert not unexpected and set(missing) == want, (unexpected, set(missing) ^ want)
    enc = torch.ao.quantization.quantize_dynamic(enc.eval(), {torch.nn.Linear}, dtype=torch.qint8)
    qmods = dict(enc.named_modules())
    for n in lin:
        p = f"tower.q8.{n}"; sc, zp = sd[f"{p}.scale"], sd[f"{p}.zero_point"]
        q = (torch._make_per_tensor_quantized_tensor(sd[f"{p}.int8"], float(sc[0]), int(zp[0])) if sc.numel() == 1
             else torch._make_per_channel_quantized_tensor(sd[f"{p}.int8"], sc, zp, 0))
        qmods[n].set_weight_bias(q, sd.get(f"{p}.bias"))
    assert len(lin) > 0
    return enc.eval()

class Kikiwake:
    def __init__(self, d=HERE, threads=None):
        if threads: torch.set_num_threads(threads)
        sd = load_file(f"{d}/kikiwake_int8.safetensors")
        self.encoder = build_encoder(f"{d}/encoder_config.json", sd)
        self.W1, self.b1 = sd["head.weight"], sd["head.bias"]
        self.W2, self.b2, self.mu, self.sd = sd["head2.weight"], sd["head2.bias"], sd["head2.mean"], sd["head2.scale"]
        j = lambda f: json.load(open(f"{d}/{f}"))
        self.l1, self.l2, self.sh, self.names = j("labels.json"), j("labels2.json"), j("shared_labels.json"), j("label_names.json")
        self.i1 = torch.tensor([self.l1.index(k) for k in self.sh]); self.i2 = torch.tensor([self.l2.index(k) for k in self.sh])
        fc = j("feature_extractor_config.json"); fc.pop("feature_extractor_type", None)
        self.features = Gemma4AudioFeatureExtractor(**fc)

    @staticmethod
    def load_audio(path):
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True); x = x.mean(1)
        if sr != SR:
            t = np.arange(int(len(x) * SR / sr)) * (sr / SR); x = np.interp(t, np.arange(len(x)), x).astype(np.float32)
        return x

    @torch.no_grad()
    def scores(self, x):
        """x: 16 kHz mono float32. Log-score per shared label (sum of both heads' log-softmax)."""
        x = np.asarray(x, np.float32)[: int(SR * MAX_S)]
        f = self.features([x], sampling_rate=SR, return_tensors="pt", padding=True)
        o = self.encoder(f["input_features"], f["input_features_mask"], return_dict=True); m = o.attention_mask[0].bool()
        v = o.last_hidden_state[0][m].mean(0)
        l1 = self.W1 @ v + self.b1
        z = (v - self.mu) / self.sd; z = z / z.norm(); l2 = self.W2 @ z + self.b2
        return torch.log_softmax(l1[self.i1], -1) + torch.log_softmax(l2[self.i2], -1)

    def __call__(self, x, k=3):
        if isinstance(x, str): x = self.load_audio(x)
        s = self.scores(x); top = torch.topk(s, k)
        return [(self.sh[i], self.names.get(self.sh[i], self.sh[i]), float(v)) for v, i in zip(top.values, top.indices)]

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    m = Kikiwake()
    for f in sys.argv[1:]:
        print(f, " ".join(f"{c} ({n}) {s:.2f}" for c, n, s in m(f)))
