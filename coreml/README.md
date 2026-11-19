# Core ML build (int8, 189 MB)

`kikiwake_int8.mlpackage`: the same cut and both heads, with the weights quantized to int8 (`linear_quantize_weights`, symmetric, per output channel).
One function per input length: `s1_5`, `s2`, `s3`, `s5`, `s10` (seconds). Input `input_features` (from the feature extractor in
`../feature_extractor_config.json`, as in `kikiwake.py`). Outputs `vec`, `logits1`, `logits2`, `combined` (the head-sum score over the shared labels).
MacBook M1, Neural Engine, 1.5 s: 19.1 ms median (p90 20.9), n=120, measured while the Mac was busy (load average ~14).
Compile to `.mlmodelc` with `xcrun coremlcompiler compile` before use.
