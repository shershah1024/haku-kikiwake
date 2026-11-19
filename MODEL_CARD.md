---
license: apache-2.0
base_model: google/embeddinggemma-2
pipeline_tag: audio-classification
library_name: pytorch
tags:
- language-identification
- audio-classification
- on-device
- coreml
---

# haku-langid (working name)

Spoken language ID from **1.5 seconds** of speech: **52 languages verified**, in one small model that runs on the device.

- One model: a cut of EmbeddingGemma 2's audio encoder (layers 0-7, 203M parameters) plus two small heads.
  - `langid_cut07_int8.safetensors` (205 MB, int8, PyTorch, CPU).
  - `coreml/langid_cut07_q8.mlpackage` (189 MB, int8 Core ML). It takes about 19 ms per 1.5 s clip on a MacBook M1 Neural Engine (median, n=120 clips, measured while the Mac was busy).
- No server and no internet needed after the first download.

## Quick start

```bash
pip install -r requirements.txt
python langid.py . clip.wav          # run from this folder
```
```python
from langid import LangID
m = LangID(".")
m("spa.wav")   # a 1.5 s Common Voice clip: [('spa', 'Spanish', -0.13), ('cat', 'Catalan', -11.63), ('ast', 'Asturian', -12.16)]
```
Audio is resampled to 16 kHz mono; the first 10 s are used. Feature extraction comes from `google/embeddinggemma-2`
(revision `f9972ac`), downloaded on first use. Answers are limited to the 96 languages both heads know; only the 52 below are verified.

## How good is it

Same 130 Common Voice test clips (13 languages, 10 per language, 130 speakers), same server CPU (8 threads), every model:

| model | size | time per 1.5 s clip | right from 1.5 s | right on full clip (median 6.1 s) |
|---|---|---|---|---|
| **This model** (int8 PyTorch) | 203 MB | 325 ms | **106/130 (82%)** | 116/130 (89%) |
| Qwen3-ASR 0.6B | 1.9 GB | 4.7 s | 110/130 (85%) | 122/130 (94%) |
| Whisper large-v3 | 3.1 GB | 14.7 s | 104/130 (80%) | 121/130 (93%) |
| MMS-LID-126 | 3.9 GB | 1.3 s | 94/130 (72%) | 124/130 (95%) |
| MMS-LID-1024 | 3.9 GB | 1.5 s | 87/130 (67%) | 122/130 (94%) |
| Whisper small | 966 MB | 2.7 s | 87/130 (67%) | 118/130 (91%) |
| NVIDIA AmberNet | 116 MB | 60 ms | 83/130 (64%) | 106/130 (82%) |
| SpeechBrain ECAPA | 82 MB | 98 ms | 73/130 (56%) | 104/130 (80%) |

- **From 1.5 s** it is in the top group with Qwen3-ASR and Whisper large-v3. 130 clips cannot tell these three apart
  (a gap needs about 10 points to show). It is clearly ahead of MMS-1024, Whisper small, AmberNet and ECAPA at 1.5 s.
- **On full clips the big models lead** (MMS-LID-126 124/130, Qwen3-ASR and MMS-1024 122, Whisper large-v3 121; ours 116).
- It is 1/9 to 1/15 of the size of Qwen3-ASR and Whisper large-v3, and 15x to 45x faster on the same CPU.
- The 13 languages: 10 this model verified that every model covers (Arabic, Czech, German, Greek, Finnish, Japanese,
  Persian, Spanish, Thai, Turkish) plus 3 every model covers that it did **not** verify (English, Hindi, Russian).
  On the 10: 86/100 at 1.5 s. On the 3: 20/30.

## Verified languages

Tested on recordings from sources and speakers the model never trained on, scored once on a fresh set after the model was frozen
(5,877 clips). Over the 52 verified languages: 3,299 of 3,560 clips right (92.7%).

**52 of 77 tested languages verified** (≥ 80% right on every never-trained source with n ≥ 10; full clips, first 10 s; 95% Wilson intervals).

| language | code | worst source | each source: % [95% CI] (n) |
|---|---|---|---|
| Arabic | `arb` | 96% | Common Voice 17 test 96% [87-99] (n=50) |
| Armenian | `hye` | 96% | VoxLingua107 videos 96% [87-99] (n=50) |
| Assamese | `asm` | 94% | IndicVoices valid 94% [84-98] (n=50) |
| Azerbaijani | `azj` | 96% | VoxLingua107 videos 96% [87-99] (n=50) |
| Bodo (India) | `brx` | 90% | IndicVoices valid 90% [79-96] (n=50) |
| Bulgarian | `bul` | 92% | Common Voice 17 test 100% [68-100] (n=8); VoxLingua107 videos 92% [81-97] (n=50) |
| Burmese | `mya` | 100% | VoxLingua107 videos 100% [93-100] (n=50) |
| Czech | `ces` | 98% | Common Voice 17 test 98% [90-100] (n=50) |
| Danish | `dan` | 96% | Common Voice 17 test 96% [79-99] (n=23); VoxLingua107 videos 96% [87-99] (n=50) |
| Dutch | `nld` | 98% | Common Voice 17 test 98% [90-100] (n=50) |
| Estonian | `est` | 98% | Common Voice 17 test 98% [90-100] (n=50) |
| Finnish | `fin` | 98% | Common Voice 17 test 98% [90-100] (n=50) |
| French | `fra` | 86% | Common Voice 17 test 86% [74-93] (n=50); VoxPopuli test 89% [76-95] (n=44) |
| Georgian | `kat` | 100% | Common Voice 17 test 100% [93-100] (n=50); VoxLingua107 videos 100% [93-100] (n=50) |
| German | `deu` | 90% | Common Voice 17 test 94% [84-98] (n=50); VoxPopuli test 90% [70-97] (n=20) |
| Greek | `ell` | 90% | Common Voice 17 test 90% [79-96] (n=50) |
| Gujarati | `guj` | 88% | IndicVoices valid 88% [76-94] (n=50) |
| Hebrew | `heb` | 94% | VoxLingua107 videos 94% [84-98] (n=50) |
| Hungarian | `hun` | 98% | Common Voice 17 test 98% [90-100] (n=50); VoxLingua107 videos 98% [90-100] (n=50) |
| Icelandic | `isl` | 92% | VoxLingua107 videos 92% [81-97] (n=50) |
| Italian | `ita` | 100% | Common Voice 17 test 100% [93-100] (n=50) |
| Japanese | `jpn` | 92% | Common Voice 17 test 92% [81-97] (n=50); VoxLingua107 videos 98% [90-100] (n=50) |
| Kannada | `kan` | 84% | IndicVoices valid 84% [71-92] (n=50); VoxLingua107 videos 94% [84-98] (n=50) |
| Kazakh | `kaz` | 88% | VoxLingua107 videos 88% [76-94] (n=50) |
| Korean | `kor` | 98% | VoxLingua107 videos 98% [90-100] (n=50) |
| Lao | `lao` | 88% | VoxLingua107 videos 88% [76-94] (n=50) |
| Latvian | `lav` | 86% | Common Voice 17 test 86% [74-93] (n=50); VoxLingua107 videos 96% [87-99] (n=50) |
| Lingala | `lin` | 94% | VoxLingua107 videos 94% [84-98] (n=50) |
| Lithuanian | `lit` | 92% | Common Voice 17 test 92% [81-97] (n=50) |
| Macedonian | `mkd` | 94% | VoxLingua107 videos 94% [84-98] (n=50) |
| Malay | `zsm` | 80% | VoxLingua107 videos 80% [67-89] (n=50) |
| Malayalam | `mal` | 92% | IndicVoices valid 96% [87-99] (n=50); VoxLingua107 videos 92% [81-97] (n=50) |
| Mandarin Chinese | `cmn` | 98% | VoxLingua107 videos 98% [90-100] (n=50) |
| Mongolian | `khk` | 100% | Common Voice 17 test 100% [93-100] (n=50); VoxLingua107 videos 100% [93-100] (n=50) |
| Norwegian | `nob` | 92% | VoxLingua107 videos 92% [81-97] (n=50) |
| Pashto | `pbt` | 84% | VoxLingua107 videos 84% [71-92] (n=50) |
| Persian | `pes` | 96% | Common Voice 17 test 100% [93-100] (n=50); VoxLingua107 videos 96% [87-99] (n=50) |
| Polish | `pol` | 94% | Common Voice 17 test 94% [84-98] (n=50); VoxLingua107 videos 98% [90-100] (n=50) |
| Portuguese | `por` | 80% | Common Voice 17 test 80% [67-89] (n=50); VoxLingua107 videos 90% [79-96] (n=50) |
| Punjabi | `pan` | 84% | IndicVoices valid 84% [71-92] (n=50); VoxLingua107 videos 84% [71-92] (n=50) |
| Romanian | `ron` | 86% | Common Voice 17 test 86% [74-93] (n=50); VoxLingua107 videos 94% [84-98] (n=50) |
| Slovenian | `slv` | 84% | Common Voice 17 test 84% [62-94] (n=19); VoxLingua107 videos 84% [71-92] (n=50) |
| Somali | `som` | 90% | VoxLingua107 videos 90% [79-96] (n=50) |
| Spanish | `spa` | 86% | Common Voice 17 test 86% [74-93] (n=50) |
| Swedish | `swe` | 88% | Common Voice 17 test 88% [76-94] (n=50); VoxLingua107 videos 92% [81-97] (n=50) |
| Telugu | `tel` | 90% | IndicVoices valid 90% [79-96] (n=50); VoxLingua107 videos 90% [79-96] (n=50) |
| Thai | `tha` | 84% | Common Voice 17 test 90% [79-96] (n=50); VoxLingua107 videos 84% [71-92] (n=50) |
| Turkish | `tur` | 94% | Common Voice 17 test 98% [90-100] (n=50); VoxLingua107 videos 94% [84-98] (n=50) |
| Uzbek | `uzn` | 96% | VoxLingua107 videos 96% [87-99] (n=50) |
| Vietnamese | `vie` | 91% | Common Voice 17 test 91% [80-97] (n=46); VoxLingua107 videos 98% [90-100] (n=50) |
| Welsh | `cym` | 96% | Common Voice 17 test 96% [87-99] (n=50) |
| Yoruba | `yor` | 88% | VoxLingua107 videos 88% [76-94] (n=50) |

Borderline: Punjabi and Spanish passed in this run but not on the earlier test set (Spanish was tested on one source here). Hindi, Tamil, Tajik and Ukrainian passed an earlier
test set but not this one, so they are not in the list.

## How it works

`x` (16 kHz) → EmbeddingGemma 2 feature extractor → audio encoder layers 0-7 → mean over frames = `v` (1024).
Head 1: `W1 @ v + b1` (scaler folded in; 101 labels). Head 2: `z = (v - mean) / scale`, `z /= |z|`, `W2 @ z + b2` (437 labels).
Answer = argmax over the shared labels of `log_softmax(head 1) + log_softmax(head 2)`.

## Limits

- The comparison is a small first pass (n=130). The 10-language group was chosen on clips from the same test pool, which flatters this model; read the 13-language total.
- Model choices were tuned on other Common Voice test speakers. Whisper, MMS and Qwen3-ASR may have trained on Common Voice.
- The 52 are verified on full clips, not at 1.5 s. Short clips are weaker.
- Read speech, video and parliament audio only were tested: no phone calls, children, singing or code-switching.
- Hindi and Urdu share one label. Close relatives (e.g. Serbian / Croatian, Marathi / Konkani) are hard.
- The comparison table times the int8 PyTorch file on a server CPU; the 19 ms figure is the Core ML file on the M1 Neural Engine.

## Training data and credits

Head 1: synthetic TTS audio (ElevenLabs), reading FLEURS sentences (from FLoRes); Rasa (AI4Bharat) studio speech for 8 Indian
languages; OpenSLR 28 room and noise copies. Head 2: FLEURS, Omnilingual ASR corpus, VoxLingua107, IndicVoices, Kathbath,
Shrutilipi, OpenSLR 28. No audio, transcripts or text from any source is included. Full list and licences: `LICENCES.md`, `NOTICE`.

## Licence

Apache-2.0 (`LICENSE`, `NOTICE`). The model file is a modified copy of EmbeddingGemma 2 by Google DeepMind (Apache-2.0);
this release is not made or endorsed by Google. Use must follow the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

Links: to be added when published.
