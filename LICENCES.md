# Licences

The code, the heads and the model files are released under **Apache-2.0** (`LICENSE`). Credits are in `NOTICE`.
No audio, transcripts or text from any source below is included in this release; only weights, labels and scores.

## Base model

| model | licence | what we did |
|---|---|---|
| EmbeddingGemma 2, Google DeepMind ([google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2), revision `f9972ac`) | Apache-2.0, not gated | kept audio encoder layers 0-7, removed the rest, converted to int8 / Core ML; `feature_extractor_config.json` is its audio feature-extractor configuration, copied unchanged. Use must follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). Not made or endorsed by Google. |

## Training data

| source | head | licence |
|---|---|---|
| Synthetic TTS audio (ElevenLabs, eleven_v4 stock voices) | 1 | ElevenLabs terms of the generating account |
| FLEURS sentences (read by the TTS voices) | 1 | FLEURS CC BY 4.0; sentences from FLoRes, CC BY-SA 4.0 |
| [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (AI4Bharat) | 1 | CC BY 4.0 |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) audio | 2 | CC BY 4.0 |
| [Omnilingual ASR corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) | 2 | CC BY 4.0 |
| [VoxLingua107](https://cs.taltech.ee/staff/tanel.alumae/data/voxlingua107/) | 2 | CC BY 4.0; video copyright stays with the owners |
| [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), [Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath), [Shrutilipi](https://huggingface.co/datasets/ai4bharat/Shrutilipi) (AI4Bharat) | 2 | CC BY 4.0 |
| [OpenSLR 28](https://www.openslr.org/28/) RIRS_NOISES | 1, 2 | Apache-2.0 |

## Test data (scores and clip ids only)

Common Voice 17 test (CC0), VoxLingua107 videos not used in training (CC BY 4.0), IndicVoices valid (CC BY 4.0),
VoxPopuli test (CC0), Multilingual LibriSpeech (CC BY 4.0), ParCzech (CC BY), ParlamentParla (CC BY 4.0),
ALFFA Amharic (MIT), NCHLT (CC BY 3.0).

Language names: ISO 639-3 code tables, SIL International.
