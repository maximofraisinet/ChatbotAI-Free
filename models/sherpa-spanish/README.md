# Spanish Voice Model (Sherpa-ONNX)

This folder should contain the Spanish TTS model files for sherpa-onnx.

## Recommended: Marta Voice from VoicePowered.ai

1. Visit: https://voicepowered.ai/app/voice
2. Search for "Marta" (Spanish female voice)
3. Download the model files
4. Place them in this folder

## Alternative: MMS Spanish (Multi-lingual)

Download from Hugging Face:

```bash
# From this directory (models/sherpa-spanish/)
wget https://huggingface.co/csukuangfj/vits-mms-spa/resolve/main/model.onnx
wget https://huggingface.co/csukuangfj/vits-mms-spa/resolve/main/tokens.txt
```

## Required Files

After downloading, this folder should contain:
- `model.onnx` (or `model.int8.onnx`)
- `tokens.txt`
- (optional) `espeak-ng-data/` directory
- (optional) `lexicon.txt`

## Verify Installation

Run from the project root:
```bash
python -c "from sherpa_wrapper import SherpaWrapper; s = SherpaWrapper(); print('OK')"
```
