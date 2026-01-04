# OpenWakeWord Wakeword Training

This project uses OpenWakeWord for wakeword detection. It runs locally with no API keys.

## Quick Start (Built-In Models)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Use the default model:
   ```bash
   WAKEWORD_MODELS=hey_jarvis
   ```

## Record Training Samples

Collect labeled audio clips for your wakeword:

```bash
python scripts/record_wakeword_samples.py --label positive --count 30 --duration 1.0
python scripts/record_wakeword_samples.py --label negative --count 100 --duration 1.0
```

Tips:
- Keep recordings at 16 kHz mono.
- Record in multiple rooms and distances to reduce false triggers.
- Include background noise examples in the negative set.

## Train a Custom Model

OpenWakeWord provides an open-source training toolkit that produces `.onnx` models.
See https://github.com/dscripka/openWakeWord for training details, and point it at your
`positive/` and `negative/` sample folders.
When training finishes, export the model as an ONNX file.

## Use Your Custom Model

1. Place the model in your repo (example):
   ```bash
   mkdir -p models/wakeword
   cp /path/to/your_model.onnx models/wakeword/
   ```
2. Configure the assistant:
   ```bash
   WAKEWORD_MODELS=models/wakeword/your_model.onnx
   WAKEWORD_THRESHOLD=0.5
   ```

You can load multiple models at once by comma-separating:

```bash
WAKEWORD_MODELS=hey_jarvis,models/wakeword/your_model.onnx
```

## Tuning

- `WAKEWORD_THRESHOLD`: raise to reduce false positives, lower to detect more easily.
- `WAKEWORD_FRAME_LENGTH`: 1280 samples (80 ms) is a good default for OpenWakeWord.
