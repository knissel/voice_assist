
import os
import torch
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import logging

logging.basicConfig(level=logging.INFO)

print("Loading model...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda",
    dtype=torch.float16,
    attn_implementation="eager"
)

# Load audio
ref_audio_path = "speaker_reference.wav"
audio_data, sr = sf.read(ref_audio_path)
if len(audio_data.shape) > 1:
    audio_data = audio_data.mean(axis=1)

print("Creating prompt...")
voice_clone_prompt = model.create_voice_clone_prompt(
    ref_audio=(audio_data, sr),
    x_vector_only_mode=True
)

test_langs = ["English", "english", "en", "auto"]

for lang in test_langs:
    print(f"Testing language: {lang}")
    try:
        wavs, sr = model.generate_voice_clone(
            text="Hello world. This is a longer sentence to test duration.",
            language=lang,
            voice_clone_prompt=voice_clone_prompt
        )
        duration = len(wavs[0]) / sr
        print(f"SUCCESS: {lang}, Duration: {duration:.2f}s")
        sf.write(f"test_{lang}.wav", wavs[0], sr)
    except Exception as e:
        print(f"FAILED: {lang} - {e}")
