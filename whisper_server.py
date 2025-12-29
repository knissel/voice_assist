#!/usr/bin/env python3
"""
Whisper inference server for RTX 5090.
Runs faster-whisper with GPU acceleration and exposes HTTP endpoint.

Usage:
    python whisper_server.py --model large-v3 --port 5000
"""
import argparse
import os
import tempfile
import time
import logging
import soundfile as sf
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel

try:
    import torch
    import torchaudio
    _HAS_TORCHAUDIO = True
except Exception:
    torch = None
    torchaudio = None
    _HAS_TORCHAUDIO = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None
MODEL_NAME = None
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
BEST_OF = int(os.getenv("WHISPER_BEST_OF", "1"))
TEMPERATURE = float(os.getenv("WHISPER_TEMPERATURE", "0.0"))
VAD_FILTER = os.getenv("WHISPER_VAD_FILTER", "false").lower() == "true"
WORD_TIMESTAMPS = os.getenv("WHISPER_WORD_TIMESTAMPS", "false").lower() == "true"
WITHOUT_TIMESTAMPS = os.getenv("WHISPER_WITHOUT_TIMESTAMPS", "true").lower() == "true"
CONDITION_ON_PREVIOUS_TEXT = os.getenv("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "false").lower() == "true"
IN_MEMORY_DECODE = os.getenv("WHISPER_IN_MEMORY", "true").lower() == "true"
TARGET_SAMPLE_RATE = 16000


def _load_audio_in_memory(audio_file):
    """Load audio from request file into a mono float32 numpy array."""
    try:
        audio_file.stream.seek(0)
        audio, sr = sf.read(audio_file.stream, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SAMPLE_RATE:
            if not _HAS_TORCHAUDIO:
                raise RuntimeError("torchaudio not available for resampling")
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_resampled = torchaudio.functional.resample(
                audio_tensor, sr, TARGET_SAMPLE_RATE
            )
            audio = audio_resampled.squeeze(0).numpy()
        return audio
    except Exception as e:
        logger.warning(f"In-memory decode failed: {e}")
        return None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": MODEL_NAME}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file.
    Expects: multipart/form-data with 'audio' file
    Returns: JSON with 'text' and 'duration'
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']

    temp_path = None
    try:
        start_time = time.time()

        audio_input = None
        if IN_MEMORY_DECODE:
            audio_input = _load_audio_in_memory(audio_file)

        if audio_input is None:
            # Save to a unique temp file to avoid collisions under concurrency
            try:
                audio_file.stream.seek(0)
            except Exception:
                pass
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_path = tmp.name
            audio_file.save(temp_path)
            audio_input = temp_path
        
        # Transcribe (optimized for speed)
        transcribe_start = time.time()
        segments, info = model.transcribe(
            audio_input,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            language=LANGUAGE,
            vad_filter=VAD_FILTER,
            temperature=TEMPERATURE,
            word_timestamps=WORD_TIMESTAMPS,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            without_timestamps=WITHOUT_TIMESTAMPS,
        )
        
        # Combine segments
        segment_start = time.time()
        text = " ".join([segment.text for segment in segments]).strip()
        segment_time = time.time() - segment_start
        
        duration = time.time() - start_time
        transcribe_time = time.time() - transcribe_start
        
        logger.info(f"Transcribed in {duration:.2f}s (model: {transcribe_time:.2f}s, segments: {segment_time:.2f}s): {text}")
        
        return jsonify({
            "text": text,
            "duration": duration,
            "language": info.language,
            "language_probability": info.language_probability
        }), 200
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper GPU inference server")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--compute-type", default="float16", help="Compute type")
    parser.add_argument("--language", default=LANGUAGE, help="Language code (e.g., en)")
    parser.add_argument("--beam-size", type=int, default=BEAM_SIZE, help="Beam size")
    parser.add_argument("--best-of", type=int, default=BEST_OF, help="Best-of candidates")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Temperature")
    parser.add_argument("--vad-filter", action="store_true", default=VAD_FILTER, help="Enable VAD filter")
    parser.add_argument("--word-timestamps", action="store_true", default=WORD_TIMESTAMPS, help="Word timestamps")
    parser.add_argument("--without-timestamps", action="store_true", default=WITHOUT_TIMESTAMPS, help="Disable timestamps")
    parser.add_argument(
        "--condition-on-previous-text",
        action="store_true",
        default=CONDITION_ON_PREVIOUS_TEXT,
        help="Condition on previous text"
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        default=IN_MEMORY_DECODE,
        help="Decode audio in-memory instead of temp files"
    )
    args = parser.parse_args()
    MODEL_NAME = args.model
    LANGUAGE = args.language
    BEAM_SIZE = args.beam_size
    BEST_OF = args.best_of
    TEMPERATURE = args.temperature
    VAD_FILTER = args.vad_filter
    WORD_TIMESTAMPS = args.word_timestamps
    WITHOUT_TIMESTAMPS = args.without_timestamps
    CONDITION_ON_PREVIOUS_TEXT = args.condition_on_previous_text
    IN_MEMORY_DECODE = args.in_memory
    
    logger.info(f"Loading Whisper model: {args.model} on {args.device}")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type
    )
    logger.info("Model loaded successfully")
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
