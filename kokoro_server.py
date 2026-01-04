#!/usr/bin/env python3
"""
Kokoro inference server.
Runs Kokoro with GPU acceleration (if available) and exposes HTTP endpoints
compatible with the XTTS server client.

Usage:
    python kokoro_server.py --port 5001 --voice af_heart
"""
import argparse
import io
import os
import time
import logging
from typing import Generator, Optional, Tuple

from flask import Flask, Response, jsonify, request, send_file
import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

pipeline = None
pipeline_device = None
pipeline_lang_code = None
pipeline_repo_id = None

DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
DEFAULT_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")
DEFAULT_REPO_ID = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
DEFAULT_SPLIT_PATTERN = os.getenv("KOKORO_SPLIT_PATTERN", r"\n+")
DEFAULT_SAMPLE_RATE = int(os.getenv("KOKORO_SAMPLE_RATE", "24000"))
DEFAULT_STREAM_CHUNK_SIZE = int(os.getenv("KOKORO_STREAM_CHUNK_SIZE", "15"))
MIN_STREAM_CHUNK_SIZE = 1
MAX_STREAM_CHUNK_SIZE = 200


def _get_float_env(name: str, fallback: float) -> float:
    try:
        return float(os.getenv(name, str(fallback)))
    except (TypeError, ValueError):
        return fallback


DEFAULT_SPEED = _get_float_env("KOKORO_SPEED", 1.0)


def load_pipeline(lang_code: str, repo_id: str, device: Optional[str]) -> "KPipeline":
    """Load Kokoro pipeline and model."""
    from kokoro import KPipeline

    logger.info("Loading Kokoro pipeline...")
    device_arg = None if device == "auto" else device
    pipeline_instance = KPipeline(
        lang_code=lang_code,
        repo_id=repo_id,
        device=device_arg,
    )
    return pipeline_instance


def _extract_audio(result) -> Optional[np.ndarray]:
    """Extract audio from a KPipeline result or tuple."""
    if hasattr(result, "audio"):
        audio = result.audio
    elif isinstance(result, (list, tuple)) and len(result) >= 3:
        audio = result[2]
    else:
        audio = result

    if audio is None:
        return None
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    return audio


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.int16)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _iter_kokoro_audio(text: str, voice: str, speed: float) -> Generator[np.ndarray, None, None]:
    results = pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=DEFAULT_SPLIT_PATTERN,
    )
    for result in results:
        audio = _extract_audio(result)
        if audio is not None:
            yield audio


def _iter_pcm_chunks(audio: np.ndarray, chunk_samples: int) -> Generator[Tuple[bytes, int], None, None]:
    audio_int16 = _float_to_int16(audio)
    if chunk_samples <= 0:
        yield audio_int16.tobytes(), len(audio_int16)
        return
    for start in range(0, len(audio_int16), chunk_samples):
        chunk = audio_int16[start:start + chunk_samples]
        if len(chunk) == 0:
            continue
        yield chunk.tobytes(), len(chunk)


def _parse_request_params(data: dict) -> Tuple[str, float]:
    voice = data.get("voice") or DEFAULT_VOICE
    speed = DEFAULT_SPEED
    if "speed" in data:
        try:
            speed = float(data["speed"])
        except (TypeError, ValueError):
            speed = DEFAULT_SPEED
    return voice, speed


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "kokoro",
        "voice": DEFAULT_VOICE,
        "lang_code": pipeline_lang_code,
        "device": pipeline_device,
        "repo_id": pipeline_repo_id,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "model_loaded": pipeline is not None,
    }), 200


@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception(f"Unhandled server error: {e}")
    return jsonify({"error": str(e)}), 500


@app.route("/synthesize", methods=["POST"])
def synthesize():
    if pipeline is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    voice, speed = _parse_request_params(data)

    try:
        start_time = time.time()
        audio_chunks = list(_iter_kokoro_audio(text, voice, speed))
        if not audio_chunks:
            return jsonify({"error": "No audio generated"}), 500

        audio = np.concatenate(audio_chunks)
        inference_time = time.time() - start_time

        buffer = io.BytesIO()
        audio_int16 = _float_to_int16(audio)
        sf.write(buffer, audio_int16, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        audio_duration = len(audio_int16) / DEFAULT_SAMPLE_RATE
        total_time = time.time() - start_time
        logger.info(
            f"Synthesized {len(text)} chars in {total_time:.2f}s "
            f"(inference: {inference_time:.2f}s, audio: {audio_duration:.1f}s)"
        )

        response = send_file(buffer, mimetype="audio/wav", as_attachment=False)
        response.headers["X-Inference-Time"] = str(inference_time)
        response.headers["X-Audio-Duration"] = str(audio_duration)
        return response

    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/synthesize_stream", methods=["POST"])
def synthesize_stream():
    if pipeline is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    voice, speed = _parse_request_params(data)
    try:
        stream_chunk_size = int(data.get("stream_chunk_size", DEFAULT_STREAM_CHUNK_SIZE))
    except (TypeError, ValueError):
        stream_chunk_size = DEFAULT_STREAM_CHUNK_SIZE
    stream_chunk_size = max(MIN_STREAM_CHUNK_SIZE, min(stream_chunk_size, MAX_STREAM_CHUNK_SIZE))

    # Each unit is roughly 50ms at 24kHz (1200 samples).
    stream_chunk_samples = stream_chunk_size * 1200

    def generate_audio_chunks():
        start_time = time.time()
        chunk_count = 0
        total_samples = 0
        first_chunk_sent = False

        try:
            for audio in _iter_kokoro_audio(text, voice, speed):
                for chunk_bytes, chunk_samples in _iter_pcm_chunks(audio, stream_chunk_samples):
                    if not first_chunk_sent:
                        ttfa = time.time() - start_time
                        logger.info(f"Stream: first chunk in {ttfa:.3f}s ({chunk_samples} samples)")
                        first_chunk_sent = True
                    total_samples += chunk_samples
                    chunk_count += 1
                    yield chunk_bytes

            if chunk_count == 0:
                logger.warning("Stream returned no audio chunks")
                return

            elapsed = time.time() - start_time
            audio_duration = total_samples / DEFAULT_SAMPLE_RATE
            logger.info(
                f"Stream complete: {chunk_count} chunks, {audio_duration:.1f}s audio in {elapsed:.2f}s"
            )

        except Exception as e:
            logger.error(f"Streaming synthesis error: {e}")
            return

    response = Response(generate_audio_chunks(), mimetype="audio/pcm")
    response.headers["X-Sample-Rate"] = str(DEFAULT_SAMPLE_RATE)
    response.headers["X-Channels"] = "1"
    response.headers["X-Format"] = "int16"
    response.headers["X-Stream-Chunk-Size"] = str(stream_chunk_size)
    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kokoro GPU inference server")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu/auto)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Default voice (e.g. af_heart)")
    parser.add_argument("--lang-code", default=DEFAULT_LANG_CODE, help="Language code (a, b, e, f, h, i, p, j, z)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id")
    args = parser.parse_args()

    DEFAULT_VOICE = args.voice
    pipeline_device = args.device
    pipeline_lang_code = args.lang_code
    pipeline_repo_id = args.repo_id

    pipeline = load_pipeline(pipeline_lang_code, pipeline_repo_id, pipeline_device)

    logger.info(f"Starting Kokoro server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
