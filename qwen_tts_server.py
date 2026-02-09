#!/usr/bin/env python3
"""
Qwen3-TTS inference server for RTX 5090.
Replaces XTTS v2 with Qwen3-TTS (Qwen/Qwen3-TTS-12Hz-1.7B-Base).
"""
import argparse
import io
import os
import time
import logging
import threading
import re
import queue
from flask import Flask, request, jsonify, send_file, Response
import torch
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None
voice_clone_prompt = None
speaker_wav_path = None
speaker_text_path = None
ref_audio_data = None
ref_text_content = None

DEFAULT_STREAM_CHUNK_SIZE = int(os.getenv("XTTS_STREAM_CHUNK_SIZE", "15")) # Kept for API compatibility
MIN_STREAM_CHUNK_SIZE = 1
MAX_STREAM_CHUNK_SIZE = 200

# Streaming here is "real" at the segment level: we split text into smaller pieces and generate
# each piece sequentially, yielding PCM bytes after each segment completes. Qwen's current public
# Python API does not expose an audio iterator like XTTS' inference_stream.
DEFAULT_STREAM_MAX_SEG_CHARS = int(os.getenv("QWEN_STREAM_MAX_SEG_CHARS", "220"))

model_lock = threading.Lock()

_SENTENCE_SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?\u3002\uFF01\uFF1F])\s+)")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:\uFF0C\uFF1B\uFF1A])\s+")


def _normalize_language(language: str) -> str:
    if not language:
        return "english"

    # Map common codes to full names (Qwen expects lowercase language names)
    lang_map = {
        "en": "english",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        "de": "german",
        "fr": "french",
        "ru": "russian",
        "pt": "portuguese",
        "es": "spanish",
        "it": "italian",
    }
    key = str(language).strip().lower()
    return lang_map.get(key, key)


def _float_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio.astype(np.int16)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16)


def _split_text_for_streaming(text: str, max_chars: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    max_chars = int(max_chars) if max_chars else DEFAULT_STREAM_MAX_SEG_CHARS
    max_chars = max(40, max_chars)

    # 1) sentence-ish split
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]

    # 2) break overly-long parts into clauses
    refined: list[str] = []
    for p in parts:
        if len(p) <= max_chars:
            refined.append(p)
            continue

        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(p) if c and c.strip()]
        if not clauses:
            refined.append(p)
            continue

        buf = ""
        for c in clauses:
            if not buf:
                buf = c
                continue
            if len(buf) + 1 + len(c) <= max_chars:
                buf = f"{buf} {c}"
            else:
                refined.append(buf)
                buf = c
        if buf:
            refined.append(buf)

    # 3) group small parts up to max_chars to reduce per-call overhead
    segments: list[str] = []
    buf = ""
    for p in refined:
        if not buf:
            buf = p
            continue
        if len(buf) + 1 + len(p) <= max_chars:
            buf = f"{buf} {p}"
        else:
            segments.append(buf)
            buf = p
    if buf:
        segments.append(buf)

    return segments or [text]


def _iter_pcm_chunks(audio_int16: np.ndarray, chunk_samples: int):
    if chunk_samples <= 0:
        yield audio_int16.tobytes()
        return

    for start in range(0, len(audio_int16), chunk_samples):
        chunk = audio_int16[start:start + chunk_samples]
        if chunk.size == 0:
            continue
        yield chunk.tobytes()

def load_model(device: str = "cuda"):
    """Load Qwen3-TTS model."""
    global model
    
    logger.info("Loading Qwen3-TTS model...")
    
    try:
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
        )
        logger.info(f"Qwen3-TTS model loaded on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback to float16 or eager if flash-attn fails
        logger.info("Retrying with float16 and eager attention...")
        try:
             model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                device_map=device,
                dtype=torch.float16,
                attn_implementation="eager",
            )
             logger.info(f"Qwen3-TTS model loaded on {device} (fallback mode)")
             
             try:
                 langs = model.get_supported_languages()
                 logger.info(f"Supported languages: {langs}")
             except Exception as ex:
                 logger.warning(f"Could not get supported languages: {ex}")

             return model
        except Exception as e2:
            logger.critical(f"FATAL: Could not load model: {e2}")
            raise e2

def load_speaker_reference(wav_path: str):
    """Load speaker reference and pre-compute prompt."""
    global voice_clone_prompt, ref_audio_data, ref_text_content, speaker_wav_path, speaker_text_path
    
    if not os.path.exists(wav_path):
        logger.warning(f"Speaker reference not found: {wav_path}")
        return False
        
    speaker_wav_path = wav_path
    
    # Check for transcript file (e.g., speaker_reference.txt)
    base_path = os.path.splitext(wav_path)[0]
    txt_path = base_path + ".txt"
    
    ref_text = None
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            logger.info(f"Found speaker transcript: {ref_text[:50]}...")
            speaker_text_path = txt_path
        except Exception as e:
            logger.warning(f"Could not read transcript file: {e}")
    
    if not ref_text:
        logger.warning("No transcript found. Quality might be lower. Using x_vector_only_mode if supported or empty text.")
        # NOTE: Qwen3-TTS base model typically needs ref_text. 
        # If missing, we can try to use a dummy text or x_vector_only_mode if the API supports it.
        # Based on docs: "If you set x_vector_only_mode=True, only the speaker embedding is used so ref_text is not required"
    
    ref_text_content = ref_text
    
    logger.info(f"Computing voice clone prompt from: {wav_path}")
    
    try:
        # Load audio using soundfile
        audio_data, sr = sf.read(wav_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1) # Mono
        
        ref_audio_data = (audio_data, sr)
        
        kwargs = {
            "ref_audio": ref_audio_data,
            "x_vector_only_mode": (ref_text_content is None)
        }
        if ref_text_content is not None:
             kwargs["ref_text"] = ref_text_content

        with model_lock:
            voice_clone_prompt = model.create_voice_clone_prompt(**kwargs)
        logger.info(f"Voice clone prompt computed. Type: {type(voice_clone_prompt)}")
        return True
    except Exception as e:
        logger.error(f"Failed to compute voice clone prompt: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "model": "qwen3_tts_1.7b",
        "speaker_loaded": voice_clone_prompt is not None,
        "ref_text_avail": ref_text_content is not None
    }), 200

@app.errorhandler(Exception)
@app.errorhandler(Exception)
def handle_exception(e):
    logger.exception(f"Unhandled server error: {e}")
    return jsonify({"error": str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """
    Synthesize speech from text.
    Api compatible with XTTS logic.
    """
    global voice_clone_prompt
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    language = _normalize_language(data.get('language', 'English'))

    # Ensure speaker is loaded
    if voice_clone_prompt is None:
        # Try loading default
        default_ref = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
        if os.path.exists(default_ref):
             if not load_speaker_reference(default_ref):
                 return jsonify({"error": "Failed to load default speaker reference"}), 500
        else:
             return jsonify({"error": "No speaker reference configured"}), 500

    try:
        start_time = time.time()
        
        logger.info(f"Synthesizing with: text='{text}', language='{language}'")
        
        # Synthesize (single utterance)
        # Qwen3-TTS returns a list of wavs (one per input text)
        with model_lock:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt
            )
        
        audio_np = wavs[0]
        
        inference_time = time.time() - start_time
        
        # Save to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sr, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        
        total_time = time.time() - start_time
        audio_duration = len(audio_np) / sr
        
        logger.info(f"Synthesized {len(text)} chars in {total_time:.2f}s "
                   f"(inference: {inference_time:.2f}s, audio: {audio_duration:.1f}s)")
        
        response = send_file(
            buffer,
            mimetype='audio/wav',
            as_attachment=False
        )
        response.headers['X-Inference-Time'] = str(inference_time)
        response.headers['X-Audio-Duration'] = str(audio_duration)
        return response
        
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/synthesize_stream', methods=['POST'])
def synthesize_stream():
    """
    Streaming synthesis (segment-based).

    Qwen3-TTS does not currently expose an audio iterator like XTTS' `inference_stream`.
    This endpoint achieves low time-to-first-audio by splitting long text into smaller segments,
    generating each segment sequentially, and streaming PCM bytes as soon as each segment finishes.

    Expects JSON: {"text": "...", "language": "en", "stream_chunk_size": 15}
    Returns: Raw PCM int16 mono audio stream at the model sample rate.
    """
    global voice_clone_prompt
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json(silent=True)
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    language = _normalize_language(data.get('language', 'English'))
        
    try:
        stream_chunk_size = int(data.get('stream_chunk_size', DEFAULT_STREAM_CHUNK_SIZE))
    except (TypeError, ValueError):
        stream_chunk_size = DEFAULT_STREAM_CHUNK_SIZE
    stream_chunk_size = max(MIN_STREAM_CHUNK_SIZE, min(stream_chunk_size, MAX_STREAM_CHUNK_SIZE))

    try:
        max_segment_chars = int(data.get("max_segment_chars", DEFAULT_STREAM_MAX_SEG_CHARS))
    except (TypeError, ValueError):
        max_segment_chars = DEFAULT_STREAM_MAX_SEG_CHARS
        
    # Ensure speaker is loaded
    if voice_clone_prompt is None:
        default_ref = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
        if os.path.exists(default_ref):
             if not load_speaker_reference(default_ref):
                 return jsonify({"error": "Failed to load default speaker reference"}), 500
        else:
             return jsonify({"error": "No speaker reference configured"}), 500

    segments = _split_text_for_streaming(text, max_chars=max_segment_chars)
    if not segments:
        return jsonify({"error": "No text provided"}), 400

    prompt_snapshot = voice_clone_prompt

    # Generate first segment up front so we can set correct headers (sample rate).
    start_time = time.time()
    with model_lock:
        wavs0, sr0 = model.generate_voice_clone(
            text=segments[0],
            language=language,
            voice_clone_prompt=prompt_snapshot,
        )
    audio0_int16 = _float_to_int16(wavs0[0])

    # Each stream_chunk_size unit is ~50ms at 24kHz, scaled to the actual sample rate.
    chunk_unit_samples = max(1, int(round(sr0 / 20.0)))  # 50ms
    stream_chunk_samples = int(stream_chunk_size) * chunk_unit_samples

    # Generate later segments in a background thread while we stream segment 0 to the client.
    seg_queue: "queue.Queue[bytes | None]" = queue.Queue(maxsize=32)
    stop_event = threading.Event()

    def _producer():
        try:
            for seg in segments[1:]:
                if stop_event.is_set():
                    break

                with model_lock:
                    wavs, sr = model.generate_voice_clone(
                        text=seg,
                        language=language,
                        voice_clone_prompt=prompt_snapshot,
                    )
                if sr != sr0:
                    logger.warning(f"Stream: sample rate changed mid-stream: {sr0} -> {sr}")

                audio_int16 = _float_to_int16(wavs[0])
                for b in _iter_pcm_chunks(audio_int16, stream_chunk_samples):
                    if stop_event.is_set():
                        break
                    # Apply backpressure so we don't buffer unbounded audio if the client reads slowly.
                    while not stop_event.is_set():
                        try:
                            seg_queue.put(b, timeout=0.25)
                            break
                        except queue.Full:
                            continue
        except Exception as e:
            logger.error(f"Streaming producer error: {e}")
        finally:
            # Signal completion (or failure) to the response generator.
            while True:
                try:
                    seg_queue.put(None, timeout=0.25)
                    break
                except queue.Full:
                    if stop_event.is_set():
                        break

    producer_thread = None
    if len(segments) > 1:
        producer_thread = threading.Thread(target=_producer, name="qwen_tts_stream_producer", daemon=True)
        producer_thread.start()

    def generate_audio_chunks():
        chunk_count = 0
        total_samples = 0
        first_chunk_sent = False

        try:
            # Segment 0
            for b in _iter_pcm_chunks(audio0_int16, stream_chunk_samples):
                if not first_chunk_sent:
                    ttfa = time.time() - start_time
                    logger.info(f"Stream: first chunk in {ttfa:.3f}s ({len(b)} bytes)")
                    first_chunk_sent = True
                total_samples += len(b) // 2
                chunk_count += 1
                yield b

            # Remaining segments from producer thread
            if producer_thread is not None:
                while True:
                    item = seg_queue.get()
                    if item is None:
                        break
                    total_samples += len(item) // 2
                    chunk_count += 1
                    yield item

            if chunk_count == 0:
                logger.warning("Stream returned no audio chunks")
                return

            elapsed = time.time() - start_time
            audio_duration = total_samples / sr0 if sr0 else 0.0
            logger.info(
                f"Stream complete: {len(segments)} segments, {chunk_count} chunks, "
                f"{audio_duration:.1f}s audio in {elapsed:.2f}s"
            )
        except GeneratorExit:
            # Client disconnected; stop generating.
            stop_event.set()
            logger.info("Stream: client disconnected")
            return
        except Exception as e:
            stop_event.set()
            logger.error(f"Streaming error: {e}")
            return

    response = Response(generate_audio_chunks(), mimetype='audio/pcm')
    response.headers['X-Sample-Rate'] = str(sr0)
    response.headers['X-Channels'] = '1'
    response.headers['X-Format'] = 'int16'
    response.headers['X-Stream-Chunk-Size'] = str(stream_chunk_size)
    return response

@app.route('/set_speaker', methods=['POST'])
def set_speaker():
    """Set speaker reference."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    try:
        ref_path = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
        audio_file.save(ref_path)
        
        # If there is a text file uploaded? currently not supported by this endpoint in single file.
        # We just reload.
        
        if load_speaker_reference(ref_path):
            return jsonify({"status": "success", "message": "Speaker reference updated"}), 200
        else:
            return jsonify({"error": "Failed to compute speaker embedding"}), 500
            
    except Exception as e:
        logger.error(f"Set speaker error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Server")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--speaker", default=None)
    args = parser.parse_args()
    
    model = load_model(args.device)
    
    if args.speaker:
        load_speaker_reference(args.speaker)
    else:
        default_ref = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
        if os.path.exists(default_ref):
            load_speaker_reference(default_ref)
            
    logger.info(f"Starting Qwen3-TTS server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
