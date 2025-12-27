#!/usr/bin/env python3
"""
Whisper inference server for RTX 5090.
Runs faster-whisper with GPU acceleration and exposes HTTP endpoint.

Usage:
    python whisper_server.py --model large-v3 --port 5000
"""
import argparse
import io
import time
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model": args.model}), 200

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
    
    try:
        start_time = time.time()
        
        # Save to temporary location
        temp_path = "/tmp/whisper_temp.wav"
        audio_file.save(temp_path)
        
        # Transcribe (optimized for speed)
        transcribe_start = time.time()
        segments, info = model.transcribe(
            temp_path,
            beam_size=1,  # Faster, still accurate for short commands
            language="en",
            vad_filter=False,  # Disable VAD for speed (client already does VAD)
            temperature=0.0
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper GPU inference server")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--compute-type", default="float16", help="Compute type")
    args = parser.parse_args()
    
    logger.info(f"Loading Whisper model: {args.model} on {args.device}")
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type
    )
    logger.info("Model loaded successfully")
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
