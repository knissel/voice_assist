#!/usr/bin/env python3
"""
XTTS v2 inference server for RTX 5090.
Runs Coqui XTTS with GPU acceleration and exposes HTTP endpoint.

Usage:
    python xtts_server.py --port 5001
    
The server will automatically download the XTTS v2 model on first run (~2GB).
"""
import argparse
import io
import os
import time
import logging
from flask import Flask, request, jsonify, send_file
import torch
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = None
speaker_wav_path = None
gpt_cond_latent = None
speaker_embedding = None


def load_model(device: str = "cuda"):
    """Load XTTS v2 model."""
    global model
    
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    
    logger.info("Loading XTTS v2 model...")
    
    # Use TTS library's model manager to download/cache the model
    from TTS.utils.manage import ModelManager
    from TTS.utils.synthesizer import Synthesizer
    
    # Get model path (downloads if not cached)
    model_manager = ModelManager()
    model_path, config_path, _ = model_manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    
    # Load config
    config = XttsConfig()
    config.load_json(config_path)
    
    # Load model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=os.path.dirname(model_path), eval=True)
    model = model.to(device)
    
    logger.info(f"XTTS v2 model loaded on {device}")
    return model


def load_speaker_reference(wav_path: str):
    """Pre-compute speaker embedding from reference audio."""
    global gpt_cond_latent, speaker_embedding
    
    if not os.path.exists(wav_path):
        logger.warning(f"Speaker reference not found: {wav_path}")
        return False
    
    logger.info(f"Computing speaker embedding from: {wav_path}")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[wav_path])
    logger.info("Speaker embedding computed")
    return True


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy", 
        "model": "xtts_v2",
        "speaker_loaded": speaker_embedding is not None
    }), 200


@app.route('/synthesize', methods=['POST'])
def synthesize():
    """
    Synthesize speech from text.
    
    Expects JSON: {"text": "Hello world", "language": "en"}
    Returns: WAV audio file
    """
    global gpt_cond_latent, speaker_embedding
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    language = data.get('language', 'en')
    
    # Use default speaker if none loaded
    if speaker_embedding is None:
        # Use a built-in speaker for now
        logger.warning("No speaker reference loaded, using default")
        # Get a default speaker from the model's speaker manager if available
        try:
            # Try to use a reference audio from the model's samples
            sample_path = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
            if os.path.exists(sample_path):
                load_speaker_reference(sample_path)
            else:
                return jsonify({"error": "No speaker reference configured"}), 500
        except Exception as e:
            return jsonify({"error": f"Speaker setup failed: {e}"}), 500
    
    try:
        start_time = time.time()
        
        # Synthesize
        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
        )
        
        inference_time = time.time() - start_time
        
        # Convert to audio bytes
        audio_tensor = torch.tensor(out["wav"]).unsqueeze(0)
        
        # Save to buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, 24000, format="wav")
        buffer.seek(0)
        
        total_time = time.time() - start_time
        audio_duration = len(out["wav"]) / 24000
        
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


@app.route('/set_speaker', methods=['POST'])
def set_speaker():
    """
    Set speaker reference from uploaded audio.
    
    Expects: multipart/form-data with 'audio' file (WAV, 6-30 seconds recommended)
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Save reference audio
        ref_path = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
        audio_file.save(ref_path)
        
        # Compute embedding
        if load_speaker_reference(ref_path):
            return jsonify({"status": "success", "message": "Speaker reference updated"}), 200
        else:
            return jsonify({"error": "Failed to compute speaker embedding"}), 500
            
    except Exception as e:
        logger.error(f"Set speaker error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS v2 GPU inference server")
    parser.add_argument("--port", type=int, default=5001, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--speaker", default=None, help="Path to speaker reference WAV")
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.device)
    
    # Load speaker reference if provided
    if args.speaker:
        speaker_wav_path = args.speaker
        load_speaker_reference(speaker_wav_path)
    else:
        # Check for default speaker reference
        default_ref = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
        if os.path.exists(default_ref):
            load_speaker_reference(default_ref)
        else:
            logger.warning("No speaker reference provided. Use /set_speaker endpoint or --speaker flag.")
    
    logger.info(f"Starting XTTS server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)
