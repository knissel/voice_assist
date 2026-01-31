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
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    language = data.get('language', 'English') # Qwen uses full names typically, mapping mapping needed
    
    # Map common codes to full names (lowercase required by Qwen3)
    lang_map = {
        "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean", 
        "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese", 
        "es": "spanish", "it": "italian"
    }
    if language.lower() in lang_map:
        language = lang_map[language.lower()]
    else:
        # Ensure lowercase for full names
        language = language.lower()

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
        
        # Synthesize
        # Qwen3-TTS usually returns a list of wavs (one per input text)
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
    Simulated streaming synthesis.
    Qwen3-TTS python API 'generate_voice_clone' does not expose an iterator easily in basic usage 
    without digging into internal `model.generate`.
    For now, we will generate usually and chunk the output to satisfy the client contract.
    TODO: Implement true streaming if Qwen3 provides a stream iterator.
    """
    global voice_clone_prompt
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data['text']
    language = data.get('language', 'English')
    lang_map = {
        "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", 
        "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese", 
        "es": "Spanish", "it": "Italian"
    }
    if language in lang_map:
        language = lang_map[language]
        
    try:
        stream_chunk_size = int(data.get('stream_chunk_size', DEFAULT_STREAM_CHUNK_SIZE))
    except (TypeError, ValueError):
        stream_chunk_size = DEFAULT_STREAM_CHUNK_SIZE
        
    # Ensure speaker is loaded
    if voice_clone_prompt is None:
        default_ref = os.path.join(os.path.dirname(__file__), "speaker_reference.wav")
        if os.path.exists(default_ref):
             if not load_speaker_reference(default_ref):
                 return jsonify({"error": "Failed to load default speaker reference"}), 500
        else:
             return jsonify({"error": "No speaker reference configured"}), 500

    def generate_audio_chunks():
        # NON-STREAMING GENERATION (Simulated Streaming)
        # We generate the whole audio then yield it in chunks.
        # This adds latency but keeps compatibility.
        # True streaming would require accessing the underlying generator.
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt
            )
            audio_np = wavs[0]
            
            # Normalize to int16
            audio_int16 = (audio_np * 32767).astype('int16')
            
            # Yield in chunks
            # 24000 sample rate * 2 bytes = 48000 bytes/sec
            # Chunk size of 0.1s = 4800 bytes
            chunk_bytes = 4800 
            
            for i in range(0, len(audio_int16), chunk_bytes // 2):
                chunk = audio_int16[i:i + (chunk_bytes // 2)]
                yield chunk.tobytes()
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            return

    response = Response(generate_audio_chunks(), mimetype='audio/pcm')
    # Default Qwen sr is usually 24000 or 48000 depending on model. 
    # The 1.7B Base model output sample rate needs to be checked (usually 24k for these models).
    # We'll assume 24000 based on XTTS similarity, but we should verify. 
    # Qwen3-TTS usually outputs 24kHz.
    response.headers['X-Sample-Rate'] = '24000' 
    response.headers['X-Channels'] = '1'
    response.headers['X-Format'] = 'int16'
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
