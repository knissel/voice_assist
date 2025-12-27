#!/usr/bin/env python3
"""
Benchmark GPU vs Local transcription performance.
Records audio once and transcribes it using both methods.
"""
import time
import pyaudio
import wave
import subprocess
import os
from tools.transcription import create_transcription_service
from dotenv import load_dotenv

load_dotenv()

CHUNK = 1024
RATE = 16000

def record_test_audio(duration=5):
    """Record a test audio sample."""
    print(f"🎤 Recording {duration} seconds of audio for benchmark...")
    print("   (Say something - the longer the better for testing)")
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    temp_path = "/tmp/benchmark_audio.wav"
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    pa.terminate()
    print("✅ Recording complete\n")
    return temp_path

def transcribe_local(audio_path):
    """Transcribe using local whisper.cpp."""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_whisper_path = os.path.join(repo_root, "whisper.cpp", "build", "bin", "whisper-cli")
    default_model_path = os.path.join(repo_root, "whisper.cpp", "models", "ggml-tiny.bin")
    
    whisper_path = os.getenv("WHISPER_PATH", default_whisper_path)
    model_path = os.getenv("MODEL_PATH", default_model_path)
    
    print("💻 Testing LOCAL CPU transcription (whisper.cpp tiny)...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [whisper_path, "-m", model_path, "-f", audio_path, "-nt"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        duration = time.time() - start_time
        
        text = result.stdout.strip()
        if text:
            lines = text.split('\n')
            text = next((line.strip() for line in reversed(lines) 
                       if line.strip() and not line.startswith('[')), "")
        
        return text, duration
        
    except Exception as e:
        return f"Error: {e}", 0

def transcribe_remote(audio_path, service):
    """Transcribe using remote GPU server."""
    import requests
    
    remote_url = os.getenv("WHISPER_REMOTE_URL")
    if not remote_url:
        return "Remote URL not configured", 0
    
    print(f"🚀 Testing REMOTE GPU transcription ({remote_url})...")
    start_time = time.time()
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'audio': f}
            response = requests.post(
                f"{remote_url}/transcribe",
                files=files,
                timeout=30
            )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '').strip()
            server_duration = result.get('duration', 0)
            return text, duration, server_duration
        else:
            return f"Error: {response.status_code}", 0, 0
            
    except Exception as e:
        return f"Error: {e}", 0, 0

def run_benchmark(duration):
    """Run benchmark for a specific duration."""
    print(f"\n{'=' * 60}")
    print(f"  Testing with {duration}s audio")
    print('=' * 60)
    
    # Record test audio
    audio_path = record_test_audio(duration)
    
    # Test 1: Local CPU
    print("-" * 60)
    local_text, local_time = transcribe_local(audio_path)
    print(f"⏱️  Time: {local_time:.2f}s")
    print(f"📝 Text: {local_text[:80]}{'...' if len(local_text) > 80 else ''}")
    print()
    
    # Test 2: Remote GPU
    print("-" * 60)
    service = create_transcription_service()
    remote_text, remote_total_time, remote_server_time = transcribe_remote(audio_path, service)
    print(f"⏱️  Total time (with network): {remote_total_time:.2f}s")
    print(f"⏱️  Server processing time: {remote_server_time:.2f}s")
    print(f"📝 Text: {remote_text[:80]}{'...' if len(remote_text) > 80 else ''}")
    print()
    
    return {
        'duration': duration,
        'local_time': local_time,
        'remote_time': remote_total_time,
        'server_time': remote_server_time
    }

def main():
    import sys
    
    print("=" * 60)
    print("  GPU vs LOCAL Transcription Benchmark")
    print("=" * 60)
    
    # Allow custom duration from command line
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
            results = [run_benchmark(duration)]
        except ValueError:
            print("Usage: python benchmark_transcription.py [duration_seconds]")
            return
    else:
        # Default: test with 20 seconds
        results = [run_benchmark(20)]
    
    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['duration']}s audio:")
        print(f"  Local CPU (tiny):     {r['local_time']:.2f}s")
        print(f"  Remote GPU (total):   {r['remote_time']:.2f}s")
        print(f"  Remote GPU (server):  {r['server_time']:.2f}s")
        
        if r['local_time'] > 0 and r['remote_time'] > 0:
            speedup = r['local_time'] / r['remote_time']
            print(f"  🚀 Speedup: {speedup:.1f}x")
            
            if speedup > 1:
                time_saved = r['local_time'] - r['remote_time']
                print(f"  ⏰ Time saved: {time_saved:.2f}s")
    
    print("\n" + "=" * 60)
    print("\n💡 TIP: For longer audio, try:")
    print("   python benchmark_transcription.py 10")
    print("   python benchmark_transcription.py 30")
    print("=" * 60)

if __name__ == "__main__":
    main()
