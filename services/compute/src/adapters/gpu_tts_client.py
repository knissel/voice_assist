"""
GPU TTS Client with Piper fallback.

Attempts to use remote XTTS server for high-quality synthesis.
Falls back to local Piper TTS if server is unavailable or slow.
"""
import io
import os
import time
import logging
import threading
import numpy as np
import requests
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUTTSConfig:
    """Configuration for GPU TTS client."""
    server_url: str = "http://localhost:5001"
    timeout_seconds: float = 3.0  # Max time to wait for GPU server
    stream_timeout_seconds: float = 30.0
    stream_chunk_size: int = 15
    stream_chunk_bytes: int = 2400
    fallback_enabled: bool = True
    language: str = "en"
    sample_rate: int = 24000  # XTTS outputs 24kHz


class GPUTTSClient:
    """
    TTS client that uses remote GPU server with local Piper fallback.
    
    Features:
    - Automatic fallback to Piper if GPU server is unavailable
    - Health checking to avoid repeated failed requests
    - Configurable timeout for latency control
    - Thread-safe
    
    Usage:
        client = GPUTTSClient(
            server_url="http://192.168.1.100:5001",
            piper_voice=piper_voice,  # Optional Piper fallback
            piper_sample_rate=22050
        )
        
        # Synthesize and get audio data
        audio_data, sample_rate = client.synthesize("Hello world")
        
        # Or synthesize and play directly
        client.speak("Hello world", audio_output)
    """
    
    def __init__(
        self,
        server_url: str = None,
        piper_voice = None,
        piper_sample_rate: int = 22050,
        timeout_seconds: float = 3.0,
        stream_timeout_seconds: float = 30.0,
        stream_chunk_size: int = 15,
        stream_chunk_bytes: int = 2400,
        language: str = "en"
    ):
        self.server_url = server_url or os.getenv("XTTS_SERVER_URL", "http://localhost:5001")
        self.piper_voice = piper_voice
        self.piper_sample_rate = piper_sample_rate
        self.timeout = timeout_seconds
        self.stream_timeout = stream_timeout_seconds
        self.stream_chunk_size = stream_chunk_size
        self.stream_chunk_bytes = stream_chunk_bytes
        self.language = language
        self._session = requests.Session()
        
        self._lock = threading.Lock()
        self._server_healthy = None  # None = unknown, True/False = known state
        self._last_health_check = 0
        self._health_check_interval = 30  # Re-check every 30 seconds
        
        # Stats
        self._gpu_calls = 0
        self._piper_calls = 0
        self._gpu_failures = 0
    
    def _check_server_health(self) -> bool:
        """Check if GPU server is available."""
        now = time.time()
        
        # Use cached result if recent
        if self._server_healthy is not None and (now - self._last_health_check) < self._health_check_interval:
            return self._server_healthy
        
        response = None
        try:
            response = self._session.get(
                f"{self.server_url}/health",
                timeout=1.0
            )
            self._server_healthy = response.status_code == 200
            self._last_health_check = now
            
            if self._server_healthy:
                logger.info(f"GPU TTS server healthy: {self.server_url}")
            else:
                logger.warning(f"GPU TTS server unhealthy: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self._server_healthy = False
            self._last_health_check = now
            logger.warning(f"GPU TTS server unreachable: {e}")
        finally:
            if response is not None:
                response.close()
        
        return self._server_healthy
    
    def _synthesize_gpu(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize using GPU server. Returns (audio_data, sample_rate) or None."""
        response = None
        try:
            start = time.time()
            
            response = self._session.post(
                f"{self.server_url}/synthesize",
                json={"text": text, "language": self.language},
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.warning(f"GPU TTS error: {response.status_code}")
                self._gpu_failures += 1
                return None
            
            # Parse WAV audio from response
            import wave
            audio_buffer = io.BytesIO(response.content)
            
            with wave.open(audio_buffer, 'rb') as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                audio_bytes = wav.readframes(n_frames)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            elapsed = time.time() - start
            self._gpu_calls += 1
            
            # Get timing from headers if available
            inference_time = response.headers.get('X-Inference-Time', 'N/A')
            logger.info(f"GPU TTS: {len(text)} chars in {elapsed:.2f}s (inference: {inference_time}s)")
            
            return audio_data, sample_rate
            
        except requests.exceptions.Timeout:
            logger.warning(f"GPU TTS timeout after {self.timeout}s")
            self._gpu_failures += 1
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"GPU TTS request failed: {e}")
            self._gpu_failures += 1
            # Mark server as unhealthy to avoid repeated failures
            self._server_healthy = False
            self._last_health_check = time.time()
            return None
        except Exception as e:
            logger.error(f"GPU TTS unexpected error: {e}")
            self._gpu_failures += 1
            return None
        finally:
            try:
                if response is not None:
                    response.close()
            except Exception:
                pass
    
    def synthesize_stream(self, text: str, audio_output) -> bool:
        """
        Stream synthesis - plays audio chunks as they're generated.
        
        This significantly reduces time-to-first-audio compared to synthesize().
        
        Args:
            text: Text to synthesize
            audio_output: Audio output object with write() and sample_rate attributes
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            return False
        
        with self._lock:
            if not self._check_server_health():
                logger.info("GPU server not healthy, falling back to non-streaming")
                return False
            
            response = None
            try:
                start = time.time()
                chunk_count = 0
                total_samples = 0
                
                # Use streaming endpoint with iter_content
                response = self._session.post(
                    f"{self.server_url}/synthesize_stream",
                    json={
                        "text": text,
                        "language": self.language,
                        "stream_chunk_size": self.stream_chunk_size
                    },
                    timeout=(self.timeout, self.stream_timeout),
                    stream=True  # Enable streaming response
                )
                
                if response.status_code != 200:
                    logger.warning(f"GPU TTS stream error: {response.status_code}")
                    self._gpu_failures += 1
                    return False
                
                sample_rate = int(response.headers.get('X-Sample-Rate', 24000))
                
                # Buffer for accumulating chunks (for resampling)
                # We need to resample from 24kHz to audio_output.sample_rate
                target_rate = getattr(audio_output, 'sample_rate', sample_rate)
                
                # Read and play chunks as they arrive
                # Use larger chunk size for network efficiency, smaller for low latency
                for raw_chunk in response.iter_content(chunk_size=self.stream_chunk_bytes):
                    if not raw_chunk:
                        continue
                    
                    audio_chunk = np.frombuffer(raw_chunk, dtype=np.int16)
                    total_samples += len(audio_chunk)
                    chunk_count += 1
                    
                    # Log time-to-first-audio
                    if chunk_count == 1:
                        ttfa = time.time() - start
                        logger.info(f"GPU TTS stream: first audio in {ttfa:.3f}s")
                    
                    # Resample if needed
                    if target_rate != sample_rate:
                        audio_chunk = self._resample(audio_chunk, sample_rate, target_rate)
                    
                    # Play immediately
                    audio_output.write(audio_chunk)
                
                elapsed = time.time() - start
                audio_duration = total_samples / sample_rate if sample_rate else 0.0
                
                if chunk_count == 0:
                    logger.warning("GPU TTS stream returned no audio chunks")
                    self._gpu_failures += 1
                    return False
                
                self._gpu_calls += 1
                logger.info(f"GPU TTS stream: {len(text)} chars, {chunk_count} chunks, "
                           f"{audio_duration:.1f}s audio in {elapsed:.2f}s")
                
                return True
                
            except requests.exceptions.Timeout:
                logger.warning(f"GPU TTS stream timeout after {self.timeout}s")
                self._gpu_failures += 1
                return False
            except requests.exceptions.RequestException as e:
                logger.warning(f"GPU TTS stream request failed: {e}")
                self._gpu_failures += 1
                self._server_healthy = False
                self._last_health_check = time.time()
                return False
            except Exception as e:
                logger.error(f"GPU TTS stream unexpected error: {e}")
                self._gpu_failures += 1
                return False
            finally:
                try:
                    if response is not None:
                        response.close()
                except Exception:
                    pass
    
    def _synthesize_piper(self, text: str) -> Optional[tuple[np.ndarray, int]]:
        """Synthesize using local Piper. Returns (audio_data, sample_rate) or None."""
        if self.piper_voice is None:
            logger.error("Piper fallback not available")
            return None
        
        try:
            start = time.time()
            
            # Collect all audio chunks
            audio_chunks = []
            for chunk in self.piper_voice.synthesize(text):
                audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
            
            if not audio_chunks:
                return None
            
            audio_data = np.concatenate(audio_chunks)
            elapsed = time.time() - start
            self._piper_calls += 1
            
            logger.info(f"Piper TTS: {len(text)} chars in {elapsed:.2f}s")
            
            return audio_data, self.piper_sample_rate
            
        except Exception as e:
            logger.error(f"Piper TTS error: {e}")
            return None
    
    def synthesize(self, text: str, prefer_gpu: bool = True) -> Optional[tuple[np.ndarray, int]]:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            prefer_gpu: If True, try GPU first then fallback to Piper
            
        Returns:
            Tuple of (audio_data as np.ndarray int16, sample_rate) or None
        """
        if not text:
            return None
        
        with self._lock:
            if prefer_gpu and self._check_server_health():
                result = self._synthesize_gpu(text)
                if result is not None:
                    return result
                # Fall through to Piper
            
            # Use Piper (either as fallback or primary)
            return self._synthesize_piper(text)
    
    def speak(
        self,
        text: str,
        audio_output,  # PersistentAudioOutput or similar
        prefer_gpu: bool = True,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """
        Synthesize and play text.
        
        Args:
            text: Text to speak
            audio_output: Audio output object with write() method
            prefer_gpu: If True, try GPU first
            on_start: Callback when speech starts
            on_end: Callback when speech ends (bool = was_gpu)
            
        Returns:
            True if successful
        """
        result = self.synthesize(text, prefer_gpu=prefer_gpu)
        if result is None:
            return False
        
        audio_data, sample_rate = result
        
        if on_start:
            on_start()
        
        try:
            # Resample if needed (GPU is 24kHz, Piper is typically 22050Hz)
            if hasattr(audio_output, 'sample_rate') and audio_output.sample_rate != sample_rate:
                audio_data = self._resample(audio_data, sample_rate, audio_output.sample_rate)
            
            audio_output.write(audio_data)
            
            if on_end:
                on_end(sample_rate == 24000)  # True if GPU was used
            
            return True
            
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False
    
    def _resample(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio using scipy for better quality and speed."""
        if from_rate == to_rate:
            return audio
        
        try:
            from scipy.signal import resample_poly
            from math import gcd
            
            # Find the GCD to minimize resampling ratio
            g = gcd(from_rate, to_rate)
            up = to_rate // g
            down = from_rate // g
            
            # resample_poly is faster and higher quality than linear interp
            resampled = resample_poly(audio.astype(np.float32), up, down)
            return np.clip(resampled, -32768, 32767).astype(np.int16)
            
        except ImportError:
            # Fallback to linear interpolation if scipy not available
            duration = len(audio) / from_rate
            new_length = int(duration * to_rate)
            old_indices = np.linspace(0, len(audio) - 1, new_length)
            new_audio = np.interp(old_indices, np.arange(len(audio)), audio.astype(np.float32))
            return new_audio.astype(np.int16)
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "gpu_calls": self._gpu_calls,
            "piper_calls": self._piper_calls,
            "gpu_failures": self._gpu_failures,
            "server_healthy": self._server_healthy,
            "server_url": self.server_url
        }
    
    def force_health_check(self) -> bool:
        """Force a health check of the GPU server."""
        self._last_health_check = 0  # Reset cache
        with self._lock:
            return self._check_server_health()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        try:
            self._session.close()
        except Exception:
            pass
