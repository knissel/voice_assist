"""
Transcription service with GPU offloading and local fallback.
Tries remote Whisper server first, falls back to local whisper.cpp.
"""
import os
import subprocess
import requests
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(
        self,
        remote_url: Optional[str] = None,
        local_whisper_path: Optional[str] = None,
        local_model_path: Optional[str] = None,
        timeout: float = 5.0,
        health_check_interval: float = 30.0
    ):
        """
        Initialize transcription service with fallback.
        
        Args:
            remote_url: URL of remote Whisper server (e.g., "http://192.168.1.100:5000")
            local_whisper_path: Path to whisper.cpp binary
            local_model_path: Path to whisper.cpp model
            timeout: Timeout for remote server requests (seconds)
            health_check_interval: How often to re-check server health (seconds)
        """
        self.remote_url = remote_url
        self.local_whisper_path = local_whisper_path
        self.local_model_path = local_model_path
        self.timeout = timeout
        self.remote_available = False
        self._last_health_check = 0.0
        self._health_check_interval = health_check_interval
        
        # Check remote server availability
        if self.remote_url:
            self._check_remote_health()
    
    def _check_remote_health(self, force: bool = False) -> bool:
        """Check if remote server is available. Uses cached result if recent."""
        import time
        now = time.time()
        
        # Use cached result if recent (unless forced)
        if not force and self._last_health_check > 0:
            if (now - self._last_health_check) < self._health_check_interval:
                return self.remote_available
        
        try:
            response = requests.get(
                f"{self.remote_url}/health",
                timeout=self.timeout
            )
            self.remote_available = response.status_code == 200
            self._last_health_check = now
            if self.remote_available:
                logger.info(f"✅ Remote Whisper server available at {self.remote_url}")
            return self.remote_available
        except (requests.RequestException, Exception) as e:
            self.remote_available = False
            self._last_health_check = now
            logger.debug(f"Remote server unavailable: {e}")
            return False
    
    def _transcribe_remote(self, audio_path: str) -> Optional[str]:
        """Transcribe using remote GPU server."""
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                response = requests.post(
                    f"{self.remote_url}/transcribe",
                    files=files,
                    timeout=30  # Longer timeout for transcription
                )
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                duration = result.get('duration', 0)
                logger.info(f"🚀 GPU transcription: {duration:.2f}s")
                return text
            else:
                logger.warning(f"Remote transcription failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"Remote transcription error: {e}")
            self.remote_available = False
            return None
    
    def _transcribe_local(self, audio_path: str) -> Optional[str]:
        """Transcribe using local whisper.cpp."""
        if not self.local_whisper_path or not self.local_model_path:
            logger.error("Local whisper.cpp not configured")
            return None
        
        try:
            logger.info("💻 Falling back to local CPU transcription...")
            result = subprocess.run(
                [self.local_whisper_path, "-m", self.local_model_path, "-f", audio_path, "-nt"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            text = result.stdout.strip()
            if text:
                lines = text.split('\n')
                text = next((line.strip() for line in reversed(lines) 
                           if line.strip() and not line.startswith('[')), "")
            return text
            
        except subprocess.TimeoutExpired:
            logger.error("Local transcription timed out")
            return None
        except Exception as e:
            logger.error(f"Local transcription error: {e}")
            return None
    
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio with automatic fallback.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text (empty string if both methods fail)
        """
        # Re-check health if cached result says unavailable (server may have recovered)
        if not self.remote_available and self.remote_url:
            self._check_remote_health()
        
        # Try remote first if available
        if self.remote_available:
            text = self._transcribe_remote(audio_path)
            if text:
                return text
            logger.warning("Remote failed, falling back to local...")
        
        # Fallback to local
        text = self._transcribe_local(audio_path)
        return text if text else ""


def create_transcription_service() -> TranscriptionService:
    """Create transcription service from environment variables."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Remote server configuration
    remote_url = os.getenv("WHISPER_REMOTE_URL")  # e.g., "http://192.168.1.100:5000"
    
    # Local fallback configuration
    default_whisper_path = os.path.join(repo_root, "whisper.cpp", "build", "bin", "whisper-cli")
    default_model_path = os.path.join(repo_root, "whisper.cpp", "models", "ggml-tiny.bin")
    
    def _resolve_path(env_value: Optional[str], fallback: str) -> str:
        """Prefer env path when it exists; otherwise use fallback."""
        if env_value and os.path.exists(env_value):
            return env_value
        return fallback

    local_whisper_path = _resolve_path(os.getenv("WHISPER_PATH"), default_whisper_path)
    local_model_path = _resolve_path(os.getenv("MODEL_PATH"), default_model_path)
    
    return TranscriptionService(
        remote_url=remote_url,
        local_whisper_path=local_whisper_path,
        local_model_path=local_model_path,
        timeout=2.0  # Quick health check
    )
