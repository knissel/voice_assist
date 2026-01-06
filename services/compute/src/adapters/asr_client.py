"""
Async ASR client with retry logic and circuit breaker.
Designed for remote Whisper server with graceful fallback.
"""
import asyncio
import logging
from typing import Optional
from dataclasses import dataclass
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from ASR transcription."""
    text: str
    duration_ms: int
    language: str = "en"
    confidence: Optional[float] = None


class ASRClientError(Exception):
    """Base exception for ASR client errors."""
    pass


class ASRServerUnavailable(ASRClientError):
    """Raised when ASR server is unavailable."""
    pass


class ASRTranscriptionFailed(ASRClientError):
    """Raised when transcription fails."""
    pass


class AsyncASRClient:
    """
    Async HTTP client for remote Whisper server.
    
    Features:
    - Async HTTP with proper timeouts
    - Automatic retry with exponential backoff
    - Health checking
    - Connection pooling
    
    Usage:
        client = AsyncASRClient("http://192.168.1.100:5000")
        await client.start()
        
        result = await client.transcribe("/tmp/audio.wav")
        print(result.text)
        
        await client.close()
    """
    
    def __init__(
        self,
        base_url: str,
        connect_timeout: float = 5.0,
        transcribe_timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.base_url = base_url.rstrip('/')
        self.connect_timeout = connect_timeout
        self.transcribe_timeout = transcribe_timeout
        self.max_retries = max_retries
        
        self._client: Optional[httpx.AsyncClient] = None
        self._healthy = False
        self._last_health_check = 0.0
        self._health_check_interval = 30.0  # seconds
    
    async def start(self) -> None:
        """Initialize the HTTP client."""
        if self._client is not None:
            return
        
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                self.transcribe_timeout,
                connect=self.connect_timeout
            ),
            limits=httpx.Limits(
                max_connections=2,
                max_keepalive_connections=1
            )
        )
        
        # Initial health check
        await self.health_check()
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    @property
    def is_healthy(self) -> bool:
        """Check if server was healthy at last check."""
        return self._healthy
    
    async def health_check(self) -> bool:
        """
        Check if the ASR server is available.
        
        Returns:
            True if server is healthy
        """
        if not self._client:
            await self.start()
        
        try:
            response = await self._client.get(
                f"{self.base_url}/health",
                timeout=self.connect_timeout
            )
            self._healthy = response.status_code == 200
            
            if self._healthy:
                logger.debug(f"ASR server healthy: {self.base_url}")
            else:
                logger.warning(f"ASR server unhealthy: {response.status_code}")
            
            return self._healthy
            
        except Exception as e:
            self._healthy = False
            logger.warning(f"ASR health check failed: {e}")
            return False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        reraise=True
    )
    async def _do_transcribe(self, audio_path: str) -> TranscriptionResult:
        """Internal transcription with retry logic."""
        with open(audio_path, 'rb') as f:
            files = {'audio': ('audio.wav', f, 'audio/wav')}
            response = await self._client.post(
                f"{self.base_url}/transcribe",
                files=files
            )
        
        if response.status_code != 200:
            raise ASRTranscriptionFailed(
                f"Transcription failed: {response.status_code} - {response.text}"
            )
        
        data = response.json()
        return TranscriptionResult(
            text=data.get('text', '').strip(),
            duration_ms=int(data.get('duration', 0) * 1000),
            language=data.get('language', 'en'),
            confidence=data.get('language_probability')
        )
    
    async def transcribe(self, audio_path: str) -> Optional[TranscriptionResult]:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file (WAV format)
        
        Returns:
            TranscriptionResult or None if failed
        
        Raises:
            ASRServerUnavailable: If server is not reachable
            ASRTranscriptionFailed: If transcription fails
        """
        if not self._client:
            await self.start()
        
        if not self._healthy:
            # Try health check first
            if not await self.health_check():
                raise ASRServerUnavailable(f"ASR server unavailable: {self.base_url}")
        
        try:
            result = await self._do_transcribe(audio_path)
            logger.info(
                f"Transcription complete: {result.duration_ms}ms, "
                f"'{result.text[:50]}{'...' if len(result.text) > 50 else ''}'"
            )
            return result
            
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            self._healthy = False
            raise ASRServerUnavailable(f"ASR server connection failed: {e}")
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise ASRTranscriptionFailed(str(e))


class SyncASRClient:
    """
    Synchronous wrapper around AsyncASRClient.
    For use in non-async code paths.
    """
    
    def __init__(self, base_url: str, **kwargs):
        self._async_client = AsyncASRClient(base_url, **kwargs)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop
    
    def _run(self, coro):
        """Run coroutine in event loop."""
        loop = self._get_loop()
        if loop.is_running():
            # We're in an async context, use run_coroutine_threadsafe
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=60)
        else:
            return loop.run_until_complete(coro)
    
    def start(self) -> None:
        self._run(self._async_client.start())
    
    def close(self) -> None:
        self._run(self._async_client.close())
        if self._loop and not self._loop.is_running():
            self._loop.close()
            self._loop = None
    
    def health_check(self) -> bool:
        return self._run(self._async_client.health_check())
    
    def transcribe(self, audio_path: str) -> Optional[TranscriptionResult]:
        return self._run(self._async_client.transcribe(audio_path))
    
    @property
    def is_healthy(self) -> bool:
        return self._async_client.is_healthy
