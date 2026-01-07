"""
TTS adapter with sentence splitting for faster time-to-first-audio.
Supports barge-in interruption.
"""
import re
import threading
import logging
from typing import Optional, Generator, Callable
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """TTS configuration."""
    sample_rate: int = 22050
    channels: int = 1
    dtype: str = 'int16'


class TTSAdapter:
    """
    Wrapper around Piper TTS with optimizations for voice assistant use.
    
    Features:
    - Sentence splitting for faster time-to-first-audio
    - Barge-in support via interrupt event
    - Pre-warming for faster first synthesis
    - Thread-safe
    
    Usage:
        adapter = TTSAdapter("/path/to/model.onnx")
        adapter.load()
        adapter.prewarm()
        
        # Simple playback
        adapter.speak("Hello, how can I help you?")
        
        # With interrupt support
        interrupt = threading.Event()
        adapter.speak("Long response...", interrupt_event=interrupt)
        # Call interrupt.set() to stop playback
    """
    
    # Sentence splitting pattern
    SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')
    
    def __init__(
        self,
        model_path: str,
        use_sentence_splitting: bool = True,
        min_sentence_length: int = 10
    ):
        self.model_path = model_path
        self.use_sentence_splitting = use_sentence_splitting
        self.min_sentence_length = min_sentence_length
        
        self._voice = None
        self._config: Optional[TTSConfig] = None
        self._lock = threading.Lock()
        self._is_speaking = False
    
    def load(self) -> bool:
        """
        Load the Piper voice model.
        
        Returns:
            True if loaded successfully
        """
        try:
            from piper.voice import PiperVoice
            
            with self._lock:
                self._voice = PiperVoice.load(self.model_path)
                self._config = TTSConfig(
                    sample_rate=self._voice.config.sample_rate,
                    channels=1,
                    dtype='int16'
                )
            
            logger.info(f"Loaded TTS model: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            return False
    
    def prewarm(self) -> None:
        """Pre-warm the model with a dummy synthesis."""
        if not self._voice:
            return
        
        try:
            # Synthesize a short phrase to warm up JIT
            _ = list(self._voice.synthesize("Ready"))
            logger.info("TTS model pre-warmed")
        except Exception as e:
            logger.warning(f"TTS prewarm failed: {e}")
    
    @property
    def is_loaded(self) -> bool:
        return self._voice is not None
    
    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
    
    @property
    def config(self) -> Optional[TTSConfig]:
        return self._config
    
    def split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences for incremental synthesis.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        if not self.use_sentence_splitting:
            return [text]
        
        # Skip splitting for short text (overhead not worth it)
        if len(text) < self.min_sentence_length:
            return [text]
        
        sentences = self.SENTENCE_PATTERN.split(text)
        
        # Merge very short sentences with the next one
        merged = []
        buffer = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if buffer:
                sentence = buffer + " " + sentence
                buffer = ""
            
            if len(sentence) < self.min_sentence_length:
                buffer = sentence
            else:
                merged.append(sentence)
        
        if buffer:
            if merged:
                merged[-1] = merged[-1] + " " + buffer
            else:
                merged.append(buffer)
        
        return merged if merged else [text]
    
    def synthesize(
        self,
        text: str,
        interrupt_event: Optional[threading.Event] = None
    ) -> Generator[bytes, None, None]:
        """
        Synthesize text to audio chunks.
        
        Args:
            text: Text to synthesize
            interrupt_event: Optional event to check for interruption
        
        Yields:
            Audio data as bytes (int16)
        """
        if not self._voice:
            logger.error("TTS model not loaded")
            return
        
        sentences = self.split_sentences(text)
        
        for sentence in sentences:
            if interrupt_event and interrupt_event.is_set():
                logger.info("TTS interrupted")
                break
            
            try:
                for audio_chunk in self._voice.synthesize(sentence):
                    if interrupt_event and interrupt_event.is_set():
                        break
                    yield audio_chunk.audio_int16_bytes
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
                break
    
    def speak(
        self,
        text: str,
        interrupt_event: Optional[threading.Event] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[bool], None]] = None
    ) -> bool:
        """
        Synthesize and play text through speakers.
        
        Args:
            text: Text to speak
            interrupt_event: Optional event to check for interruption
            on_start: Callback when speech starts
            on_end: Callback when speech ends (bool = was_interrupted)
        
        Returns:
            True if completed, False if interrupted
        """
        if not self._voice:
            logger.error("TTS model not loaded")
            return False
        
        if not text:
            return True
        
        try:
            import sounddevice as sd
            
            self._is_speaking = True
            was_interrupted = False
            
            stream = sd.OutputStream(
                samplerate=self._config.sample_rate,
                channels=self._config.channels,
                dtype=self._config.dtype
            )
            stream.start()
            
            if on_start:
                on_start()
            
            try:
                for audio_bytes in self.synthesize(text, interrupt_event):
                    if interrupt_event and interrupt_event.is_set():
                        was_interrupted = True
                        break
                    
                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    stream.write(audio_data)
                    
            finally:
                stream.stop()
                stream.close()
                self._is_speaking = False
                
                if on_end:
                    on_end(was_interrupted)
            
            return not was_interrupted
            
        except Exception as e:
            logger.error(f"TTS playback error: {e}")
            self._is_speaking = False
            return False
    
    def speak_async(
        self,
        text: str,
        interrupt_event: Optional[threading.Event] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_end: Optional[Callable[[bool], None]] = None
    ) -> threading.Thread:
        """
        Speak text in a background thread.
        
        Returns:
            Thread object (already started)
        """
        thread = threading.Thread(
            target=self.speak,
            args=(text, interrupt_event, on_start, on_end),
            daemon=True,
            name="TTS"
        )
        thread.start()
        return thread
    
    def estimate_duration_ms(self, text: str) -> int:
        """
        Estimate speech duration in milliseconds.
        Rough estimate: ~150 words per minute.
        """
        words = len(text.split())
        return int((words / 150) * 60 * 1000)
