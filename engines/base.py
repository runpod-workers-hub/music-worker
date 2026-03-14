"""Base interface for music generation engines."""

from abc import ABC, abstractmethod


class MusicEngine(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        duration: float = 60.0,
        **kwargs,
    ) -> tuple[bytes, int, str]:
        """
        Generate music from a text prompt.

        Returns (audio_bytes, sample_rate, format).
        """
