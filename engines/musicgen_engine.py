"""MusicGen engine — Meta's instrumental music generation."""

import io
import os

import numpy as np
import soundfile as sf

from engines.base import MusicEngine

MODEL_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


class MusicGenEngine(MusicEngine):
    def __init__(self):
        from audiocraft.models import MusicGen

        model_size = os.environ.get("MUSICGEN_MODEL", "medium")
        model_name = MODEL_MAP.get(model_size, model_size)

        print(f"Loading MusicGen engine ({model_name})...")
        self.model = MusicGen.get_pretrained(model_name)
        self.sample_rate = self.model.sample_rate  # 32000
        print("MusicGen engine ready.")

    def generate(
        self,
        prompt: str,
        duration: float = 30.0,
        **kwargs,
    ) -> tuple[bytes, int, str]:
        # MusicGen max is ~30s reliably
        duration = min(duration, 30.0)
        self.model.set_generation_params(duration=duration)

        wav = self.model.generate([prompt])  # [batch, channels, samples]
        audio = wav[0].cpu().numpy()

        # Convert from [channels, samples] to [samples, channels]
        if audio.ndim == 2:
            audio = audio.T

        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        return buf.getvalue(), self.sample_rate, "wav"
