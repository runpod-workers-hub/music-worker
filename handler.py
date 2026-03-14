"""
RunPod Serverless Music Generation Handler.

Supports ACE-Step (full songs with vocals) and MusicGen (instrumentals).
Select engine via MUSIC_ENGINE env var.
"""

import base64
import os

import runpod

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MUSIC_ENGINE = os.environ.get("MUSIC_ENGINE", "acestep")

# ---------------------------------------------------------------------------
# Load engine at startup
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"Music RunPod Worker — Loading engine: {MUSIC_ENGINE}")
print("=" * 60)

if MUSIC_ENGINE == "acestep":
    from engines.acestep_engine import ACEStepEngine
    engine = ACEStepEngine()
elif MUSIC_ENGINE == "musicgen":
    from engines.musicgen_engine import MusicGenEngine
    engine = MusicGenEngine()
else:
    raise ValueError(f"Unknown MUSIC_ENGINE: {MUSIC_ENGINE}. Use 'acestep' or 'musicgen'.")


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    """
    Input format:
    {
        "prompt": "upbeat electronic dance music with strong bass",
        "lyrics": "[verse]\\nDancing through the night\\n[chorus]\\nFeel the beat",
        "duration": 60,
        "format": "wav",
        "seed": 42,
        "infer_step": 27,
        "guidance_scale": 15.0
    }

    MusicGen-specific:
    {
        "prompt": "lo-fi hip hop beats to relax to",
        "duration": 30
    }
    """
    job_input = job["input"]

    prompt = job_input.get("prompt", "")
    if not prompt:
        return {"error": "Missing 'prompt' field"}

    duration = float(job_input.get("duration", 60.0))

    # Pass through all extra params to the engine
    kwargs = {}
    for key in ("lyrics", "format", "seed", "infer_step", "guidance_scale",
                "scheduler_type", "cfg_type", "omega_scale"):
        if key in job_input:
            kwargs[key] = job_input[key]

    try:
        audio_bytes, sample_rate, fmt = engine.generate(
            prompt=prompt,
            duration=duration,
            **kwargs,
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
        "format": fmt,
        "sample_rate": sample_rate,
        "duration": duration,
        "engine": MUSIC_ENGINE,
    }


runpod.serverless.start({"handler": handler})
