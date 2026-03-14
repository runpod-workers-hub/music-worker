"""ACE-Step engine — full-song AI music with vocals and lyrics."""

import os
import tempfile

from engines.base import MusicEngine

ACESTEP_DEFAULTS = {
    "infer_step": 27,
    "guidance_scale": 15.0,
    "scheduler_type": "euler",
    "cfg_type": "apg",
    "omega_scale": 10.0,
    "use_erg_tag": True,
    "use_erg_lyric": True,
    "use_erg_diffusion": True,
}


class ACEStepEngine(MusicEngine):
    def __init__(self):
        from acestep.pipeline_ace_step import ACEStepPipeline

        print("Loading ACE-Step engine...")
        self.pipe = ACEStepPipeline(
            checkpoint_dir=None,  # auto-downloads to ~/.cache/ace-step
            device_id=0,
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=bool(os.environ.get("CPU_OFFLOAD", "")),
        )
        print("ACE-Step engine ready.")

    def generate(
        self,
        prompt: str,
        duration: float = 60.0,
        **kwargs,
    ) -> tuple[bytes, int, str]:
        lyrics = kwargs.get("lyrics", "[Instrumental]")
        audio_format = kwargs.get("format", "wav")
        seed = kwargs.get("seed", -1)
        manual_seeds = [seed] if seed >= 0 else None

        with tempfile.TemporaryDirectory() as tmpdir:
            params = {**ACESTEP_DEFAULTS}
            for key in ("infer_step", "guidance_scale", "scheduler_type",
                        "cfg_type", "omega_scale"):
                if key in kwargs:
                    params[key] = kwargs[key]

            output_paths = self.pipe(
                prompt=prompt,
                lyrics=lyrics,
                audio_duration=duration,
                infer_step=params["infer_step"],
                guidance_scale=params["guidance_scale"],
                scheduler_type=params["scheduler_type"],
                cfg_type=params["cfg_type"],
                omega_scale=params["omega_scale"],
                use_erg_tag=params["use_erg_tag"],
                use_erg_lyric=params["use_erg_lyric"],
                use_erg_diffusion=params["use_erg_diffusion"],
                format=audio_format,
                save_path=tmpdir,
                batch_size=1,
                manual_seeds=manual_seeds,
            )

            # Find the generated audio file
            audio_file = None
            for item in output_paths:
                if isinstance(item, str) and item.endswith(f".{audio_format}"):
                    audio_file = item
                    break

            if not audio_file:
                # Search the output directory
                for f in os.listdir(tmpdir):
                    if f.endswith(f".{audio_format}"):
                        audio_file = os.path.join(tmpdir, f)
                        break

            if not audio_file:
                raise RuntimeError("ACE-Step did not produce an audio file")

            with open(audio_file, "rb") as f:
                audio_bytes = f.read()

        return audio_bytes, 48000, audio_format
