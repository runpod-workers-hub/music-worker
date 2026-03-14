# Music Generation RunPod Worker

RunPod Serverless worker for AI music generation:

- **ACE-Step** — Full songs with vocals and lyrics. 48kHz, 50+ languages, up to 4 minutes. ~2s/song on A100.
- **MusicGen** — Meta's instrumental music model. 32kHz, up to 30s clips.

## Quick Start

Set `MUSIC_ENGINE` to `acestep` or `musicgen`.

```bash
docker build --platform linux/amd64 -t yourusername/music-worker:latest .
docker push yourusername/music-worker:latest
```

## Input Format

### ACE-Step (Full Song with Vocals)
```json
{
  "input": {
    "prompt": "upbeat electronic dance music with strong bass",
    "lyrics": "[verse]\nDancing through the night\n[chorus]\nFeel the beat",
    "duration": 60,
    "seed": 42,
    "infer_step": 27,
    "guidance_scale": 15.0,
    "format": "wav"
  }
}
```

Use `"lyrics": "[Instrumental]"` for instrumental only.

### MusicGen (Instrumentals)
```json
{
  "input": {
    "prompt": "lo-fi hip hop beats to relax to",
    "duration": 30
  }
}
```

## Output
```json
{
  "audio": "<base64-encoded audio>",
  "format": "wav",
  "sample_rate": 48000,
  "duration": 60,
  "engine": "acestep"
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSIC_ENGINE` | `acestep` | Engine: `acestep` or `musicgen` |
| `MUSICGEN_MODEL` | `medium` | MusicGen size: `small`, `medium`, `large`, `melody` |
| `CPU_OFFLOAD` | | Set to `1` for ACE-Step on 8GB GPUs |

## License

- ACE-Step: Apache 2.0
- MusicGen: MIT (code), CC-BY-NC-4.0 (pretrained weights)
