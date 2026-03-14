# =============================================================================
# Music Generation Worker — ACE-Step + MusicGen on RunPod Serverless
# =============================================================================
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# System dependencies (build-essential needed for thinc/spacy compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev python3-wheel \
    build-essential g++ cmake \
    ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev \
    libsndfile1 git espeak-ng pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA 12.6
RUN pip install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu126

# Install ACE-Step (has spacy/thinc deps that need C++ compiler)
RUN pip install --no-cache-dir \
    git+https://github.com/ace-step/ACE-Step.git

# Install MusicGen (audiocraft needs PyAV which needs ffmpeg dev headers)
RUN pip install --no-cache-dir audiocraft

# Install RunPod SDK and utilities
RUN pip install --no-cache-dir runpod>=1.7.0 soundfile numpy

# Copy application code
COPY engines/ engines/
COPY handler.py .

ENV PYTHONUNBUFFERED=1

# Default: ACE-Step. Override with MUSIC_ENGINE=musicgen
ENV MUSIC_ENGINE="acestep" \
    CPU_OFFLOAD="" \
    MUSICGEN_MODEL="medium"

CMD ["python3", "-u", "handler.py"]
