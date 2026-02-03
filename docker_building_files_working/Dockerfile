FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf \
    ATTENTION_IMPL=flash_attention_2 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    sox \
    git \
    python3-venv \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Verify CUDA installation
RUN nvcc --version

# Create virtual environment
RUN python3 -m venv /opt/venv

# Upgrade pip and install base packages
RUN /opt/venv/bin/python -m pip install -U pip setuptools wheel packaging

# Install qwen-tts and gradio
RUN /opt/venv/bin/python -m pip install -U qwen-tts gradio

# Install flash-attention specifically for RTX 3090 (compute capability 8.6)
RUN FLASH_ATTENTION_FORCE_ARCH=1 \
    FLASH_ATTENTION_CUDA_ARCHS=86 \
    TORCH_CUDA_ARCH_LIST=8.6 \
    NVCC_THREADS=2 \
    MAX_JOBS=8 \
    CUDA_HOME=/usr/local/cuda \
    /opt/venv/bin/python -m pip install -U flash-attn --no-build-isolation -v

# Verify installations
RUN /opt/venv/bin/python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); import flash_attn; print('flash_attn: import ok')"

# Create data directory
RUN mkdir -p /data

# Set working directory
WORKDIR /data

# Expose port
EXPOSE 8000

# Set entrypoint to use virtual environment
ENTRYPOINT ["/opt/venv/bin/python"]
CMD ["/data/app.py"]
