# Qwen3-TTS Docker Image for RTX 3090

This Docker setup builds a custom image optimized for NVIDIA RTX 3090 GPUs (CUDA compute capability 8.6).

## Files

- `Dockerfile` - The Docker image definition
- `build.sh` - Script to build the image
- `docker-compose.yml` - Docker Compose configuration for TrueNAS

## Building the Image

### Method 1: Using the build script

```bash
chmod +x build.sh
./build.sh
```

### Method 2: Manual build

```bash
docker build -t qwen3-tts-rtx3090:latest .
```

### Method 3: Build with docker-compose

Uncomment the build section in docker-compose.yml and run:

```bash
docker-compose build
```

## Build Details

The image is optimized for RTX 3090 with:
- CUDA compute capability: 8.6
- Flash Attention 2 compiled specifically for architecture 86
- PyTorch configured for CUDA 8.6

## Running the Container

### Using docker-compose (recommended for TrueNAS)

```bash
docker-compose up -d
```

### Using docker run

```bash
docker run -d \
  --name qwen3-tts-truenas \
  --gpus all \
  -p 8000:8000 \
  -v /mnt/MEGA-Storage/Docker-tests/qwen3-tts/data:/data \
  -e TZ=Etc/UTC \
  -e HF_HOME=/data/hf \
  -e HF_HUB_CACHE=/data/hf \
  qwen3-tts-rtx3090:latest
```

## Requirements

- NVIDIA RTX 3090 GPU
- NVIDIA Docker runtime installed
- Docker Compose (for TrueNAS deployment)
- Sufficient disk space for the build (~15GB)

## Build Time

The build process, especially the flash-attention compilation, can take 30-60 minutes depending on your system.

## Deployment to Registry (Optional)

If you want to push this to a Docker registry:

```bash
# Tag for your registry
docker tag qwen3-tts-rtx3090:latest your-registry.com/qwen3-tts-rtx3090:latest

# Push to registry
docker push your-registry.com/qwen3-tts-rtx3090:latest

# Update docker-compose.yml to use the registry image
# image: your-registry.com/qwen3-tts-rtx3090:latest
```

## Troubleshooting

### Build fails during flash-attention compilation
- Ensure you have enough RAM (16GB+ recommended)
- The build uses MAX_JOBS=1 and NVCC_THREADS=1 to avoid OOM errors
- Building inside a container may take longer but is more reliable

### GPU not detected
- Verify nvidia-docker is installed: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi`
- Check NVIDIA driver version: `nvidia-smi`
- Ensure driver supports CUDA 12.8

### Container crashes on startup
- Check logs: `docker logs qwen3-tts-truenas`
- Ensure /data/app.py exists in your volume mount
- Verify GPU accessibility inside container: `docker exec qwen3-tts-truenas nvidia-smi`
