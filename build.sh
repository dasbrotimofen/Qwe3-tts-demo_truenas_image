#!/bin/bash

# Build script for qwen3-tts Docker image optimized for RTX 3090

set -e

IMAGE_NAME="qwen3-tts-rtx3090"
IMAGE_TAG="latest"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Optimized for NVIDIA RTX 3090 (CUDA compute capability 8.6)"
echo ""

docker build \
    --tag ${IMAGE_NAME}:${IMAGE_TAG} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo ""
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "To push to a registry:"
echo "  docker tag ${IMAGE_NAME}:${IMAGE_TAG} your-registry/${IMAGE_NAME}:${IMAGE_TAG}"
echo "  docker push your-registry/${IMAGE_NAME}:${IMAGE_TAG}"
