#!/bin/bash
# docker/build.sh

echo "Building Fashion MNIST Classifier Docker Image"

# Suppress Deprecation Warning
export DOCKER_BUILDKIT=0

# Build the Docker image
docker build -f docker/Dockerfile -t fashion-mnist-classifier:latest .

echo "Docker image built successfully"
echo "Run with: docker run --rm fashion-mnist-classifier:latest"
