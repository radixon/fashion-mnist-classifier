#!/bin/bash
# docker/run_training.sh

echo "Running Fashion MNIST training in Docker"

docker run --rm \
    -v  $(pwd)/data:/app/data \
    -v  $(pwd)/models:/app/models \
    -v  $(pwd)/logs:/app/logs \
    fashion-mnist-classifier:latest \
    python scripts/train.py

echo "Training Completed"