#!/bin/bash
# docker/run_prediction.sh

echo "Running Fashion MNIST prediction in Docker"

docker run --rm \
    -v  $(pwd)/data:/app/data \
    -v  $(pwd)/models:/app/models \
    -v  $(pwd)/logs:/app/logs \
    fashion-mnist-classifier:latest \
    python scripts/predict.py

echo "Prediction Completed"