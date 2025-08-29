#!/bin/bash
# docker/run_evaluation.sh

echo "Running Fashion MNIST evaluation in Docker"

docker run --rm \
    -v  $(pwd)/data:/app/data \
    -v  $(pwd)/models:/app/models \
    -v  $(pwd)/logs:/app/logs \
    fashion-mnist-classifier:latest \
    python scripts/evaluate.py

echo "Evaluation Completed"