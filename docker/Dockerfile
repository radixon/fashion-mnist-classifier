# Use a slim Python 3.9 image as a base
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Set up environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install System Dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file to leverage Docker's layer caching
COPY requirements.txt .

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy setup.py and install package
COPY setup.py .
COPY src/ ./src/
RUN pip install -e .

# Copy necessary files
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Create Directories
RUN mkdir -p models data/raw data/processed logs

# Set Permissions
RUN chmod +x scripts/*.py

# Expose the default port for Streamlit
EXPOSE 8501

# Default Command
CMD ["python", "scripts/train.py"]