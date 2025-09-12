# Use a slim Python 3.9 image as a base
FROM python:3.11-slim as base

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

# Dependencies Installation
FROM base as dependencies

# Copy requirements file to leverage Docker's layer caching
COPY requirements.txt .

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Application
FROM dependencies as application

# Copy application code
COPY . .

# Copy setup.py and install package
# COPY setup.py .
# COPY src/ ./src/
# RUN pip install -e .

# Copy application files
# COPY scripts/ ./scripts/
# COPY configs/ ./configs/
# COPY app/ ./app/

# # Copy best trained model
# COPY src/models/ ./src/models/

# Create Directories
RUN mkdir -p models results/figures results/metrics results/logs data/raw

# Set Permissions
RUN chmod +x scripts/*.py

# Expose the default port for Streamlit
EXPOSE 8501 5000

# Default Command
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=true"]