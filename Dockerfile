# Use an official Python image with CUDA support for GPU
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Avoid buffering logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg is required by whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first (for Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler file
COPY handler.py .

# Set environment variable for Whisper model (can be overridden)
ENV WHISPER_MODEL=medium

# Run the handler
CMD ["python", "-u", "handler.py"]
