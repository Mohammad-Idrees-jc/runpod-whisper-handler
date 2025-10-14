# Use an official lightweight Python image
FROM python:3.10-slim

# Avoid buffering logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirement list first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Set environment variable for RunPod handler
ENV RUNPOD_HANDLER_ENTRYPOINT=handler.py

# Expose port (RunPod sometimes expects it)
EXPOSE 8000

# Run the handler
CMD ["python", "handler.py"]
