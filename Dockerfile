# Use Python 3.9 base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System dependencies: ffmpeg (for audio), build tools, libsndfile (for whisper)
RUN apt-get update && \
    apt-get install -y ffmpeg git build-essential libsndfile1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt .
COPY pipeline.py .
COPY src/ ./src/

# Default command (adjust if your entry point is different)
CMD ["python", "pipeline.py"]