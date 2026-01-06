# services/compute/Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python and audio dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from root
COPY requirements_gpu_server.txt .

RUN pip3 install --upgrade pip
# RTX 5090 (Compute 12.0/Blackwell) needs recent CUDA. 12.4 should be enough for basic runtime.
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install everything else
RUN pip3 install --no-cache-dir -r requirements_gpu_server.txt
RUN pip3 install --no-cache-dir fastapi uvicorn python-dotenv google-genai requests

# Copy source code and config
COPY .env .
COPY src/ .

# Ensure server.py knows about the .env
ENV PYTHONPATH=/app

CMD ["python3", "server.py"]
