# services/edge/Dockerfile
FROM python:3.10-slim

# Install audio dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-dev \
    libasound2-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from root
COPY requirements_pi.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_pi.txt
RUN pip install --no-cache-dir requests python-dotenv torch --index-url https://download.pytorch.org/whl/cpu

# Copy source code and config
COPY .env .
COPY src/ .

# Configuration defaults
ENV COMPUTE_SERVER_URL=http://localhost:8000
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
