# Distributed Docker Deployment Guide

This guide explains how to set up the Voice Assistant in a distributed "Edge/Compute" architecture using Docker.

## Architecture Overview

- **Compute Node (Server)**:
  - **Hardware**: High-end PC with NVIDIA GPU (e.g., RTX 5090, 4090).
  - **Service**: Runs the "Brain" (STT, LLM, TTS).
  - **Container**: `services/compute/Dockerfile`
  - **Port**: 8000 (FastAPI processed requests).

- **Edge Node (Endpoint)**:
  - **Hardware**: Mini PC (N100), Raspberry Pi 5, or Laptop.
  - **Service**: Runs the "Ears" and "Mouth" (Mic input, Wake Word detection, Speaker output).
  - **Container**: `services/edge/Dockerfile`
  - **Communication**: Streams audio to the Compute Node via WebSocket (HTTP fallback available).

## Step 1: Compute Node Setup

1.  **NVIDIA Container Toolkit**: Ensure your GPU server has the toolkit installed so Docker can access the GPU.
    ```bash
    # Verify installation
    nvidia-smi
    docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.0-base nvidia-smi
    ```

2.  **Environment Configuration**: Create a `.env` file in the root directory.
    ```bash
    GEMINI_API_KEY=your_key_here
    MODEL_NAME=gemini-2.5-flash-lite
    XTTS_SERVER_URL=http://localhost:5001
    ```

3.  **Run with Docker Compose**:
    ```bash
    docker compose up compute --build
    ```

## Step 2: Edge Node Setup

1.  **Hardware Access**: Docker needs permission to access `/dev/snd`.
    ```bash
    # Add your user to the audio group
    sudo usermod -aG audio $USER
    # Log out and back in
    ```

2.  **Environment Configuration**: In the root `.env`, set the server URL.
    ```bash
    COMPUTE_SERVER_URL=http://<COMPUTE_NODE_IP>:8000
    # Optional override (defaults to ws://<COMPUTE_NODE_IP>:8000/ws/audio)
    # COMPUTE_WS_URL=ws://<COMPUTE_NODE_IP>:8000/ws/audio
    MIC_DEVICE_INDEX=0  # Use scripts/find_devices.py to identify
    ```

3.  **Run with Docker Compose**:
    ```bash
    docker compose up edge --build
    ```

## Performance Tuning

### Latency
- **Network**: Ensure both nodes are on the same local network (wired LAN preferred).
- **GPU**: If transcription is slow, verify the Compute Node is actually using the GPU (check `nvidia-smi` during a command).

### Hardware Audio
- If you have multiple audio interfaces, use the `scripts/find_devices.py` script to find the correct index and set `MIC_DEVICE_INDEX` and `TTS_OUTPUT_DEVICE` in your `.env`.

## Troubleshooting

### "Permission Denied" on `/dev/snd`
If the Edge container fails with audio errors, try running it with host networking or privileged mode (not recommended for production):
```yaml
# In docker-compose.yml
edge:
  ...
  privileged: true
  network_mode: host
```

### Dependency Issues
The Dockerfiles are configured to install necessary libraries. If a specific tool (like Control4) fails, ensure its environment variables are set correctly in the root `.env` which is shared with both containers.
