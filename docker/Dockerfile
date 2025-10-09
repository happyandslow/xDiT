# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.09-py3

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip
# RUN pip install --no-cache-dir "diffusers>=0.35.1" "transformers==4.37.2"
RUN pip install --no-cache-dir "diffusers>=0.35.1" "transformers==4.56.2"

# Uninstall apex first
RUN pip uninstall -y apex
RUN pip uninstall -y xfuser

# RUN pip install --no-cache-dir xfuser[diffusers,flash-attn,ray]
# RUN pip install -e ".[diffusers,flash-attn,ray]"
COPY . /app/xDiT
RUN cd /app/xDiT && pip install --no-cache-dir -e ".[diffusers,flash-attn,ray]"
RUN cd /app
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install flask
RUN pip uninstall -y opencv-python opencv-python-headless || true
RUN pip install --no-cache-dir "numpy<2" "opencv-python-headless==4.8.0.74"
RUN pip install --no-cache-dir imageio imageio-ffmpeg

COPY ./entrypoints /app/entrypoints
COPY ./examples /app/examples   

WORKDIR /app

# Set ENTRYPOINT with CMD as default arguments
# ENTRYPOINT ["python", "/app/comfyui-xdit/launch_host.py"]
# CMD ["--config", "./comfyui-xdit/config.json"]
