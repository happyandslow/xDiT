# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*



# Update pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Uninstall apex first
RUN pip uninstall -y apex
RUN pip uninstall -y xfuser

# # Install flash_attn separately with --use-pep517 flag
# RUN pip install --no-cache-dir --use-pep517 flash-attn==2.6.3 flask

#RUN pip install --no-cache-dir xfuser[diffusers,flash-attn,ray]
#RUN pip install -e ".[diffusers,flash-attn,ray]"
RUN git clone --branch lexu/custom-endpoint https://github.com/happyandslow/xDiT.git /app/xDiT
RUN cd /app/xDiT && pip install -e ".[diffusers,flash-attn,ray]"
RUN cd /app
RUN pip install fastapi
RUN pip install uvicorn

RUN pip install flask
# ensure numpy stays <2 to keep compatibility with the base pytorch wheel
RUN pip uninstall -y opencv-python opencv-python-headless || true

# install headless (safe for containers) and pin numpy
RUN pip install --no-cache-dir "numpy<2" "opencv-python-headless==4.8.0.74"

RUN pip install --no-cache-dir imageio imageio-ffmpeg

# Copy the entire comfyui-xdit directory into the container
COPY ./entrypoints /app/entrypoints
COPY ./examples /app/examples   

# Change to the xDiT directory
WORKDIR /app

# Set ENTRYPOINT with CMD as default arguments
# ENTRYPOINT ["python", "/app/comfyui-xdit/launch_host.py"]
# CMD ["--config", "./comfyui-xdit/config.json"]
