# ISLES26 Docker Image for Grand Challenge Submission
# Base: PyTorch with CUDA 12.1, cuDNN 8
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /opt/algorithm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Note: We use smaller versions of large models to fit in 32GB RAM
RUN pip install --no-cache-dir \
    nibabel==3.2.6 \
    monai==1.3.0 \
    omegaconf==2.3.0 \
    scikit-learn==1.5.0 \
    pandas==2.2.2 \
    scipy==1.14.1 \
    tqdm==4.66.4 \
    && pip install --no-cache-dir \
    transformers==4.44.2 \
    torchmetrics==1.4.0

# Copy pipeline modules (inference only - no training code)
COPY pipeline/model.py pipeline/conditioning.py pipeline/preprocessing.py pipeline/augmentation.py ./pipeline/
COPY utils/ /opt/algorithm/utils/
COPY pipeline/__init__.py pipeline/conditioning.py pipeline/model.py pipeline/preprocessing.py ./pipeline/

# Copy config
COPY configs/config.yaml /opt/algorithm/configs/

# Copy entrypoint
COPY entrypoint.py /opt/algorithm/

# Set environment variables
ENV ISLES26_INPUT_PATH=/input/image.nii.gz
ENV ISLES26_OUTPUT_PATH=/output/mask.nii.gz
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set NVIDIA runtime options
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create input/output directories (for Grand Challenge)
RUN mkdir -p /input /output

# Default entrypoint
ENTRYPOINT ["python", "/opt/algorithm/entrypoint.py"]

# Health check (for long-running jobs)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('OK')" || exit 1
