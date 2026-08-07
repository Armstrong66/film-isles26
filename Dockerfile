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
    fastapi==0.115.0 \
    uvicorn==0.30.1 \
    SimpleITK==2.4.0 \
    && pip install --no-cache-dir \
    transformers==4.44.2 \
    torchmetrics==1.4.0

# Copy pipeline modules (inference only - no training code)
COPY pipeline/ ./pipeline/
COPY utils/ /opt/algorithm/utils/
COPY configs/ /opt/algorithm/configs/

# Copy entrypoint
COPY entrypoint.py /opt/algorithm/

# Create checkpoints directory (mount at runtime)
RUN mkdir -p /opt/algorithm/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set NVIDIA runtime options
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Create input/output directories (for Grand Challenge)
RUN mkdir -p /input/images/t1-brain-mri /output/images/stroke-lesion-segmentation

# Expose port for Grand Challenge API
EXPOSE 4743

# Default entrypoint - runs FastAPI server
CMD ["python", "entrypoint.py"]
