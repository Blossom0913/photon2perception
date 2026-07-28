FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    libraw-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional: OpenMMLab packages (uncomment if needed)
# RUN pip install --no-cache-dir openmim && \
#     mim install mmdet==3.3.0 mmsegmentation==1.2.0

# Copy project
COPY . .

# Default command: run tests
CMD ["python", "-m", "pytest", "tests/test_core.py", "-v"]
