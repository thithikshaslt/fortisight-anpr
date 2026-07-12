FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System deps for OpenCV + ONNX
RUN printf "Types: deb\nURIs: http://ftp.us.debian.org/debian\nSuites: bookworm bookworm-updates\nComponents: main\n\nTypes: deb\nURIs: http://security.debian.org/debian-security\nSuites: bookworm-security\nComponents: main\n" > /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
RUN pip install --no-cache-dir \
    opencv-python-headless \
    redis \
    requests \
    fast-alpr \
    onnxruntime

# Copy engine code
COPY engine.py .

CMD ["python", "engine.py"]
