FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies for llama.cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY . .

# Install Sawyer
RUN pip install --no-cache-dir -e .


# ── Router image ──────────────────────────────────────────────────

FROM base AS router

EXPOSE 50051 8443 8080

# Router entry point: gRPC + dashboard
CMD ["python", "-m", "sawyer.router.server"]


# ── Node image ────────────────────────────────────────────────────

FROM base AS node

# Install llama.cpp from source
RUN git clone https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=ON \
    && cmake --build build --config Release -j$(nproc) \
    && cp build/bin/llama-server /usr/local/bin/ \
    && cp build/bin/llama-cli /usr/local/bin/ \
    && rm -rf /opt/llama.cpp

# Create cache directories
RUN mkdir -p /root/.sawyer/cache /root/.sawyer/experts

VOLUME ["/root/.sawyer/cache", "/root/.sawyer/experts"]

EXPOSE 50052 8080

# Node entry point
CMD ["python", "-m", "sawyer.cli", "serve"]