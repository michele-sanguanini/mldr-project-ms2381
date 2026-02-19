FROM debian:bookworm-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install Python 3.10 via uv
RUN uv python install 3.10

# Install DVC with S3 support (MinIO is S3-compatible)
RUN uv tool install "dvc[s3]"

# Ensure uv-managed tools are on PATH
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

CMD ["bash"]
