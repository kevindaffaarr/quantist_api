# Example: https://github.com/GoogleCloudPlatform/cloud-run-microservice-template-python/blob/main/Dockerfile

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.14-slim AS base

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# uv installation
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /bin/
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# kaleido >=1.0 drives an external Chrome over CDP instead of bundling one.
RUN apt-get update && \
    apt-get install -y --no-install-recommends chromium && \
    rm -rf /var/lib/apt/lists/*

# Copy local code to the container image.
ADD . /quantist_api
# Create and change to the app directory.
WORKDIR /quantist_api

# Sync the project into a new environment, using the frozen lockfile
RUN uv sync --compile-bytecode

# Clean up
RUN apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Run the web service on container startup.
CMD ["uv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
