# Example: https://github.com/GoogleCloudPlatform/cloud-run-microservice-template-python/blob/main/Dockerfile

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12.7-slim as base

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1

# uv installation
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /bin/
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Copy local code to the container image.
ADD . ./
# Create and change to the app directory.
WORKDIR /

# Sync the project into a new environment, using the frozen lockfile
RUN uv sync --frozen --compile-bytecode

# Clean up
RUN apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Run the web service on container startup.
CMD ["uv", "run", "fastapi", "run", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
