# Example: https://github.com/GoogleCloudPlatform/cloud-run-microservice-template-python/blob/main/Dockerfile

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12.7-slim as base

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED 1

# Create and change to the app directory.
WORKDIR /

# Install Git
RUN apt-get update && \
    rm -rf /var/lib/apt/lists/*

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY requirements.txt ./

# Install dependencies.
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy local code to the container image.
COPY . ./

# Uninstall Git and clean up
RUN apt-get purge -y git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Run the web service on container startup.
CMD ["fastapi", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
