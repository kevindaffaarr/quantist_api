# Example: https://github.com/GoogleCloudPlatform/cloud-run-microservice-template-python/blob/main/Dockerfile

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
# FROM python:3.10.7-slim as base

# Use optimized uvicorn-gunicorn-fastapi image from tiangolo
# https://github.com/tiangolo/uvicorn-gunicorn-fastapi-docker
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10 as base

# Allow statements and log messages to immediately appear in the Cloud Run logs
ENV PYTHONUNBUFFERED 1

# Create and change to the app directory.
WORKDIR /

# Copy application dependency manifests to the container image.
# Copying this separately prevents re-running pip install on every code change.
COPY requirements.txt ./

# Install dependencies.
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy local code to the container image.
COPY . ./

# Dev
FROM base as dev
COPY --from=base / /

# Prod
FROM base as prod
COPY --from=base / /

# Already using tiangolo/uvicorn-gunicorn-fastapi image, so no need to specify
# Run the web service on container startup.
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]