# NimbusGuard Base Image with ALL Dependencies
# This base image contains ALL dependencies from all services
# to completely eliminate download time for service builds

FROM python:3.11-slim AS base

# Update package lists and install common system dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    curl \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install common Python packages
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Copy consolidated requirements file
COPY docker/requirements.txt /tmp/requirements.txt

# Install ALL dependencies from consolidated requirements file
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# Create virtual environment template with ALL packages
RUN python -m venv /opt/venv-template
ENV PATH="/opt/venv-template/bin:$PATH"

# Install everything into the template venv using the same consolidated requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv-template/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv-template/bin/pip install -r /tmp/requirements.txt

# Label for identification
LABEL maintainer="NimbusGuard Team"
LABEL description="Base image for NimbusGuard services with ALL dependencies pre-installed"
LABEL version="2.0" 