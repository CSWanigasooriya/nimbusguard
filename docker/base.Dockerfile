FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app

# Default command (can be overridden)
CMD ["python"] 