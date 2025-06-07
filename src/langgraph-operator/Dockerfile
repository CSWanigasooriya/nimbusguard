FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Create non-root user for security
RUN groupadd -r nimbusguard && useradd -r -g nimbusguard nimbusguard
RUN chown -R nimbusguard:nimbusguard /app
USER nimbusguard

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Expose ports
EXPOSE 8080

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Entry point
ENTRYPOINT ["./entrypoint.sh"] 