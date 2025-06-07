# Optimized Dockerfile using shared base image with ALL dependencies
FROM nimbusguard/base:latest AS runtime

# Copy virtual environment from base (already has ALL dependencies)
COPY --from=nimbusguard/base:latest /opt/venv-template /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r nimbusguard && useradd -r -g nimbusguard nimbusguard

# Copy application code (this layer will change most often)
COPY --chown=nimbusguard:nimbusguard . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Switch to non-root user
USER nimbusguard

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Expose ports
EXPOSE 8080

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Entry point
ENTRYPOINT ["./entrypoint.sh"] 