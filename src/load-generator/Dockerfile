# Optimized Dockerfile using shared base image with ALL dependencies
FROM nimbusguard/base:latest AS runtime

# Copy virtual environment from base (already has ALL dependencies)
COPY --from=nimbusguard/base:latest /opt/venv-template /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r loadgen && useradd -r -g loadgen -m loadgen

# Copy application code (this layer will change most often)
COPY --chown=loadgen:loadgen . .

# Switch to non-root user
USER loadgen

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Expose port
EXPOSE 8081

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["python", "main.py"] 