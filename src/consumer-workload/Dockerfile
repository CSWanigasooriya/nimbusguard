# Optimized Dockerfile using shared base image with ALL dependencies
FROM nimbusguard/base:latest AS runtime

# Copy virtual environment from base (already has ALL dependencies)
COPY --from=nimbusguard/base:latest /opt/venv-template /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r consumer && useradd -r -g consumer consumer

# Create logs directory with proper permissions
RUN mkdir -p /app/logs /tmp/logs && \
    chown -R consumer:consumer /app/logs /tmp/logs

# Copy application code (this layer will change most often)
COPY --chown=consumer:consumer . .

# Switch to non-root user
USER consumer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"] 