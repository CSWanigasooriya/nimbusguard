# NimbusGuard DQN Operator
# ============================================================================
# Lightweight operator image for KServe-based DQN agent
# Uses KServe for model serving, no local PyTorch dependencies
# ============================================================================

FROM nimbusguard/base:latest

WORKDIR /app

# Copy the virtual environment from base image
COPY --from=nimbusguard/base:latest /opt/venv-template /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create operator user and necessary directories
RUN id -u operator >/dev/null 2>&1 || useradd -r -g operator -m operator && \
    mkdir -p /app/logs /tmp/logs && \
    chown -R operator:operator /app/logs /tmp/logs

# Copy the DQN operator code
COPY --chown=operator:operator . .

USER operator

# Set environment variables for DQN mode
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Health check for KServe connectivity
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9080/healthz || exit 1

# Run the DQN operator
CMD ["kopf", "run", \
     "--standalone", \
     "--liveness=http://0.0.0.0:9080/healthz", \
     "--all-namespaces", \
     "main.py"]