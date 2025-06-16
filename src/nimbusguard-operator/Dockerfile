FROM nimbusguard/base:latest

WORKDIR /app

# Copy the virtual environment from base image
COPY --from=nimbusguard/base:latest /opt/venv-template /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the operator code first (as root)
COPY . .

# Install the operator package as root (before switching to operator user)
RUN pip install -e .

# Create operator user and necessary directories
RUN id -u operator >/dev/null 2>&1 || useradd -r -g operator -m operator && \
    mkdir -p /app/logs /tmp/logs && \
    chown -R operator:operator /app /tmp/logs

# Switch to operator user
USER operator

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1

# Run the refactored operator
CMD ["python", "main.py"]
