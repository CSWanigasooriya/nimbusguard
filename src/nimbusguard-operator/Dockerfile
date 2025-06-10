FROM nimbusguard/base:latest

WORKDIR /app

# Copy the virtual environment from base image
COPY --from=nimbusguard/base:latest /opt/venv-template /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create operator user and necessary directories
RUN id -u operator >/dev/null 2>&1 || useradd -r -g operator -m operator && \
    mkdir -p /app/logs /tmp/logs && \
    chown -R operator:operator /app/logs /tmp/logs

# Copy the operator code
COPY --chown=operator:operator . .

USER operator

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run the operator
CMD ["python", "nimbusguard_operator.py"] 