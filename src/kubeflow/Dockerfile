# NimbusGuard Kubeflow Unified Image
# Handles both DQN training (Katib) and model serving (KServe)
# Usage controlled by different commands/entrypoints

ARG BASE_IMAGE=nimbusguard/base:latest
FROM ${BASE_IMAGE}

# Copy virtual environment from base (already has ALL dependencies including Kubeflow)
COPY --from=nimbusguard/base:latest /opt/venv-template /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy Kubeflow-specific source code
COPY src/kubeflow/experiments/katib_training.py /app/katib_training.py
COPY src/kubeflow/serving/transformer.py /app/transformer.py
COPY src/kubeflow/continuous_training.py /app/continuous_training.py
COPY src/kubeflow/data_preprocessing.py /app/data_preprocessing.py
COPY src/nimbusguard-operator /app/nimbusguard-operator

# Set working directory
WORKDIR /app

# Make scripts executable
RUN chmod +x /app/katib_training.py /app/transformer.py /app/continuous_training.py

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports for serving
EXPOSE 8080 8090

# Default command (can be overridden)
CMD ["python", "/app/transformer.py"] 