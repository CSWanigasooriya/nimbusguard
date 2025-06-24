# Use base image
FROM nimbusguard-base:latest

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the load generator scripts
COPY load_generator.py .
COPY quick_test.py .

# Make scripts executable
RUN chmod +x load_generator.py quick_test.py

# Switch to non-root user
USER app

# Default command
ENTRYPOINT ["python", "load_generator.py"] 