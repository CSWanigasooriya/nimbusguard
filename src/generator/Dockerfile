# Use base image
FROM nimbusguard-base:latest

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the load generator scripts
COPY . .

# Make scripts executable
RUN chmod +x load_generator.py

# Default entrypoint
ENTRYPOINT ["python", "load_generator.py"] 