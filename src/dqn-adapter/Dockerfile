# Python 3.9 as a base image
FROM nimbusguard-base:latest

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Copy the scientifically selected feature scaler
COPY feature_scaler.gz /app/feature_scaler.gz
COPY dqn_features.parquet /app/dqn_features.parquet

# Expose the ports the app runs on
EXPOSE 8080 9090

# Command to run the application using Kopf with liveness probe
CMD ["kopf", "run", "--namespace", "nimbusguard", "--liveness=http://0.0.0.0:8081/healthz", "/app/main.py"] 