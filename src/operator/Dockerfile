# Python 3.9 as a base image
FROM nimbusguard-base:latest

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml file into the container for dependency management
COPY pyproject.toml .

# Install the dependencies using modern pyproject.toml approach
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -e .

# Copy the application code into the container
COPY . .

# Copy the feature scaler into /tmp for runtime use
COPY forecaster.keras forecaster_scaler.pkl /tmp/


# Expose the ports the app runs on
EXPOSE 8080 8081

# Command to run the application using Kopf with liveness probe
CMD ["kopf", "run", "--namespace", "nimbusguard", "--liveness=http://0.0.0.0:8081/healthz", "/app/main.py"] 