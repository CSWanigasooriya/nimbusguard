# Use the base image
FROM nimbusguard-base:latest

# Set the working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the application code into the container
COPY . .

# Copy the pre-trained models into /tmp for runtime use
COPY memory.keras memory.pkl /tmp/

# Expose the ports the app runs on
EXPOSE 8080 8081

# Command to run the application using Kopf with liveness probe
CMD ["kopf", "run", "--namespace", "nimbusguard", "--liveness=http://0.0.0.0:8081/healthz", "/app/app.py"] 