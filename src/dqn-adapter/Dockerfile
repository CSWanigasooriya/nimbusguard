# Python 3.9 as a base image
FROM nimbusguard-base:latest

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY src/dqn-adapter/requirements.txt .

# Install the dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the application code into the container
COPY src/dqn-adapter/ .

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application using Kopf
CMD ["kopf", "run", "--namespace", "nimbusguard", "/app/main.py"] 