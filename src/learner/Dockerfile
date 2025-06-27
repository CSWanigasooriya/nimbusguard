# Use the official Python image from the Docker Hub
FROM nimbusguard-base:latest

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY src/learner/requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy the rest of the application's code
COPY src/learner/ .

# Command to run the application
CMD ["python", "main.py"]
