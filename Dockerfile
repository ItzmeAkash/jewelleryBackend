# Use a specific Python version with platform specification
FROM python:3.10.16-slim

# Install system dependencies and clean up to reduce image size
RUN apt-get update

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application code
COPY . /app

# Copy and install Python dependencies
RUN pip install -r requirements.txt


# Start Gunicorn to serve the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]