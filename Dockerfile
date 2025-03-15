# Use a specific Python version with platform specification
FROM python:3.10.16-slim

# Install system dependencies and clean up to reduce image size
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the rest of the application code
COPY . /app

# Copy and install Python dependencies
RUN pip install -r requirements.txt

# Add a fallback for when camera isn't available
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0
ENV OPENCV_VIDEOIO_DEBUG=1

# Start Gunicorn to serve the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]