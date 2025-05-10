FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for better cache utilization
COPY requirements.txt .

# Install specific versions of libraries to avoid compatibility issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir gradio==4.2.0 && \
    pip install --no-cache-dir pandasai==2.0.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    # Pin gradio-client to a compatible version
    pip install --no-cache-dir gradio-client==0.7.0

# Copy application code
COPY app.py .
COPY .env.example .env

# Create a directory for sample data and temp files
RUN mkdir -p /app/sample_data /app/temp

# Copy sample data files if they exist
COPY sample_data*.* /app/sample_data/

# Set environment variables for the application
ENV TEMP_DIR=/app/temp

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application with debugging enabled
CMD ["python", "-u", "app.py"]