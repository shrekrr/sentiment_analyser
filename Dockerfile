FROM python:3.11-slim

WORKDIR /app

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY . .

# Expose the API port
EXPOSE 8000

# Set environment variable gracefully
ENV MODEL_CHECKPOINT="checkpoints/full_model_best.pt"

# Run Gunicorn with Uvicorn workers for asynchronous production
CMD ["gunicorn", "frontend.app:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]
