# Use a Python base image
FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Expose port and run the FastAPI server (required by OpenEnv URL checks)
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
