# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train model
RUN python train.py

# Command to run on container start
CMD ["python", "test.py"]