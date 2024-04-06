# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /usr/src/app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train model
RUN python train.py

ENV MODEL_PATH=/usr/src/app/model.pkl

# Command to run on container start
CMD ["python", "test.py"]