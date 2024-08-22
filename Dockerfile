# Use the official Python image as a base image
FROM python:3.11.9

# Set the working directory
WORKDIR /app

# Install PyTorch early to leverage caching
RUN pip install torch

# Copy only the requirements file to leverage caching
COPY docker_requirements.txt .
Run pip install ogb
COPY wheels ./wheels
# Install dependencies without using cache to reduce image size
RUN pip install --no-cache-dir -r docker_requirements.txt

# Copy the remaining application files
COPY fedgraph /app/fedgraph
COPY setup.py .
COPY README.md .

# Install the application
RUN pip install .

# Copy documentation and examples
COPY docs/examples /app/docs/examples

# Specify the command to run the application
# CMD ["python", "/app/docs/examples/example_LP.py"]