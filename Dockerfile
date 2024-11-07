# Use the official Python image as a base image
FROM python:3.11.9

# Set the working directory
WORKDIR /app

# Install PyTorch early to leverage caching
RUN pip install torch

# # Copy the wheels directory
# COPY wheels ./wheels

# # Install torch-geometric related wheels from the local directory
# RUN pip install --no-cache-dir --find-links=./wheels \
#     torch-cluster \
#     torch-scatter \
#     torch-sparse \
#     torch-spline-conv

# Copy the requirements file (excluding torch-geometric wheels as they are pre-installed)
COPY docker_requirements.txt .

# Install remaining dependencies from the requirements file
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
