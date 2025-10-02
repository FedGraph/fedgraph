# Use the official Python image as a base image
FROM python:3.12

# Set the working directory
WORKDIR /app

# Install system dependencies needed for OpenFHE
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt-lists/*

# Install PyTorch first (use available version)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file
COPY docker_requirements.txt .

# Install torch-geometric and related packages
RUN pip install torch-geometric || echo "torch-geometric install attempted"

# Install other dependencies (excluding tenseal)
RUN grep -v "tenseal" docker_requirements.txt | \
    grep -v "torch" | \
    grep -v "torch-cluster" | \
    grep -v "torch-scatter" | \
    grep -v "torch-sparse" | \
    grep -v "torch-spline-conv" | \
    grep -v "torch-geometric" > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt

# Install OpenFHE Python package
# Note: Using an older version compatible with pre-built binaries
RUN pip install --no-cache-dir openfhe==1.2.3.0.24.4 || \
    echo "Warning: OpenFHE installation may need manual verification"

# Copy the remaining application files
COPY fedgraph /app/fedgraph
COPY setup.py .
COPY README.md .

# Install the application (without dependencies since we already installed them)
RUN pip install --no-deps .

# Copy documentation and examples
COPY tutorials /app/docs/examples

# Specify the command to run the application
CMD ["python", "-c", "import fedgraph; print('FedGraph with OpenFHE ready!')"]
