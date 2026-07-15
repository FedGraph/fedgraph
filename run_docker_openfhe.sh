#!/bin/bash

# Script to build and run FedGraph with OpenFHE in Docker

echo "🐳 Building FedGraph + OpenFHE Docker image..."

# Build the Docker image
docker build -t fedgraph-openfhe .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "✅ Docker image built successfully!"
echo ""

echo "🚀 Running FedGraph + OpenFHE container..."

# Run the container interactively
docker run -it --rm \
    -v "$(pwd):/app/workspace" \
    -w /app/workspace \
    fedgraph-openfhe \
    /bin/bash

echo "👋 Container stopped."
