#!/bin/bash
# Quick test script to verify OpenFHE implementation in Docker

set -e

echo "🐳 Building FedGraph + OpenFHE Docker image..."
docker build -t fedgraph-openfhe .

echo ""
echo "✅ Docker image built successfully!"
echo ""

echo "🧪 Running OpenFHE smoke test..."
docker run --rm fedgraph-openfhe python /app/workspace/test_openfhe_smoke.py

echo ""
echo "🧪 Running OpenFHE threshold wrapper test..."
docker run --rm fedgraph-openfhe python -c "
from fedgraph.openfhe_threshold import test_threshold_he
test_threshold_he()
"

echo ""
echo "🧪 Testing NC integration structure..."
docker run --rm fedgraph-openfhe python -c "
import sys
sys.path.insert(0, '/app')

# Test imports
print('Testing imports...')
from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS
from fedgraph.server_class import Server
from fedgraph.trainer_class import Trainer_General
print('✓ All imports successful')

# Test that methods exist
print('\\nChecking Server methods...')
assert hasattr(Server, '_aggregate_openfhe_feature_sums')
print('✓ Server has _aggregate_openfhe_feature_sums')

print('\\nChecking Trainer methods...')
assert hasattr(Trainer_General, 'setup_openfhe_nonlead')
assert hasattr(Trainer_General, 'set_openfhe_public_key')
assert hasattr(Trainer_General, 'openfhe_partial_decrypt_main')
print('✓ Trainer has all required methods')

print('\\n🎉 All structure checks passed!')
"

echo ""
echo "🎉 All Docker tests passed! The implementation is ready."
echo ""
echo "To run the full NC HE tutorial:"
echo "  docker run -it --rm -v \$(pwd):/app/workspace -w /app/workspace fedgraph-openfhe python tutorials/FGL_NC_HE.py"

