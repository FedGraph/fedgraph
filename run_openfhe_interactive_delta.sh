#!/bin/bash

# FedGraph OpenFHE NC - Interactive Session on Delta
# This script starts an interactive job and sets up the environment

# Replace with your account name
ACCOUNT="YOUR_ACCOUNT"

echo "Starting interactive session on Delta..."
echo "This will request:"
echo "  - 1 CPU node"
echo "  - 16 cores"
echo "  - 32GB memory"
echo "  - 2 hour time limit"
echo ""

srun --account=$ACCOUNT --partition=cpu-interactive \
  --nodes=1 --tasks=1 --tasks-per-node=1 \
  --cpus-per-task=16 --mem=32g \
  --time=02:00:00 \
  --pty bash << 'EOF'

# Inside interactive session
echo "=========================================="
echo "Interactive session started on: $(hostname)"
echo "=========================================="
echo ""

# Check GLIBC version
echo "Checking GLIBC version..."
ldd --version | head -n1
echo ""

# Load Python
module load python/3.11
echo "Python loaded: $(which python3)"
echo ""

# Create and activate venv
if [ ! -d "$HOME/openfhe_env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv $HOME/openfhe_env
fi

source $HOME/openfhe_env/bin/activate
echo "Virtual environment activated"
echo ""

# Install dependencies if needed
if [ ! -f "$HOME/openfhe_env/.installed" ]; then
    echo "Installing dependencies (this takes ~5 minutes)..."
    pip install -q --upgrade pip
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q torch-geometric
    pip install -q openfhe==1.2.3.0.24.4
    pip install -q ray[default] attridict ogb pyyaml networkx scipy scikit-learn
    
    # Clone repo
    cd $HOME
    if [ ! -d "$HOME/fedgraph" ]; then
        git clone -b gcn_v2 https://github.com/FedGraph/fedgraph.git
    fi
    cd $HOME/fedgraph
    pip install -q --no-deps .
    
    touch $HOME/openfhe_env/.installed
    echo "Dependencies installed!"
fi

echo ""
echo "=========================================="
echo "Setup Complete! You can now run:"
echo "  cd ~/fedgraph/tutorials"
echo "  python FGL_NC_HE.py"
echo ""
echo "Or test quickly with:"
echo "  cd ~/fedgraph/tutorials"
echo "  python -c 'from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS; print(\"OpenFHE loaded successfully\")'"
echo "=========================================="
echo ""

# Start interactive bash
bash

EOF

