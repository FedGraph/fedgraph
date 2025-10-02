#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --account=YOUR_ACCOUNT    # Replace with your Delta account name
#SBATCH --job-name=agentic
#SBATCH --time=02:00:00      # 2 hours should be enough
#SBATCH --constraint="scratch"
#SBATCH -e agentic-%j.err
#SBATCH -o agentic-%j.out

# Agentic System - Slurm Batch Script for Delta
# Federated graph learning experiments

echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $SLURM_SUBMIT_DIR"

# Load modules
module reset
module load python/3.11    # or python/3.12 if available
module list

# Create virtual environment (first time only)
if [ ! -d "$HOME/openfhe_env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv $HOME/openfhe_env
fi

# Activate virtual environment
source $HOME/openfhe_env/bin/activate

# Install dependencies (first time only)
if [ ! -f "$HOME/openfhe_env/.installed" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    
    # Install PyTorch first
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    
    # Install torch-geometric and its dependencies
    pip install torch-geometric
    pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
    
    # Install OpenFHE
    pip install openfhe==1.2.3.0.24.4
    
    # Install other dependencies
    pip install ray[default] attridict ogb pyyaml networkx scipy scikit-learn
    
    # Clone and install fedgraph
    cd $HOME
    if [ ! -d "$HOME/fedgraph" ]; then
        git clone -b gcn_v2 https://github.com/FedGraph/fedgraph.git
    fi
    cd $HOME/fedgraph
    pip install --no-deps .
    
    touch $HOME/openfhe_env/.installed
    echo "Dependencies installed successfully"
fi

# Change to tutorial directory
cd $HOME/fedgraph/tutorials

# Run the OpenFHE NC tutorial
echo ""
echo "=========================================="
echo "Running OpenFHE NC Tutorial"
echo "=========================================="
python FGL_NC_HE.py

echo ""
echo "Job finished at $(date)"

