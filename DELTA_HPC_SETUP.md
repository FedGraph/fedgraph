# Running OpenFHE NC on Delta HPC

## Prerequisites

1. Access to Delta HPC (NCSA)
2. Active allocation/account
3. SSH access to Delta login nodes

---

## Quick Start

### Step 1: Upload Files to Delta

From your local machine:

```bash
# Upload the batch script
scp run_openfhe_delta.slurm YOUR_USERNAME@login.delta.ncsa.illinois.edu:~/

# Or clone directly on Delta (recommended)
ssh YOUR_USERNAME@login.delta.ncsa.illinois.edu
git clone -b gcn_v2 https://github.com/FedGraph/fedgraph.git
cd fedgraph
```

### Step 2: Check Your Account

On Delta login node:

```bash
accounts
```

Note your account name under "Project" column. You'll need this for the batch script.

### Step 3: Edit Batch Script

```bash
cd ~/fedgraph
nano run_openfhe_delta.slurm
```

Change this line:
```bash
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
```

To your actual account, for example:
```bash
#SBATCH --account=bbka-delta-cpu
```

### Step 4: Submit Job

```bash
sbatch run_openfhe_delta.slurm
```

### Step 5: Monitor Job

```bash
# Check job status
squeue -u $USER

# Check output (replace JOBID with your actual job ID)
tail -f openfhe-JOBID.out

# Check errors
tail -f openfhe-JOBID.err
```

---

## Option 2: Interactive Session (Faster Testing)

### Quick Interactive Test

```bash
# On Delta login node
srun --account=YOUR_ACCOUNT --partition=cpu-interactive \
  --nodes=1 --tasks=1 --cpus-per-task=8 --mem=16g \
  --time=01:00:00 --pty bash

# Once on compute node, check GLIBC
ldd --version

# Load Python and test OpenFHE
module load python/3.11
pip install --user openfhe==1.2.3.0.24.4

# Quick test
python3 -c "import openfhe; print('OpenFHE works!')"
```

### Full Interactive Setup

```bash
# 1. Make the script executable
chmod +x run_openfhe_interactive_delta.sh

# 2. Edit the script to add your account
nano run_openfhe_interactive_delta.sh
# Change: ACCOUNT="REPLACE_WITH_YOUR_ACCOUNT"

# 3. Run it
./run_openfhe_interactive_delta.sh
```

---

## Comparing Plaintext vs OpenFHE

Create a custom comparison script:

```bash
nano compare_openfhe.slurm
```

Add this content:

```bash
#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --job-name=compare_openfhe
#SBATCH --time=03:00:00
#SBATCH -e compare-%j.err
#SBATCH -o compare-%j.out

source $HOME/openfhe_env/bin/activate
cd $HOME/fedgraph

python3 << 'PYEOF'
from fedgraph.federated_methods import run_NC
from attridict import AttriDict

# Test 1: Plaintext
print("\n" + "="*60)
print("TEST 1: PLAINTEXT (Baseline)")
print("="*60)
config = {
    "fedgraph_task": "NC",
    "method": "FedGCN",
    "use_encryption": False,
    "dataset": "cora",
    "num_trainers": 3,
    "num_rounds": 10,
    "seed": 42,
}
run_NC(AttriDict(config))

# Test 2: OpenFHE
print("\n" + "="*60)
print("TEST 2: OPENFHE (Two-Party Threshold HE)")
print("="*60)
config["use_encryption"] = True
config["he_backend"] = "openfhe"
run_NC(AttriDict(config))

print("\n" + "="*60)
print("COMPARISON COMPLETE")
print("Check the output above for accuracy difference")
print("Expected: < 1% difference")
print("="*60)
PYEOF
```

Then submit:

```bash
sbatch compare_openfhe.slurm
```

---

## File System Usage

Following Delta best practices:

- **HOME** (`$HOME`): Store scripts, environments, small files
- **SCRATCH** (`$SCRATCH`): Store datasets, temporary outputs
- **WORK/PROJECTS** (`$WORK`): Store results, checkpoints

Example directory structure:

```bash
$HOME/
  ├── openfhe_env/           # Python virtual environment
  └── fedgraph/              # Git repository

$SCRATCH/
  └── fedgraph_results/      # Training outputs, logs
```

---

## GPU Version (Optional)

To use GPU nodes for faster training:

```bash
#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4          # A40 GPU partition
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gpus-per-node=1
#SBATCH --job-name=openfhe_nc_gpu
#SBATCH --time=01:00:00
#SBATCH -e openfhe-gpu-%j.err
#SBATCH -o openfhe-gpu-%j.out

module reset
module load python/3.11
module load cuda/11.8   # or appropriate CUDA version

source $HOME/openfhe_env/bin/activate
cd $HOME/fedgraph/tutorials

# Run with GPU support
python FGL_NC_HE.py
```

---

## Troubleshooting

### Issue: GLIBC version error

```bash
# Check GLIBC version on compute node
srun --account=YOUR_ACCOUNT --partition=cpu-interactive \
  --nodes=1 --cpus-per-task=1 --mem=4g --time=00:05:00 \
  ldd --version
```

**Expected:** GLIBC 2.31+ (Delta has RedHat 8.4)
**Required:** GLIBC 2.29+ for OpenFHE 1.2.3

If version is too old, try:
```bash
pip install openfhe==1.1.0  # Earlier version
```

### Issue: Module not found

```bash
# Check available Python modules
module avail python

# Try different Python version
module load python/3.12
```

### Issue: Out of memory

Increase memory in SBATCH directive:
```bash
#SBATCH --mem=64g  # instead of 32g
```

### Issue: Job pending for long time

```bash
# Check why job is pending
squeue -u $USER -l

# Try interactive partition for testing
#SBATCH --partition=cpu-interactive

# Or use preempt partition (cheaper, may be interrupted)
#SBATCH --partition=cpu-preempt
```

---

## Expected Resource Usage

Based on the FedGCN NC implementation:

| Resource | Cora Dataset | Citeseer | Pubmed |
|----------|--------------|----------|--------|
| **Memory** | ~16GB | ~20GB | ~32GB |
| **Cores** | 8-16 | 8-16 | 16-32 |
| **Time** | ~30 min | ~45 min | ~90 min |
| **Storage** | ~2GB | ~3GB | ~5GB |

---

## Monitoring Your Job

### While job is running:

```bash
# SSH to compute node
squeue -u $USER  # Get node name
ssh NODE_NAME    # e.g., ssh cn042

# Once on node:
top -u $USER
htop
nvidia-smi       # if using GPU
```

### Check output in real-time:

```bash
# Get job ID
JOBID=$(squeue -u $USER -h -o %i | head -1)

# Tail output
tail -f openfhe-${JOBID}.out

# Or use watch
watch -n 5 tail -20 openfhe-${JOBID}.out
```

---

## Batch Job Array (Multiple Experiments)

To run multiple configurations:

```bash
#!/bin/bash
#SBATCH --array=0-4           # 5 jobs
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --time=02:00:00
#SBATCH -e array-%A_%a.err   # %A=job ID, %a=array index
#SBATCH -o array-%A_%a.out

# Define datasets
DATASETS=("cora" "citeseer" "pubmed" "cora" "citeseer")
HE_BACKENDS=("openfhe" "openfhe" "openfhe" "tenseal" "tenseal")

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
HE_BACKEND=${HE_BACKENDS[$SLURM_ARRAY_TASK_ID]}

echo "Running: Dataset=$DATASET, Backend=$HE_BACKEND"

source $HOME/openfhe_env/bin/activate
cd $HOME/fedgraph

python3 << PYEOF
from fedgraph.federated_methods import run_NC
from attridict import AttriDict

config = {
    "fedgraph_task": "NC",
    "method": "FedGCN",
    "use_encryption": True,
    "he_backend": "$HE_BACKEND",
    "dataset": "$DATASET",
    "num_trainers": 3,
    "num_rounds": 10,
    "seed": 42,
}
run_NC(AttriDict(config))
PYEOF
```

Submit array job:
```bash
sbatch array_experiment.slurm
```

---

## Next Steps

1. **First time setup**: Run interactive session to verify everything works
2. **Single experiment**: Use `run_openfhe_delta.slurm` for single runs
3. **Comparisons**: Use custom scripts to compare plaintext vs OpenFHE
4. **Production**: Use batch arrays for multiple experiments

**See also:**
- `README_OPENFHE.md` - Implementation details
- `OPENFHE_NC_IMPLEMENTATION.md` - Technical documentation
- Delta docs: https://docs.ncsa.illinois.edu/systems/delta/

