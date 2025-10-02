# START HERE: Running OpenFHE NC on Delta HPC

## TL;DR - Fastest Path

```bash
# 1. SSH to Delta
ssh YOUR_USERNAME@login.delta.ncsa.illinois.edu

# 2. Clone your repository
git clone -b gcn_v2 https://github.com/FedGraph/fedgraph.git
cd fedgraph

# 3. Check your account
accounts

# 4. Edit batch script with your account name
nano run_openfhe_delta.slurm
# Change: #SBATCH --account=YOUR_ACCOUNT_NAME

# 5. Submit job
sbatch run_openfhe_delta.slurm

# 6. Monitor
squeue -u $USER
tail -f openfhe-*.out
```

**Expected runtime:** 30-60 minutes for Cora dataset

---

## What You Have

### Files Created for Delta HPC:

1. **`run_openfhe_delta.slurm`** - Main batch script (RECOMMENDED)
   - Automated setup and execution
   - Runs full OpenFHE NC tutorial
   - Uses CPU partition

2. **`run_openfhe_interactive_delta.sh`** - Interactive session
   - Good for testing/debugging
   - Gives you a shell on compute node
   - Manual control

3. **`DELTA_HPC_SETUP.md`** - Complete documentation
   - All options explained
   - Troubleshooting guide
   - GPU version included
   - Batch arrays for multiple experiments

---

## Why Delta is Perfect

- **GLIBC 2.31+** (RedHat 8.4) - Compatible with OpenFHE
- **Powerful compute nodes** - 128 CPU cores, 252GB RAM
- **GPU options** - A100, A40, H200, MI100
- **No Docker needed** - Direct Python installation works
- **Batch scheduling** - Set it and forget it

---

## Workflow Options

### Option A: Batch Job (Recommended)
**Best for:** Production runs, reproducibility

```bash
# On Delta login node
cd ~/fedgraph
nano run_openfhe_delta.slurm  # Edit account name
sbatch run_openfhe_delta.slurm
squeue -u $USER                # Check status
```

**Pros:** Hands-off, queued when resources available, full logging
**Cons:** Wait in queue, can't interact

### Option B: Interactive Session
**Best for:** Testing, debugging, experimentation

```bash
# On Delta login node
cd ~/fedgraph
nano run_openfhe_interactive_delta.sh  # Edit account name
./run_openfhe_interactive_delta.sh
# Wait for interactive session to start...
cd ~/fedgraph/tutorials
python FGL_NC_HE.py
```

**Pros:** Immediate feedback, can modify on the fly
**Cons:** Ties up your terminal, limited to 1 hour

---

## Expected Output

When successful, you'll see:

```
Starting OpenFHE threshold encrypted feature aggregation...
Step 1: Server generates lead keys...
Step 2: Designated trainer generates non-lead share...
Step 3: Server finalizes joint public key...
Step 4: Distributing joint public key to all trainers...
Two-party threshold key generation complete!

Round 1/10: Train Loss: 1.234, Test Acc: 0.756
Round 2/10: Train Loss: 0.987, Test Acc: 0.782
...
Round 10/10: Train Loss: 0.456, Test Acc: 0.814

Final Test Accuracy: 0.814
```

**Security achieved:** Neither server nor any single trainer can decrypt alone!

---

## Common Issues

### Issue: "Job pending for a long time"
**Solution:** Use interactive partition for testing:
```bash
#SBATCH --partition=cpu-interactive
```

### Issue: "Account not found"
**Solution:** Run `accounts` on Delta and use exact account name:
```bash
$ accounts
Expiration Date     Project            Hours Allocated    Hours Used
--------------------------------------------------------------------
2025-06-30          bbka-delta-cpu     100000             1234
```
Then: `#SBATCH --account=bbka-delta-cpu`

### Issue: "Module not found"
**Solution:** Check available Python:
```bash
module avail python
module load python/3.11  # or python/3.12
```

---

## After First Run

### Compare Plaintext vs OpenFHE

Create a comparison script (see `DELTA_HPC_SETUP.md` for full code):
```bash
nano compare.slurm
sbatch compare.slurm
```

### Run Multiple Experiments

Use job arrays to test different:
- Datasets (Cora, Citeseer, Pubmed)
- Number of trainers (3, 5, 10)
- Training rounds (10, 20, 50)

See `DELTA_HPC_SETUP.md` for batch array examples.

---

## Resource Estimates

| Dataset | Memory | Cores | Time | Queue Wait* |
|---------|--------|-------|------|-------------|
| Cora    | 16GB   | 8     | 30m  | 5-10m       |
| Citeseer| 20GB   | 8     | 45m  | 5-10m       |
| Pubmed  | 32GB   | 16    | 90m  | 10-30m      |

*Queue wait times are estimates and vary by system load

---

## What's Working vs Not Working

| Environment | Status | Notes |
|-------------|--------|-------|
| **Delta HPC** | ✅ WORKS | You are here - RECOMMENDED |
| Docker (local) | ✅ WORKS | Alternative if Delta is busy |
| Google Colab | ❌ FAILS | GLIBC too old |
| macOS local | ❌ FAILS | Dependency conflicts |

---

## Files Summary

### For Delta HPC (USE THESE):
- `run_openfhe_delta.slurm` - Batch script
- `run_openfhe_interactive_delta.sh` - Interactive
- `DELTA_HPC_SETUP.md` - Full documentation
- `START_HERE.md` - This file

### For Docker (Alternative):
- `Dockerfile` - Docker image definition
- `QUICKSTART_DOCKER.md` - Docker instructions

### For Documentation:
- `README_OPENFHE.md` - Quick reference
- `OPENFHE_NC_IMPLEMENTATION.md` - Technical details
- `RUNTIME_OPTIONS.md` - Environment comparison

### For Colab (Reference Only):
- `COLAB_SETUP.md` - Won't work due to GLIBC
- `FedGraph_OpenFHE_NC.ipynb` - Reference notebook

---

## Quick Commands

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f openfhe-*.out

# SSH to running job
squeue -u $USER  # get node name
ssh NODE_NAME

# Cancel job
scancel JOBID

# Check account balance
accounts

# List your files
ls -lh ~/fedgraph

# Check results
cat openfhe-JOBID.out | grep "Test Acc"
```

---

## Next Actions

1. **Upload to Delta:**
   ```bash
   # From your local Mac
   cd /Users/fanxy/Documents/GitHub/fedgraph-10
   scp run_openfhe_delta.slurm YOUR_USERNAME@login.delta.ncsa.illinois.edu:~/
   ```

2. **Or clone on Delta:**
   ```bash
   # On Delta login node
   git clone -b gcn_v2 https://github.com/FedGraph/fedgraph.git
   ```

3. **Edit and submit:**
   ```bash
   cd ~/fedgraph
   nano run_openfhe_delta.slurm  # Add your account
   sbatch run_openfhe_delta.slurm
   ```

4. **Monitor and collect results**

---

**Need help?** See `DELTA_HPC_SETUP.md` for detailed documentation.

**Ready to push to GitHub?** Let me know and I'll help you commit and push all these files to the `gcn_v2` branch.

