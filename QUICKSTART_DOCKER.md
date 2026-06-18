# QuickStart: OpenFHE NC with Docker

## Prerequisites

- Docker Desktop installed and running
- 4GB+ free RAM
- 5GB+ free disk space

## 3-Step Quick Start

### 1. Build Docker Image (5-10 minutes, one-time only)

```bash
cd /Users/fanxy/Documents/GitHub/fedgraph-10
docker build -t fedgraph-openfhe .
```

### 2. Run Interactive Container

```bash
docker run -it --rm fedgraph-openfhe bash
```

You're now inside the container!

### 3. Run OpenFHE NC Tutorial

```bash
cd /app/tutorials
python FGL_NC_HE.py
```

**Expected output:**
```
Starting OpenFHE threshold encrypted feature aggregation...
Step 1: Server generates lead keys...
Step 2: Designated trainer generates non-lead share...
Step 3: Server finalizes joint public key...
Step 4: Distributing joint public key to all trainers...
Two-party threshold key generation complete!

Training Round 1/10...
...
Final Test Accuracy: ~0.81
```

---

## Quick Comparison Test

Inside the Docker container, compare plaintext vs OpenFHE:

```bash
python << 'EOF'
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
print("TEST 2: OPENFHE (Secure)")
print("="*60)
config["use_encryption"] = True
config["he_backend"] = "openfhe"
run_NC(AttriDict(config))

print("\n" + "="*60)
print("DONE! Compare the two accuracies above.")
print("Expected: < 1% difference")
print("="*60)
EOF
```

---

## Troubleshooting

### Issue: Docker daemon not running
**Error:** `Cannot connect to the Docker daemon`

**Solution:** Start Docker Desktop application

### Issue: Permission denied
**Error:** `permission denied while trying to connect to the Docker daemon socket`

**Solution (Mac/Linux):**
```bash
sudo usermod -aG docker $USER
# Then logout and login again
```

### Issue: Out of memory
**Error:** `Killed` during build

**Solution:** Increase Docker memory limit:
- Docker Desktop > Settings > Resources > Memory
- Set to at least 4GB

---

## Custom Configuration

To run with different settings:

```bash
python << 'EOF'
from fedgraph.federated_methods import run_NC
from attridict import AttriDict

config = {
    "fedgraph_task": "NC",
    "method": "FedGCN",
    "use_encryption": True,
    "he_backend": "openfhe",
    
    # Customize these:
    "dataset": "citeseer",      # Options: cora, citeseer, pubmed
    "num_trainers": 5,           # Number of federated clients
    "num_rounds": 20,            # Training rounds
    "iid_beta": 10000,           # Data distribution (higher = more IID)
    "seed": 42,
}

run_NC(AttriDict(config))
EOF
```

---

## What's Happening?

1. **Two-Party Key Generation:**
   - Server generates lead key share
   - Designated trainer generates non-lead key share
   - Joint public key is created and distributed

2. **Secure Feature Aggregation:**
   - Trainers encrypt local features with joint public key
   - Server aggregates encrypted features (homomorphically)
   - Server produces partial decryption (cannot decrypt alone)
   - Designated trainer produces partial decryption
   - Server fuses both partials to get final result

3. **Security Guarantee:**
   - Neither server nor any single trainer can decrypt alone
   - Requires cooperation between server and designated trainer

---

## Next Steps

- See `README_OPENFHE.md` for detailed documentation
- See `OPENFHE_NC_IMPLEMENTATION.md` for technical details
- Modify `tutorials/FGL_NC_HE.py` for custom experiments

---

## Cleaning Up

Exit container:
```bash
exit
```

Remove Docker image (to free space):
```bash
docker rmi fedgraph-openfhe
```

