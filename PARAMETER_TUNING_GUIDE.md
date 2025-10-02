# OpenFHE Parameter Tuning Guide for NC FedGCN

## Current Parameters (Default)

### OpenFHE CKKS Parameters
```python
# In fedgraph/openfhe_threshold.py
ring_dim = 16384              # Polynomial ring dimension
security_level = 128          # bits
multiplicative_depth = 2      # Number of multiplications supported
scaling_mod_size = 59         # bits
first_mod_size = 60           # bits
scale = 2**50                 # Encryption scale
scaling_technique = FLEXIBLEAUTOEXT  # Automatic rescaling
```

## Parameters Affecting Accuracy

### 1. Scale (Most Important for Accuracy)

**Current**: `2**50` (line 130 in openfhe_threshold.py)

**Impact on Accuracy**:
- **Higher scale** → Better precision → Less noise
- **Lower scale** → Worse precision → More noise

**Tuning Recommendations**:
```python
# For high precision (minimal accuracy loss)
scale = 2**55  # Excellent precision, slower

# Current (good balance)
scale = 2**50  # Good precision, moderate speed

# For faster computation (some accuracy loss)
scale = 2**45  # Lower precision, faster
```

**Expected Accuracy Impact**:
- `2**55`: < 0.5% accuracy loss vs plaintext
- `2**50`: < 1% accuracy loss vs plaintext (current)
- `2**45`: 1-3% accuracy loss vs plaintext

**How to Change**:
```python
# Edit fedgraph/openfhe_threshold.py, line 130
scale = 2**55  # Change this value
```

---

### 2. Ring Dimension

**Current**: `16384`

**Impact on Accuracy**:
- **Higher ring_dim** → Can pack more values → Better for large features
- **Lower ring_dim** → Fewer values per ciphertext → May truncate

**Tuning Recommendations**:
```python
# For large feature dimensions (e.g., > 1000)
ring_dim = 32768  # More slots, higher security

# Current (good for most cases)
ring_dim = 16384  # Standard, 128-bit security

# For small feature dimensions (< 500)
ring_dim = 8192   # Fewer slots, faster
```

**Expected Accuracy Impact**:
- Ring dimension itself doesn't affect precision
- Only matters if feature vectors don't fit (rare)

**How to Change**:
```python
# Edit fedgraph/federated_methods.py, lines 286, 293
server.openfhe_cc = OpenFHEThresholdCKKS(
    security_level=128, 
    ring_dim=32768  # Change this
)
```

---

### 3. Multiplicative Depth

**Current**: `2`

**Impact on Accuracy**:
- Determines how many multiplications before rescaling
- In pretrain, we only do additions (no multiplications needed)
- **Should not affect accuracy** for NC pretrain

**Tuning Recommendations**:
```python
# For pretrain (additions only)
multiplicative_depth = 1  # Sufficient, faster

# Current (safe default)
multiplicative_depth = 2  # Safe for future extensions

# For complex operations
multiplicative_depth = 3  # If adding multiplications later
```

**How to Change**:
```python
# Edit fedgraph/openfhe_threshold.py, line 63
params.SetMultiplicativeDepth(1)  # Change this
```

---

### 4. Scaling Modulus Sizes

**Current**: 
- `scaling_mod_size = 59`
- `first_mod_size = 60`

**Impact on Accuracy**:
- Related to scale parameter
- Determines precision of intermediate computations

**Tuning Recommendations**:
```python
# For higher scale (2**55)
scaling_mod_size = 60
first_mod_size = 61

# Current (matches scale 2**50)
scaling_mod_size = 59
first_mod_size = 60

# For lower scale (2**45)
scaling_mod_size = 50
first_mod_size = 51
```

**Rule of Thumb**: `log2(scale) ≈ scaling_mod_size`

**How to Change**:
```python
# Edit fedgraph/openfhe_threshold.py, lines 64-65
params.SetScalingModSize(60)  # Change this
params.SetFirstModSize(61)    # Change this
```

---

## Recommended Parameter Sets

### Option 1: Maximum Accuracy (Minimal Loss < 0.5%)
```python
# In openfhe_threshold.py
scale = 2**55                    # Line 130
params.SetMultiplicativeDepth(2) # Line 63
params.SetScalingModSize(60)     # Line 64
params.SetFirstModSize(61)       # Line 65

# In federated_methods.py
ring_dim = 16384  # or 32768 for very large features
```

**Trade-offs**:
- ✅ Best accuracy
- ✅ Minimal noise
- ❌ Slower (~1.2x baseline time)
- ❌ More memory

---

### Option 2: Balanced (Current - Good Default)
```python
# Current settings (no changes needed)
scale = 2**50
params.SetMultiplicativeDepth(2)
params.SetScalingModSize(59)
params.SetFirstModSize(60)
ring_dim = 16384
```

**Trade-offs**:
- ✅ Good accuracy (< 1% loss)
- ✅ Reasonable speed
- ✅ Moderate memory
- ✅ Works for most cases

---

### Option 3: Fast (Some Accuracy Loss ~2%)
```python
# In openfhe_threshold.py
scale = 2**45                    # Line 130
params.SetMultiplicativeDepth(1) # Line 63 (pretrain only needs additions)
params.SetScalingModSize(50)     # Line 64
params.SetFirstModSize(51)       # Line 65

# In federated_methods.py
ring_dim = 16384
```

**Trade-offs**:
- ✅ Faster (~1.5x faster than Option 1)
- ✅ Less memory
- ❌ Some accuracy loss (1-3%)

---

## Testing Methodology

### 1. Establish Baseline
```bash
# Run plaintext version first
python test_accuracy.py  # Just Test 1

# Record the final test accuracy
# Example: "Final Test Accuracy: 0.825"
```

### 2. Test with Encryption
```bash
# Run with current parameters
python tutorials/FGL_NC_HE.py

# Compare test accuracy
# Acceptable: ≤ 2% absolute drop (e.g., 0.825 → 0.805)
```

### 3. Tune if Needed
If accuracy loss > 2%:
```python
# Try increasing scale
# Edit fedgraph/openfhe_threshold.py line 130
scale = 2**55  # Increase from 2**50

# Rebuild and test
docker build -t fedgraph-openfhe .
# Run again
```

---

## Quick Parameter Change Guide

### To Increase Scale (Better Accuracy):

**File**: `fedgraph/openfhe_threshold.py`
**Line**: 130
```python
# Change from:
scale = 2**50

# To:
scale = 2**55  # or 2**52, 2**53, etc.
```

**Also update** (Lines 64-65):
```python
params.SetScalingModSize(60)  # Increase to ~log2(scale)
params.SetFirstModSize(61)
```

### To Increase Ring Dimension (More Packing):

**File**: `fedgraph/federated_methods.py`
**Line**: 286
```python
# Change from:
server.openfhe_cc = OpenFHEThresholdCKKS(security_level=128, ring_dim=16384)

# To:
server.openfhe_cc = OpenFHEThresholdCKKS(security_level=128, ring_dim=32768)
```

**Also update** Line 465 (trainer initialization):
```python
self.openfhe_cc = OpenFHEThresholdCKKS(security_level=128, ring_dim=32768, cc=crypto_context)
```

---

## Accuracy Monitoring

### Metrics to Track

1. **Test Accuracy** (primary metric)
   - Plaintext baseline: Track this first
   - Encrypted: Compare with baseline
   - Acceptable drop: ≤ 2%

2. **Training Loss**
   - Should converge similarly to plaintext
   - If diverging, increase scale

3. **Precision Error**
   - Monitor decryption errors in logs
   - If errors > 1e-3, increase scale

### Example Comparison

```
Configuration       | Test Accuracy | Δ vs Plaintext | Time  | Memory
--------------------|---------------|----------------|-------|--------
Plaintext           | 0.825         | -              | 1.0x  | 1.0x
TenSEAL (scale 2^40)| 0.818         | -0.7%          | 1.3x  | 1.5x
OpenFHE (scale 2^50)| 0.820         | -0.5%          | 1.4x  | 1.6x
OpenFHE (scale 2^55)| 0.824         | -0.1%          | 1.6x  | 1.8x
```

---

## Advanced Tuning Tips

### 1. Feature Normalization
Before encryption, normalize features to similar scales:
```python
# In your data preprocessing
features = (features - mean) / std
```
**Why**: CKKS works better when all values are similar magnitude

### 2. Batch Size
```python
# In config
batch_size = -1  # Full batch (current - good)
# vs
batch_size = 64  # Mini-batch (more noise accumulation)
```
**Recommendation**: Use full batch (`-1`) for better accuracy

### 3. Learning Rate
With encryption, you may need to adjust:
```python
# Current
learning_rate = 0.5

# If accuracy drops significantly, try:
learning_rate = 0.3  # Lower LR for more stable training
```

---

## Debugging Accuracy Issues

### If accuracy drops > 5%:

1. **Check scale parameter**:
   ```python
   scale = 2**55  # Increase significantly
   ```

2. **Check for overflow/underflow**:
   - Look for NaN in logs
   - Feature values too large? Normalize first

3. **Check decryption precision**:
   ```python
   # Add logging in server_class.py after fusion
   print(f"Decryption error: {torch.abs(decrypted - expected).max()}")
   ```

4. **Try higher ring dimension**:
   ```python
   ring_dim = 32768  # If features are large
   ```

---

## Testing Command

```bash
# Quick test (10 rounds)
python test_accuracy.py

# Full test (100 rounds - for final evaluation)
# Edit test_accuracy.py: global_rounds = 100
python test_accuracy.py

# Compare results
tensorboard --logdir ./runs
```

---

## Expected Results

### Target Performance
- **Accuracy Drop**: < 1% (absolute)
- **Time Overhead**: 1.3-1.5x
- **Memory Overhead**: 1.5-2x

### If Results Don't Meet Target
1. Increase scale to 2**55
2. Check feature normalization
3. Verify no numeric issues (NaN, Inf)
4. Consider adjusting learning rate

---

## Summary: Quick Start Tuning

**For minimal accuracy loss (< 0.5%)**:
```bash
# 1. Edit fedgraph/openfhe_threshold.py line 130
scale = 2**55

# 2. Edit lines 64-65
params.SetScalingModSize(60)
params.SetFirstModSize(61)

# 3. Test
python test_accuracy.py
```

**Current parameters are already well-tuned** for < 1% accuracy loss. Only tune if you observe > 2% drop.

---

**Last Updated**: October 2, 2025

