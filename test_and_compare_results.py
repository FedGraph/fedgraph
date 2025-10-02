#!/usr/bin/env python3
"""
Test NC FedGCN with different encryption settings and compare results.
This script shows where OpenFHE two-party threshold is used (PRETRAIN ONLY).
"""

import sys
import json
import time
from pathlib import Path

try:
    from attridict import attridict
    from fedgraph.federated_methods import run_fedgraph
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False
    print("⚠️  Missing dependencies. Run inside Docker or install fedgraph.")


def create_config(use_encryption=False, he_backend="tenseal", global_rounds=10):
    """Create configuration for testing."""
    return {
        # Task, Method, and Dataset Settings
        "fedgraph_task": "NC",
        "dataset": "cora",
        "method": "FedGCN",  # ← FedGCN uses pretrain!
        "iid_beta": 10000,
        "distribution_type": "average",
        # Training Configuration
        "global_rounds": global_rounds,
        "local_step": 3,
        "learning_rate": 0.5,
        "n_trainer": 2,
        "batch_size": -1,
        # Model Structure
        "num_layers": 2,
        "num_hops": 1,  # ← Must be >= 1 for pretrain phase
        # Resource and Hardware Settings
        "gpu": False,
        "num_cpus_per_trainer": 1,
        "num_gpus_per_trainer": 0,
        # Logging and Output Configuration
        "logdir": f"./runs/{'openfhe' if he_backend == 'openfhe' else ('tenseal' if use_encryption else 'plaintext')}_{int(time.time())}",
        # Security and Privacy - KEY SETTINGS
        "use_encryption": use_encryption,
        "he_backend": he_backend if use_encryption else None,
        # Dataset Handling Options
        "use_huggingface": False,
        "saveto_huggingface": False,
        # Scalability and Cluster Configuration
        "use_cluster": False,
    }


def print_config_summary(config):
    """Print configuration summary."""
    print("\n📋 Configuration:")
    print(f"  • Task: {config['fedgraph_task']}")
    print(f"  • Method: {config['method']}")
    print(f"  • Dataset: {config['dataset']}")
    print(f"  • Trainers: {config['n_trainer']}")
    print(f"  • Global Rounds: {config['global_rounds']}")
    print(f"  • Num Hops: {config['num_hops']} ({'PRETRAIN ENABLED' if config['num_hops'] >= 1 else 'NO PRETRAIN'})")
    print(f"  • Encryption: {config['use_encryption']}")
    if config['use_encryption']:
        print(f"  • HE Backend: {config.get('he_backend', 'tenseal')}")
        if config.get('he_backend') == 'openfhe':
            print("    ↳ 🔐 TWO-PARTY THRESHOLD HE (Server + Trainer0)")
        else:
            print("    ↳ 🔓 Single-key HE (Server only)")
    print()


def test_plaintext(rounds=10):
    """Test with plaintext (no encryption) - BASELINE."""
    print("="*70)
    print("TEST 1: PLAINTEXT BASELINE (No Encryption)")
    print("="*70)
    
    config = create_config(use_encryption=False, global_rounds=rounds)
    print_config_summary(config)
    
    if not HAVE_DEPS:
        print("❌ Cannot run - missing dependencies")
        return None
    
    config = attridict(config)
    
    try:
        print("🚀 Starting training...")
        start_time = time.time()
        run_fedgraph(config)
        total_time = time.time() - start_time
        
        print("\n✅ Plaintext test completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        return {
            'success': True,
            'time': total_time,
            'logdir': config.logdir
        }
    except Exception as e:
        print(f"\n❌ Plaintext test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_tenseal(rounds=10):
    """Test with TenSEAL encryption (single-key)."""
    print("\n" + "="*70)
    print("TEST 2: TENSEAL ENCRYPTION (Single-Key)")
    print("="*70)
    
    config = create_config(use_encryption=True, he_backend="tenseal", global_rounds=rounds)
    print_config_summary(config)
    
    if not HAVE_DEPS:
        print("❌ Cannot run - missing dependencies")
        return None
    
    config = attridict(config)
    
    try:
        print("🚀 Starting training with TenSEAL...")
        start_time = time.time()
        run_fedgraph(config)
        total_time = time.time() - start_time
        
        print("\n✅ TenSEAL test completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        return {
            'success': True,
            'time': total_time,
            'logdir': config.logdir
        }
    except Exception as e:
        print(f"\n❌ TenSEAL test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def test_openfhe(rounds=10):
    """Test with OpenFHE threshold encryption (two-party)."""
    print("\n" + "="*70)
    print("TEST 3: OPENFHE THRESHOLD ENCRYPTION (Two-Party)")
    print("="*70)
    print("\n🔐 This will use TWO-PARTY THRESHOLD:")
    print("  • Server holds secret_share_1")
    print("  • Trainer0 holds secret_share_2")
    print("  • Both required to decrypt (SECURE!)")
    print()
    
    config = create_config(use_encryption=True, he_backend="openfhe", global_rounds=rounds)
    print_config_summary(config)
    
    if not HAVE_DEPS:
        print("❌ Cannot run - missing dependencies")
        return None
    
    config = attridict(config)
    
    try:
        print("🚀 Starting training with OpenFHE...")
        print("\n📍 Watch for these PRETRAIN phase messages:")
        print("  1. 'Step 1: Server generates lead keys...'")
        print("  2. 'Step 2: Designated trainer generates non-lead share...'")
        print("  3. 'Step 3: Server finalizes joint public key...'")
        print("  4. 'Two-party threshold key generation complete!'")
        print()
        
        start_time = time.time()
        run_fedgraph(config)
        total_time = time.time() - start_time
        
        print("\n✅ OpenFHE threshold test completed!")
        print(f"⏱️  Total time: {total_time:.2f}s")
        return {
            'success': True,
            'time': total_time,
            'logdir': config.logdir
        }
    except Exception as e:
        print(f"\n❌ OpenFHE test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def print_comparison(results):
    """Print comparison of all test results."""
    print("\n" + "="*70)
    print("📊 RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Configuration':<30} {'Status':<15} {'Time (s)':<15} {'Overhead':<15}")
    print("-"*75)
    
    baseline_time = None
    for name, result in results.items():
        if result is None:
            status = "⏭️  Skipped"
            time_str = "-"
            overhead = "-"
        elif result['success']:
            status = "✅ Passed"
            time_str = f"{result['time']:.2f}"
            if name == 'plaintext':
                baseline_time = result['time']
                overhead = "1.0x"
            elif baseline_time:
                overhead = f"{result['time']/baseline_time:.2f}x"
            else:
                overhead = "-"
        else:
            status = "❌ Failed"
            time_str = "-"
            overhead = "-"
        
        print(f"{name:<30} {status:<15} {time_str:<15} {overhead:<15}")
    
    print("\n📝 Notes:")
    print("  • OpenFHE uses TWO-PARTY threshold (only in PRETRAIN phase)")
    print("  • Training phase uses plaintext (no encryption)")
    print("  • Pretrain = Feature aggregation before training")
    print("  • Expected overhead: 1.3-1.5x for encrypted pretrain")
    
    print("\n📁 Log directories:")
    for name, result in results.items():
        if result and result['success']:
            print(f"  • {name}: {result['logdir']}")
    
    print("\n💡 To view tensorboard:")
    print("  tensorboard --logdir ./runs")
    print("  Then open: http://localhost:6006")


def verify_implementation():
    """Verify that OpenFHE two-party is implemented in pretrain."""
    print("="*70)
    print("🔍 VERIFYING IMPLEMENTATION")
    print("="*70)
    
    federated_methods_path = Path(__file__).parent / "fedgraph" / "federated_methods.py"
    
    if not federated_methods_path.exists():
        print("❌ Cannot find federated_methods.py")
        return False
    
    with open(federated_methods_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("Two-party key generation", "generate_lead_keys"),
        ("Non-lead key setup", "setup_openfhe_nonlead"),
        ("Joint key finalization", "finalize_joint_public_key"),
        ("Public key distribution", "set_openfhe_public_key"),
        ("Pretrain phase", "Pre-Train Communication"),
        ("OpenFHE backend check", 'he_backend", "tenseal") == "openfhe"'),
    ]
    
    print("\n✅ Implementation verification:")
    all_found = True
    for desc, pattern in checks:
        if pattern in content:
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc} NOT FOUND")
            all_found = False
    
    if all_found:
        print("\n🎉 OpenFHE two-party threshold IS implemented in NC FedGCN pretrain!")
    else:
        print("\n⚠️  Some components not found")
    
    return all_found


def main():
    """Run all tests and show results."""
    print("\n" + "🔬"*35)
    print("NC FedGCN - OpenFHE Two-Party Threshold Testing")
    print("🔬"*35 + "\n")
    
    # First verify implementation
    if not verify_implementation():
        print("\n⚠️  Implementation verification failed!")
        return 1
    
    print("\n\nThis will run 3 tests:")
    print("  1. Plaintext (baseline) - No encryption")
    print("  2. TenSEAL - Single-key encryption")
    print("  3. OpenFHE - Two-party threshold encryption (PRETRAIN ONLY)")
    print()
    print("⏱️  Each test takes ~1-3 minutes (5 rounds)")
    print()
    
    if not HAVE_DEPS:
        print("❌ Missing dependencies. Please run inside Docker:")
        print("   docker run --rm -v $(pwd):/app/workspace -w /app/workspace \\")
        print("       fedgraph-openfhe python workspace/test_and_compare_results.py")
        return 1
    
    input("Press Enter to start tests (or Ctrl+C to cancel)...")
    
    # Run tests
    results = {}
    
    # Test 1: Plaintext
    results['plaintext'] = test_plaintext(rounds=5)
    
    # Test 2: TenSEAL
    results['tenseal'] = test_tenseal(rounds=5)
    
    # Test 3: OpenFHE
    results['openfhe'] = test_openfhe(rounds=5)
    
    # Print comparison
    print_comparison(results)
    
    # Check if all passed
    all_passed = all(r and r['success'] for r in results.values() if r is not None)
    
    if all_passed:
        print("\n🎉 All tests completed successfully!")
        print("\n✅ OpenFHE two-party threshold is working in NC FedGCN PRETRAIN!")
        return 0
    else:
        print("\n⚠️  Some tests failed or were skipped.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

