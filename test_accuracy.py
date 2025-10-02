#!/usr/bin/env python3
"""
Test and compare accuracy between plaintext and encrypted NC FedGCN.
This script runs both versions and compares their performance.
"""

import sys
import attridict
from fedgraph.federated_methods import run_fedgraph

def test_plaintext_baseline():
    """Run NC FedGCN without encryption to establish baseline accuracy."""
    print("="*70)
    print("🔍 Running PLAINTEXT baseline (no encryption)")
    print("="*70)
    
    config = {
        # Task, Method, and Dataset Settings
        "fedgraph_task": "NC",
        "dataset": "cora",
        "method": "FedGCN",
        "iid_beta": 10000,
        "distribution_type": "average",
        # Training Configuration
        "global_rounds": 10,  # Reduced for faster testing
        "local_step": 3,
        "learning_rate": 0.5,
        "n_trainer": 2,
        "batch_size": -1,
        # Model Structure
        "num_layers": 2,
        "num_hops": 1,
        # Resource and Hardware Settings
        "gpu": False,
        "num_cpus_per_trainer": 1,
        "num_gpus_per_trainer": 0,
        # Logging and Output Configuration
        "logdir": "./runs/plaintext",
        # Security and Privacy
        "use_encryption": False,  # ← NO ENCRYPTION (baseline)
        # Dataset Handling Options
        "use_huggingface": False,
        "saveto_huggingface": False,
        # Scalability and Cluster Configuration
        "use_cluster": False,
    }
    
    config = attridict.attridict(config)
    
    print("\n📊 Config:")
    print(f"  • Method: {config.method}")
    print(f"  • Dataset: {config.dataset}")
    print(f"  • Trainers: {config.n_trainer}")
    print(f"  • Global rounds: {config.global_rounds}")
    print(f"  • Encryption: {config.use_encryption}")
    print()
    
    try:
        run_fedgraph(config)
        print("\n✅ Plaintext baseline completed!")
        return True
    except Exception as e:
        print(f"\n❌ Plaintext baseline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tenseal_encryption():
    """Run NC FedGCN with TenSEAL encryption for comparison."""
    print("\n" + "="*70)
    print("🔍 Running TENSEAL encryption (single-key)")
    print("="*70)
    
    config = {
        "fedgraph_task": "NC",
        "dataset": "cora",
        "method": "FedGCN",
        "iid_beta": 10000,
        "distribution_type": "average",
        "global_rounds": 10,
        "local_step": 3,
        "learning_rate": 0.5,
        "n_trainer": 2,
        "batch_size": -1,
        "num_layers": 2,
        "num_hops": 1,
        "gpu": False,
        "num_cpus_per_trainer": 1,
        "num_gpus_per_trainer": 0,
        "logdir": "./runs/tenseal",
        "use_encryption": True,  # ← WITH ENCRYPTION
        "he_backend": "tenseal",  # ← TenSEAL (default)
        "use_huggingface": False,
        "saveto_huggingface": False,
        "use_cluster": False,
    }
    
    config = attridict.attridict(config)
    
    print("\n📊 Config:")
    print(f"  • Method: {config.method}")
    print(f"  • Dataset: {config.dataset}")
    print(f"  • Encryption: {config.use_encryption}")
    print(f"  • HE Backend: {config.he_backend}")
    print()
    
    try:
        run_fedgraph(config)
        print("\n✅ TenSEAL encryption completed!")
        return True
    except Exception as e:
        print(f"\n❌ TenSEAL encryption failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openfhe_encryption():
    """Run NC FedGCN with OpenFHE threshold encryption."""
    print("\n" + "="*70)
    print("🔍 Running OPENFHE threshold encryption (two-party)")
    print("="*70)
    
    config = {
        "fedgraph_task": "NC",
        "dataset": "cora",
        "method": "FedGCN",
        "iid_beta": 10000,
        "distribution_type": "average",
        "global_rounds": 10,
        "local_step": 3,
        "learning_rate": 0.5,
        "n_trainer": 2,
        "batch_size": -1,
        "num_layers": 2,
        "num_hops": 1,
        "gpu": False,
        "num_cpus_per_trainer": 1,
        "num_gpus_per_trainer": 0,
        "logdir": "./runs/openfhe",
        "use_encryption": True,  # ← WITH ENCRYPTION
        "he_backend": "openfhe",  # ← OpenFHE threshold
        "use_huggingface": False,
        "saveto_huggingface": False,
        "use_cluster": False,
    }
    
    config = attridict.attridict(config)
    
    print("\n📊 Config:")
    print(f"  • Method: {config.method}")
    print(f"  • Dataset: {config.dataset}")
    print(f"  • Encryption: {config.use_encryption}")
    print(f"  • HE Backend: {config.he_backend}")
    print()
    
    try:
        run_fedgraph(config)
        print("\n✅ OpenFHE threshold encryption completed!")
        return True
    except Exception as e:
        print(f"\n❌ OpenFHE encryption failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests and compare results."""
    print("\n" + "="*70)
    print("🧪 NC FedGCN Accuracy Testing")
    print("="*70)
    print("\nThis will test three configurations:")
    print("  1. Plaintext (no encryption) - BASELINE")
    print("  2. TenSEAL (single-key) - COMPARISON")
    print("  3. OpenFHE (two-party threshold) - OUR IMPLEMENTATION")
    print()
    
    results = {}
    
    # Test 1: Plaintext baseline
    print("\n" + "🔹"*35)
    print("TEST 1: Plaintext Baseline")
    print("🔹"*35)
    results['plaintext'] = test_plaintext_baseline()
    
    # Test 2: TenSEAL encryption
    print("\n" + "🔹"*35)
    print("TEST 2: TenSEAL Encryption")
    print("🔹"*35)
    results['tenseal'] = test_tenseal_encryption()
    
    # Test 3: OpenFHE encryption
    print("\n" + "🔹"*35)
    print("TEST 3: OpenFHE Threshold Encryption")
    print("🔹"*35)
    results['openfhe'] = test_openfhe_encryption()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"\n{'Configuration':<30} {'Status':<15}")
    print("-"*45)
    print(f"{'Plaintext (baseline)':<30} {'✅ Passed' if results['plaintext'] else '❌ Failed':<15}")
    print(f"{'TenSEAL (single-key)':<30} {'✅ Passed' if results['tenseal'] else '❌ Failed':<15}")
    print(f"{'OpenFHE (threshold)':<30} {'✅ Passed' if results['openfhe'] else '❌ Failed':<15}")
    
    print("\n📝 Note: Check tensorboard logs in ./runs/ for detailed metrics:")
    print("  • Plaintext: ./runs/plaintext")
    print("  • TenSEAL:   ./runs/tenseal")
    print("  • OpenFHE:   ./runs/openfhe")
    
    print("\n💡 To view results:")
    print("  tensorboard --logdir ./runs")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests completed successfully!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

