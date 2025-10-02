#!/usr/bin/env python3
"""
Test OpenFHE two-party threshold integration for NC FedGCN pretrain.
This test verifies the implementation without running the full federated learning pipeline.
"""

import sys
import os

# Add the fedgraph directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_openfhe_import():
    """Test that OpenFHE can be imported"""
    try:
        import openfhe
        print("✓ OpenFHE imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import OpenFHE: {e}")
        print("  Please install OpenFHE: pip install openfhe==1.4.0.1.24.4")
        return False

def test_threshold_wrapper():
    """Test the OpenFHE threshold wrapper"""
    try:
        from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS
        print("✓ OpenFHE threshold wrapper imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import threshold wrapper: {e}")
        return False

def test_two_party_protocol():
    """Test the two-party key generation protocol"""
    try:
        from fedgraph.openfhe_threshold import OpenFHEThresholdCKKS
        
        print("\nTesting Two-Party Threshold Protocol:")
        print("=" * 50)
        
        # 1. Server (lead party) generates initial keys
        print("Step 1: Server generates lead keys...")
        server = OpenFHEThresholdCKKS(security_level=128, ring_dim=16384)
        kp1 = server.generate_lead_keys()
        print("  ✓ Server generated lead keys")
        
        # 2. Designated trainer (non-lead party) generates share
        print("Step 2: Trainer generates non-lead share...")
        trainer = OpenFHEThresholdCKKS(security_level=128, ring_dim=16384, cc=server.cc)
        kp2 = trainer.generate_nonlead_share(kp1.publicKey)
        print("  ✓ Trainer generated non-lead share")
        
        # 3. Server finalizes joint public key
        print("Step 3: Server finalizes joint public key...")
        joint_pk = server.finalize_joint_public_key(kp2.publicKey)
        print("  ✓ Server finalized joint public key")
        
        # 4. Set joint public key on trainer
        print("Step 4: Setting joint public key on trainer...")
        trainer.set_public_key(joint_pk)
        print("  ✓ Joint public key set on trainer")
        
        # 5. Test encryption and threshold decryption
        print("\nStep 5: Testing encryption and threshold decryption...")
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Encrypt with server's public key
        ct1 = server.encrypt(test_data)
        print("  ✓ Server encrypted data")
        
        # Encrypt with trainer's public key
        ct2 = trainer.encrypt(test_data)
        print("  ✓ Trainer encrypted data")
        
        # Add ciphertexts
        ct_sum = server.add_ciphertexts(ct1, ct2)
        print("  ✓ Added ciphertexts")
        
        # Threshold decryption: both parties do partial decrypt
        partial_lead = server.partial_decrypt(ct_sum)
        print("  ✓ Server performed partial decryption (lead)")
        
        partial_main = trainer.partial_decrypt(ct_sum)
        print("  ✓ Trainer performed partial decryption (main)")
        
        # Fuse partial decryptions
        result = server.fuse_partial_decryptions(partial_lead, partial_main)
        print("  ✓ Fused partial decryptions")
        
        # Verify result
        expected = [2.0, 4.0, 6.0, 8.0, 10.0]
        result_slice = result[:len(expected)]
        
        print(f"\n  Expected: {expected}")
        print(f"  Result:   {result_slice}")
        
        # Check accuracy
        errors = [abs(e - r) for e, r in zip(expected, result_slice)]
        max_error = max(errors)
        print(f"  Max error: {max_error:.2e}")
        
        if max_error < 1e-1:
            print("\n✓ Two-party threshold protocol works correctly!")
            return True
        else:
            print(f"\n✗ Error too large: {max_error}")
            return False
            
    except Exception as e:
        print(f"✗ Two-party protocol test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_trainer_integration():
    """Test that server and trainer classes have the required methods"""
    try:
        print("\nTesting Server/Trainer Integration:")
        print("=" * 50)
        
        # Check server methods
        from fedgraph.server_class import Server
        
        required_server_methods = [
            '_aggregate_openfhe_feature_sums',
        ]
        
        for method in required_server_methods:
            if hasattr(Server, method):
                print(f"  ✓ Server has method: {method}")
            else:
                print(f"  ✗ Server missing method: {method}")
                return False
        
        # Check trainer methods
        from fedgraph.trainer_class import Trainer_General
        
        required_trainer_methods = [
            'setup_openfhe_nonlead',
            'set_openfhe_public_key',
            'openfhe_partial_decrypt_main',
            '_get_openfhe_encrypted_local_feature_sum',
        ]
        
        for method in required_trainer_methods:
            if hasattr(Trainer_General, method):
                print(f"  ✓ Trainer has method: {method}")
            else:
                print(f"  ✗ Trainer missing method: {method}")
                return False
        
        print("\n✓ Server and Trainer integration looks good!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🔍 Testing OpenFHE Two-Party Threshold Integration for NC FedGCN")
    print("=" * 60)
    print()
    
    tests = [
        ("OpenFHE Import", test_openfhe_import),
        ("Threshold Wrapper", test_threshold_wrapper),
        ("Server/Trainer Integration", test_server_trainer_integration),
    ]
    
    # Only run two-party protocol test if OpenFHE is available
    if test_openfhe_import():
        tests.append(("Two-Party Protocol", test_two_party_protocol))
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print('='*60)
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {passed}/{total}")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! OpenFHE threshold integration is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        if passed < 2:
            print("\n💡 Tip: Run this inside Docker for full OpenFHE support:")
            print("   ./run_docker_openfhe.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())

