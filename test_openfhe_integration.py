#!/usr/bin/env python3
"""
Simple test script to verify OpenFHE threshold integration works.
This tests the basic OpenFHE wrapper functionality.
"""

import sys
import os

# Add the fedgraph directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fedgraph'))

def test_openfhe_import():
    """Test that OpenFHE wrapper can be imported"""
    try:
        from openfhe_threshold import OpenFHEThresholdCKKS
        print("✓ OpenFHE wrapper imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import OpenFHE wrapper: {e}")
        return False

def test_openfhe_initialization():
    """Test that OpenFHE context can be initialized"""
    try:
        from openfhe_threshold import OpenFHEThresholdCKKS
        cc = OpenFHEThresholdCKKS()
        print("✓ OpenFHE context initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize OpenFHE context: {e}")
        return False

def test_openfhe_encryption():
    """Test basic encryption/decryption"""
    try:
        from openfhe_threshold import OpenFHEThresholdCKKS
        cc = OpenFHEThresholdCKKS()
        
        # Test data
        test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Encrypt
        encrypted = cc.encrypt(test_data)
        print("✓ OpenFHE encryption successful")
        
        # Decrypt
        decrypted = cc.decrypt(encrypted)
        print("✓ OpenFHE decryption successful")
        
        # Check values are close (floating point precision)
        import numpy as np
        if np.allclose(test_data, decrypted, atol=1e-3):
            print("✓ Encrypted/decrypted values match")
            return True
        else:
            print(f"✗ Values don't match: {test_data} vs {decrypted}")
            return False
            
    except Exception as e:
        print(f"✗ Encryption/decryption test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing OpenFHE threshold integration...")
    print("=" * 50)
    
    tests = [
        test_openfhe_import,
        test_openfhe_initialization,
        test_openfhe_encryption,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! OpenFHE integration is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 