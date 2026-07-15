#!/usr/bin/env python3
"""
Minimal OpenFHE CKKS smoke test to verify setup before trying multiparty.
"""

import time

import openfhe


def test_basic_ckks():
    """Test basic CKKS functionality (no multiparty)."""
    print("🔍 Testing basic OpenFHE CKKS...")

    # Create context with conservative parameters
    params = openfhe.CCParamsCKKSRNS()
    params.SetSecurityLevel(openfhe.HEStd_128_classic)
    params.SetRingDim(16384)
    params.SetMultiplicativeDepth(1)
    params.SetScalingModSize(40)
    params.SetFirstModSize(50)
    print("✅ Parameters set")

    cc = openfhe.GenCryptoContext(params)
    print("✅ Context created")

    # Enable basic features
    cc.Enable(openfhe.PKE)
    cc.Enable(openfhe.LEVELEDSHE)
    print("✅ Features enabled")

    # Generate keys
    kp = cc.KeyGen()
    print("✅ Keys generated")

    # Test data
    x = [1.0, 2.0, 3.0]
    scale = 2**40
    pt = cc.MakeCKKSPackedPlaintext(x, scale)
    print("✅ Plaintext created")

    # Encrypt
    ct = cc.Encrypt(kp.publicKey, pt)
    print("✅ Encrypted")

    # Decrypt
    decrypted = cc.Decrypt(ct, kp.secretKey)
    decrypted.SetLength(len(x))  # Set logical length
    result = decrypted.GetRealPackedValue()
    print("✅ Decrypted")

    # Check result
    print(f"Expected: {x}")
    print(f"Result:   {result[:len(x)]}")

    # Verify accuracy
    errors = [abs(e - r) for e, r in zip(x, result[: len(x)])]
    max_error = max(errors)
    print(f"Max error: {max_error:.2e}")

    if max_error < 1e-3:
        print("🎉 Basic CKKS test PASSED!")
        return True
    else:
        print("❌ Basic CKKS test FAILED!")
        return False


def test_import_speed():
    """Test OpenFHE import speed."""
    print("🔍 Testing OpenFHE import speed...")
    start = time.time()
    import openfhe

    import_time = time.time() - start
    print(f"✅ Import took {import_time:.2f} seconds")

    if import_time < 5.0:
        print("🎉 Import speed OK!")
        return True
    else:
        print("⚠️  Import is slow (possible emulation)")
        return False


if __name__ == "__main__":
    print("🚀 OpenFHE Smoke Test")
    print("=" * 50)

    # Test import speed first
    import_ok = test_import_speed()
    print()

    # Test basic CKKS
    ckks_ok = test_basic_ckks()
    print()

    # Summary
    print("📊 Summary:")
    print(f"  Import speed: {'✅' if import_ok else '❌'}")
    print(f"  Basic CKKS:   {'✅' if ckks_ok else '❌'}")

    if import_ok and ckks_ok:
        print("\n🎉 All tests PASSED! Ready for threshold HE.")
        exit(0)
    else:
        print("\n❌ Some tests FAILED. Check environment setup.")
        exit(1)
