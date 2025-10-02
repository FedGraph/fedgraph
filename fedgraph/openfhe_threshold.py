"""
OpenFHE Threshold Homomorphic Encryption Wrapper

This module provides a two-party threshold HE implementation using OpenFHE CKKS.
Supports distributed key generation, encryption, addition, and threshold decryption.
"""

import openfhe
import numpy as np
from typing import List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class OpenFHEThresholdCKKS:
    """
    Two-party threshold homomorphic encryption using OpenFHE CKKS.
    
    This class implements threshold HE where:
    - All clients share the same public key
    - Server holds one secret share
    - Designated trainer holds the other secret share
    - Decryption requires both parties
    """
    
    def __init__(self, security_level: int = 128, ring_dim: int = 16384, cc=None):
        """
        Initialize OpenFHE threshold context.
        
        Args:
            security_level: Security level (128, 192, or 256 bits)
            ring_dim: Ring dimension (must be power of 2, >= 16384 for 128-bit security)
            cc: Optional shared CryptoContext (if None, creates new one)
        """
        self.security_level = security_level
        self.ring_dim = ring_dim
        self.cc = cc
        self.public_key = None
        self.secret_key_share = None
        self.is_lead_party = False
        
        if self.cc is None:
            # Map security levels to OpenFHE constants
            security_map = {
                128: openfhe.HEStd_128_classic,
                192: openfhe.HEStd_192_classic,
                256: openfhe.HEStd_256_classic
            }
            
            if security_level not in security_map:
                raise ValueError(f"Security level must be 128, 192, or 256, got {security_level}")
            
            self._setup_context(security_map[security_level])
    
    def _setup_context(self, security_constant):
        """Setup the OpenFHE crypto context."""
        params = openfhe.CCParamsCKKSRNS()
        params.SetSecurityLevel(security_constant)
        params.SetRingDim(self.ring_dim)
        
        # More headroom for multiparty fusion:
        params.SetMultiplicativeDepth(2)
        params.SetScalingModSize(59)
        params.SetFirstModSize(60)
        
        # More forgiving automatic scaling in multiparty:
        if hasattr(params, "SetScalingTechnique"):
            # FLEXIBLEAUTOEXT is recommended for tricky CKKS pipelines
            params.SetScalingTechnique(openfhe.FLEXIBLEAUTOEXT)
        
        self.cc = openfhe.GenCryptoContext(params)
        
        feats = openfhe.PKESchemeFeature
        for name in ("PKE", "SHE", "LEVELEDSHE", "PRE", "MULTIPARTY"):
            if hasattr(feats, name):
                self.cc.Enable(getattr(feats, name))
        
        logger.info(f"OpenFHE context initialized with ring_dim={self.ring_dim}")
    
    def generate_lead_keys(self):
        """Lead party: produce initial key pair."""
        self.is_lead_party = True
        kp1 = self.cc.KeyGen()
        self.public_key = kp1.publicKey
        self.secret_key_share = kp1.secretKey
        logger.info("Lead party: KeyGen done")
        return kp1
    
    def generate_nonlead_share(self, lead_public_key):
        """Non-lead party: derive secret share from the lead's public key."""
        self.is_lead_party = False
        kp2 = self.cc.MultipartyKeyGen(lead_public_key)
        # Save our share; public_key will be set to the final joint PK later.
        self.secret_key_share = kp2.secretKey
        logger.info("Non-lead party: MultipartyKeyGen done")
        return kp2
    
    def finalize_joint_public_key(self, nonlead_public_key):
        """Lead party: finalize the joint public key using the non-lead's contribution."""
        assert self.is_lead_party and self.secret_key_share is not None
        kp_final = self.cc.MultipartyKeyGen(nonlead_public_key)
        self.public_key = kp_final.publicKey
        logger.info("Lead party: joint public key finalized")
        return self.public_key
    
    def set_public_key(self, public_key: openfhe.PublicKey):
        """Set the public key (for non-lead parties)."""
        self.public_key = public_key
        logger.info("Public key set for threshold HE")
    
    def encrypt(self, data: Union[List[float], np.ndarray]) -> openfhe.Ciphertext:
        """
        Encrypt data using the public key.
        
        Args:
            data: List or numpy array of float values to encrypt
            
        Returns:
            Encrypted ciphertext
        """
        if self.public_key is None:
            raise RuntimeError("Public key not set. Call generate_keys() or set_public_key() first.")
        
        # Convert to list if numpy array
        if isinstance(data, np.ndarray):
            data = data.tolist()
        
        # Stable high scale for multiparty fusion
        scale = 2**50
        plaintext = self.cc.MakeCKKSPackedPlaintext(data, scale)
        
        ciphertext = self.cc.Encrypt(self.public_key, plaintext)
        
        logger.debug(f"Encrypted {len(data)} values")
        return ciphertext
    
    def add_ciphertexts(self, ct1: openfhe.Ciphertext, ct2: openfhe.Ciphertext) -> openfhe.Ciphertext:
        """
        Add two ciphertexts homomorphically.
        
        Args:
            ct1: First ciphertext
            ct2: Second ciphertext
            
        Returns:
            Sum of the ciphertexts
        """
        return self.cc.EvalAdd(ct1, ct2)
    
    def add_ciphertext_list(self, ciphertexts: List[openfhe.Ciphertext]) -> openfhe.Ciphertext:
        """
        Add multiple ciphertexts homomorphically.
        
        Args:
            ciphertexts: List of ciphertexts to add
            
        Returns:
            Sum of all ciphertexts
        """
        if not ciphertexts:
            raise ValueError("Empty ciphertext list")
        
        result = ciphertexts[0]
        for ct in ciphertexts[1:]:
            result = self.cc.EvalAdd(result, ct)
        
        logger.debug(f"Added {len(ciphertexts)} ciphertexts")
        return result
    
    def partial_decrypt(self, ciphertext: openfhe.Ciphertext) -> openfhe.Plaintext:
        """
        Perform partial decryption using this party's secret key share.
        
        Args:
            ciphertext: Ciphertext to partially decrypt
            
        Returns:
            Partially decrypted plaintext
        """
        if self.secret_key_share is None:
            raise RuntimeError("Secret key share not set. Call generate_lead_keys() or generate_nonlead_share() first.")
        
        if self.is_lead_party:
            pt_list = self.cc.MultipartyDecryptLead([ciphertext], self.secret_key_share)
        else:
            pt_list = self.cc.MultipartyDecryptMain([ciphertext], self.secret_key_share)
        
        logger.debug(f"Performed partial decryption (lead_party={self.is_lead_party})")
        return pt_list[0]
    
    def fuse_partial_decryptions(self, partial1: openfhe.Plaintext, partial2: openfhe.Plaintext) -> List[float]:
        """
        Fuse two partial decryptions to get the final result.
        
        Args:
            partial1: First partial decryption (plaintext)
            partial2: Second partial decryption (plaintext)
            
        Returns:
            Decrypted plaintext values as list of floats
        """
        # Be strict about lead/main ordering at fusion
        fused = self.cc.MultipartyDecryptFusion([partial1, partial2])
        # Optional: set logical length to your input length before reading values
        # fused.SetLength(N)  # uncomment if you see trailing zeros
        
        # Extract the plaintext values
        result = fused.GetRealPackedValue()
        
        logger.debug(f"Fused partial decryptions, got {len(result)} values")
        return result
    
    def decrypt(self, ciphertext: openfhe.Ciphertext) -> List[float]:
        """
        Decrypt a ciphertext using the full secret key (for testing only).
        NOTE: This is NOT valid for threshold mode - only for non-threshold tests.
        
        Args:
            ciphertext: Ciphertext to decrypt
            
        Returns:
            Decrypted plaintext values
        """
        if self.secret_key_share is None:
            raise RuntimeError("Secret key share not set. Call generate_lead_keys() or generate_nonlead_share() first.")
        
        # For testing purposes, use regular decryption
        # In production, this should use threshold decryption
        decrypted = self.cc.Decrypt(self.secret_key_share, ciphertext)
        result = decrypted.GetRealPackedValue()
        
        logger.debug(f"Decrypted {len(result)} values")
        return result
    
    def get_context_info(self) -> dict:
        """Get information about the crypto context."""
        return {
            "security_level": self.security_level,
            "ring_dim": self.ring_dim,
            "has_public_key": self.public_key is not None,
            "has_secret_share": self.secret_key_share is not None,
            "is_lead_party": self.is_lead_party
        }


# Convenience functions for easy integration
def create_threshold_context(security_level: int = 128, ring_dim: int = 16384) -> OpenFHEThresholdCKKS:
    """Create a new threshold HE context."""
    return OpenFHEThresholdCKKS(security_level, ring_dim)


def test_simple_he():
    """Test basic OpenFHE functionality without threshold."""
    print("Testing basic OpenFHE HE...")
    
    # Create a simple context
    server = create_threshold_context()
    
    # Generate regular (non-threshold) keys
    kp = server.cc.KeyGen()
    server.public_key = kp.publicKey
    server.secret_key_share = kp.secretKey
    
    # Test simple encryption/decryption
    x = [0.1, 0.2, 0.3]
    ct_x = server.encrypt(x)
    decrypted = server.decrypt(ct_x)
    
    print("Expected:", x)
    print("Result:  ", decrypted[:len(x)])
    
    # Check if it's close enough
    if all(abs(e - r) < 1e-1 for e, r in zip(x, decrypted[:len(x)])):
        print("Basic HE test passed!")
        return True
    else:
        print("Basic HE test failed!")
        return False


def test_threshold_he():
    """Test the threshold HE implementation."""
    import signal
    import sys
    
    def timeout_handler(signum, frame):
        print("Test timed out after 30 seconds")
        sys.exit(1)
    
    # Set a 30-second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    try:
        print("Testing OpenFHE Threshold HE...")
        
        # ONE context for both roles
        server = create_threshold_context()
        trainer = OpenFHEThresholdCKKS(security_level=128, ring_dim=16384, cc=server.cc)
        
        # 1) Lead generates initial keys
        kp1 = server.generate_lead_keys()
        
        # 2) Non-lead derives its share from lead's PK (same cc)
        kp2 = trainer.generate_nonlead_share(kp1.publicKey)
        
        # 3) Lead finalizes the joint public key
        joint_pk = server.finalize_joint_public_key(kp2.publicKey)
        
        # 4) Distribute the joint public key (same cc)
        trainer.set_public_key(joint_pk)
        
        # Quick integrity checks
        print("Joint PK set on both? ", server.public_key is not None, trainer.public_key is not None)
        print("Lead/Main flags: ", server.is_lead_party, trainer.is_lead_party)
        
        x = [0.1, 0.2, 0.3]  # Test vectors
        y = [0.05, 0.1, 0.15]  # Test vectors
        ct_x = server.encrypt(x)
        ct_y = trainer.encrypt(y)
        
        ct_sum = server.add_ciphertexts(ct_x, ct_y)
        
        p_lead = server.partial_decrypt(ct_sum)    # lead
        p_main = trainer.partial_decrypt(ct_sum)   # non-lead
        out = server.fuse_partial_decryptions(p_lead, p_main)
        
        exp = [a+b for a,b in zip(x,y)]
        print("Expected:", exp)
        print("Result:  ", out[:len(exp)])
        assert all(abs(e - r) < 1e-1 for e, r in zip(exp, out[:len(exp)]))
        print("Threshold HE test completed!")
        
    finally:
        signal.alarm(0)  # Cancel the alarm


if __name__ == "__main__":
    test_threshold_he()