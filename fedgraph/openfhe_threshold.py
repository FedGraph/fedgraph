"""
OpenFHE Threshold Homomorphic Encryption Wrapper

This module provides a two-party threshold HE implementation using OpenFHE CKKS.
Supports distributed key generation, encryption, addition, and threshold decryption.

The protocol follows the official OpenFHE multiparty CKKS example:
- Party A (lead/server): calls cc.KeyGen()
- Party B (non-lead/trainer): calls cc.MultipartyKeyGen(kp1.publicKey)
- kp2.publicKey IS the joint public key (no separate finalization needed)
- Encryption uses kp2.publicKey (the joint key)
- Decryption: lead calls MultipartyDecryptLead, non-lead calls MultipartyDecryptMain
- Fusion combines both partial decryptions
"""

import openfhe
import numpy as np
from typing import List, Tuple, Optional, Union
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


class OpenFHEThresholdCKKS:
    """
    Two-party threshold homomorphic encryption using OpenFHE CKKS.

    This class implements threshold HE where:
    - Server (lead) and designated trainer (non-lead) each hold a secret share
    - The joint public key is kp2.publicKey (the non-lead party's output)
    - All parties encrypt with the joint public key
    - Decryption requires both parties' partial decryptions
    """

    def __init__(self, security_level: int = 128, ring_dim: int = 16384, cc=None):
        self.security_level = security_level
        self.ring_dim = ring_dim
        self.cc = cc
        self.public_key = None
        self.secret_key_share = None
        self.is_lead_party = False

        if self.cc is None:
            self._setup_context()

    def _setup_context(self):
        """Setup the OpenFHE crypto context following the official CKKS multiparty example."""
        params = openfhe.CCParamsCKKSRNS()
        # Follow official example: only set depth and scaling mod size.
        # Let OpenFHE auto-select ring dimension and security parameters.
        params.SetMultiplicativeDepth(3)
        params.SetScalingModSize(50)
        params.SetBatchSize(self.ring_dim // 2)

        self.cc = openfhe.GenCryptoContext(params)

        # Enable all features needed for threshold CKKS
        self.cc.Enable(openfhe.PKE)
        self.cc.Enable(openfhe.KEYSWITCH)
        self.cc.Enable(openfhe.LEVELEDSHE)
        self.cc.Enable(openfhe.ADVANCEDSHE)
        self.cc.Enable(openfhe.MULTIPARTY)

        logger.info(f"OpenFHE context initialized (ring_dim={self.cc.GetRingDimension()})")

    def generate_lead_keys(self):
        """Lead party (server): generate initial key pair."""
        self.is_lead_party = True
        kp1 = self.cc.KeyGen()
        self.public_key = kp1.publicKey
        self.secret_key_share = kp1.secretKey
        logger.info("Lead party: KeyGen done")
        return kp1

    def generate_nonlead_share(self, lead_public_key):
        """
        Non-lead party (trainer): derive secret share from the lead's public key.

        IMPORTANT: kp2.publicKey is the joint public key that everyone uses for
        encryption. There is no separate finalization step.
        """
        self.is_lead_party = False
        kp2 = self.cc.MultipartyKeyGen(lead_public_key)
        self.secret_key_share = kp2.secretKey
        # kp2.publicKey IS the joint public key
        self.public_key = kp2.publicKey
        logger.info("Non-lead party: MultipartyKeyGen done")
        return kp2

    def set_public_key(self, public_key):
        """Set the joint public key (for parties that didn't generate it)."""
        self.public_key = public_key
        logger.info("Public key set for threshold HE")

    def encrypt(self, data: Union[List[float], np.ndarray]):
        """Encrypt data using the joint public key."""
        if self.public_key is None:
            raise RuntimeError("Public key not set.")

        if isinstance(data, np.ndarray):
            data = data.tolist()

        plaintext = self.cc.MakeCKKSPackedPlaintext(data)
        ciphertext = self.cc.Encrypt(self.public_key, plaintext)

        logger.debug(f"Encrypted {len(data)} values")
        return ciphertext

    def add_ciphertexts(self, ct1, ct2):
        """Add two ciphertexts homomorphically."""
        return self.cc.EvalAdd(ct1, ct2)

    def add_ciphertext_list(self, ciphertexts):
        """Add multiple ciphertexts homomorphically."""
        if not ciphertexts:
            raise ValueError("Empty ciphertext list")

        result = ciphertexts[0]
        for ct in ciphertexts[1:]:
            result = self.cc.EvalAdd(result, ct)

        logger.debug(f"Added {len(ciphertexts)} ciphertexts")
        return result

    def partial_decrypt(self, ciphertext):
        """Perform partial decryption using this party's secret key share."""
        if self.secret_key_share is None:
            raise RuntimeError("Secret key share not set.")

        if self.is_lead_party:
            pt_list = self.cc.MultipartyDecryptLead([ciphertext], self.secret_key_share)
        else:
            pt_list = self.cc.MultipartyDecryptMain([ciphertext], self.secret_key_share)

        logger.debug(f"Performed partial decryption (lead_party={self.is_lead_party})")
        return pt_list[0]

    def fuse_partial_decryptions(self, partial_lead, partial_main) -> List[float]:
        """
        Fuse two partial decryptions to get the final result.
        Order matters: lead partial first, then main partial.
        """
        fused = self.cc.MultipartyDecryptFusion([partial_lead, partial_main])
        result = fused.GetRealPackedValue()

        logger.debug(f"Fused partial decryptions, got {len(result)} values")
        return result

    def serialize_context(self) -> bytes:
        """Serialize the CryptoContext to bytes for transfer via Ray."""
        tmpdir = tempfile.mkdtemp()
        cc_path = os.path.join(tmpdir, "cc.bin")
        try:
            self.cc.SerializeToFile(cc_path, openfhe.BINARY)
            with open(cc_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(cc_path):
                os.remove(cc_path)
            os.rmdir(tmpdir)

    def serialize_public_key(self) -> bytes:
        """Serialize the public key to bytes for transfer via Ray."""
        tmpdir = tempfile.mkdtemp()
        pk_path = os.path.join(tmpdir, "pk.bin")
        try:
            self.cc.SerializeToFile(pk_path, openfhe.BINARY)  # context must be serialized first
            if not openfhe.SerializeToFile(pk_path, self.public_key, openfhe.BINARY):
                raise RuntimeError("Failed to serialize public key")
            with open(pk_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(pk_path):
                os.remove(pk_path)
            os.rmdir(tmpdir)

    def serialize_ciphertext(self, ct) -> bytes:
        """Serialize a ciphertext to bytes for transfer via Ray."""
        tmpdir = tempfile.mkdtemp()
        ct_path = os.path.join(tmpdir, "ct.bin")
        try:
            if not openfhe.SerializeToFile(ct_path, ct, openfhe.BINARY):
                raise RuntimeError("Failed to serialize ciphertext")
            with open(ct_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(ct_path):
                os.remove(ct_path)
            os.rmdir(tmpdir)

    def deserialize_ciphertext(self, ct_bytes: bytes):
        """Deserialize a ciphertext from bytes."""
        tmpdir = tempfile.mkdtemp()
        ct_path = os.path.join(tmpdir, "ct.bin")
        try:
            with open(ct_path, "wb") as f:
                f.write(ct_bytes)
            ct, success = openfhe.DeserializeCiphertext(ct_path, openfhe.BINARY)
            if not success:
                raise RuntimeError("Failed to deserialize ciphertext")
            return ct
        finally:
            if os.path.exists(ct_path):
                os.remove(ct_path)
            os.rmdir(tmpdir)

    def get_context_info(self) -> dict:
        """Get information about the crypto context."""
        return {
            "security_level": self.security_level,
            "ring_dim": self.cc.GetRingDimension() if self.cc else None,
            "has_public_key": self.public_key is not None,
            "has_secret_share": self.secret_key_share is not None,
            "is_lead_party": self.is_lead_party,
        }


def create_threshold_context(security_level: int = 128, ring_dim: int = 16384) -> OpenFHEThresholdCKKS:
    """Create a new threshold HE context."""
    return OpenFHEThresholdCKKS(security_level, ring_dim)


def test_threshold_he():
    """Test the threshold HE implementation following the official OpenFHE pattern."""
    import signal
    import sys

    def timeout_handler(signum, frame):
        print("Test timed out after 60 seconds")
        sys.exit(1)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)

    try:
        print("Testing OpenFHE Threshold HE...")

        # ONE shared context
        server = create_threshold_context()
        trainer = OpenFHEThresholdCKKS(cc=server.cc)

        # 1) Lead (server) generates initial key pair
        kp1 = server.generate_lead_keys()

        # 2) Non-lead (trainer) derives its share from lead's public key
        #    kp2.publicKey IS the joint public key
        kp2 = trainer.generate_nonlead_share(kp1.publicKey)

        # 3) Server also sets the joint public key (kp2.publicKey)
        server.set_public_key(kp2.publicKey)

        print("Joint PK set on both?", server.public_key is not None, trainer.public_key is not None)
        print("Lead/Main flags:", server.is_lead_party, trainer.is_lead_party)

        # Encrypt test vectors
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
        y = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ct_x = server.encrypt(x)
        ct_y = trainer.encrypt(y)

        # Homomorphic addition
        ct_sum = server.add_ciphertexts(ct_x, ct_y)

        # Threshold decryption
        p_lead = server.partial_decrypt(ct_sum)
        p_main = trainer.partial_decrypt(ct_sum)
        out = server.fuse_partial_decryptions(p_lead, p_main)

        exp = [a + b for a, b in zip(x, y)]
        print("Expected:", exp)
        print("Result:  ", [round(v, 2) for v in out[: len(exp)]])
        assert all(abs(e - r) < 0.1 for e, r in zip(exp, out[: len(exp)]))
        print("Threshold HE test PASSED!")

    finally:
        signal.alarm(0)


if __name__ == "__main__":
    test_threshold_he()
