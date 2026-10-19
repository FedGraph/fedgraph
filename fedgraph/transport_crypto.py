"""Point-to-point transport encryption between parties of the pretraining round.

Partial decryptions and forwarded aggregate rows that are addressed to one
client are relayed through the server.  They are sealed under the recipient's
transport key (X25519 key agreement with an ephemeral sender key, HKDF-SHA256,
ChaCha20-Poly1305), so the relay learns only the ciphertext length.  This is
the "end-to-end encrypted relay" of Phase 4: the server never holds both
partial decryptions of a chunk in a form it can fuse.
"""
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_INFO = b"fedgcn-v2-transport"


def _derive(shared: bytes, epk: bytes, rpk: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_INFO + epk + rpk).derive(shared)


class TransportKey:
    """A party's long-lived transport key pair."""

    def __init__(self):
        self._sk = X25519PrivateKey.generate()
        self._pk = self._sk.public_key().public_bytes_raw()

    def public_bytes(self) -> bytes:
        return self._pk

    def open(self, blob: bytes) -> bytes:
        """Open a blob produced by :func:`seal` for this key."""
        epk, nonce, ct = blob[:32], blob[32:44], blob[44:]
        shared = self._sk.exchange(X25519PublicKey.from_public_bytes(epk))
        return ChaCha20Poly1305(_derive(shared, epk, self._pk)).decrypt(nonce, ct, None)


def seal(recipient_pk: bytes, data: bytes) -> bytes:
    """Seal ``data`` so that only the holder of ``recipient_pk`` can open it."""
    esk = X25519PrivateKey.generate()
    epk = esk.public_key().public_bytes_raw()
    shared = esk.exchange(X25519PublicKey.from_public_bytes(recipient_pk))
    nonce = os.urandom(12)
    return epk + nonce + ChaCha20Poly1305(_derive(shared, epk, recipient_pk)).encrypt(nonce, data, None)
