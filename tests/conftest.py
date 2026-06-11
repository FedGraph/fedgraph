"""Pytest configuration shared by the smoke-test suite."""
from __future__ import annotations

import pytest


def _openfhe_available() -> bool:
    try:
        import openfhe  # noqa: F401
        return True
    except Exception:  # pragma: no cover - exercised only on systems without openfhe
        return False


def _tenseal_available() -> bool:
    try:
        import tenseal  # noqa: F401
        return True
    except Exception:  # pragma: no cover
        return False


needs_openfhe = pytest.mark.skipif(
    not _openfhe_available(),
    reason="OpenFHE wheel not installed; threshold-HE tests skipped.",
)

needs_tenseal = pytest.mark.skipif(
    not _tenseal_available(),
    reason="TenSEAL wheel not installed; TenSEAL-backend tests skipped.",
)
