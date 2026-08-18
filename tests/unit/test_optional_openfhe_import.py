"""Regression coverage for plaintext imports without the optional OpenFHE wheel."""

from __future__ import annotations

import subprocess
import sys


def test_plaintext_trainer_import_does_not_require_openfhe() -> None:
    program = """
import importlib.abc
import sys


class BlockOpenFHE(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openfhe":
            raise ModuleNotFoundError("No module named 'openfhe'", name="openfhe")
        return None


sys.meta_path.insert(0, BlockOpenFHE())
import fedgraph
from fedgraph.trainer_class import Trainer_General

assert Trainer_General is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
