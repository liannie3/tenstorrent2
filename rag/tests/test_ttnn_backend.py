"""Host-only contract tests for the provisional TT-NN v0.80 adapter."""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
import tempfile
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


def load_backend(repo: Path):
    path = repo / "tools" / "ttnn_backend.py"
    spec = importlib.util.spec_from_file_location("ttnn_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    backend = load_backend(repo)

    assert backend.parse_major_minor("v0.80.0") == (0, 80)
    assert backend.parse_major_minor("0.80.0+wheel") == (0, 80)
    backend.require_supported_version("0.80.3")
    assert backend.installed_ttnn_version(
        SimpleNamespace(__version__="0.80.3")
    ) == "0.80.3"
    with patch.object(backend.importlib.metadata, "version", return_value="0.80.0"):
        assert backend.installed_ttnn_version(SimpleNamespace()) == "0.80.0"
    try:
        backend.require_supported_version("0.81.0")
        raise AssertionError("version mismatch was accepted")
    except RuntimeError:
        pass

    normalized = backend.normalize_queries([3.0, 4.0, 0.0, 2.0], 2, 2)
    assert normalized == [0.6, 0.8, 0.0, 1.0]
    assert all(math.isclose(
        sum(value * value for value in normalized[row * 2 : row * 2 + 2]),
        1.0,
    ) for row in range(2))

    scores = [0.5, 0.75, 0.75, 100.0, -1.0, 0.2, 0.1, 200.0]
    matches = backend.select_matches(scores, 2, 4, 3, 0.7)
    assert matches[0] == backend.Match(True, 1, 0.75)
    assert matches[1] == backend.Match(False, 1, 0.2)

    with tempfile.TemporaryDirectory() as folder:
        bank = Path(folder) / "bank.bin"
        with bank.open("wb") as output:
            output.write(struct.pack("<8s4I", b"RAGBANK1", 2, 3, 101, 1024))
            output.write(struct.pack("<6f", 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        entries, dimension, keys = backend.load_bank_keys(bank)
        assert (entries, dimension) == (2, 3)
        assert keys == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

        packed = Path(folder) / "bank.ttbank"
        payload = bytearray(32 * 32 * 2)
        # Transposed rows: key[entry 0, dimension 0] and key[entry 1, dimension 1].
        struct.pack_into("<H", payload, 0, 0x3F80)
        struct.pack_into("<H", payload, (32 + 1) * 2, 0x3F80)
        packed.write_bytes(struct.pack(
            "<8s8I", b"RAGTTB01", 1, 2, 3, 32, 32, 32, 1, 1
        ) + payload)
        packed.with_suffix(".ttbank.json").write_text(json.dumps({
            "payload_sha256": hashlib.sha256(payload).hexdigest()
        }), encoding="utf-8")
        loaded = backend.load_packed_bank(packed)
        assert loaded[:4] == (2, 3, 32, 32)
        assert loaded[4][0] == 1.0 and loaded[4][33] == 1.0


if __name__ == "__main__":
    main()
