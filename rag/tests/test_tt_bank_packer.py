"""Host-only tests for deterministic Tenstorrent bank packing."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import tempfile
from pathlib import Path


def load_packer(repo: Path):
    path = repo / "tools" / "pack_tt_bank.py"
    spec = importlib.util.spec_from_file_location("pack_tt_bank", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_source(path: Path) -> list[list[float]]:
    keys = [
        [1.0, 0.5, -0.25],
        [-1.0, 0.0, 0.75],
        [0.125, -0.5, 1.0],
    ]
    with path.open("wb") as output:
        output.write(struct.pack("<8s4I", b"RAGBANK1", 3, 3, 101, 1024))
        output.write(struct.pack("<9f", *(value for row in keys for value in row)))
    return keys


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    packer = load_packer(repo)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "memory.bank"
        keys = write_source(source)
        first = root / "first.ttbank"
        second = root / "second.ttbank"

        manifest = packer.pack(source, first, "test-release")
        packer.pack(source, second, "test-release")

        assert first.read_bytes() == second.read_bytes()
        assert first.with_suffix(".ttbank.json").read_bytes() == (
            second.with_suffix(".ttbank.json").read_bytes()
        )
        assert manifest["entries"] == 3
        assert manifest["embedding_dimension"] == 3
        assert manifest["padded_entries"] == 32
        assert manifest["padded_embedding_dimension"] == 32
        assert manifest["tt_metal_version"] == "test-release"
        assert manifest["source_bank_sha256"] == hashlib.sha256(
            source.read_bytes()
        ).hexdigest()

        raw = first.read_bytes()
        header = packer.HEADER.unpack(raw[: packer.HEADER.size])
        assert header == (b"RAGTTB01", 1, 3, 3, 32, 32, 32, 1, 1)
        payload = raw[packer.HEADER.size :]
        assert len(payload) == 32 * 32 * 2

        def packed_value(column: int, entry: int) -> int:
            offset = (column * 32 + entry) * 2
            return struct.unpack_from("<H", payload, offset)[0]

        for column in range(3):
            for entry in range(3):
                assert packed_value(column, entry) == packer.float_to_bf16(
                    keys[entry][column]
                )
        assert all(
            packed_value(column, entry) == 0
            for column in range(32)
            for entry in range(32)
            if column >= 3 or entry >= 3
        )

        on_disk_manifest = json.loads(
            first.with_suffix(".ttbank.json").read_text(encoding="utf-8")
        )
        assert on_disk_manifest == manifest


if __name__ == "__main__":
    main()
