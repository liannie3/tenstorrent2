"""Pack RAGBANK1 keys into a deterministic Tenstorrent-oriented BF16 matrix.

The output stores the normalized key matrix transposed as
``[padded_dimension, padded_entries]`` in row-major BF16. Both axes are padded
to 32 elements for tile-based device operations. Padded entries must still be
masked by the search implementation and are never valid candidate IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

SOURCE_MAGIC = b"RAGBANK1"
PACKED_MAGIC = b"RAGTTB01"
PACKED_VERSION = 1
TILE = 32
BF16_DTYPE = 1
TRANSPOSED_ROW_MAJOR_LAYOUT = 1
HEADER = struct.Struct("<8s8I")
SOURCE_HEADER = struct.Struct("<8s4I")


def round_up(value: int, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError("dimensions and alignment must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def float_to_bf16(value: float) -> int:
    """Convert finite float32 to BF16 using round-to-nearest, ties-to-even."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    exponent = bits & 0x7F800000
    fraction = bits & 0x007FFFFF
    if exponent == 0x7F800000:
        if fraction:
            return ((bits >> 16) | 0x0040) & 0xFFFF
        return bits >> 16
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def read_source_keys(path: Path) -> tuple[int, int, list[float]]:
    with path.open("rb") as source:
        raw_header = source.read(SOURCE_HEADER.size)
        if len(raw_header) != SOURCE_HEADER.size:
            raise ValueError("truncated RAGBANK1 header")
        magic, entries, dimension, visual_tokens, visual_dimension = (
            SOURCE_HEADER.unpack(raw_header)
        )
        if magic != SOURCE_MAGIC:
            raise ValueError("input is not a RAGBANK1 file")
        if not entries or not dimension:
            raise ValueError("source bank has invalid dimensions")
        if visual_tokens != 101 or visual_dimension != 1024:
            raise ValueError("source bank has unsupported visual dimensions")

        value_count = entries * dimension
        raw_keys = source.read(value_count * 4)
        if len(raw_keys) != value_count * 4:
            raise ValueError("truncated RAGBANK1 key matrix")
        keys = list(struct.unpack(f"<{value_count}f", raw_keys))
        if not all(math.isfinite(value) for value in keys):
            raise ValueError("source bank contains non-finite keys")
        return entries, dimension, keys


def pack(source: Path, target: Path, tt_metal_version: str = "unknown") -> dict:
    entries, dimension, keys = read_source_keys(source)
    padded_entries = round_up(entries, TILE)
    padded_dimension = round_up(dimension, TILE)

    target.parent.mkdir(parents=True, exist_ok=True)
    payload_hash = hashlib.sha256()
    with target.open("wb") as output:
        output.write(HEADER.pack(
            PACKED_MAGIC,
            PACKED_VERSION,
            entries,
            dimension,
            padded_entries,
            padded_dimension,
            TILE,
            BF16_DTYPE,
            TRANSPOSED_ROW_MAJOR_LAYOUT,
        ))
        row = bytearray(padded_entries * 2)
        for column in range(padded_dimension):
            row[:] = bytes(len(row))
            if column < dimension:
                for entry in range(entries):
                    value = keys[entry * dimension + column]
                    struct.pack_into("<H", row, entry * 2, float_to_bf16(value))
            output.write(row)
            payload_hash.update(row)

    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "schema": "rag-tt-bank-manifest-v1",
        "format": "RAGTTB01",
        "packing_version": PACKED_VERSION,
        "source_bank": source.name,
        "source_bank_sha256": source_hash,
        "entries": entries,
        "valid_entries": entries,
        "embedding_dimension": dimension,
        "padded_entries": padded_entries,
        "padded_embedding_dimension": padded_dimension,
        "tile_alignment": TILE,
        "storage_dtype": "bfloat16_little_endian",
        "layout": "transposed_row_major",
        "payload_sha256": payload_hash.hexdigest(),
        "tt_metal_version": tt_metal_version,
    }
    manifest_path = target.with_suffix(target.suffix + ".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"packed {entries} entries from {source} into {target}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source RAGBANK1 file")
    parser.add_argument("output", type=Path, help="packed Tenstorrent bank")
    parser.add_argument(
        "--tt-metal-version",
        default="unknown",
        help="release, tag, or commit used to consume this artifact",
    )
    arguments = parser.parse_args()
    pack(arguments.input, arguments.output, arguments.tt_metal_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
