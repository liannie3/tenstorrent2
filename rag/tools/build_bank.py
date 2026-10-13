"""Pack precomputed RAG features into the RAGBANK1 format.

Input JSONL rows: {"description": str, "text_key": [2048 floats],
"visual_f32": "path/to/101x1024-float32-little-endian.bin"}.
Paths are relative to the JSONL file.
"""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

TEXT_DIM = 2048
VISUAL_BYTES = 101 * 1024 * 4
MAGIC = b"RAGBANK1"


# Input parsing
def rows(path):
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_no}: invalid JSON") from exc


# Key preprocessing
def normalized_key(row):
    values = row["text_key"]
    if len(values) != TEXT_DIM:
        raise ValueError("text_key must contain 2048 values")
    values = [float(x) for x in values]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("text_key contains non-finite values")
    norm = math.sqrt(sum(x * x for x in values))
    if not norm:
        raise ValueError("text_key has zero norm")
    return [x / norm for x in values]


# Bank serialization
def build(source, target, metadata=None):
    count = sum(1 for _ in rows(source))
    if not 0 < count <= 1_000_000:
        raise ValueError("bank must contain 1 to 1,000,000 entries")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<4I", count, TEXT_DIM, 101, 1024))
        for row in rows(source):
            out.write(struct.pack("<2048f", *normalized_key(row)))
        offsets_pos = out.tell()
        out.write(bytes(count * 8))
        offsets = []
        for row in rows(source):
            offsets.append(out.tell())
            description = row["description"].encode("utf-8")
            if len(description) > 1024 * 1024:
                raise ValueError("description exceeds 1 MiB")
            visual_path = source.parent / row["visual_f32"]
            if visual_path.stat().st_size != VISUAL_BYTES:
                raise ValueError(f"{visual_path}: expected {VISUAL_BYTES} bytes")
            out.write(struct.pack("<I", len(description)))
            out.write(description)
            with visual_path.open("rb") as visual:
                while chunk := visual.read(1024 * 1024):
                    out.write(chunk)
        out.seek(offsets_pos)
        out.write(struct.pack(f"<{count}Q", *offsets))

    manifest = {
        "format": "RAGBANK1",
        "entries": count,
        "text_dim": TEXT_DIM,
        "visual_shape": [101, 1024],
        "storage_dtype": "float32_little_endian",
        "input_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metadata": metadata or {},
    }
    target.with_suffix(target.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {count} entries to {target}")


# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path, help="JSON with checkpoint and preprocessing identifiers")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8")) if args.metadata else None
    build(args.input, args.output, metadata)
