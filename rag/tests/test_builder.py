"""Small end-to-end format check, requiring only the Python standard library."""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    # Fixture setup
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        visual = root / "visual.bin"
        visual.write_bytes(bytes(101 * 1024 * 4))
        entries = root / "entries.jsonl"
        key = [0.0] * 2048
        key[0] = 2.0
        entries.write_text(json.dumps({
            "description": "A pedestrian is crossing",
            "text_key": key,
            "visual_f32": visual.name,
        }) + "\n", encoding="utf-8")

        # Bank generation
        bank = root / "memory.bank"
        subprocess.run([sys.executable, str(repo / "tools" / "build_bank.py"),
                        str(entries), str(bank)], check=True)
        # Format validation
        with bank.open("rb") as source:
            assert source.read(8) == b"RAGBANK1"
            assert struct.unpack("<4I", source.read(16)) == (1, 2048, 101, 1024)
            assert struct.unpack("<f", source.read(4))[0] == 1.0
        manifest = json.loads((root / "memory.bank.json").read_text(encoding="utf-8"))
        assert manifest["entries"] == 1
        assert manifest["visual_shape"] == [101, 1024]
        # CLI integration
        if len(sys.argv) > 1:
            query = root / "query.f32"
            query.write_bytes(struct.pack("<2048f", *key))
            output = subprocess.check_output([sys.argv[1], str(bank), str(query)], text=True)
            assert "accepted=1 id=0" in output
            assert "description=A pedestrian is crossing" in output
            output = subprocess.check_output([sys.argv[1], str(bank), str(query), "0.7"], text=True)
            assert "visual_shape=101x1024" in output


if __name__ == "__main__":
    main()
