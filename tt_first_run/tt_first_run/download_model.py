"""Download once, record resolved revision, then inference runs offline."""
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument('--repo', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
p.add_argument('--revision', default='main')
p.add_argument('--destination', default='models/tinyllama-chat')
a = p.parse_args()
sha = HfApi().model_info(a.repo, revision=a.revision).sha
snapshot_download(repo_id=a.repo, revision=sha, local_dir=a.destination,
                  allow_patterns=['*.json', '*.safetensors', 'tokenizer.model', '*.jinja', '*.txt'])
Path(a.destination, 'download_provenance.json').write_text(
    json.dumps({'repo_id': a.repo, 'revision': sha}, indent=2) + '\n')
print(f'Model saved in {a.destination}; resolved revision {sha}')
