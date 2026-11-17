# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            config.py
# CREATED:         10/15/2026
# REVISION:        1.0
# ======================================= #

"""Layered config: common/configs/base.yaml overlaid with one profile.

base.yaml holds everything that must be identical across platforms (model,
data, parity). A profile -- common/configs/cpu.yaml, nvidia/configs/jetson.yaml,
tt/configs/tt.yaml -- overrides only the platform choices. Keeping that split means you cannot accidentally benchmark
128 images on one target and 404 on another and still call it a comparison.
"""

from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # clipbench/common/config.py -> project root
PACKAGE_ROOT = PROJECT_ROOT / 'clipbench'
BASE_CONFIG = PACKAGE_ROOT / 'common' / 'configs' / 'base.yaml'

## Config values holding filesystem paths. Resolved against the project root so
## the harness behaves the same whatever directory you launch it from.
PATH_KEYS = (('data', 'path'), ('parity', 'golden_path'), ('output', 'results_dir'))

## Profile name -> yaml path, from every platform's configs/ folder
def _profile_paths():
    return {p.stem: p for p in PACKAGE_ROOT.glob('*/configs/*.yaml') if p.stem != 'base'}

# Determines which profiles are available (i.e. CPU, CUDA, etc.)
def available_profiles():
    return sorted(_profile_paths())

# Reads one YAML file
def _read(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}

# Loads the config of the profile
def load_config(profile):
    profiles = available_profiles()     # Lists available profiles
    if profile not in profiles:         # Ensures the platform is in the list of available profiles
        raise ValueError(f'Unknown profile {profile!r}, expected one of {profiles}') 

    ## Configs are two levels deep (section -> key), so overriding run.device in
    ## a profile keeps the run.seed that profile never mentioned.
    config = _read(BASE_CONFIG)
    for section, values in _read(_profile_paths()[profile]).items():
        config.setdefault(section, {}).update(values)

    config['profile'] = profile  

    for section, key in PATH_KEYS:
        config[section][key] = str(PROJECT_ROOT / config[section][key])

    ## Fail loudly rather than quietly benchmarking the CPU when a profile asked
    ## for CUDA and did not get it.
    if config['run']['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            f'profile {profile!r} requests device cuda but torch.cuda.is_available() '
            'is False - wrong venv, or a CPU-only torch wheel'
        )

    return config
