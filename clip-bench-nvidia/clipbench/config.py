# ======================================= #
# AUTHOR:          Travis Mora
# ORGANIZATION:    Texas A&M University
# FILE:            config.py
# CREATED:         10/15/2026
# REVISION:        1.0
# ======================================= #

"""Layered config: configs/base.yaml overlaid with one profile.

base.yaml holds everything that must be identical across platforms (model,
data, parity). A profile -- cpu.yaml, jetson.yaml -- overrides only the
platform choices. Keeping that split means you cannot accidentally benchmark
128 images on one target and 404 on another and still call it a comparison.
"""

from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / 'configs'

## Config values holding filesystem paths. Resolved against the project root so
## the harness behaves the same whatever directory you launch it from.
PATH_KEYS = (('data', 'path'), ('parity', 'golden_path'), ('output', 'results_dir'))

# Determines which profiles are available (i.e. CPU, CUDA, etc.)
def available_profiles(): 
    return sorted(p.stem for p in CONFIG_DIR.glob('*.yaml') if p.stem != 'base')

# Reads the YAML file for whichever profile is running
def _read(name):
    with open(CONFIG_DIR / name) as f:
        return yaml.safe_load(f) or {}

# Loads the config of the profile
def load_config(profile):
    profiles = available_profiles()     # Lists available profiles
    if profile not in profiles:         # Ensures the platform is in the list of available profiles
        raise ValueError(f'Unknown profile {profile!r}, expected one of {profiles}') 

    ## Configs are two levels deep (section -> key), so overriding run.device in
    ## a profile keeps the run.seed that profile never mentioned.
    config = _read('base.yaml')
    for section, values in _read(f'{profile}.yaml').items():
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
