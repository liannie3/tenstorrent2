from pathlib import Path

import pytest

from clipbench.config import available_profiles, load_config

CAMERAS = {
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
}

## Sections that must not differ between profiles, or the cross-platform
## comparison in README section 4 is not comparing like with like
SHARED_SECTIONS = ('model', 'data', 'parity')


@pytest.fixture
def config():
    return load_config('cpu')


def test_model_name_expected(config):
    assert config['model']['name'] == 'openai/clip-vit-large-patch14-336'


def test_data_image_size_matches_model(config):
    assert config['data']['image_size'] == 336, 'Image size does not match model requirement: 336'


## Unpinned, `CLIPImageProcessor` picks torchvision where it is installed and
## PIL where it is not, which would preprocess differently per platform
def test_image_processor_backend_is_pinned(config):
    assert config['model'].get('image_processor') in ('pil', 'torchvision')


def test_data_path_is_a_nuscenes_keyframe_camera(config):
    camera = Path(config['data']['path']).name
    assert camera in CAMERAS, f'Unknown camera: {camera}, expected one of {sorted(CAMERAS)}'


def test_data_frames_exist(config):
    root = Path(config['data']['path'])
    if not root.is_dir():
        pytest.skip(f'nuScenes frames not downloaded at {root}')
    assert any(root.glob('*.jpg')), f'{root} contains no .jpg frames'


## The whole point of the base.yaml / <profile>.yaml split: a profile may
## override device, precision and output paths, and nothing else
@pytest.mark.parametrize('profile', [p for p in available_profiles() if p != 'cpu'])
def test_profiles_share_model_and_data(profile, config):
    try:
        other = load_config(profile)
    except RuntimeError as exc:          # e.g. the jetson profile on a CPU-only box
        pytest.skip(str(exc))

    for section in SHARED_SECTIONS:
        assert other[section] == config[section], (
            f'{profile}.yaml overrides {section}, which must stay identical across targets'
        )


def main():
    return pytest.main([__file__, '-v'])


if __name__ == '__main__':
    raise SystemExit(main())
