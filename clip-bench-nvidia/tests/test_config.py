import pytest
from harness.config import load_config
import os

@pytest.fixture
def config():
    return load_config()

def test_model_name_expected(config):
    assert config['model']['name'] == 'openai/clip-vit-large-patch14-336'

def test_data_camera_is_nuscenes_keyframe_camera(config):
    cameras = {
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
    }
    assert config['data']['camera'] in cameras, \
        f"Unknown camera: {config['data']['camera']}, expected one of {sorted(cameras)}"

def test_data_nuscenes_root_exists(config):
    root = config['data']['nuscenes_root']
    if not os.path.exists(root):
        pytest.skip(f"nuScenes not downloaded at {root}")
    assert os.path.isdir(os.path.join(root, 'samples')), \
        f"{root} has no samples/ directory - is this the v1.0-mini root?"


def test_data_image_size_matches_model(config):
    assert config['data']['image_size'] == 336, f"Image size does not match model requirement: 336"

def main():
    return pytest.main([__file__, "-v"])
 
 
if __name__ == "__main__":
    raise SystemExit(main())
