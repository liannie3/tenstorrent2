import pytest
from harness.config import load_config
import os

@pytest.fixture
def config():
    return load_config()

def test_model_name_expected(config):
    assert config['model']['name'] == 'openai/clip-vit-large-patch14-336'

def test_data_image_dir_exists(config):
    image_dir = config['data']['image_dir']
    assert os.path.exists(image_dir), f"Image directory does not exist: {image_dir}"
    
def test_data_image_size_matches_model(config):
    assert config['data']['image_size'] == 336, f"Image size does not match model requirement: 336"

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))