from harness.config import load_config
config = load_config()
print(config['model']['layer_index'], config['run']['batch_size'])