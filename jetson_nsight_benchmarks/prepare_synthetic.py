"""Create explicitly synthetic fixtures; this does not establish trained alignment."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent/'synthetic_inputs'
root.mkdir(exist_ok=True)
for name in ('current','reference'):
    (root/(name+'.ppm')).write_text('P3\n1 1\n255\n0 0 0\n')
manifest={'alignment_id':'SYNTHETIC_TEST_ONLY','training_reference':'NONE: synthetic interface test, not trained',
          'model_path':str(Path(a.model).resolve()),'text_embedding_id':'SYNTHETIC_TEST_ONLY_TEXT',
          'fusion_id':'retrieved_current_chat_v1','current_token_count':576,'retrieved_token_count':101}
(root/'alignment.json').write_text(json.dumps(manifest,indent=2))
print('Synthetic fixtures created. Outputs cannot validate visual understanding or upstream performance.')
