"""TEST ONLY: fabricated detections, zero visual tensors and fixed retrieval."""
from pathlib import Path
import os
import torch
from contracts import VisualOutput, RagHit


def build(llm):
    if llm.alignment['alignment_id'] != 'SYNTHETIC_TEST_ONLY':
        raise ValueError('Synthetic adapters require the synthetic test manifest.')
    class Yolo:
        def hybrid_prompt(self, image_path, question):
            return question + '\nSYNTHETIC TEST detections (not actual observations): car, pedestrian.'
    class Vision:
        alignment_id = 'SYNTHETIC_TEST_ONLY'
        def encode(self, image_path, *, role):
            n = 576 if role == 'current' else 101
            return VisualOutput(torch.zeros((1,n,llm.hidden_size),device=llm.device,dtype=llm.dtype))
    class Rag:
        query_mode = 'tinyllama_text'
        embedding_id = llm.text_embedding_id
        def retrieve(self, hybrid_prompt, query_embedding):
            # Fixed score exercises acceptance/rejection, not a database search.
            score = float(os.environ.get('SYNTHETIC_RAG_SCORE','0.9'))
            return RagHit('synthetic-reference',score,'SYNTHETIC reference: a car near a pedestrian.',
                          str(Path(__file__).with_name('synthetic_inputs')/'reference.ppm'))
    return {'yolo':Yolo(),'vision':Vision(),'rag':Rag()}
