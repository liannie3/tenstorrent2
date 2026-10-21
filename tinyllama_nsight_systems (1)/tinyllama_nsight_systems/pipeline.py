"""Connect real YOLO, CLIP/MLP, optional RAG, and TinyLlama objects.

The team supplies models/weights through the adapter factory. No dummy outputs
or model implementations are substituted here. Normal runs install no profiler.
"""
from contextlib import nullcontext
from pathlib import Path
from contracts import PipelineAnswer


class AVPipeline:
    def __init__(self, llm, yolo, vision, rag=None, *, rag_query_mode='tinyllama_text',
                 embedding_id=None, threshold=0.7):
        if rag_query_mode not in ('tinyllama_text', 'clip') or not -1 <= threshold <= 1:
            raise ValueError('Invalid RAG query mode or similarity threshold.')
        # Alignment identifies a trained visual-projector/LLM pairing.
        llm.require_alignment(vision.alignment_id)
        if rag is not None:
            expected = llm.text_embedding_id if rag_query_mode == 'tinyllama_text' else embedding_id
            if not expected or rag.embedding_id != expected or rag.query_mode != rag_query_mode:
                raise ValueError('RAG bank does not match the selected query encoder.')
        self.llm, self.yolo, self.vision, self.rag = llm, yolo, vision, rag
        self.mode, self.threshold = rag_query_mode, threshold

    def run(self, image_path, question, *, max_new_tokens=128, benchmark=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('Question must be nonempty.')
        if not Path(image_path).is_file():
            raise FileNotFoundError(image_path)
        # NVTX labels are added only when an observer is explicitly passed.
        def span(name):
            return benchmark.span(name) if hasattr(benchmark, 'span') else nullcontext()

        with span('yolo'):
            hybrid = self.yolo.hybrid_prompt(image_path, question)
        if not isinstance(hybrid, str) or not hybrid.strip():
            raise ValueError('YOLO must return a nonempty hybrid prompt.')
        with span('vision_current'):
            current = self.vision.encode(image_path, role='current')
            self.llm.validate_visual(current.projected_tokens, role='current')

        hit, reference_tokens = None, None
        if self.rag is not None:
            with span('rag_query_embedding'):
                query = self.llm.embed_query(hybrid) if self.mode == 'tinyllama_text' else current.retrieval_embedding
                if query is None:
                    raise ValueError('CLIP retrieval needs a retrieval embedding.')
            with span('rag_search'):
                candidate = self.rag.retrieve(hybrid, query)
                if candidate is not None:
                    candidate.validate()
                    if candidate.similarity >= self.threshold:
                        hit = candidate
            # Accept reference text and image together, or omit both.
            if hit is not None:
                if not Path(hit.image_path).is_file():
                    raise FileNotFoundError(hit.image_path)
                with span('vision_reference'):
                    reference = self.vision.encode(hit.image_path, role='retrieved')
                    reference_tokens = reference.projected_tokens
                    self.llm.validate_visual(reference_tokens, role='retrieved')

        optional = {'benchmark': benchmark} if benchmark is not None else {}
        with span('llm_answer'):
            answer = self.llm.answer(hybrid, current_tokens=current.projected_tokens,
                                     retrieved_tokens=reference_tokens,
                                     retrieved_description=hit.description if hit else None,
                                     max_new_tokens=max_new_tokens, **optional)
        return PipelineAnswer(answer, hybrid, hit.entry_id if hit else None)
