"""Stage 3 — concept bottleneck construction & scoring.

Rubric lives in configs/concepts.yaml, not hardcoded here.
"""
import yaml

def load_concepts(path: str = "configs/concepts.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["concepts"]

def score_review_llm(text: str, concepts: dict) -> dict:
    """Single LLM call, structured JSON output, one score per concept.
    Batch via async/concurrent requests; cache by review_id (or school_id,
    if scoring concatenated documents) to avoid re-billing.
    """
    raise NotImplementedError

def aggregate_to_school(review_scores, cfg: dict):
    """If aggregation.strategy == 'concatenate' (see stage0.yaml), scoring
    already happens once per school — this becomes a pass-through plus
    residualization on log(total_concatenated_word_count). Otherwise,
    residualize per-review scores on log(word_count) before averaging.
    """
    raise NotImplementedError
