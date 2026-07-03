from evaluation.metrics import (
    compute_exact_match,
    compute_rouge_l,
    compute_bert_score,
    hallucination_rate,
    temporal_ordering_accuracy,
    sound_event_recall,
    hallucinated_events,
    full_evaluation,
)

__all__ = [
    "compute_exact_match",
    "compute_rouge_l",
    "compute_bert_score",
    "hallucination_rate",
    "temporal_ordering_accuracy",
    "sound_event_recall",
    "hallucinated_events",
    "full_evaluation",
]

# BenchmarkRunner needs torch + transformers; keep it optional so the
# metrics work on machines without a GPU stack installed.
try:
    from evaluation.benchmark import BenchmarkRunner
    from evaluation.compare_models import compare_models
    __all__ += ["BenchmarkRunner", "compare_models"]
except ImportError:
    pass
