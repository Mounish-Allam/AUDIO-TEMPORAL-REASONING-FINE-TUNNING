"""
Evaluation metrics for audio temporal reasoning.

Design notes (honest limitations, worth knowing before quoting numbers):

- hallucination_rate:  a prediction counts as hallucinated if it names a
  sound event (from a fixed vocabulary of common AudioCaps sounds) that
  does not appear in the reference caption. This catches "invented sounds"
  but not subtler errors, and the vocabulary is not exhaustive.

- temporal_ordering_accuracy:  compares the ORDER in which shared content
  words appear in the prediction vs. the reference. It only scores words
  that appear in both texts, so a prediction that misses events entirely
  is penalised by sound_event_recall, not by this metric.

These are simple word-level metrics — no LLM judge, no embeddings.
They are easy to explain and easy to audit, which matters more here
than squeezing out the last bit of measurement accuracy.
"""
import re
import logging

logger = logging.getLogger(__name__)

# rouge-score and bert-score are optional: metrics degrade gracefully
# so the demo and tests run on machines without them installed.
try:
    from rouge_score import rouge_scorer
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False


def _norm(word: str) -> str:
    """Light stemming so 'barks'/'barking' match 'bark'.
    Applied to text AND vocabulary, so both sides always agree."""
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


# Common sound events in AudioCaps captions. Used to detect when a
# prediction names a sound that the reference never mentions.
_RAW_SOUND_EVENTS = {
    # people
    "man", "woman", "child", "baby", "person", "people", "crowd",
    "speech", "speak", "talk", "voice", "laugh", "cry", "scream",
    "shout", "whistle", "sing", "cough", "sneeze", "snore", "breath",
    "footstep", "clap", "cheer", "applause",
    # animals
    "dog", "bark", "cat", "meow", "bird", "chirp", "tweet", "crow",
    "rooster", "duck", "quack", "goat", "sheep", "bleat", "cow", "moo",
    "horse", "neigh", "pig", "oink", "insect", "buzz", "frog", "croak",
    "owl", "hoot",
    # vehicles / machines
    "car", "truck", "bus", "engine", "motor", "motorcycle", "train",
    "horn", "honk", "siren", "alarm", "helicopter", "airplane", "plane",
    "boat", "traffic", "brake", "tire", "accelerate", "rev", "idle",
    "machine", "drill", "saw", "hammer", "vacuum", "fan", "clock",
    "tick", "bell", "ring", "phone", "typing", "keyboard", "camera",
    # nature / environment
    "rain", "thunder", "wind", "water", "wave", "ocean", "stream",
    "splash", "drip", "fire", "crackle", "explosion", "gunshot", "gun",
    "fireworks", "leaves", "rustle",
    # household / misc
    "music", "song", "instrument", "guitar", "piano", "drum", "knock",
    "door", "slam", "creak", "squeak", "glass", "shatter", "pour",
    "sizzle", "frying", "microwave", "toilet", "flush", "spray",
    "whoosh", "thump", "bang", "beep", "click", "hiss", "rumble",
}

_SOUND_EVENTS = {_norm(w) for w in _RAW_SOUND_EVENTS}

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
    "to", "of", "and", "or", "with", "this", "that", "it", "by",
    "from", "be", "as", "for", "then", "first", "next", "after",
    "before", "finally", "followed", "while", "several", "some",
    "loud", "loudly", "quiet", "background", "distance", "nearby",
    "sound", "sounds", "audio", "clip", "recording", "heard", "hear",
}


def _words(text: str) -> list:
    """Lowercase, lightly stemmed words from a text."""
    return [_norm(w) for w in re.findall(r"\b[a-z]+\b", text.lower())]


def compute_exact_match(predictions: list, references: list) -> float:
    """Exact match accuracy for MCQ tasks."""
    correct = sum(
        p.strip().lower() == r.strip().lower()
        for p, r in zip(predictions, references)
    )
    return round(correct / len(predictions) * 100, 2)


def compute_rouge_l(predictions: list, references: list) -> float:
    """ROUGE-L F-measure for open-ended generation quality."""
    if not _ROUGE_AVAILABLE:
        logger.warning("rouge-score not installed — skipping ROUGE-L")
        return -1.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_rouge_1(predictions: list, references: list) -> float:
    """ROUGE-1 F-measure for unigram overlap."""
    if not _ROUGE_AVAILABLE:
        logger.warning("rouge-score not installed — skipping ROUGE-1")
        return -1.0
    scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    scores = [
        scorer.score(ref, pred)["rouge1"].fmeasure
        for pred, ref in zip(predictions, references)
    ]
    return round(sum(scores) / len(scores) * 100, 2)


def compute_bert_score(predictions: list, references: list) -> float:
    """BERTScore F1 for semantic similarity."""
    if not _BERT_AVAILABLE:
        logger.warning("bert-score not installed — skipping BERTScore")
        return -1.0
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    return round(F1.mean().item() * 100, 2)


def hallucination_rate(predictions: list, references: list) -> float:
    """
    % of predictions that mention a sound event NOT present in the reference.

    Example:
        reference:  "A dog barks and a car passes by"
        prediction: "A dog barks while music plays"   -> hallucinated ("music")
    """
    hallucinated = 0
    for pred, ref in zip(predictions, references):
        pred_events = set(_words(pred)) & _SOUND_EVENTS
        ref_words   = set(_words(ref))
        if pred_events - ref_words:
            hallucinated += 1
    return round(hallucinated / len(predictions) * 100, 2)


def hallucinated_events(prediction: str, reference: str) -> list:
    """The specific sound events a prediction invented — for error analysis."""
    pred_events = set(_words(prediction)) & _SOUND_EVENTS
    ref_words   = set(_words(reference))
    return sorted(pred_events - ref_words)


def temporal_ordering_accuracy(predictions: list, references: list) -> float:
    """
    Do shared content words appear in the SAME ORDER in prediction and
    reference?

    Per sample: take content words that appear in both texts (in order of
    first appearance), form all pairs (i, j) from the reference order, and
    count the fraction of pairs the prediction keeps in that order.
    Samples with fewer than 2 shared words are skipped.
    """
    scores = []
    for pred, ref in zip(predictions, references):
        # content words in order of first appearance
        def first_appearance_order(text):
            seen, ordered = set(), []
            for w in _words(text):
                if w not in _STOPWORDS and w not in seen:
                    seen.add(w)
                    ordered.append(w)
            return ordered

        ref_order  = first_appearance_order(ref)
        pred_order = first_appearance_order(pred)
        shared     = [w for w in ref_order if w in pred_order]

        if len(shared) < 2:
            continue

        pred_pos = {w: i for i, w in enumerate(pred_order)}
        total, correct = 0, 0
        for i in range(len(shared)):
            for j in range(i + 1, len(shared)):
                total += 1
                if pred_pos[shared[i]] < pred_pos[shared[j]]:
                    correct += 1
        scores.append(correct / total)

    if not scores:
        return 0.0
    return round(sum(scores) / len(scores) * 100, 2)


def sound_event_recall(predictions: list, references: list) -> float:
    """
    What fraction of the reference's sound events does the prediction
    mention? (Complements hallucination_rate, which checks the reverse.)
    """
    recalls = []
    for pred, ref in zip(predictions, references):
        ref_events = set(_words(ref)) & _SOUND_EVENTS
        if not ref_events:
            continue
        pred_words = set(_words(pred))
        recalls.append(len(ref_events & pred_words) / len(ref_events))

    if not recalls:
        return 0.0
    return round(sum(recalls) / len(recalls) * 100, 2)


def full_evaluation(predictions: list, references: list, task: str = "temporal") -> dict:
    """
    Run all metrics and return results dict.

    task="temporal"  → ROUGE-1, ROUGE-L, BERTScore, hallucination rate,
                        temporal ordering accuracy, sound event recall
    task="mcq"       → exact match only
    task="open"      → ROUGE-L, BERTScore, hallucination rate

    Metrics whose library is not installed are reported as -1.0.
    """
    if not predictions or not references:
        return {}

    results = {}

    if task == "mcq":
        results["exact_match"] = compute_exact_match(predictions, references)

    elif task == "temporal":
        results["rouge_1"]                    = compute_rouge_1(predictions, references)
        results["rouge_l"]                    = compute_rouge_l(predictions, references)
        results["bert_score"]                 = compute_bert_score(predictions, references)
        results["hallucination_rate"]         = hallucination_rate(predictions, references)
        results["temporal_ordering_accuracy"] = temporal_ordering_accuracy(predictions, references)
        results["sound_event_recall"]         = sound_event_recall(predictions, references)

    else:  # "open"
        results["rouge_l"]            = compute_rouge_l(predictions, references)
        results["bert_score"]         = compute_bert_score(predictions, references)
        results["hallucination_rate"] = hallucination_rate(predictions, references)

    logger.info(f"Evaluation results: {results}")
    return results
