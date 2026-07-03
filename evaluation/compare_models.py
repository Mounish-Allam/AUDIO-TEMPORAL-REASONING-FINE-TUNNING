import json
import time
import logging
import pandas as pd
from evaluation.metrics import full_evaluation

logger = logging.getLogger(__name__)

def compare_models(results_files: dict, references_file: str, task: str = "temporal") -> pd.DataFrame:
    """
    Compare multiple model outputs against ground truth.

    Args:
        results_files: dict of {model_name: path_to_predictions_json}
        references_file: path to ground truth JSON
        task: 'mcq' or 'open'

    Returns:
        DataFrame with metrics per model
    """
    with open(references_file, 'r') as f:
        references = [s["answer"] for s in json.load(f)]

    rows = []
    for model_name, results_path in results_files.items():
        with open(results_path, 'r') as f:
            predictions = [s["model_prediction"] for s in json.load(f)]

        metrics = full_evaluation(predictions, references, task=task)
        metrics["model"] = model_name
        rows.append(metrics)
        logger.info(f"{model_name}: {metrics}")

    df = pd.DataFrame(rows).set_index("model")
    print("\n=== Model Comparison ===")
    print(df.to_string())
    return df