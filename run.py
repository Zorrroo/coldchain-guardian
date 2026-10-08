"""Run the entire ColdChain Guardian pipeline end-to-end.

    python run.py

Steps: generate data -> EDA figures -> train/tune/compare -> evaluate.
Each stage is idempotent and writes its outputs to data/, figures/ and
models/. On completion it prints the headline metrics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import evaluate, generate_data, train  # noqa: E402
from src.eda import run_eda  # noqa: E402


def main() -> None:
    start = time.time()

    print("\n[1/4] Generating synthetic shipment dataset ...")
    df = generate_data.main()

    print("\n[2/4] Running exploratory data analysis ...")
    run_eda(df)

    print("\n[3/4] Training, tuning and comparing models ...")
    train.train(df)

    print("\n[4/4] Evaluating best model and saving figures ...")
    metrics = evaluate.evaluate()

    elapsed = time.time() - start
    test = metrics.get("test_metrics", {})
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Best model:      {metrics.get('best_model')}")
    print(f"Test ROC-AUC:    {test.get('roc_auc', float('nan')):.3f}")
    print(f"Test accuracy:   {test.get('accuracy', float('nan')):.3f}")
    print(f"Test F1:         {test.get('f1', float('nan')):.3f}")
    print(f"Total time:      {elapsed:.1f}s")
    print("=" * 70)
    print("Next: `python src/predict.py` to score example shipments.")


if __name__ == "__main__":
    main()
