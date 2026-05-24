"""
Legacy entry point.

Full experiments use evaluation/run_all.py (via evaluation/evaluator.py).
After the eye-image path fix, re-run:
  1. python data/extract_normalized_images.py
  2. python data/generate_low_light_dataset.py
  3. python evaluation/run_all.py
"""

if __name__ == "__main__":
    from evaluation.run_all import run_all

    run_all()
