import os
import sys
import csv

# Add parent directory to path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluation.config import LIGHTING_FOLDERS, LLIE_METHODS, PARAMS
from evaluation.evaluator import run_single_eval

DATA_ROOT = "data"

def run_all(max_samples=1000):

    os.makedirs("results", exist_ok=True)

    with open("results/results.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "lighting", "method", "params",
            "mean_error", "std_error",
            "avg_time_ms", "fps"
        ])

        for lighting_name, folder in LIGHTING_FOLDERS.items():

            root = os.path.join(DATA_ROOT, folder)

            print(f"\n=== {lighting_name} ===")

            for method in LLIE_METHODS:

                param_list = PARAMS.get(method, PARAMS["default"])

                for params in param_list:

                    result = run_single_eval(root, method, params, max_samples)

                    if result is None:
                        continue

                    mean_err, std_err, avg_time, fps = result

                    writer.writerow([
                        lighting_name,
                        method.value,
                        str(params),
                        round(mean_err, 4),
                        round(std_err, 4),
                        round(avg_time, 4),
                        round(fps, 2)
                    ])

                    print(f"{method.value} {params} → {mean_err:.4f}, FPS={fps:.2f}")

if __name__ == "__main__":
    run_all()