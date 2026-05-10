import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/average_results.csv"
)

plt.figure(figsize=(10,5))

plt.bar(
    df["method"],
    df["mean_error"]
)

plt.xlabel("LLIE Method")

plt.ylabel("Mean Angular Error")

plt.title(
    "Mean Angular Error Comparison"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/mean_error_comparison.png"
)

plt.show()