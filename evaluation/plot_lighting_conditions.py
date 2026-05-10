import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/results.csv"
)

methods = df["method"].unique()

plt.figure(figsize=(12,6))

for method in methods:

    subset = df[
        df["method"] == method
    ]

    plt.plot(
        subset["lighting"],
        subset["mean_error"],
        marker='o',
        label=method
    )

plt.ylabel("Mean Angular Error")

plt.xlabel("Lighting Condition")

plt.title(
    "LLIE Performance Across Lighting Conditions"
)

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/lighting_condition_comparison.png"
)

plt.show()