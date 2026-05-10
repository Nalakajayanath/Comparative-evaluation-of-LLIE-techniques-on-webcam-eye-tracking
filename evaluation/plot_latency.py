import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/average_results.csv"
)

plt.figure(figsize=(10,5))

plt.bar(
    df["method"],
    df["avg_time_ms"]
)

plt.xlabel("LLIE Method")

plt.ylabel("Average Latency (ms)")

plt.title("Latency Comparison")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/latency_comparison.png"
)

plt.show()