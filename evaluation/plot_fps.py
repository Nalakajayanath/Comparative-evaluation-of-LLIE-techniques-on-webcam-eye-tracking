import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/average_results.csv"
)

plt.figure(figsize=(10,5))

plt.bar(
    df["method"],
    df["fps"]
)

plt.xlabel("LLIE Method")

plt.ylabel("FPS")

plt.title("FPS Comparison")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/fps_comparison.png"
)

plt.show()