import pandas as pd

# Load all experiment results
df = pd.read_csv("results/results.csv")

# Group by LLIE method
summary = df.groupby("method").agg({
    "mean_error": "mean",
    "avg_time_ms": "mean",
    "fps": "mean"
}).reset_index()

# Round values
summary["mean_error"] = summary["mean_error"].round(4)
summary["avg_time_ms"] = summary["avg_time_ms"].round(4)
summary["fps"] = summary["fps"].round(2)

# Sort by accuracy
summary = summary.sort_values("mean_error")

# Save
summary.to_csv(
    "results/average_results.csv",
    index=False
)

print(summary)