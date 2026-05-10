import pandas as pd

df = pd.read_csv("results/results.csv")

# Get best method for each lighting condition
best = df.loc[
    df.groupby("lighting")["mean_error"].idxmin()
]

best = best[[
    "lighting",
    "method",
    "mean_error"
]]

best["mean_error"] = best[
    "mean_error"
].round(4)

# Save
best.to_csv(
    "results/best_methods.csv",
    index=False
)

print(best)