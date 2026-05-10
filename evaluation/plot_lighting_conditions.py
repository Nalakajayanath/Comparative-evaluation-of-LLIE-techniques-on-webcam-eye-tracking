import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "results/results.csv"
)

# Explanation text
label_info = (
    "normal = Normal Lighting\n"
    "low_i = Intensity Reduction\n"
    "low_g = Gamma Darkening\n"
    "low_left = Left Directional Lighting\n"
    "low_right = Right Directional Lighting\n"
    "low_i_g_left = Combined (low_i + low_g + low_left)\n"
    "low_i_g_right = Combined (low_i + low_g + low_right)"
)

methods = df["method"].unique()

plt.figure(figsize=(14,7))

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

plt.xticks(rotation=20)

plt.legend()

# Add explanation box
plt.gcf().text(
    0.75,
    0.20,
    label_info,
    fontsize=8,
    bbox=dict(facecolor='white')
)

plt.tight_layout()

plt.savefig(
    "results/lighting_condition_comparison.png",
    dpi=300
)

plt.show()