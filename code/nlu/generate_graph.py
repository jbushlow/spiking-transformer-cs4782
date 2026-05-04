import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

datasets = ["SST-2", "SST-5", "MR", "Subj"]

paper_46m = {
    "SST-2": 80.39,
    "SST-5": 37.69,
    "MR":    69.23,
    "Subj":  88.45,
}

my_results = {
    "SST-2": 81.3073,
    "SST-5": 38.5068,
    "MR":    48.45,
    "Subj":  88.125,
}

paper_vals = [paper_46m[d] for d in datasets]
my_vals    = [my_results[d] for d in datasets]
deltas     = [my_results[d] - paper_46m[d] for d in datasets]

x = np.arange(len(datasets))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#fafafa")
ax.set_facecolor("#fafafa")

bars1 = ax.bar(x - width / 2, paper_vals, width, label="Paper (46M)",
               color="#3266ad", zorder=3)
bars2 = ax.bar(x + width / 2, my_vals,    width, label="Reimplementation",
               color="#c0392b", zorder=3)

for bar, delta in zip(bars2, deltas):
    sign = "+" if delta >= 0 else ""
    color = "#2ecc71" if delta >= 0 else "#e74c3c"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        f"{sign}{delta:.2f}%",
        ha="center", va="bottom",
        fontsize=8.5, color=color, fontweight="bold"
    )

ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_title("SpikeGPT 46M — Paper vs. Reimplementation", fontsize=13, fontweight="bold", pad=14)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11)
ax.set_ylim(0, 105)
ax.yaxis.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.legend(frameon=False, fontsize=10)

fig.tight_layout()
plt.savefig("spikegpt_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to spikegpt_comparison.png")