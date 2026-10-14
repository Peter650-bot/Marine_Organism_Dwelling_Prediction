import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

handles = [
    Patch(facecolor="#E86A33", edgecolor="none", label="Label 1: inside 50% contour"),
    Patch(facecolor="#DCE6F0", edgecolor="none", label="Label 0: outside"),
    Line2D([0], [0], color="black", lw=1.5, label="50% volume contour of weighted KDE"),
    Line2D([0], [0], marker="o", ls="", mfc="none", mec="#444444", label="Sightings, sized by weight"),
    Patch(facecolor="#BDB9B3", edgecolor="none", label="Land"),
]
fig = plt.figure(figsize=(3.2, 1.6))
fig.legend(handles=handles, loc="center", frameon=False, fontsize=9)
fig.savefig("label_map_legend.pdf", bbox_inches="tight")
