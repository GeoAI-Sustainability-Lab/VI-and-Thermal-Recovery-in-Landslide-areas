"""Population-level absolute LST & NDVI series (新圖 5): event patches vs
matched intact-forest controls per calendar epoch, one column per cohort
(typhoon 2024, rainfall 2025, earthquake 2024, Hansen loss 2022).
Both series carry the same inter-annual climate signal; the gap that opens
at the event is the disturbance effect, and recovery is the gap closing.
Output: figures/F11_population.{pdf,png}
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
R = json.load(open(f"{O}/abs_series.json"))

COLS = [("typhoon_2024", "Typhoon landslides 2024", WONG["vermillion"]),
        ("rainfall_2025", "Rainfall landslides 2025", WONG["orange"]),
        ("earthquake_2024", "Earthquake landslides 2024", WONG["blue"]),
        ("hansen_2022", "Forest loss 2022 (Hansen)", WONG["purple"])]

fig, axes = plt.subplots(2, 4, figsize=(183 * MM, 95 * MM), constrained_layout=True,
                         sharex=True)
for j, (key, title, col) in enumerate(COLS):
    d = R[key]
    tev = d["t_event_med"]
    for i, band in enumerate(["lst", "ndvi"]):
        ax = axes[i][j]
        xs, pm, ps, cm_, cs = [], [], [], [], []
        for tag, e in d["epochs"].items():
            if band in e:
                xs.append(e["mid"])
                pm.append(e[band]["patch"]); ps.append(e[band]["patch_se"])
                cm_.append(e[band]["ctrl"]); cs.append(e[band]["ctrl_se"])
        xs = np.array(xs); order = np.argsort(xs)
        xs = xs[order]
        pm = np.array(pm)[order]; ps = np.array(ps)[order]
        cm_ = np.array(cm_)[order]; cs = np.array(cs)[order]
        ax.axvline(tev, color="grey", lw=0.9, ls="--", zorder=1)
        ax.fill_between(xs, cm_ - 1.96 * cs, cm_ + 1.96 * cs, color="#4d4d4d",
                        alpha=0.18, lw=0)
        ax.plot(xs, cm_, "-o", ms=2.4, lw=1.0, color="#4d4d4d", zorder=3)
        ax.fill_between(xs, pm - 1.96 * ps, pm + 1.96 * ps, color=col, alpha=0.18,
                        lw=0)
        ax.plot(xs, pm, "-o", ms=2.4, lw=1.1, color=col, zorder=4)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_title(f"{chr(97+j)}  {title}\n(n = {d['n_patches']})", loc="left",
                         fontweight="bold", fontsize=6.6)
        if j == 0:
            ax.set_ylabel("Summer LST (°C)" if band == "lst" else "Summer NDVI")
        if i == 1:
            ax.set_xlabel("Year")
            ax.set_xticks([2021, 2023, 2025])
axes[0][0].legend(handles=[
    Line2D([], [], marker="o", ms=3, lw=1.1, color=WONG["vermillion"],
           label="Disturbed patches (colour)"),
    Line2D([], [], marker="o", ms=3, lw=1.0, color="#4d4d4d", label="Matched controls"),
    Line2D([], [], ls="--", lw=0.9, color="grey", label="Event")],
    frameon=False, fontsize=5.4, loc="upper left")
fig.savefig(f"{F}/F11_population.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F11_population.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F11_population")
