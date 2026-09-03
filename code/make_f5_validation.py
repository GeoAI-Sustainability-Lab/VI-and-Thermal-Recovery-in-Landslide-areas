"""F5 validation (v19, 2 panels): (a) control matching quality,
(b) compositing method check. The old pre-event/DiD boxplot panel duplicated
Table 3 and was removed. Output: figures/F5_validation.{pdf,png}
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)


R2 = json.load(open(f"{O}/results2.json"))
DDC = R2["did_cohorts"]
pdel = pd.read_parquet(f"{D}/patches_deltas2.parquet")
fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 76 * MM), constrained_layout=True)
ax = axes[0]
dq = pdel[pdel.n_ctrl >= 30]
ax.hist(dq.ctrl_elev - dq.elev, bins=np.linspace(-250, 250, 51), color=WONG["blue"],
        alpha=0.7)
ax.set_xlabel("Control − patch elevation (m)", color=WONG["blue"])
ax.tick_params(axis="x", colors=WONG["blue"])
ax.set_ylabel("Patches")
ax2 = ax.twiny()
ax2.hist(dq.ctrl_slope - dq.slope, bins=np.linspace(-15, 15, 51), histtype="step",
         color=WONG["vermillion"], lw=1.8)
ax2.set_xlabel("Control − patch slope (°)", color=WONG["vermillion"])
ax2.tick_params(axis="x", colors=WONG["vermillion"])
ax.set_title("a  Control matching quality", loc="left", fontweight="bold", fontsize=9.5)
style_ax(ax)

ax = axes[1]
mv = json.load(open(f"{O}/median_validation.json"))
names = list(mv)
x = np.arange(len(names))
ax.bar(x - 0.18, [mv[n]["mean_minus_median_bias"] for n in names], width=0.36,
       color=WONG["blue"], label="mean − median bias")
ax.bar(x + 0.18, [mv[n]["mean_minus_median_rmse"] for n in names], width=0.36,
       color=WONG["orange"], label="RMSE")
ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names],
                                     fontsize=8.3)
ax.axhline(0, color="grey", lw=0.5)
ax.set_ylabel("LST difference (°C)")
ax.set_ylim(None, 3.6)
ax.legend(frameon=False, fontsize=8.3, loc="upper left")
ax.set_title("b  Compositing method check", loc="left", fontweight="bold", fontsize=9.5)
style_ax(ax)
fig.savefig(f"{F}/F5_validation.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F5_validation.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F5_validation")
