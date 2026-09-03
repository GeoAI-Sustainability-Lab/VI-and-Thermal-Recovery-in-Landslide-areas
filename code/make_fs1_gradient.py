"""Supplementary Fig. S1: why the apparent elevation gradient in canopy-height
sensitivity is an artifact of canopy-height support.
(a) terrain-adjusted LST vs canopy height by elevation band with each band's
p10-p90 canopy-height support drawn beneath, so it is visible that bands are
compared over different segments of one saturating curve;
(b) marginal sensitivity per band under four estimators, with the trend test
against elevation for each.
Inputs: outputs/results.json, outputs/gradient_check.json
Output: figures/FS1_gradient_estimators.{pdf,png}
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

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
R = json.load(open(f"{O}/results.json"))
G = json.load(open(f"{O}/gradient_check.json"))
bs = R["band_stats"]
COLS = [WONG["skyblue"], WONG["green"], WONG["yellow"], WONG["orange"],
        WONG["blue"], WONG["purple"]]
LAB = {"0-500": "0–500 m", "500-1000": "500–1,000 m",
       "1000-1500": "1,000–1,500 m", "1500-2000": "1,500–2,000 m",
       "2000-2500": "2,000–2,500 m", "2500-3600": "2,500–3,600 m"}

# stacked, full width: at full font size neither panel's legend fits beside
# the other's title in a side-by-side layout
fig, (axA, axB) = plt.subplots(2, 1, figsize=(183 * MM, 156 * MM),
                               constrained_layout=True)

ax = axA
ymin = 99
for b, c in zip(bs, COLS):
    bc = np.array(b["bin_centers"]); med = np.array(b["bin_median"], float)
    q25 = np.array(b["bin_q25"], float); q75 = np.array(b["bin_q75"], float)
    m = np.isfinite(med)
    ax.plot(bc[m], med[m], color=c, lw=1.4, label=LAB.get(b["band"], b["band"]))
    ax.fill_between(bc[m], q25[m], q75[m], color=c, alpha=0.13, lw=0)
    ymin = min(ymin, np.nanmin(q25[m]))
y0 = ymin - 0.8
for i, (b, c) in enumerate(zip(bs, COLS)):
    gb = G["bands"].get(b["band"])
    if not gb:
        continue
    yy = y0 - i * 0.55
    ax.plot([gb["chm_p10"], gb["chm_p90"]], [yy, yy], color=c, lw=2.6,
            solid_capstyle="butt")
    ax.plot([gb["chm_p50"]], [yy], marker="|", color="black", ms=4, mew=0.9)
ax.axvspan(20, 30, color="grey", alpha=0.10, lw=0)
ax.text(25, ax.get_ylim()[1], "common\nwindow", ha="center", va="top",
        fontsize=8.3, color="#555555")
ax.set_xlim(8, 50)
ax.set_xlabel("Canopy height (m)")
ax.set_ylabel("Terrain-adjusted LST (°C)")
ax.legend(frameon=False, fontsize=8.3, title="Elevation band", title_fontsize=8.3,
          loc="upper right", borderaxespad=0.1, handlelength=1.4)
ax.set_title("a  Each band occupies a different part of one saturating curve",
             loc="left", fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

ax = axB
keys = list(G["bands"])
x = np.arange(len(keys))
series = [("slope_full", "linear fit over each band's own support (conventional)",
           WONG["vermillion"], "o", None),
          ("slope_common", "linear fit, common 20–35 m window", WONG["orange"],
           "s", "slope_common_se"),
          ("slope_np", "non-parametric, 20–25 vs 25–30 m", WONG["blue"], "^",
           "slope_np_se"),
          ("slope_gbm_local", "gradient-boosting local slope at 25 m",
           WONG["green"], "D", None)]
for (k, lab, c, mk, sek), dx in zip(series, np.linspace(-0.22, 0.22, 4)):
    v = np.array([G["bands"][b].get(k) for b in keys], float)
    e = (np.array([G["bands"][b].get(sek) or np.nan for b in keys], float) * 1.96
         if sek else None)
    ax.errorbar(x + dx, v, yerr=e, fmt=mk, ms=3.2, color=c, lw=0.9, capsize=1.6,
                label=lab)
ax.axhline(0, color="grey", lw=0.6, ls=":")
gt = G["gradient_test"]
ax.set_xticks(x)
ax.set_xticklabels([LAB.get(k, k).replace(" m", "").replace(",", "")
                    for k in keys], fontsize=8.3, rotation=30, ha="right")
ax.set_ylim(-1.62, 0.34)
ax.set_ylabel("Sensitivity of LST to canopy height (°C per 10 m)")
ax.legend(frameon=False, fontsize=8.3, loc="upper right", handlelength=1.2,
          labelspacing=0.28, ncol=2, columnspacing=0.9)
ax.text(0.015, 0.10,
        "Trend of sensitivity against elevation\n"
        f"conventional per-band fit:  {gt['slope_full']['slope_per_1000m']:+.2f} per 1,000 m, "
        f"p = {gt['slope_full']['p']:.3f}\n"
        f"common-window fit:  {gt['slope_common']['slope_per_1000m']:+.2f}, "
        f"p = {gt['slope_common']['p']:.3f}\n"
        f"non-parametric:  {gt['slope_np']['slope_per_1000m']:+.2f}, "
        f"p = {gt['slope_np']['p']:.2f}   (n.s.)\n"
        f"GBM local slope:  {gt['slope_gbm_local']['slope_per_1000m']:+.2f}, "
        f"p = {gt['slope_gbm_local']['p']:.2f}   (n.s.)",
        transform=ax.transAxes, fontsize=8.3, color="#333333", va="bottom",
        linespacing=1.45)
ax.set_title("b  The elevation gradient depends on the estimator", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

fig.savefig(f"{F}/FS1_gradient_estimators.pdf", bbox_inches="tight")
fig.savefig(f"{F}/FS1_gradient_estimators.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved FS1_gradient_estimators")
