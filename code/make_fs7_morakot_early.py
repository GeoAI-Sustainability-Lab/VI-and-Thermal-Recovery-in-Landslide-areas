"""Supplementary figure S7: Morakot ages 1-3 through the L5/7 cross-sensor
bridge, joined to the production Landsat-8 record (ages 4-17). Bridged points
carry two error components: the sampling CI (whisker) and the calibration band
(translucent bar spanning the 2013/2014 overlap-year transfers; L5 chained).
Display only - these points enter no fit. Output: figures/FS7_morakot_early.*
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
E = json.load(open(f"{O}/morakot_early.json"))
MK = json.load(open(f"{O}/morakot.json"))
SEN_MK = {"L5": "s", "L7": "o"}

fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 84 * MM), constrained_layout=True)

for ax, key, ylab, prodkey, col in [
        (axs[0], "lst", "ΔLST patch − control (°C, Landsat-8 scale)", "dlst",
         WONG["vermillion"]),
        (axs[1], "ndvi", "ΔNDVI patch − control (Landsat-8 scale)", "dndvi",
         WONG["green"])]:
    # production L8 record
    ages = [r["age"] for r in MK["cohort"] if r.get(prodkey) is not None]
    vals = [r[prodkey] for r in MK["cohort"] if r.get(prodkey) is not None]
    ses = [r.get(prodkey + "_se", 0) or 0 for r in MK["cohort"]
           if r.get(prodkey) is not None]
    ax.errorbar(ages, vals, yerr=[1.96 * s for s in ses], fmt="D-", ms=3.2,
                lw=1.0, capsize=1.6, color=WONG["purple"],
                label="Landsat 8 record (ages 4–17)")
    # bridged points
    for p in E["points"]:
        lo, hi = p[f"{key}_eq_lo"], p[f"{key}_eq_hi"]
        mid, se = p[f"{key}_eq_mid"], p.get(f"{key}_se_eq") or p.get(f"{key}_se")
        if lo is None or mid is None:
            continue
        ax.add_patch(plt.Rectangle((p["age"] - 0.22, lo), 0.44, hi - lo,
                     fc=col, alpha=0.28, ec="none", zorder=2))
        ax.errorbar([p["age"]], [mid], yerr=[1.96 * (se or 0)],
                    fmt=SEN_MK[p["sensor"]], ms=4.2, mfc="white", mec=col,
                    color=col, lw=1.0, capsize=1.8, zorder=4)
        ax.annotate(f"{p['sensor']}\nn={p['n']:,}", (p["age"], hi),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=8.3, color="#555555")
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.axvline(3.5, color="#999999", lw=0.7, ls=":")
    ax.text(3.35, ax.get_ylim()[0], " bridge | production ", fontsize=8.3,
            color="#777777", rotation=90, va="bottom", ha="right")
    ax.set_xlim(0, 18)
    ax.set_xlabel("Years since Typhoon Morakot (Aug 2009)")
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)

axs[0].set_title("a  Thermal: early peak, then the observed decay",
                 loc="left", fontweight="bold", fontsize=9.5)
axs[1].set_title("b  Greenness: deepest in year 2", loc="left",
                 fontweight="bold", fontsize=9.5)
h = [Line2D([], [], marker="D", ls="-", color=WONG["purple"], ms=3.2,
            label="Landsat 8 record"),
     Line2D([], [], marker="o", ls="", mfc="white", mec=WONG["vermillion"],
            color=WONG["vermillion"], ms=4.2, label="L7 bridged (±95% CI)"),
     Line2D([], [], marker="s", ls="", mfc="white", mec=WONG["vermillion"],
            color=WONG["vermillion"], ms=4.2, label="L5 bridged, chained"),
     plt.Rectangle((0, 0), 1, 1, fc=WONG["vermillion"], alpha=0.28,
                   label="calibration band (2013/2014 transfers)")]
axs[0].legend(handles=h, frameon=False, fontsize=8.3, loc="upper right",
              handlelength=1.4)

fig.savefig(f"{F}/FS7_morakot_early.pdf", bbox_inches="tight")
fig.savefig(f"{F}/FS7_morakot_early.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved FS7_morakot_early")
