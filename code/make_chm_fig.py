"""Figure: ETH 10 m vs Meta/WRI CHMv2 1 m canopy height comparison (圖 4).
(a) joint distribution at 30 m analysis cells (hexbin) with 1:1 line;
(b) elevation-band LST-height slopes estimated with each model.
Inputs: data/chm2_sample.parquet, outputs/chm2_compare.json
Output: figures/F10_chm_compare.{pdf,png}
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

s = pd.read_parquet(f"{D}/chm2_sample.parquet")
v = s[np.isfinite(s.chm2) & np.isfinite(s.chm)]
C = json.load(open(f"{O}/chm2_compare.json"))

fig, axes = plt.subplots(1, 2, figsize=(140 * MM, 62 * MM), constrained_layout=True)

ax = axes[0]
hb = ax.hexbin(v.chm, v.chm2, gridsize=48, bins="log", cmap="viridis",
               extent=(0, 50, 0, 50), lw=0.1)
ax.plot([0, 50], [0, 50], ls="--", color="black", lw=0.8)
cb = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Cells (log scale)")
ax.set_xlabel("ETH canopy height (m)")
ax.set_ylabel("CHMv2 canopy height, 30 m mean (m)")
ax.text(0.03, 0.97, f"n = {C['n']:,}\nr = {C['r']:.2f}\n"
        f"bias = {C['bias_mean']:+.1f} m\nRMSD = {C['rmsd']:.1f} m",
        transform=ax.transAxes, va="top", fontsize=6)
ax.set_title("a  ETH vs CHMv2 at analysis cells", loc="left", fontweight="bold",
             fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
bands = list(C["band_slopes"])
x = np.arange(len(bands))
eth = [C["band_slopes"][b]["eth"]["slope_per_m"] * 10 for b in bands]
eth_se = [C["band_slopes"][b]["eth"]["se"] * 10 * 1.96 for b in bands]
ch2 = [C["band_slopes"][b]["chm2"]["slope_per_m"] * 10 for b in bands]
ch2_se = [C["band_slopes"][b]["chm2"]["se"] * 10 * 1.96 for b in bands]
ax.bar(x - 0.18, eth, width=0.36, yerr=eth_se, color=WONG["blue"], alpha=0.85,
       error_kw=dict(lw=0.7), label="ETH 10 m")
ax.bar(x + 0.18, ch2, width=0.36, yerr=ch2_se, color=WONG["orange"], alpha=0.85,
       error_kw=dict(lw=0.7), label="CHMv2 1 m")
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks(x)
ax.set_xticklabels([b.replace("-", "-\n") + "\nm" for b in bands], fontsize=5.4)
ax.set_ylabel("dLST/dh (°C per 10 m)")
ax.legend(frameon=False, fontsize=6, loc="lower right")
ax.set_title("b  Buffering slope by canopy source", loc="left", fontweight="bold",
             fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

fig.savefig(f"{F}/F10_chm_compare.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F10_chm_compare.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F10_chm_compare")
