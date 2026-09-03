"""Sentinel-1 pilot figure: radar structural recovery clock vs greenness and
thermal clocks. (a) Δγ0 VH (patch − control, dB) vs years since event,
Morakot-cohort-excluded bins + exponential fit, Morakot cohort shown
separately as purple diamonds (matching Fig. 6a convention); (b) the three
time constants side by side.
Inputs: outputs/s1_pilot.json, outputs/results2.json, data/s1_pilot_rows.parquet
Output: figures/F12_s1_pilot.{pdf,png}
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
O = f"{_TROOT}/outputs"
D = f"{_TROOT}/data"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
S1 = json.load(open(f"{O}/s1_pilot.json"))
R2 = json.load(open(f"{O}/results2.json"))
TR = R2["tau_ratios"]["event_pooled"]
PURPLE = "#9467bd"

# Morakot cohort yearly bins from rows
df = pd.read_parquet(f"{D}/s1_pilot_rows.parquet")
dd = pd.read_parquet(f"{D}/patches_deltas2.parquet", columns=["t_event"])
df = df.merge(dd, left_on="pid", right_index=True)
mor = df[(df.t_event >= 2009.5) & (df.t_event <= 2009.75) & df.dvh_db.notna()]
mb = mor.groupby(mor.age.round())["dvh_db"].agg(["mean", "sem", "count"])

fig, axes = plt.subplots(1, 2, figsize=(183 * MM, 78 * MM), constrained_layout=True)

ax = axes[0]
fvh = S1["fit_vh_xmor"]
b = fvh["bins"]
ax.errorbar(b["center"], b["mean"], yerr=1.96 * np.array(b["se"]), fmt="s", ms=2.8,
            color=WONG["blue"], capsize=1.6, lw=0.8,
            label="VH bins, excl. Morakot")
tt = np.linspace(0.6, 21, 150)
ax.plot(tt, fvh["A"] * np.exp(-tt / fvh["tau"]), color=WONG["blue"], lw=1.3,
        label=f"VH fit τ = {fvh['tau']:.1f} ± {fvh['tau_se']:.1f} yr")
fvv = S1.get("fit_vv_xmor")
if fvv:
    ax.plot(tt, fvv["A"] * np.exp(-tt / fvv["tau"]), color=WONG["skyblue"], lw=0.9,
            ls="--", alpha=0.9, label=f"VV fit τ = {fvv['tau']:.0f} ± {fvv['tau_se']:.0f} yr")
ax.errorbar(mb.index, mb["mean"], yerr=1.96 * mb["sem"], fmt="D", ms=3.2,
            mfc="none", color=PURPLE, capsize=1.6, lw=0.8,
            label="Morakot 2009")
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Years since event")
ax.set_ylabel(r"$\Delta\gamma^0$ patch − control (dB)")
# 圖例壓到資料上：改為頂端留白帶、兩欄排列，並縮短標籤
ax.set_ylim(-3.4, 1.45)
ax.legend(frameon=False, fontsize=8.3, loc="upper left", ncol=2,
          columnspacing=0.9, handlelength=1.4, labelspacing=0.3,
          borderaxespad=0.2)
ax.set_title("a  Radar backscatter recovery",
             loc="left", fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
taus = [("Greenness\nΔNDVI", TR["g"], None, WONG["green"]),
        ("Thermal\nΔLST", TR["t"], None, WONG["vermillion"]),
        (r"Structure" + "\n" + r"$\Delta\gamma^0$ VH", fvh["tau"], fvh["tau_se"],
         WONG["blue"])]
x = np.arange(3)
for i, (lab, t, se, c) in enumerate(taus):
    ax.bar(i, t, width=0.55, color=c, alpha=0.85, edgecolor="black", lw=0.4,
           yerr=(1.96 * se) if se else None, error_kw=dict(lw=0.8, capsize=2))
    ax.annotate(f"{t:.1f}", (i, t + (11.5 if se else 0.6)), ha="center",
                fontsize=8.3, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([t[0] for t in taus], fontsize=8.3)
ax.set_ylabel("Recovery e-folding time τ (yr)")
ax.set_title("b  Three recovery clocks", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

fig.savefig(f"{F}/F12_s1_pilot.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F12_s1_pilot.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F12_s1_pilot")
