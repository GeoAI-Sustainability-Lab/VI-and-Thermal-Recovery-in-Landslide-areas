"""Typhoon Morakot deep dive.
(a) 26-year Landsat series of three Morakot scars spanning three orders of
    magnitude in area and 900 m of elevation;
(b) the 2,396-patch cohort against the pooled recovery fit of all other events;
(c, d) area and elevation dose-response WITHIN the single event, where trigger,
    date and rainfall are held constant by construction.
Inputs: outputs/morakot.json, outputs/recovery_clocks.json
Output: figures/F15_morakot.{pdf,png}
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
from grid_utils import apply_mpl_standards, WONG, AREA_COL

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
M = json.load(open(f"{O}/morakot.json"))
C = json.load(open(f"{O}/recovery_clocks.json"))
CT, CG = WONG["vermillion"], WONG["green"]

fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 134 * MM),
                         constrained_layout=True)
# extra row gap: the top row's "Year" axis label must clear the c/d titles
fig.get_layout_engine().set(h_pad=10 / 72)

# ---------------- (a) three 26-year case series ----------------
ax = axes[0][0]
CASES = [("high_elev_event", "92 ha · 2,008 m", AREA_COL[2], "-"),
         ("hansen_2009_morakot", "58 ha · 1,427 m", AREA_COL[1], "-"),
         ("small_morakot_event", "2 ha · 1,131 m", AREA_COL[0], "-")]
for key, lab, col, ls in CASES:
    c = M["cases"].get(key)
    if not c:
        continue
    ax.plot(c["year"], c["dlst"], ls, marker="o", ms=2.2, lw=1.1, color=col,
            label=lab)
ax.axvline(2009.6, color="grey", lw=1.0, ls="--")
# label at the foot of the event line, where no series ever goes
ax.text(2010.1, -0.95, "Morakot\nAug 2009", fontsize=8.3, va="top",
        color="#444444")
ax.axhline(0, color="black", lw=0.7, ls=":")
ax.set_ylim(-2.6, 9.2)
ax.set_xlabel("Year"); ax.set_ylabel("ΔLST patch − control (°C)")
ax.legend(frameon=False, fontsize=8.3, loc="upper left", title="scar area · elevation",
          title_fontsize=8.3, borderaxespad=0.2, labelspacing=0.3,
          handlelength=1.2, handletextpad=0.5)
ax.set_title("a  Three Morakot scars, 2000–2026", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- (b) cohort against all other events ----------------
ax = axes[0][1]
co, ot = M["cohort"], M["others"]
tt = np.linspace(0.6, 22, 150)
TH = C["thermal"]
ax.plot(tt, TH["A"] * np.exp(-tt / TH["tau"]), color="#888888", lw=1.4, ls="--",
        label=f"pooled fit, all events (τ = {TH['tau']:.1f} yr)")
ax.errorbar([r["age"] for r in ot], [r["dlst"] for r in ot],
            yerr=[1.96 * r["dlst_se"] for r in ot], fmt="o", ms=2.0,
            color="#888888", lw=0.7, capsize=1.2, alpha=0.7,
            label="other events, 1-yr bins")
ax.errorbar([r["age"] for r in co], [r["dlst"] for r in co],
            yerr=[1.96 * r["dlst_se"] for r in co], fmt="D", ms=4.0,
            color=WONG["purple"], lw=1.1, capsize=1.8,
            label=f"Morakot cohort ({M['n_patches']:,} patches)")
ax.axhline(0, color="black", lw=0.7, ls="--")
ax.set_xlim(0, 22.5)
ax.set_ylim(-0.15, 3.55)     # headroom so the legend clears the early bins
ax.set_xlabel("Years since disturbance")
ax.set_ylabel("ΔLST patch − control (°C)")
ax.legend(frameon=False, fontsize=8.3, loc="upper right", borderaxespad=0.2,
          labelspacing=0.3)
# the late-window statement is carried by the caption,
# so the panel itself stays clean
ax.set_title("b  The largest event tracks the pooled curve, at its slow end",
             loc="left", fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- (c, d) dose-response within the single event ----------
for ax, key, title, cols in [
        (axes[1][0], "area", "c  Scar area, within Morakot only", AREA_COL),
        (axes[1][1], "elev", "d  Elevation, within Morakot only",
         [WONG["skyblue"], WONG["green"], WONG["purple"]])]:
    grp = M[key]
    for (lab, e), col in zip(grp.items(), cols):
        ser = e["series"]
        ax.errorbar([r["age"] for r in ser], [r["dlst"] for r in ser],
                    yerr=[1.96 * r["dlst_se"] for r in ser], fmt="-o", ms=2.6,
                    color=col, lw=1.2, capsize=1.4,
                    label=f"{lab} · {e['n_patches']:,} · +{e['dlst']:.2f} °C")
    ax.axhline(0, color="black", lw=0.7, ls="--")
    ax.set_xlabel("Years since disturbance")
    ax.set_ylabel("ΔLST patch − control (°C)")
    ax.set_ylim(-0.2, 5.75)   # headroom so the legend clears the top whiskers
    ax.legend(frameon=False, fontsize=8.3, loc="upper right", handlelength=1.2,
              borderaxespad=0.3, labelspacing=0.3)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)

fig.savefig(f"{F}/F15_morakot.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F15_morakot.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F15_morakot")
