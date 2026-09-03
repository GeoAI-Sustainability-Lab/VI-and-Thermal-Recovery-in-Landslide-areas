"""Fig. 4 (v24): event study, seven pre-event years and two after.
Three panels: (a) thermal, (b) greenness, (c) thermal split by trigger.
No explanatory text is drawn on the figure - the legends carry only the series
names and the patch counts, everything else lives in the caption and in 3.2.1.
Every pre-event year that enters the pooled lead statistic is now drawn, so
the flat pre-event path is visible rather than asserted. The by-cohort-year split is covered by
the cohort DiD of 3.2.2 and Table 3, so it is no longer drawn here.
Input: outputs/eventstudy.json  Output: figures/F16_eventstudy.{pdf,png}
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
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG, AGENT_LAB, AGENT_COL

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
E = json.load(open(f"{O}/eventstudy.json"))
KLO, KHI = -7, 2   # display window: every pre-event year that is tested,
                   # and the two years after the event


def _win(rows):
    return [r for r in rows if KLO <= r["k"] <= KHI]


fig = plt.figure(figsize=(183 * MM, 68 * MM), constrained_layout=True)
gs = fig.add_gridspec(1, 3)


def draw(ax, rows, col_lag, ylab, title, ylim, loc, ev_top):
    rows = _win(rows)
    k = np.array([r["k"] for r in rows], float)
    b = np.array([r["beta"] for r in rows], float)
    lo = np.array([r["lo"] for r in rows], float)
    hi = np.array([r["hi"] for r in rows], float)
    lead = k <= -2
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=1)
    ax.axvspan(k.min() - 0.6, -0.5, color="#000000", alpha=0.045, lw=0, zorder=0)
    # the event line spans the data band only, so it never crosses the legend
    ax.vlines(0, ylim[0], ev_top, color=WONG["vermillion"], lw=0.9, ls=":", zorder=1)
    for m, c, lab in [(lead, "#7A7A7A", "before"), (~lead, col_lag, "after")]:
        if not m.any():
            continue
        ax.errorbar(k[m], b[m], yerr=[b[m] - lo[m], hi[m] - b[m]], fmt="o",
                    ms=2.9, lw=0.9, capsize=1.5, color=c, mfc=c, mec=c,
                    label=lab, zorder=3)
    # the reference summer k = -1 is zero by construction
    ax.plot([-1], [0], marker="o", ms=3.4, mfc="white", mec="#7A7A7A",
            color="#7A7A7A", ls="", zorder=4, label="reference summer")
    ax.plot(list(k[lead]) + [-1], list(b[lead]) + [0], color="#7A7A7A",
            lw=0.9, alpha=0.8, zorder=2)
    ax.plot([-1] + list(k[~lead]), [0] + list(b[~lead]), color=col_lag,
            lw=1.0, alpha=0.75, zorder=2)
    ax.set_xlabel("Years relative to the disturbance")
    ax.set_ylabel(ylab)
    ax.set_xlim(k.min() - 0.7, k.max() + 0.7)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8.3, loc=loc, handlelength=1.3,
              title=f"n = {max(r['n_patch'] for r in rows):,} patches",
              title_fontsize=8.3, alignment="left", labelspacing=0.3)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)


axA = fig.add_subplot(gs[0, 0])
draw(axA, E["lst"], WONG["vermillion"],
     "ΔLST relative to the reference summer (°C)",
     "a  Thermal event study", ylim=(-0.32, 2.30), loc="upper left",
     ev_top=1.05)

axB = fig.add_subplot(gs[0, 1])
draw(axB, E["ndvi"], WONG["green"],
     "ΔNDVI relative to the reference summer",
     "b  Greenness event study", ylim=(-0.245, 0.175), loc="upper left",
     ev_top=0.175)

# ---------------- (c) by trigger ----------------
axC = fig.add_subplot(gs[0, 2])
axC.axhline(0, color="black", lw=0.7, ls="--", zorder=1)
kk = [r["k"] for rows in E.get("lst_by_agent", {}).values() for r in _win(rows)]
if kk:
    axC.axvspan(min(kk) - 0.6, -0.5, color="#000000", alpha=0.045, lw=0, zorder=0)
for ag, rows in E.get("lst_by_agent", {}).items():
    rows = _win(rows)
    k = np.array([r["k"] for r in rows], float)
    b = np.array([r["beta"] for r in rows], float)
    lo = np.array([r["lo"] for r in rows], float)
    hi = np.array([r["hi"] for r in rows], float)
    c = AGENT_COL.get(ag, "#888888")
    axC.fill_between(k, lo, hi, color=c, alpha=0.14, lw=0, zorder=2)
    pre_m = k <= -1
    kp = np.append(k[pre_m], -1); bp = np.append(b[pre_m], 0)
    axC.plot(kp, bp, color="#8a8a8a", lw=1.1, zorder=3)
    axC.plot(k[pre_m], b[pre_m], color="#8a8a8a", ls="", marker="o", ms=2.2, zorder=3)
    axC.plot(np.insert(k[k >= 1], 0, -1), np.insert(b[k >= 1], 0, 0),
             color=c, lw=1.2, zorder=3)
    axC.plot(k[k >= 1], b[k >= 1], color=c, ls="", marker="o", ms=2.2, zorder=3,
             label=f"{AGENT_LAB.get(ag, ag)}  n = {max(r['n_patch'] for r in rows):,}")
axC.set_xlabel("Years relative to the disturbance")
axC.xaxis.set_major_locator(MultipleLocator(2))
axC.set_ylabel("ΔLST relative to the reference summer (°C)")
_hiC = max(r["hi"] for rows in E.get("lst_by_agent", {}).values() for r in _win(rows))
axC.set_ylim(top=_hiC + 0.95)
axC.vlines(0, axC.get_ylim()[0], _hiC + 0.10, color=WONG["vermillion"], lw=0.9,
           ls=":", zorder=1)
axC.legend(frameon=False, fontsize=8.3, loc="upper left", handlelength=1.3,
           borderaxespad=0.25, labelspacing=0.3)
axC.set_title("c  By trigger", loc="left", fontweight="bold", fontsize=9.5)
axC.spines[["top", "right"]].set_visible(False)

fig.savefig(f"{F}/F16_eventstudy.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F16_eventstudy.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F16_eventstudy v23")
