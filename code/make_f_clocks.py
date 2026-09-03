"""Recovery-clock figure: the pooled thermal and greenness
recovery of landslide scars, 2004-2025.
(a) raw anomalies against time since disturbance, every patch-epoch shown
    semi-transparently, with the exponential fits on twin axes;
(b) the same two fits normalised to their own initial amplitude on ONE axis,
    with patch-level bootstrap bands - the comparison of the two clocks is made
    here, where it cannot depend on twin-axis scaling;
(c) the joint LST-NDVI trajectory (hysteresis);
(d) bootstrap distribution of tau_thermal / tau_greenness.
Inputs: outputs/recovery_clocks.json, outputs/results2.json, chrono2_long.parquet
Output: figures/F13_recovery_clocks.{pdf,png}
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
from matplotlib.lines import Line2D

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
C = json.load(open(f"{O}/recovery_clocks.json"))
R2 = json.load(open(f"{O}/results2.json"))
TH, GR = C["thermal"], C["greenness"]
CT, CG = WONG["vermillion"], WONG["green"]

L = pd.read_parquet(f"{O}/chrono2_long.parquet")
ev = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= 22) & (L.src == "event")]

# (a) and (b) carry the figure and get the height; (c) is a wide, short strip
# where the milestone gaps are read horizontally. The bootstrap ratio moved to
# the supplementary figure drawn at the end of this script.
fig = plt.figure(figsize=(183 * MM, 132 * MM), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.42])
axA = fig.add_subplot(gs[0, 0]); axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[1, :])

# ---------------- (a) raw anomalies, every patch-epoch ----------------
ax = axA
d1 = ev[ev.dlst.notna()]
ax.scatter(d1.age, d1.dlst, s=0.9, color=CT, alpha=0.035, lw=0, rasterized=True)
tt = np.linspace(0.6, 22, 200)
ax.plot(tt, TH["A"] * np.exp(-tt / TH["tau"]), color=CT, lw=2.0, zorder=6)
ax.errorbar(TH["bin_center"], TH["bin_mean"],
            yerr=1.96 * np.array(TH["bin_se"]), fmt="o", ms=2.6, color=CT,
            mec="white", mew=0.4, lw=0.8, capsize=1.4, zorder=7)
ax.axhline(0, color="black", lw=0.7, ls="--", zorder=5)
ax.set_ylim(-4.5, 11)
ax.set_xlabel("Years since disturbance")
ax.set_ylabel("ΔLST patch − control (°C)", color=CT)
ax.tick_params(axis="y", colors=CT)
ax2 = ax.twinx()
d2 = ev[ev.dndvi.notna()]
ax2.scatter(d2.age, d2.dndvi, s=0.9, color=CG, alpha=0.035, lw=0, rasterized=True)
ax2.plot(tt, GR["A"] * np.exp(-tt / GR["tau"]), color=CG, lw=2.0, zorder=6)
ax2.errorbar(GR["bin_center"], GR["bin_mean"],
             yerr=1.96 * np.array(GR["bin_se"]), fmt="s", ms=2.4, color=CG,
             mec="white", mew=0.4, lw=0.8, capsize=1.4, zorder=7)
ax2.set_ylim(-0.92, 0.30)
ax2.set_ylabel("ΔNDVI patch − control", color=CG)
ax2.tick_params(axis="y", colors=CG)
ax2.spines[["top"]].set_visible(False)
ax.spines[["top"]].set_visible(False)
ax.legend(handles=[
    Line2D([], [], color=CT, lw=2, label="ΔLST (left axis)"),
    Line2D([], [], color=CG, lw=2, label="ΔNDVI (right axis)"),
    Line2D([], [], marker="o", ls="", ms=3, color="grey",
           label="1-yr bins")],
    frameon=False, fontsize=8.3, loc="upper left")
ax.set_title(f"a  {C['n_rows']:,} patch-epochs from {C['n_patches']:,} landslide scars",
             loc="left", fontweight="bold", fontsize=9.5)

# ---------------- (b) normalised, single axis, bootstrap bands -------------
ax = axB
for key, col, lab, mk in [("greenness", CG, "Greenness", "s"),
                          ("thermal", CT, "Thermal", "o")]:
    k = C[key]
    t = np.array(k["norm_t"])
    ax.fill_between(t, k["norm_lo"], k["norm_hi"], color=col, alpha=0.22, lw=0)
    ax.plot(t, np.exp(-t / k["tau"]), color=col, lw=2.0,
            label=f"{lab}  τ = {k['tau']:.1f} "
                  f"[{k['tau_ci'][0]:.1f}–{k['tau_ci'][1]:.1f}] yr")
    obs = np.array(k["bin_mean"]) / k["A"]
    err = 1.96 * np.array(k["bin_se"]) / abs(k["A"])
    ax.errorbar(k["bin_center"], obs, yerr=err, fmt="none", ecolor=col,
                elinewidth=0.6, capsize=1.2, alpha=0.7, zorder=4)
    ax.plot(k["bin_center"], obs, mk, ms=2.6, color=col, mec="white", mew=0.4,
            alpha=0.85)
# the 0.5 line marks "half of the anomaly gone"; the years between the two
# crossings are quantified by the 50% milestone in panel (c), not annotated here
ax.axhline(0.5, color="grey", lw=0.6, ls=":")
ax.text(0.5, 0.53, "half of the\nanomaly gone", fontsize=8.3, color="#555555",
        va="bottom")
ax.set_xlim(0, 22); ax.set_ylim(0, 1.05)
ax.set_xlabel("Years since disturbance")
ax.set_ylabel("Fraction of the initial anomaly remaining")
ax.legend(frameon=False, fontsize=8.3, loc="upper right",
          bbox_to_anchor=(1.0, 1.03))
ax.set_title("b  Both clocks on one scale, patch-level bootstrap", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- (c) recovery milestones: greenness arrives first ----------
ax = axC
import json as _json
_rc = C
tT, tG = TH["tau"], GR["tau"]
tT_ci, tG_ci = TH["tau_ci"], GR["tau_ci"]
lag50, lag_ci = _rc["lag_t50"]["mean"], _rc["lag_t50"]["ci"]
MS = [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]
for i, (ffrac, lab) in enumerate(MS):
    fac = np.log(1 / (1 - ffrac))
    aT, aG = tT * fac, tG * fac
    ciT = [tT_ci[0] * fac, tT_ci[1] * fac]
    ciG = [tG_ci[0] * fac, tG_ci[1] * fac]
    lg = lag50 * fac / np.log(2)
    lg_lo, lg_hi = lag_ci[0] * fac / np.log(2), lag_ci[1] * fac / np.log(2)
    y = len(MS) - 1 - i
    beyond = aT > 22.0                       # thermal 75% lies past the window
    # the span is dashed once it runs past the 22-yr observation window
    if beyond:
        ax.plot([aG, min(aT, 22.0)], [y, y], color="#999999", lw=1.0, zorder=2)
        ax.plot([min(aT, 22.0), aT], [y, y], color="#999999", lw=1.0, ls=(0, (3, 2)),
                zorder=2)
    else:
        ax.plot([aG, aT], [y, y], color="#999999", lw=1.0, zorder=2)
    ax.errorbar([aG], [y], xerr=[[aG - ciG[0]], [ciG[1] - aG]], fmt="s", ms=5,
                color=CG, capsize=2, lw=1.1, zorder=4)
    if beyond:
        # extrapolated milestone: dashed interval and an open marker
        ax.hlines(y, ciT[0], ciT[1], color=CT, lw=1.1, ls=(0, (3, 2)), zorder=4)
        ax.vlines([ciT[0], ciT[1]], y - 0.07, y + 0.07, color=CT, lw=1.1, zorder=4)
        ax.plot([aT], [y], "o", ms=5, color=CT, mfc="white", zorder=5)
    else:
        ax.errorbar([aT], [y], xerr=[[aT - ciT[0]], [ciT[1] - aT]], fmt="o", ms=5,
                    color=CT, capsize=2, lw=1.1, zorder=4)
    ax.annotate(f"+{lg:.1f} yr [{lg_lo:.1f}–{lg_hi:.1f}]",
                ((aG + aT) / 2, y + 0.16), ha="center", fontsize=8.3,
                color="#333333", fontweight="bold")
ax.set_yticks(range(len(MS)))
ax.set_yticklabels([lab for _, lab in reversed(MS)], fontsize=8.3)
ax.set_ylabel("Anomaly cleared")
ax.set_ylim(-0.62, len(MS) - 0.35)
ax.set_xlim(0, 30)
ax.set_xlabel("Years since disturbance (from fitted τ)")
from matplotlib.lines import Line2D as _L2
ax.legend(handles=[
    _L2([], [], marker="s", ls="", color=CG, ms=5, label="greenness reaches it"),
    _L2([], [], marker="o", ls="", color=CT, ms=5, label="thermal reaches it"),
    _L2([], [], marker="o", ls=(0, (3, 2)), color=CT, mfc="white", ms=5,
        label="beyond 22-yr window (extrapolated)")],
    frameon=False, fontsize=8.3, loc="upper right", bbox_to_anchor=(0.995, 0.99),
    handlelength=1.2, labelspacing=0.35)
ax.set_title("c  Greenness reaches every milestone first", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

fig.savefig(f"{F}/F13_recovery_clocks.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F13_recovery_clocks.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F13_recovery_clocks (a, b, c)")

# ---------------- supplementary: bootstrap ratio ----------------
figS, ax = plt.subplots(figsize=(183 * MM, 78 * MM), layout="constrained")
r = np.array(C["ratio"]["samples"])
ax.hist(r, bins=32, color=WONG["blue"], alpha=0.80, edgecolor="white", lw=0.4)
ax.axvline(1.0, color="black", lw=1.1, ls="--")
ax.axvline(C["ratio"]["mean"], color=WONG["vermillion"], lw=1.4)
lo, hi = C["ratio"]["ci"]
ax.axvspan(lo, hi, color=WONG["vermillion"], alpha=0.14, lw=0)
ax.set_xlabel("τ thermal / τ greenness")
ax.set_ylabel(f"Bootstrap resamples (n = {C['ratio']['n_boot']})")
# top-left block sits right of the 1.00 line, above the short left-tail bars;
# the by-patch resampling is stated in the caption, not repeated here
ax.text(0.065, 0.97,
        f"ratio = {C['ratio']['mean']:.3f}\n95% CI [{lo:.3f}, {hi:.3f}]\n"
        f"P(ratio > 1) = {C['ratio']['p_gt1']:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=8.3,
        linespacing=1.5)
ax.set_title("Thermal recovery is slower in every resample", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
figS.savefig(f"{F}/FS9_ratio_bootstrap.pdf", bbox_inches="tight")
figS.savefig(f"{F}/FS9_ratio_bootstrap.png", bbox_inches="tight", dpi=500)
plt.close(figS)
print("saved FS9_ratio_bootstrap")
