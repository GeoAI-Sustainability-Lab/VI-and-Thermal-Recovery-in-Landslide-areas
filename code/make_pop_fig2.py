"""Population figure v10: per cohort column — absolute LST with a narrow
patch−control difference strip below, absolute NDVI with its strip. The two
strata rows were removed in v19 (they duplicated Table 3 and Fig 8).
Inputs: outputs/abs_series.json, outputs/abs_series2.json
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
from grid_utils import apply_mpl_standards, WONG, AGENT_COL

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
R = json.load(open(f"{O}/abs_series.json"))
R2 = json.load(open(f"{O}/abs_series2.json"))

# same four agent-years, same order and same labels as the case chips of Fig. 6
COLS = [("typhoon_2024", "Typhoon 2024", AGENT_COL["typhoon_rain"]),
        ("rainfall_2025", "Heavy rain 2025", AGENT_COL["rainfall"]),
        ("earthquake_2024", "Earthquake 2024", AGENT_COL["earthquake"]),
        ("hansen_2023", "Annual loss 2023", AGENT_COL["hansen"])]
ELEV_ST = [("lt1000", "<1,000 m", WONG["skyblue"]),
           ("1000_2000", "1,000–2,000 m", WONG["green"]),
           ("gt2000", ">2,000 m", WONG["purple"])]
AREA_ST = [("lt2", "<2 ha", WONG["yellow"]),
           ("2_10", "2–10 ha", WONG["orange"]),
           ("ge10", "≥10 ha", WONG["vermillion"])]

fig = plt.figure(figsize=(183 * MM, 150 * MM), constrained_layout=True)
gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 0.30, 1.0, 0.30],
                      hspace=0.05)
XT = [2014, 2018, 2022, 2026]   # the epoch stack now runs 2013-2026
XLIM = (2012.6, 2026.9)                # every panel spans the full stack


def diff_strip(ax, series, col, tev, ylab=None, ylim=None, ticks=True):
    xs = sorted(series, key=lambda t: series[t]["mid"])
    x = [series[t]["mid"] for t in xs]
    y = [series[t]["diff"] for t in xs]
    e = [1.96 * series[t]["se"] for t in xs]
    ax.axvline(tev, color="grey", lw=0.8, ls="--", zorder=1)
    ax.axhline(0, color="black", lw=0.7, zorder=2)
    # bars, not lines: reads instantly as a different statistic (patch - control)
    # and the up/down direction is unambiguous
    ax.bar(x, y, width=0.62, color=col, alpha=0.85, lw=0, zorder=3)
    ax.errorbar(x, y, yerr=e, fmt="none", ecolor="black", elinewidth=0.6,
                capsize=1.2, zorder=4)
    ax.spines[["top", "right"]].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.3)
    if not ticks:
        ax.tick_params(labelleft=False)


# one common scale per difference row, so the four cohorts compare directly
YL = {}
for band in ("lst", "ndvi"):
    lo = hi = 0.0
    for key, _, _ in COLS:
        ser = R2[key]["diff"][band]
        for t in ser:
            lo = min(lo, ser[t]["diff"] - 1.96 * ser[t]["se"])
            hi = max(hi, ser[t]["diff"] + 1.96 * ser[t]["se"])
    pad = 0.06 * (hi - lo)
    YL[band] = (lo - pad, hi + pad)

ABS = {}
for band in ("lst", "ndvi"):
    lo = hi = None
    for key, _, _ in COLS:
        for e in R[key]["epochs"].values():
            if band not in e:
                continue
            b = e[band]
            for v in (b["patch"] - 1.96 * b["patch_se"], b["patch"] + 1.96 * b["patch_se"],
                      b["ctrl"] - 1.96 * b["ctrl_se"], b["ctrl"] + 1.96 * b["ctrl_se"]):
                lo = v if lo is None else min(lo, v); hi = v if hi is None else max(hi, v)
    pad = 0.05 * (hi - lo)
    ABS[band] = (lo - pad, hi + pad)

for j, (key, title, col) in enumerate(COLS):
    d = R[key]; d2 = R2[key]
    tev = d["t_event_med"]
    # row 0: absolute LST; row 1: LST diff strip
    ax = fig.add_subplot(gs[0, j])
    for band_row, band in [(0, "lst")]:
        xs, pm, ps, cm_, cs = [], [], [], [], []
        for tag, e in d["epochs"].items():
            if band in e:
                xs.append(e["mid"])
                pm.append(e[band]["patch"]); ps.append(e[band]["patch_se"])
                cm_.append(e[band]["ctrl"]); cs.append(e[band]["ctrl_se"])
        o = np.argsort(xs)
        xs = np.array(xs)[o]; pm = np.array(pm)[o]; ps = np.array(ps)[o]
        cm_ = np.array(cm_)[o]; cs = np.array(cs)[o]
        ax.axvline(tev, color="grey", lw=0.9, ls="--", zorder=1)
        ax.fill_between(xs, cm_ - 1.96 * cs, cm_ + 1.96 * cs, color="#4d4d4d",
                        alpha=0.18, lw=0)
        ax.plot(xs, cm_, "-o", ms=2.2, lw=1.0, color="#4d4d4d", zorder=3)
        ax.fill_between(xs, pm - 1.96 * ps, pm + 1.96 * ps, color=col,
                        alpha=0.18, lw=0)
        ax.plot(xs, pm, "-o", ms=2.2, lw=1.1, color=col, zorder=4)
    ax.set_title(f"{chr(97+j)}  {title}\n(n = {d['n_patches']})", loc="left",
                 fontweight="bold", fontsize=9.5)
    ax.set_xlim(*XLIM); ax.set_xticks(XT); ax.tick_params(labelbottom=False)
    ax.set_ylim(*ABS["lst"])
    ax.spines[["top", "right"]].set_visible(False)
    if j:
        ax.tick_params(labelleft=False)
    if j == 0:
        ax.set_ylabel("Summer LST (°C)")
        ax.legend(handles=[
            Line2D([], [], marker="o", ms=3, lw=1.1, color=col,
                   label="Disturbed patches"),
            Line2D([], [], marker="o", ms=3, lw=1.0, color="#4d4d4d",
                   label="Matched controls"),
            Line2D([], [], ls="--", lw=0.9, color="grey", label="Event")],
            frameon=False, fontsize=8.3, loc="upper left")
    ax = fig.add_subplot(gs[1, j])
    diff_strip(ax, d2["diff"]["lst"], col, tev,
               ylab="patch − ctrl (°C)" if j == 0 else None, ylim=YL["lst"],
               ticks=(j == 0))
    ax.set_xlim(*XLIM); ax.set_xticks(XT); ax.tick_params(labelbottom=False)

    # row 2: absolute NDVI; row 3: NDVI diff strip
    ax = fig.add_subplot(gs[2, j])
    xs, pm, ps, cm_, cs = [], [], [], [], []
    for tag, e in d["epochs"].items():
        if "ndvi" in e:
            xs.append(e["mid"])
            pm.append(e["ndvi"]["patch"]); ps.append(e["ndvi"]["patch_se"])
            cm_.append(e["ndvi"]["ctrl"]); cs.append(e["ndvi"]["ctrl_se"])
    o = np.argsort(xs)
    xs = np.array(xs)[o]; pm = np.array(pm)[o]; ps = np.array(ps)[o]
    cm_ = np.array(cm_)[o]; cs = np.array(cs)[o]
    ax.axvline(tev, color="grey", lw=0.9, ls="--", zorder=1)
    ax.fill_between(xs, cm_ - 1.96 * cs, cm_ + 1.96 * cs, color="#4d4d4d",
                    alpha=0.18, lw=0)
    ax.plot(xs, cm_, "-o", ms=2.2, lw=1.0, color="#4d4d4d", zorder=3)
    ax.fill_between(xs, pm - 1.96 * ps, pm + 1.96 * ps, color=col, alpha=0.18,
                    lw=0)
    ax.plot(xs, pm, "-o", ms=2.2, lw=1.1, color=col, zorder=4)
    ax.set_xlim(*XLIM); ax.set_xticks(XT); ax.tick_params(labelbottom=False)
    ax.set_ylim(*ABS["ndvi"])
    ax.spines[["top", "right"]].set_visible(False)
    if j:
        ax.tick_params(labelleft=False)
    if j == 0:
        ax.set_ylabel("Summer NDVI")
    ax = fig.add_subplot(gs[3, j])
    diff_strip(ax, d2["diff"]["ndvi"], col, tev,
               ylab="patch − ctrl" if j == 0 else None, ylim=YL["ndvi"],
               ticks=(j == 0))
    ax.set_xlim(*XLIM); ax.set_xticks(XT)
    ax.set_xlabel("Year")

fig.align_ylabels()          # left-column y-axis titles share one x position
fig.savefig(f"{F}/F11_population.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F11_population.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F11_population v9")
