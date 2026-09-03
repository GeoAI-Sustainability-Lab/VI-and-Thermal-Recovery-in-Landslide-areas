"""Stratified recovery figure. One common set of strata is
applied to the whole 2004-2025 database - trigger agent, elevation band and the
official area classes - so that the three comparisons are read on the same
footing. Top two rows: age-binned anomalies with exponential fits. Bottom row:
e-folding times with patch-level bootstrap intervals, thermal against greenness.
Inputs: outputs/strata_curves.json, outputs/recovery_clocks.json
Output: figures/F14_strata.{pdf,png}
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
from grid_utils import apply_mpl_standards, WONG, AGENT_COL, AREA_COL

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
S = json.load(open(f"{O}/strata_curves.json"))
C = json.load(open(f"{O}/recovery_clocks.json"))
CT, CG = WONG["vermillion"], WONG["green"]

GROUPS = [
    # trigger colours come from the shared agent palette (same in Figs 2 and 4)
    ("Trigger agent", [("typhoon", "Typhoon", AGENT_COL["typhoon_rain"]),
                       ("rainfall", "Heavy rainfall", AGENT_COL["rainfall"]),
                       ("earthquake", "Earthquake", AGENT_COL["earthquake"]),
                       ("hansen", "Annual loss 2015–23", AGENT_COL["hansen"])]),
    ("Elevation", [("lt1000", "<1,000 m", WONG["skyblue"]),
                   ("1000_2000", "1,000–2,000 m", WONG["green"]),
                   ("gt2000", ">2,000 m", WONG["purple"])]),
    ("Patch area", [("lt2", "<2 ha", AREA_COL[0]),
                    ("2_10", "2–10 ha", AREA_COL[1]),
                    ("ge10", "≥10 ha", AREA_COL[2])]),
]

fig = plt.figure(figsize=(183 * MM, 188 * MM),
                 layout="constrained")
fig.get_layout_engine().set(w_pad=0.018, wspace=0.028)
# rows 2 and 4 host the legends, so no legend sits on the data panels
gs = fig.add_gridspec(5, 3, height_ratios=[1.0, 1.0, 0.14, 1.30, 0.085])
tt = np.linspace(0.6, 22, 150)

# short legend names; τ per stratum lives in panel (d) and Suppl. Table S2
SHORT = {"Annual loss 2015–23": "Annual loss"}

_first = {}
for j, (gname, members) in enumerate(GROUPS):
    for i, (val, ylab, ylim) in enumerate(
            [("thermal", "ΔLST patch − control (°C)", (-1.15, 7.0)),
             ("greenness", "ΔNDVI patch − control", (-0.60, 0.06))]):
        # one y-axis per row: only the left column carries ticks and label,
        # so the panels themselves get the width
        ax = fig.add_subplot(gs[i, j], sharey=_first.get(i))
        _first.setdefault(i, ax)
        for key, lab, col in members:
            e = S.get(key)
            if not e:
                continue
            k = e[val]
            ax.errorbar(k["bin_center"], k["bin_mean"],
                        yerr=1.96 * np.array(k["bin_se"]), fmt="o", ms=2.0,
                        color=col, lw=0.7, capsize=1.2, alpha=0.75)
            if k["tau"]:
                ax.plot(tt, k["A"] * np.exp(-tt / k["tau"]), color=col, lw=1.6,
                        label=SHORT.get(lab, lab))
            else:
                ax.plot([], [], color=col, lw=1.6, label=SHORT.get(lab, lab))
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.set_xlim(0, 22.5); ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 1:
            ax.set_xlabel("Years since disturbance")
            # one shared legend per column, on its own strip below the pair
            axL = fig.add_subplot(gs[2, j])
            axL.axis("off")
            h_, l_ = ax.get_legend_handles_labels()
            axL.legend(h_, l_, loc="upper center", ncol=2, frameon=False,
                       fontsize=8.3, handlelength=1.4, columnspacing=1.0,
                       labelspacing=0.3, borderaxespad=0.0)
        else:
            ax.tick_params(labelbottom=False)
            ax.set_title(f"{chr(97+j)}  {gname}", loc="left", fontweight="bold",
                         fontsize=9.5)
        if j == 0:
            ax.set_ylabel(ylab)
        else:
            ax.tick_params(labelleft=False)

# ---------------- bottom: tau forest plot ----------------
X_RATIO, X_NPAT = 45.0, 51.0      # the two right-hand text columns, centred
ax = fig.add_subplot(gs[3, :])
rows, ypos, labels, seps = [], [], [], []
y = 0
for gname, members in GROUPS:
    for key, lab, col in members:
        e = S.get(key)
        if not e:
            continue
        if not e["thermal"]["tau"] and not e["greenness"]["tau"]:
            continue          # e.g. earthquake: stated in the text and Table S2
        rows.append((e, lab)); ypos.append(y); labels.append(lab); y += 1
    seps.append(y - 0.5); y += 0.6
pooled_t, pooled_g = C["thermal"], C["greenness"]
for (e, lab), yy in zip(rows, ypos):
    for val, col, dy, mk in [("thermal", CT, +0.16, "o"),
                             ("greenness", CG, -0.16, "s")]:
        k = e[val]
        if not k["tau"]:
            if e["thermal"]["tau"] or e["greenness"]["tau"]:
                ax.text(0.5, yy + dy, "no stable fit", fontsize=8.3, color=col,
                        va="center")
            continue
        ci = k.get("tau_ci")
        if ci:
            hi_c = min(ci[1], 38.0)
            ax.plot([ci[0], hi_c], [yy + dy] * 2, color=col, lw=1.1)
            if ci[1] > 38.0:                       # CI runs off the axis
                ax.plot([38.0], [yy + dy], marker=">", ms=3, color=col)
        ax.plot([k["tau"]], [yy + dy], mk, ms=3.6, color=col, mec="white", mew=0.5)
    r = e.get("ratio", {})
    if r.get("mean") and e["thermal"]["tau"] and e["greenness"]["tau"]:
        star = "*" if (r["p_gt1"] or 0) >= 0.95 else ""
        ax.plot([], [])
        ax.text(X_RATIO, yy, f"×{r['mean']:.2f}{star}", fontsize=8.3, va="center",
                ha="center", fontweight="bold" if star else "normal")
    ax.text(X_NPAT, yy, f"{e['n_patches']:,}", fontsize=8.3, va="center",
            ha="center", color="#555555")
for sy in seps[:-1]:
    ax.axhline(sy + 0.3, color="#dddddd", lw=0.7)
# pooled band and lines span the data rows only, leaving the bottom strip
# free for the legend
_y0b, _y1b = -1.05, max(ypos) + 0.5
for _pk, _pc in ((pooled_t, CT), (pooled_g, CG)):
    ax.fill_betweenx([_y0b, _y1b], _pk["tau_ci"][0], _pk["tau_ci"][1],
                     color=_pc, alpha=0.10, lw=0, zorder=0)
    ax.vlines(_pk["tau"], _y0b, _y1b, color=_pc, lw=0.9, ls=":")
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8.3)
ax.set_ylim(-1.45, max(ypos) + 0.7)
ax.set_xlim(0, 54.5)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
ax.set_xlabel("Recovery e-folding time τ (yr), with patch-level bootstrap 95% CI")
ax.text(X_RATIO, -0.95, "τ ratio", fontsize=8.3, fontweight="bold", ha="center")
ax.text(X_NPAT, -0.95, "patches", fontsize=8.3, fontweight="bold", ha="center")
ax.set_title("d  Recovery times by stratum, thermal versus greenness",
             loc="left", fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
ax.invert_yaxis()

axL2 = fig.add_subplot(gs[4, :])
axL2.axis("off")
axL2.legend(handles=[
    Line2D([], [], marker="o", ls="", ms=4, color=CT, label="Thermal ΔLST"),
    Line2D([], [], marker="s", ls="", ms=4, color=CG, label="Greenness ΔNDVI"),
    Line2D([], [], ls=":", color="grey",
           label="pooled estimate (shaded: 95% CI)")],
    frameon=False, fontsize=8.3, loc="upper center", ncol=3,
    columnspacing=1.6, handlelength=1.6, borderaxespad=0.0)

fig.savefig(f"{F}/F14_strata.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F14_strata.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F14_strata")
