"""Step 8f: v2 figures from the unified multi-year event database.
F3 (replaced): event-based recovery clocks + tau-ratio panel
F8 (new): cohort placebo/DiD ladder + Morakot direct trajectory + era check
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
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
R = json.load(open(f"{O}/results2.json"))
F2 = R["fits2"]; FX = R.get("fits2_extra", {}); TR = R["tau_ratios"]
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6)]
ev = post[post.src == "event"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def draw_fit(ax, fit, color, label, tmax=22):
    if not fit or "tau" not in fit:
        return
    tt = np.linspace(0.6, tmax, 150)
    ax.plot(tt, fit["A"] * np.exp(-tt / fit["tau"]), color=color, lw=1.3, label=label)


# ================= F3 v2 =================
fig, axes4 = plt.subplots(2, 2, figsize=(183 * MM, 140 * MM), constrained_layout=True)
axes = [axes4[0][0], axes4[0][1], axes4[1][0]]
axD = axes4[1][1]

ax = axes[0]
b = F2["event_dlst"]["bins"]
ax.errorbar(b["center"], b["mean"], yerr=1.96 * np.array(b["se"]), fmt="s", ms=2.6,
            color=WONG["black"], capsize=1.5, lw=0.7, zorder=3,
            label="Event landslides (all, binned)")
draw_fit(ax, F2["agent_typhoon_rain_dlst"], AGENT_COL["typhoon_rain"],
         f"Typhoon fit (τ={F2['agent_typhoon_rain_dlst']['tau']:.1f} yr)")
draw_fit(ax, F2["agent_rainfall_dlst"], AGENT_COL["rainfall"],
         f"Rainfall fit (τ={F2['agent_rainfall_dlst']['tau']:.1f} yr)")
mk = R["morakot_traj"]
ages = [v["age"] for k, v in mk.items() if v["n"] >= 500]
vals = [v["dlst"] for k, v in mk.items() if v["n"] >= 500]
ses = [v["dlst_se"] for k, v in mk.items() if v["n"] >= 500]
ax.errorbar(ages, vals, yerr=1.96 * np.array(ses), fmt="D", ms=3.4, capsize=2,
            color=WONG["purple"], lw=0.9, zorder=5,
            label="Morakot 2009 cohort (direct)")
eq = ev[(ev.agent == "earthquake") & ev.dlst.notna()]
for a_lo, a_hi in [(0.6, 2), (2, 4.5), (6, 14)]:
    g = eq[(eq.age >= a_lo) & (eq.age < a_hi)]
    if len(g) >= 8:
        ax.errorbar(g.age.mean(), g.dlst.mean(), yerr=1.96 * g.dlst.std() / np.sqrt(len(g)),
                    fmt="o", ms=3.4, color=AGENT_COL["earthquake"], capsize=2, lw=0.9, zorder=4)
ax.plot([], [], "o", ms=3.4, color=AGENT_COL["earthquake"], label="Earthquake landslides (binned)")
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Years since event (day-precision)")
ax.set_ylabel("ΔLST patch − control (°C)")
ax.legend(frameon=False, fontsize=8.3, loc="upper right")
ax.set_title("a  Thermal anomaly, 22 event years", loc="left", fontweight="bold", fontsize=9.5)
style(ax)

ax = axes[1]
bn = F2["event_dndvi"]["bins"]
ax.errorbar(bn["center"], bn["mean"], yerr=1.96 * np.array(bn["se"]), fmt="s", ms=2.6,
            color=WONG["black"], capsize=1.5, lw=0.7)
draw_fit(ax, F2["event_dndvi"], WONG["green"],
         f"All-event fit (τ={F2['event_dndvi']['tau']:.1f} yr)")
agesn = [v["age"] for k, v in mk.items() if v["n"] >= 500 and v.get("dndvi") is not None]
valsn = [v["dndvi"] for k, v in mk.items() if v["n"] >= 500 and v.get("dndvi") is not None]
ax.plot(agesn, valsn, "D", ms=3.4, color=WONG["purple"])
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Years since event")
ax.set_ylabel("ΔNDVI patch − control")
ax.legend(frameon=False, fontsize=8.3)
ax.set_title("b  Greenness anomaly", loc="left", fontweight="bold", fontsize=9.5)
style(ax)

ax = axes[2]
# yearly ΔLST of the FIXED pre-2020 population, by source type: within-cohort
# recovery over calendar epochs (composition held constant)
EP_MID = {"c20": 2020.62, "c21": 2021.62, "c22": 2022.62, "c23": 2023.62,
          "c24": 2024.62, "c25": 2025.62, "c26": 2026.55}
old = post[post.year <= 2019]
SERIES = [("typhoon_rain", WONG["vermillion"], "Typhoon landslide"),
          ("rainfall", WONG["orange"], "Rainfall landslide"),
          ("hansen", WONG["purple"], "Forest loss (Hansen)")]
for key, col, lab in SERIES:
    g = old[old.agent == key] if key != "hansen" else old[old.src == "hansen"]
    xs, ms_, ss = [], [], []
    for tag, mid in EP_MID.items():
        gg = g[(g.epoch == tag) & g.dlst.notna()]
        if len(gg) >= 30:
            xs.append(mid); ms_.append(gg.dlst.mean())
            ss.append(gg.dlst.std() / np.sqrt(len(gg)))
    xs, ms_, ss = np.array(xs), np.array(ms_), np.array(ss)
    o = np.argsort(xs)
    ax.errorbar(xs[o], ms_[o], yerr=1.96 * ss[o], fmt="-o", ms=2.6, lw=1.0,
                capsize=1.6, color=col, label=lab)
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Year")
ax.set_ylabel("ΔLST patch − control (°C)")
ax.set_xticks([2021, 2023, 2025])
ax.legend(frameon=False, fontsize=8.3, loc="upper right")
ax.set_title("c  Yearly ΔLST, pre-2020 cohorts", loc="left", fontweight="bold",
             fontsize=9.5)
style(ax)

# ---- (d) ΔLST-ΔNDVI hysteresis (paired age-binned means) ----
ax = axD
hy = R["hysteresis"]
hx = np.array([h["dndvi"] for h in hy])
hyv = np.array([h["dlst"] for h in hy])
ages_h = np.array([h["age"] for h in hy])
ax.plot([hx[0], 0], [hyv[0], 0], ls="--", color="grey", lw=0.8,
        label="Proportional recovery")
ax.errorbar(hx, hyv, xerr=1.96 * np.array([h["dndvi_se"] for h in hy]),
            yerr=1.96 * np.array([h["dlst_se"] for h in hy]),
            fmt="none", ecolor="#bbbbbb", elinewidth=0.6, capsize=0, zorder=2)
sc = ax.scatter(hx, hyv, c=ages_h, cmap="viridis", s=18, zorder=4,
                edgecolor="black", linewidths=0.3)
ax.plot(hx, hyv, color="#888888", lw=0.7, zorder=3)
cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Years since event")
ax.annotate("year 1", (hx[0], hyv[0]), xytext=(hx[0] + 0.02, hyv[0] + 0.15),
            fontsize=8.3)
ax.annotate("green again,\nstill warm", (hx[-3], hyv[-3]),
            xytext=(-0.205, 0.55), fontsize=8.3, color=WONG["vermillion"],
            arrowprops=dict(arrowstyle="->", lw=0.7, color=WONG["vermillion"]))
ax.axhline(0, color="grey", lw=0.5, ls=":")
ax.axvline(0, color="grey", lw=0.5, ls=":")
ax.set_xlabel("ΔNDVI patch − control")
ax.set_ylabel("ΔLST patch − control (°C)")
ax.legend(frameon=False, fontsize=8.3, loc="upper right")
ax.set_title("d  Joint trajectory: thermal hysteresis", loc="left",
             fontweight="bold", fontsize=9.5)
style(ax)
fig.savefig(f"{F}/F3_chronosequence.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F3_chronosequence.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F3 v2")

# ================= F8 =================
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 74 * MM), constrained_layout=True)

ax = axes[0]
dd = R["did_cohorts"]
ys = sorted(dd)
for i, y in enumerate(ys):
    v = dd[y]
    ax.plot([i - 0.15, i + 0.15], [v["pre"], v["post"]], "-", color="grey", lw=0.8)
    ax.plot(i - 0.15, v["pre"], "o", ms=4, color=WONG["skyblue"])
    ax.plot(i + 0.15, v["post"], "o", ms=4, color=WONG["vermillion"])
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xticks(range(len(ys)))
ax.set_xticklabels([str(y) for y in ys], fontsize=8.3, rotation=90)
ax.set_xlabel("Event cohort")
ax.set_ylabel("ΔLST patch − control (°C)")
ax.set_ylim(-0.5, 5.4)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=WONG["skyblue"], label="Pre-event"),
                   Line2D([], [], marker="o", ls="", color=WONG["vermillion"], label="Post-event (+1 yr)")],
          frameon=False, fontsize=8.3, loc="lower left")
ax.set_title("a  Pre- vs post-event ΔLST", loc="left", fontweight="bold",
             fontsize=9.5)
style(ax)

ax = axes[1]
ages = [v["age"] for v in mk.values()]
vals = [v["dlst"] for v in mk.values()]
ses = [v["dlst_se"] for v in mk.values()]
ns = [v["n"] for v in mk.values()]
ax.errorbar(ages, vals, yerr=1.96 * np.array(ses), fmt="D-", ms=3.6, capsize=2,
            lw=0.9, color=WONG["purple"], label="ΔLST")
ax2 = ax.twinx()
agesn = [v["age"] for v in mk.values() if v.get("dndvi") is not None]
valsn = [v["dndvi"] for v in mk.values() if v.get("dndvi") is not None]
ax2.plot(agesn, valsn, "s--", ms=3, lw=0.8, color=WONG["green"], label="ΔNDVI")
ax2.set_ylabel("ΔNDVI", color=WONG["green"])
ax2.tick_params(axis="y", colors=WONG["green"])
ax2.spines[["top"]].set_visible(False)
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Years since Typhoon Morakot (Aug 2009)")
ax.set_ylabel("ΔLST (°C)", color=WONG["purple"])
ax.set_title("b  Morakot cohort", loc="left", fontweight="bold",
             fontsize=9.5)
style(ax)

ax = axes[2]
for key, col, lab in [("era_annual_swcb_dlst", WONG["blue"], "2004-17 era (annual mapping)"),
                      ("era_event_ardswc_dlst", WONG["orange"], "2018-25 era (event mapping)")]:
    f_ = F2[key]
    if f_ and "bins" in f_:
        b_ = f_["bins"]
        ax.errorbar(b_["center"], b_["mean"], yerr=1.96 * np.array(b_["se"]), fmt="o",
                    ms=2.6, capsize=1.5, lw=0.7, color=col, label=lab)
        if "tau" in f_:
            tt = np.linspace(0.6, max(b_["center"]) + 1, 100)
            ax.plot(tt, f_["A"] * np.exp(-tt / f_["tau"]), color=col, lw=1.0, alpha=0.7)
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_xlabel("Years since event")
ax.set_ylabel("ΔLST patch − control (°C)")
ax.legend(frameon=False, fontsize=8.3)
ax.set_title("c  Inventory-era robustness", loc="left", fontweight="bold", fontsize=9.5)
style(ax)

fig.savefig(f"{F}/F8_multiyear.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F8_multiyear.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F8")
