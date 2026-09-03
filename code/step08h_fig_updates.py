"""Step 8h: regenerate F1 (panel c) and F5 (panels a, b) from the UNIFIED
2004-2025 event database (patches_raw2 / patches_deltas2 / results2.json),
so every figure derives from the single unified run.
F1 a/b unchanged in content (composite + CHM); c now shows all 15,679 event
patches coloured by agent. F5 a: 2024/2025 cohort placebo/post distributions
+ DiD from results2; b: matching quality from the unified patch table;
c: compositing check (version-independent QA, unchanged).
Previous renderings preserved in figures/versions/v2/.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, apply_mpl_standards, WONG, X0, Y1, RES, W, H

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4

AGENT_STYLE = {
    "typhoon_rain": (WONG["vermillion"], "Typhoon landslide"),
    "rainfall":     (WONG["orange"],     "Rainfall landslide"),
    "earthquake":   (WONG["blue"],       "Earthquake landslide"),
}


def save(fig, name):
    fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{F}/{name}.png", bbox_inches="tight", dpi=500)
    plt.close(fig)
    print("saved", name, flush=True)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)


def km_extent():
    return [X0 / 1000, (X0 + W * RES) / 1000, (Y1 - H * RES) / 1000, Y1 / 1000]


def ds(a, f=3):
    return a[::f, ::f]


wc = read_grid(f"{D}/worldcover2021.tif")
land = (wc != 0) & (wc != 80)
dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan

gy, gx = np.gradient(np.nan_to_num(dem), RES)
az, alt = np.radians(315), np.radians(45)
sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
hs = np.clip(np.sin(alt) * np.cos(sl) + np.cos(alt) * np.sin(sl) * np.cos(az - asp), 0, 1)
ext = km_extent()

# ============ F1 study area (panel c unified) ============
lst = read_grid(f"{D}/lst_c2425.tif"); lst[lst == -9999] = np.nan
chm = read_grid(f"{D}/chm_eth30.tif"); chm[chm == -9999] = np.nan
lossyear = read_grid(f"{D}/hansen_lossyear30.tif")
patches = pd.read_parquet(f"{D}/patches_raw2.parquet",
                          columns=["src", "agent", "row", "col", "n_px"])

fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 95 * MM), constrained_layout=True)
lst_m = np.where(land, lst, np.nan)
v1, v2 = np.nanpercentile(lst_m, [2, 98])
axes[0].imshow(ds(np.where(land, hs, np.nan)), cmap="Greys_r", extent=ext,
               vmin=0, vmax=1.3, interpolation="nearest")
im = axes[0].imshow(ds(lst_m), cmap="inferno", vmin=v1, vmax=v2, extent=ext,
                    alpha=0.82, interpolation="nearest")
cb = fig.colorbar(im, ax=axes[0], shrink=0.55, pad=0.02)
cb.set_label("Summer daytime LST 2024-25 (°C)")
axes[0].set_title("a  Landsat summer LST", loc="left", fontweight="bold", fontsize=8)

chm_m = np.where(land & np.isfinite(chm), chm, np.nan)
axes[1].imshow(ds(np.where(land, hs, np.nan)), cmap="Greys_r", extent=ext, vmin=0,
               vmax=1.3, interpolation="nearest")
im = axes[1].imshow(ds(chm_m), cmap="viridis", vmin=0, vmax=35, extent=ext, alpha=0.85,
                    interpolation="nearest")
cb = fig.colorbar(im, ax=axes[1], shrink=0.55, pad=0.02)
cb.set_label("Canopy height 2020 (m)")
axes[1].set_title("b  ETH 10 m canopy height", loc="left", fontweight="bold", fontsize=8)

axes[2].imshow(ds(np.where(land, hs, np.nan)), cmap="Greys_r", extent=ext, vmin=0,
               vmax=1.3, interpolation="nearest")
ly_m = np.where((lossyear > 0) & (lossyear <= 24), lossyear, np.nan)
im = axes[2].imshow(ds(ly_m), cmap="magma", vmin=1, vmax=24, extent=ext,
                    interpolation="nearest")
cb = fig.colorbar(im, ax=axes[2], shrink=0.55, pad=0.02, ticks=[1, 9, 15, 21, 24])
cb.ax.set_yticklabels(["2001", "2009", "2015", "2021", "2024"])
cb.set_label("Forest loss year")
ev = patches[patches.src == "event"]
for agent in ["typhoon_rain", "rainfall", "earthquake"]:
    s = ev[ev.agent == agent]
    axes[2].scatter(X0 / 1000 + s.col * RES / 1000, Y1 / 1000 - s.row * RES / 1000,
                    s=0.7, c=AGENT_STYLE[agent][0], lw=0, alpha=0.5)
fp = patches[patches.src == "fire"].nlargest(1, "n_px").iloc[0]
axes[2].scatter([X0 / 1000 + fp.col * RES / 1000], [Y1 / 1000 - fp.row * RES / 1000],
                marker="*", s=55, c=WONG["purple"], edgecolors="black", lw=0.4, zorder=5)
hnd1 = [Line2D([], [], marker="o", ls="", ms=3.2, color=AGENT_STYLE[a][0],
               label=AGENT_STYLE[a][1]) for a in AGENT_STYLE]
hnd1.append(Line2D([], [], marker="*", ls="", ms=7, color=WONG["purple"],
                   markeredgecolor="black", markeredgewidth=0.4, label="Huisun fire 2021"))
axes[2].legend(handles=hnd1, loc="lower right", frameon=False, handletextpad=0.1,
               fontsize=5.6)
axes[2].set_title("c  Disturbance events 2004-2025", loc="left", fontweight="bold",
                  fontsize=8)
for ax in axes:
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("TWD97 X (km)")
    style_ax(ax)
    ax.plot([ext[0] + 15, ext[0] + 65], [ext[2] + 22, ext[2] + 22], color="black", lw=1.4)
    ax.text(ext[0] + 40, ext[2] + 32, "50 km", ha="center", fontsize=6)
axes[0].set_ylabel("TWD97 Y (km)")
save(fig, "F1_study_area")

# ============ F5 validation (a, b unified) ============
R2 = json.load(open(f"{O}/results2.json"))
DDC = R2["did_cohorts"]
pdel = pd.read_parquet(f"{D}/patches_deltas2.parquet")
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), constrained_layout=True)
ax = axes[0]
p24 = pdel[(pdel.src == "event") & (pdel.year == 2024) & (pdel.pre_tag == "c23")]
p25 = pdel[(pdel.src == "event") & (pdel.year == 2025) & (pdel.pre_tag == "c24")]
data, labels, cols = [], [], []
for series, lab, col in [
    (p24.dlst_c23.dropna(), "2024 events\npre (placebo)", WONG["skyblue"]),
    (p24.dlst_c25.dropna(), "2024 events\npost (+1 yr)", WONG["vermillion"]),
    (p25.dlst_c24.dropna(), "2025 events\npre (placebo)", WONG["skyblue"]),
    (p25.dlst_c26.dropna(), "2025 events\npost (+1 yr)", WONG["vermillion"])]:
    if len(series) > 10:
        data.append(series.values); labels.append(lab); cols.append(col)
bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=1), widths=0.55)
for patch, col in zip(bp["boxes"], cols):
    patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor("black")
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_ylabel("ΔLST patch − control (°C)")
txt = [f"{y} DiD: +{DDC[y]['did']:.2f}±{1.96*DDC[y]['se']:.2f} °C (n={DDC[y]['n']})"
       for y in ["2024", "2025"] if y in DDC]
ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top",
        fontsize=5.6, color=WONG["black"])
ax.set_title("a  Pre-warm bias & DiD effect", loc="left", fontweight="bold", fontsize=8)
ax.tick_params(axis="x", labelsize=5.4)
style_ax(ax)

ax = axes[1]
dq = pdel[pdel.n_ctrl >= 30]
ax.hist(dq.ctrl_elev - dq.elev, bins=np.linspace(-250, 250, 51), color=WONG["blue"],
        alpha=0.75, label="Δ elevation (m)")
ax.set_xlabel("Control − patch elevation (m)")
ax.set_ylabel("Patches")
ax2 = ax.twiny()
ax2.hist(dq.ctrl_slope - dq.slope, bins=np.linspace(-15, 15, 51), histtype="step",
         color=WONG["orange"], lw=1.1, label="Δ slope (°)")
ax2.set_xlabel("Control − patch slope (°)", color=WONG["orange"])
ax2.tick_params(axis="x", colors=WONG["orange"])
ax.set_title("b  Control matching quality", loc="left", fontweight="bold", fontsize=8)
style_ax(ax)

ax = axes[2]
mv = json.load(open(f"{O}/median_validation.json"))
names = list(mv)
x = np.arange(len(names))
ax.bar(x - 0.18, [mv[n]["mean_minus_median_bias"] for n in names], width=0.36,
       color=WONG["blue"], label="mean − median bias")
ax.bar(x + 0.18, [mv[n]["mean_minus_median_rmse"] for n in names], width=0.36,
       color=WONG["orange"], label="RMSE")
ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=5.6)
ax.axhline(0, color="grey", lw=0.5)
ax.set_ylabel("LST difference (°C)")
ax.legend(frameon=False, fontsize=5.6)
ax.set_title("c  Compositing method check", loc="left", fontweight="bold", fontsize=8)
style_ax(ax)
save(fig, "F5_validation")
print("STEP8H COMPLETE")
