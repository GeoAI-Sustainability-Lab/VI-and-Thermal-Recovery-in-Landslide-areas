"""Step 8b: figures (Wong palette, sans-serif, editable-text PDF + 450dpi PNG).
All numbers plotted are computed from the real pipeline outputs — nothing simulated.
F1 study area & data | F2 buffering | F3 chronosequence | F4 case trajectories
F5 validation/QA | F6 Sentinel-2 case chips
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
from matplotlib import colors as mcolors

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, apply_mpl_standards, WONG, X0, Y1, RES, W, H

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
os.makedirs(F, exist_ok=True)
MM = 1 / 25.4

AGENT_STYLE = {
    "hansen_loss":  (WONG["black"],      "Annual-map loss (2001-23)"),
    "typhoon_rain": (WONG["vermillion"], "Typhoon landslide"),
    "rainfall":     (WONG["orange"],     "Rainfall landslide"),
    "earthquake":   (WONG["blue"],       "Earthquake landslide"),
    "fire":         (WONG["purple"],     "Fire (Huisun 2021)"),
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


# ---------------- shared rasters ----------------
wc = read_grid(f"{D}/worldcover2021.tif")
land = (wc != 0) & (wc != 80)
dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan
ls, lsy = None, None

# hillshade
gy, gx = np.gradient(np.nan_to_num(dem), RES)
az, alt = np.radians(315), np.radians(45)
sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
hs = (np.sin(alt) * np.cos(sl) + np.cos(alt) * np.sin(sl) * np.cos(az - asp))
hs = np.clip(hs, 0, 1)

ext = km_extent()

# ============ F1 study area ============
try:
    lst = read_grid(f"{D}/lst_c2425.tif"); lst[lst == -9999] = np.nan
    chm = read_grid(f"{D}/chm_eth30.tif"); chm[chm == -9999] = np.nan
    lossyear = read_grid(f"{D}/hansen_lossyear30.tif")
    patches = pd.read_parquet(f"{D}/patches_raw.parquet")
    ev = {int(k): v for k, v in json.load(open(f"{D}/event_codes.json")).items()}

    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 95 * MM), constrained_layout=True)
    lst_m = np.where(land, lst, np.nan)
    v1, v2 = np.nanpercentile(lst_m, [2, 98])
    im0 = axes[0].imshow(ds(np.where(land, hs, np.nan)), cmap="Greys_r", extent=ext,
                         vmin=0, vmax=1.3, interpolation="nearest")
    im = axes[0].imshow(ds(lst_m), cmap="inferno", vmin=v1, vmax=v2, extent=ext,
                        alpha=0.82, interpolation="nearest")
    cb = fig.colorbar(im, ax=axes[0], shrink=0.55, pad=0.02)
    cb.set_label("Summer daytime LST 2024-25 (°C)")
    axes[0].set_title("a  Landsat summer LST", loc="left", fontweight="bold", fontsize=8)

    chm_m = np.where(land & np.isfinite(chm), chm, np.nan)
    axes[1].imshow(ds(np.where(land, hs, np.nan)), cmap="Greys_r", extent=ext, vmin=0, vmax=1.3,
                   interpolation="nearest")
    im = axes[1].imshow(ds(chm_m), cmap="viridis", vmin=0, vmax=35, extent=ext, alpha=0.85,
                        interpolation="nearest")
    cb = fig.colorbar(im, ax=axes[1], shrink=0.55, pad=0.02)
    cb.set_label("Canopy height 2020 (m)")
    axes[1].set_title("b  ETH 10 m canopy height", loc="left", fontweight="bold", fontsize=8)

    axes[2].imshow(ds(np.where(land, hs, np.nan)), cmap="Greys_r", extent=ext, vmin=0, vmax=1.3,
                   interpolation="nearest")
    ly_m = np.where((lossyear > 0) & (lossyear <= 24), lossyear, np.nan)
    im = axes[2].imshow(ds(ly_m), cmap="magma", vmin=1, vmax=24, extent=ext,
                        interpolation="nearest")
    cb = fig.colorbar(im, ax=axes[2], shrink=0.55, pad=0.02,
                      ticks=[1, 9, 15, 21, 24])
    cb.ax.set_yticklabels(["2001", "2009", "2015", "2021", "2024"])
    cb.set_label("Forest loss year")
    ard = patches[patches.src == "ardswc"]
    ard_ag = ard.event_code.map(lambda c: ev[int(c)]["agent"])
    for agent in ["typhoon_rain", "rainfall", "earthquake"]:
        s = ard[ard_ag == agent]
        axes[2].scatter(X0 / 1000 + s.col * RES / 1000, Y1 / 1000 - s.row * RES / 1000,
                        s=1.2, c=AGENT_STYLE[agent][0], lw=0, alpha=0.65,
                        label=AGENT_STYLE[agent][1])
    fp = patches[patches.src == "fire2021"].nlargest(1, "n_px").iloc[0]
    axes[2].scatter([X0 / 1000 + fp.col * RES / 1000], [Y1 / 1000 - fp.row * RES / 1000],
                    marker="*", s=55, c=WONG["purple"], edgecolors="black", lw=0.4,
                    zorder=5)
    hnd1 = [Line2D([], [], marker="o", ls="", ms=3.2, color=AGENT_STYLE[a][0],
                   label=AGENT_STYLE[a][1]) for a in ["typhoon_rain", "rainfall", "earthquake"]]
    hnd1.append(Line2D([], [], marker="*", ls="", ms=7, color=WONG["purple"],
                       markeredgecolor="black", markeredgewidth=0.4, label="Huisun fire 2021"))
    axes[2].legend(handles=hnd1, loc="lower right", frameon=False, handletextpad=0.1,
                   fontsize=5.6)
    axes[2].set_title("c  Disturbance events", loc="left", fontweight="bold", fontsize=8)

    for ax in axes:
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("TWD97 X (km)")
        style_ax(ax)
        ax.plot([ext[0] + 15, ext[0] + 65], [ext[2] + 22, ext[2] + 22], color="black", lw=1.4)
        ax.text(ext[0] + 40, ext[2] + 32, "50 km", ha="center", fontsize=6)
    axes[0].set_ylabel("TWD97 Y (km)")
    save(fig, "F1_study_area")
except Exception as e:
    print("F1 fail:", repr(e))

# ============ F2 buffering ============
try:
    R = json.load(open(f"{O}/results.json"))
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM), constrained_layout=True)
    band_cols = [WONG["skyblue"], WONG["green"], WONG["orange"], WONG["vermillion"],
                 WONG["blue"], WONG["purple"]]
    ax = axes[0]
    for i, b in enumerate(R["band_stats"]):
        x = np.array(b["bin_centers"]); m = np.array(b["bin_median"], dtype=float)
        q1 = np.array(b["bin_q25"], dtype=float); q3 = np.array(b["bin_q75"], dtype=float)
        ok = np.isfinite(m)
        ax.fill_between(x[ok], q1[ok], q3[ok], color=band_cols[i], alpha=0.15, lw=0)
        ax.plot(x[ok], m[ok], color=band_cols[i], lw=1.2,
                label=f"{b['band']} m ({b['slope_per_m']*10:+.2f} °C/10 m)")
    ax.set_xlabel("Canopy height (m)"); ax.set_ylabel("Terrain-adjusted LST (°C)")
    ax.legend(frameon=False, fontsize=5.2, loc="upper right", title="Elevation band",
              title_fontsize=5.6)
    ax.set_title("a  LST vs canopy height by elevation", loc="left", fontweight="bold", fontsize=8)
    style_ax(ax)

    z = np.load(f"{O}/shap_values.npz", allow_pickle=True)
    sv, Xd, feats = z["values"], z["data"], list(z["feats"])
    ic, ie = feats.index("chm"), feats.index("elev")
    ax = axes[1]
    sc = ax.scatter(Xd[:, ic], sv[:, ic], c=Xd[:, ie], cmap="viridis", s=1.5, lw=0,
                    alpha=0.6, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02); cb.set_label("Elevation (m)")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Canopy height (m)"); ax.set_ylabel("SHAP value for LST (°C)")
    ax.set_title("b  SHAP dependence: canopy height", loc="left", fontweight="bold", fontsize=8)
    style_ax(ax)

    ax = axes[2]
    ma = R["shap_mean_abs"]
    labs = {"chm": "Canopy height", "elev": "Elevation", "slope": "Slope",
            "cos_i": "Illumination cos(i)", "northness": "Northness", "eastness": "Eastness"}
    items = sorted(ma.items(), key=lambda kv: kv[1])
    ax.barh([labs[k] for k, v in items], [v for k, v in items], color=WONG["blue"], height=0.6)
    ax.set_xlabel("Mean |SHAP| (°C)")
    ax.set_title(f"c  Attribution (GBM R²={R['gbm_r2_test']:.2f})", loc="left",
                 fontweight="bold", fontsize=8)
    style_ax(ax)
    save(fig, "F2_buffering")
except Exception as e:
    print("F2 fail:", repr(e))

# ============ F3 chronosequence ============
try:
    R = json.load(open(f"{O}/results.json"))
    long = pd.read_parquet(f"{O}/chronosequence_long.parquet")
    q = long[long.quality]
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM), constrained_layout=True)

    def agent_points(ax, val):
        for agent, (col, lab) in AGENT_STYLE.items():
            g = q[(q.agent == agent) & q[val].notna() & (q.age > 0.3)]
            if agent == "hansen_loss" or len(g) < 5:
                continue
            for a_lo, a_hi in [(0.3, 1.5), (1.5, 2.5)]:
                gg = g[(g.age >= a_lo) & (g.age < a_hi)]
                if len(gg) < 5:
                    continue
                m = gg[val].mean(); se = gg[val].std() / np.sqrt(len(gg))
                ax.errorbar(gg.age.mean(), m, yerr=1.96 * se, fmt="o", ms=3.5,
                            color=col, capsize=2, lw=0.9, zorder=5)

    ax = axes[0]
    full = R["recovery_fits"]["hansen_dlst"]
    fit = R["recovery_fits"].get("recent2015_dlst") or full
    b = full["bins"]
    ax.errorbar(b["center"], b["mean"], yerr=1.96 * np.array(b["se"]), fmt="s", ms=2.6,
                color=WONG["black"], capsize=1.5, lw=0.7, label="Annual-map loss (all cohorts)")
    if "tau" in fit:
        tt = np.linspace(0.4, 24, 120)
        ax.plot(tt, fit["A"] * np.exp(-tt / fit["tau"]), color=WONG["vermillion"], lw=1.3,
                label=f"2015-23 cohort fit: τ={fit['tau']:.1f}±{fit['tau_se']:.1f} yr")
    bc = np.array(b["center"]); bm = np.array(b["mean"])
    bump = (bc >= 11) & (bc <= 16)
    if bump.any():
        ax.annotate("2009 Morakot\nmega-scar cohort", (bc[bump][np.argmax(bm[bump])],
                    float(np.max(bm[bump]))), textcoords="offset points", xytext=(6, 8),
                    fontsize=5.4, color=WONG["blue"],
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=WONG["blue"]))
    agent_points(ax, "dlst")
    fq = q[(q.agent == "fire") & q.dlst.notna()]
    ax.plot(fq.age, fq.dlst, marker="*", ls="", ms=7, color=WONG["purple"],
            markeredgecolor="black", markeredgewidth=0.4, zorder=6)
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Years since disturbance"); ax.set_ylabel("ΔLST patch − control (°C)")
    hnd = [Line2D([], [], marker="o", ls="", ms=3.5, color=c, label=l)
           for a, (c, l) in AGENT_STYLE.items() if a not in ("hansen_loss", "fire")]
    hnd.insert(0, Line2D([], [], marker="s", ls="", ms=3, color=WONG["black"],
                         label="Annual-map loss"))
    hnd.append(Line2D([], [], marker="*", ls="", ms=7, color=WONG["purple"],
                      markeredgecolor="black", markeredgewidth=0.4,
                      label="Huisun fire (single case)"))
    if "tau" in fit:
        hnd.append(Line2D([], [], ls="-", lw=1.3, color=WONG["vermillion"],
                          label=f"2015-23 cohort fit (τ={fit['tau']:.1f} yr)"))
    ax.legend(handles=hnd, frameon=False, fontsize=5.2, loc="upper right")
    ax.set_title("a  Thermal anomaly vs time", loc="left", fontweight="bold", fontsize=8)
    style_ax(ax)

    ax = axes[1]
    fitn = R["recovery_fits"].get("recent2015_dndvi") or R["recovery_fits"]["hansen_dndvi"]
    bn = R["recovery_fits"]["hansen_dndvi"]["bins"]
    ax.errorbar(bn["center"], bn["mean"], yerr=1.96 * np.array(bn["se"]), fmt="s", ms=2.6,
                color=WONG["black"], capsize=1.5, lw=0.7)
    if "tau" in fitn:
        tt = np.linspace(0.4, 24, 120)
        ax.plot(tt, fitn["A"] * np.exp(-tt / fitn["tau"]), color=WONG["green"], lw=1.3,
                label=f"fit: τ={fitn['tau']:.1f}±{fitn['tau_se']:.1f} yr")
    agent_points(ax, "dndvi")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("Years since disturbance"); ax.set_ylabel("ΔNDVI patch − control")
    ax.legend(frameon=False, fontsize=5.6)
    ax.set_title("b  Greenness anomaly vs time", loc="left", fontweight="bold", fontsize=8)
    style_ax(ax)

    ax = axes[2]
    # paired e-folding times by elevation band (annual-map loss patches, all cohorts)
    TAU_CAP = 59.0    # fit upper bound: values at/above are unconstrained
    pairs = []
    for lab_, key in [("<1000 m", "elev_low"), ("1000-2000 m", "elev_mid"),
                      (">2000 m", "elev_high")]:
        fl = R["recovery_fits"].get(f"{key}_dlst") or {}
        fn_ = R["recovery_fits"].get(f"{key}_dndvi") or {}
        if "tau" in fl and "tau" in fn_:
            pairs.append((lab_, fl["tau"], fl["tau_se"], fn_["tau"], fn_["tau_se"]))
    xb = np.arange(len(pairs))
    for i, (lab_, tl, tle, tn, tne) in enumerate(pairs):
        cap_l = tl >= TAU_CAP
        ax.bar(i - 0.17, min(tl, TAU_CAP), width=0.34, color=WONG["vermillion"],
               alpha=0.85, hatch="//" if cap_l else None, edgecolor="black", lw=0.4)
        ax.bar(i + 0.17, tn, width=0.34, color=WONG["green"], alpha=0.85,
               edgecolor="black", lw=0.4)
        if not cap_l:
            ax.errorbar(i - 0.17, tl, yerr=1.96 * tle, fmt="none", ecolor="black",
                        lw=0.7, capsize=2)
        else:
            ax.annotate("≥40 yr\n(unconstrained)", (i - 0.17, TAU_CAP), ha="center",
                        va="bottom", fontsize=5)
        ax.errorbar(i + 0.17, tn, yerr=1.96 * tne, fmt="none", ecolor="black",
                    lw=0.7, capsize=2)
        ax.annotate(f"×{tl/tn:.1f}" if not cap_l else "", (i, max(tl, tn) * 0.5),
                    ha="center", fontsize=6, fontweight="bold")
    ax.set_xticks(xb); ax.set_xticklabels([p[0] for p in pairs])
    ax.set_xlabel("Elevation band")
    ax.set_ylabel("Recovery e-folding time τ (yr)")
    ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=6, color=WONG["vermillion"],
                              label="Thermal (ΔLST)"),
                       Line2D([], [], marker="s", ls="", ms=6, color=WONG["green"],
                              label="Greenness (ΔNDVI)")],
              frameon=False, fontsize=6, loc="upper left")
    ax.set_title("c  Thermal recovery lags greenness", loc="left", fontweight="bold", fontsize=8)
    style_ax(ax)
    save(fig, "F3_chronosequence")
except Exception as e:
    print("F3 fail:", repr(e))

# ============ F4 case trajectories ============
try:
    tr = pd.read_parquet(f"{D}/case_trajectories.parquet")
    tr["dlst"] = tr.lst_p - tr.lst_c
    tr["dndvi"] = tr.ndvi_p - tr.ndvi_c
    CASE_LAB = {
        "huisun_fire_2021": "Huisun fire, May 2021",
        "hualien_eq_2024": "Hualien EQ slide, Apr 2024",
        "gaemi_typhoon_2024": "Typhoon Gaemi slide, Jul 2024",
        "hansen_2004": "2004 loss patch", "hansen_2009_morakot": "2009 loss (Morakot)",
        "hansen_2015": "2015 loss patch"}
    cases = [c for c in CASE_LAB if c in set(tr.case)]
    n = len(cases)
    ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(183 * MM, 55 * nrow * MM),
                             constrained_layout=True, squeeze=False)
    for k, case in enumerate(cases):
        ax = axes[k // ncol][k % ncol]
        d = tr[tr.case == case]
        ann = d.groupby("year").agg(dlst=("dlst", "mean"), dndvi=("dndvi", "mean"),
                                    nn=("dlst", "count"))
        ann = ann[ann.nn >= 2]
        t_ev = d.t_event.iloc[0]
        ax.axvline(t_ev, color="grey", lw=0.8, ls="--")
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.plot(ann.index, ann.dlst, "-o", ms=2.2, lw=0.9, color=WONG["vermillion"],
                label="ΔLST")
        ax.set_ylabel("ΔLST (°C)", color=WONG["vermillion"])
        ax.tick_params(axis="y", colors=WONG["vermillion"])
        ax2 = ax.twinx()
        ax2.plot(ann.index, ann.dndvi, "-s", ms=2.0, lw=0.9, color=WONG["green"],
                 label="ΔNDVI")
        ax2.set_ylabel("ΔNDVI", color=WONG["green"])
        ax2.tick_params(axis="y", colors=WONG["green"])
        ax2.spines[["top"]].set_visible(False)
        pre = ann[ann.index < int(t_ev)]
        if len(pre) >= 3:
            ax.axhspan(pre.dlst.mean() - pre.dlst.std(), pre.dlst.mean() + pre.dlst.std(),
                       color=WONG["skyblue"], alpha=0.18, lw=0)
        ax.set_title(f"{chr(97+k)}  {CASE_LAB[case]}\n{d.area_ha.iloc[0]:.1f} ha · "
                     f"{d.elev.iloc[0]:.0f} m", loc="left", fontweight="bold", fontsize=6.5)
        ax.set_xlim(1999.5, 2027); ax.spines[["top"]].set_visible(False)
        ax.set_xlabel("Year")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    save(fig, "F4_trajectories")
except Exception as e:
    print("F4 fail:", repr(e))

# ============ F5 validation ============
try:
    R = json.load(open(f"{O}/results.json"))
    pdel = pd.read_parquet(f"{D}/patches_deltas.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), constrained_layout=True)
    ax = axes[0]
    data, labels, cols = [], [], []
    p24 = pdel[(pdel.src == "ardswc") & (pdel.year == 2024)]
    p25 = pdel[(pdel.src == "ardswc") & (pdel.year == 2025)]
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
    did = R.get("did", {})
    d24, d25 = did.get("events2024_lst"), did.get("events2025_lst")
    txt = []
    if d24:
        txt.append(f"2024 DiD: +{d24['mean']:.2f}±{1.96*d24['se']:.2f} °C (n={d24['n']})")
    if d25:
        txt.append(f"2025 DiD: +{d25['mean']:.2f}±{1.96*d25['se']:.2f} °C (n={d25['n']})")
    if txt:
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
    try:
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
    except FileNotFoundError:
        ax.text(0.5, 0.5, "median validation pending", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("c  Compositing method check", loc="left", fontweight="bold", fontsize=8)
    style_ax(ax)
    save(fig, "F5_validation")
except Exception as e:
    print("F5 fail:", repr(e))

# ============ F6 Sentinel-2 chips ============
try:
    z = np.load(f"{O}/s2_chips.npz", allow_pickle=True)
    CASES = ["huisun_fire_2021", "hualien_eq_2024", "gaemi_typhoon_2024"]
    TITLES = {"huisun_fire_2021": "Huisun fire 2021", "hualien_eq_2024": "0403 Hualien EQ 2024",
              "gaemi_typhoon_2024": "Typhoon Gaemi 2024"}
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling
    fig, axes = plt.subplots(2, 3, figsize=(183 * MM, 122 * MM), constrained_layout=True)
    for j, case in enumerate(CASES):
        geom = z[f"{case}_geom"]
        x0c, y1c = geom[0], geom[1]
        tr10 = from_origin(x0c, y1c, 10, 10)
        with rasterio.open(f"{D}/events30.tif") as src:
            with WarpedVRT(src, crs="EPSG:3826", transform=tr10, width=300, height=300,
                           resampling=Resampling.nearest) as vrt:
                evw = vrt.read(1)
        with rasterio.open(f"{D}/hansen_lossyear30.tif") as src:
            with WarpedVRT(src, crs="EPSG:3826", transform=tr10, width=300, height=300,
                           resampling=Resampling.nearest) as vrt:
                lyw = vrt.read(1)
        overlay = (evw > 0) if case != "hansen" else (lyw > 0)
        for i, phase in enumerate(["pre", "post"]):
            ax = axes[i][j]
            key = f"{case}_{phase}"
            if key in z:
                rgb = np.moveaxis(z[key], 0, -1)
                ax.imshow(rgb, interpolation="nearest")
                if overlay.any():
                    ax.contour(overlay.astype(float), levels=[0.5],
                               colors=[WONG["yellow"]], linewidths=0.7)
                meta = z[f"{key}_meta"]
                ax.set_title(f"{chr(97+i*3+j)}  {TITLES[case]} — {phase} ({meta[1]})",
                             loc="left", fontweight="bold", fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[1][j].plot([15, 115], [285, 285], color="white", lw=1.6)
        axes[1][j].text(65, 274, "1 km", color="white", ha="center", fontsize=6)
    save(fig, "F6_s2_cases")
except Exception as e:
    print("F6 fail:", repr(e))

print("STEP8B COMPLETE")
