"""Fig. 3 (v23): the buffering baseline, answering one question - how much
cooler is summer daytime LST for every extra metre of canopy in intact forest?
(a) LST relative to a 25 m canopy at the same elevation and terrain, against
canopy height, over the full 400k-pixel intact-forest sample (density) with the
binned median and 95% CI; (b) the same relationship within each elevation band,
showing it holds throughout; (c) mean |SHAP| attribution of the joint model.
The estimator-sensitivity diagnostic lives in the supplementary figure
FS1_gradient_estimators.
Inputs: outputs/results.json, outputs/gradient_check.json, data/buffering_sample.parquet
Output: figures/F2_buffering.{pdf,png}
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
from matplotlib.colors import LogNorm, LinearSegmentedColormap

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
REF_LO, REF_HI = 24.0, 26.0        # reference canopy height window (25 m)
R = json.load(open(f"{O}/results.json"))
G = json.load(open(f"{O}/gradient_check.json"))
BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500),
         (2500, 3600)]
COLS = [WONG["skyblue"], WONG["green"], WONG["yellow"], WONG["orange"],
        WONG["blue"], WONG["purple"]]
LAB = ["0–500 m", "500–1,000 m", "1,000–1,500 m", "1,500–2,000 m",
       "2,000–2,500 m", "2,500–3,600 m"]

df = pd.read_parquet(f"{D}/buffering_sample.parquet").dropna(
    subset=["lst", "chm", "elev", "slope", "northness", "cos_i"])
# The anomaly is built inside 250 m elevation slices, not inside the 500-1,100 m
# display bands: canopy height correlates with elevation within a wide band
# (r = -0.20 to +0.41), so a wide-band anomaly leaves a residual lapse rate that
# bends the tall-canopy tail (upwards where tall canopy sits lower in the band).
df["slice"] = (df.elev // 250).astype(int)
parts = []
for sl_, g in df.groupby("slice"):
    if len(g) < 3000:
        continue
    ref_n = ((g.chm >= REF_LO) & (g.chm < REF_HI)).sum()
    if ref_n < 100:
        continue
    X = np.column_stack([g.cos_i, g.slope, g.northness, np.ones(len(g))])
    beta, *_ = np.linalg.lstsq(X, g.lst, rcond=None)
    res = g.lst.values - X @ beta                      # terrain removed
    ref = np.median(res[(g.chm >= REF_LO) & (g.chm < REF_HI)])
    parts.append(g.assign(anom=res - ref))             # zero = 25 m canopy
d = pd.concat(parts)
d["band"] = pd.cut(d.elev, [b[0] for b in BANDS] + [BANDS[-1][1]],
                   labels=LAB, right=False)
d = d[d.band.notna()]
print("pixels:", len(d))


def binned(x, y, edges, nmin=80):
    c, m, lo_, hi_ = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = y[(x >= a) & (x < b)]
        if len(s) >= nmin:
            mu = s.mean(); se = s.std() / np.sqrt(len(s))
            c.append(0.5 * (a + b)); m.append(mu)
            lo_.append(mu - 1.96 * se); hi_.append(mu + 1.96 * se)
    return map(np.array, (c, m, lo_, hi_))


# (a) is one tall element: the density panel with its own derivative strip
# directly beneath, sharing the canopy-height axis. Both columns use the same
# row split, so (b) lines up with the density panel and (c) with the strip.
HR, HSP = [2.5, 1.0], 0.28
fig = plt.figure(figsize=(183 * MM, 136 * MM), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])
gsL = gs[0, 0].subgridspec(2, 1, height_ratios=HR, hspace=HSP)
gsR = gs[0, 1].subgridspec(2, 1, height_ratios=HR, hspace=HSP)
axA = fig.add_subplot(gsL[0])
axDer = fig.add_subplot(gsL[1], sharex=axA)     # lower strip of panel (a)
axB = fig.add_subplot(gsR[0]); axC = fig.add_subplot(gsR[1])

# ---------------- (a) pooled relationship with density ----------------
ax = axA
sel = (d.chm >= 8) & (d.chm <= 50) & (d.anom > -6.2) & (d.anom < 6.2)
GREY = LinearSegmentedColormap.from_list(
    "GreysLight", plt.get_cmap("Greys")(np.linspace(0.02, 0.52, 256)))
hb = ax.hexbin(d.chm[sel], d.anom[sel], gridsize=(96, 66), cmap=GREY,
               norm=LogNorm(vmin=1, vmax=4000), mincnt=1, linewidths=0,
               rasterized=True)
# horizontal colourbar inside the panel, under the legend: the panel keeps the
# full figure width instead of giving a column to the bar
cax = ax.inset_axes([0.075, 0.885, 0.20, 0.026])
cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
cb.ax.tick_params(labelsize=8.3, length=2, pad=1.5)
ax.text(0.295, 0.898, "intact-forest pixels per cell", transform=ax.transAxes,
        fontsize=8.3, ha="left", va="center", color="#333333")
e = np.arange(8, 50.5, 1.0)
c, m, lo_, hi_ = binned(d.chm[sel].values, d.anom[sel].values, e)
ax.fill_between(c, lo_, hi_, color=WONG["vermillion"], alpha=0.30, lw=0)
ax.plot(c, m, color=WONG["vermillion"], lw=1.8, zorder=5,
        label="binned mean")
ax.axhline(0, color="black", lw=0.7, ls="--", zorder=4)
ax.axvline(25, color="black", lw=0.7, ls=":", zorder=4)
ax.text(25.8, -6.1, "reference: 25 m canopy", fontsize=8.3,
        va="bottom", ha="left", color="#333333")
def local_slope(h, half=3.0):          # slope of the drawn binned curve
    return (np.interp(h + half, c, m) - np.interp(h - half, c, m)) / (2 * half) * 10
BAND_SL = {int(h): float(local_slope(h)) for h in (15, 20, 25, 30, 35, 40)}
json.dump(BAND_SL, open(f"{O}/baseline_slope_by_height.json", "w"), indent=1)
print("local slopes:", {k: round(v, 2) for k, v in BAND_SL.items()})
ax.set_xlim(8, 50); ax.set_ylim(-6.4, 6.4)
ax.tick_params(labelbottom=False)          # x-axis carried by the derivative panel
ax.set_ylabel("Summer LST relative to a 25 m canopy\nat the same elevation and terrain (°C)")
ax.legend(frameon=False, fontsize=8.3, loc="upper left", borderaxespad=0.4)
ax.set_title("a  Taller canopy, cooler surface", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

# ------------- lower strip of (a): derivative, shared x-axis ----------------
ax = axDer
hh = np.arange(11, 46, 0.5)
dv = np.array([local_slope(float(x)) for x in hh])
ax.plot(hh, dv, color=WONG["blue"], lw=1.6)
ax.fill_between(hh, dv, 0, color=WONG["blue"], alpha=0.10, lw=0)
ax.axhline(0, color="black", lw=0.6, ls="--")
ax.axvline(25, color="black", lw=0.7, ls=":")
# the 15/25/35 m slope values are quoted in the text (from the same JSON),
# so the curve carries no point labels
ax.set_xlabel("Canopy height (m)")
ax.set_ylabel("Marginal cooling\n(°C per extra 10 m)")
ax.set_xlim(8, 50)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- (b) same relationship inside every elevation band ----------
ax = axB
for lab, col in zip(LAB, COLS):
    g = d[(d.band == lab) & (d.chm >= 10) & (d.chm <= 48)]
    c, m, lo_, hi_ = binned(g.chm.values, g.anom.values, np.arange(10, 48.5, 2.0))
    if len(c) < 3:
        continue
    ax.plot(c, m, color=col, lw=1.4, label=lab)
    ax.fill_between(c, lo_, hi_, color=col, alpha=0.16, lw=0)
ax.axhline(0, color="black", lw=0.7, ls="--", zorder=0)
ax.axvline(25, color="black", lw=0.7, ls=":", zorder=0)
ax.set_xlim(10, 48); ax.set_ylim(-2.2, 4.2)
ax.set_xlabel("Canopy height (m)")
ax.set_ylabel("LST relative to a 25 m canopy (°C)")
ax.legend(frameon=False, fontsize=8.3, title="Elevation band", title_fontsize=8.3,
          loc="upper right", handlelength=1.4)
ax.set_title("b  Consistent at every elevation", loc="left",
             fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

# ---------------- (c) SHAP beeswarm ----------------
ax = axC
z = np.load(f"{O}/shap_values.npz", allow_pickle=True)
sv, Xd, feats = z["values"], z["data"], list(z["feats"])
names = {"elev": "Elevation", "chm": "Canopy height", "eastness": "Eastness",
         "slope": "Slope", "cos_i": "Illumination", "northness": "Northness"}
order = np.argsort([np.abs(sv[:, i]).mean() for i in range(len(feats))])
order = order[-4:]          # the four leading predictors; the rest sit at zero
rs = np.random.default_rng(3)
keep = rs.choice(len(sv), size=min(4000, len(sv)), replace=False)
for row, fi in enumerate(order):
    v = sv[keep, fi]
    x = Xd[keep, fi]
    lo_, hi_ = np.nanpercentile(x, [2, 98])
    cval = np.clip((x - lo_) / max(hi_ - lo_, 1e-9), 0, 1)
    # density-aware vertical jitter, the standard beeswarm look
    edges = np.linspace(v.min(), v.max(), 90)
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
    yj = np.zeros(len(v))
    for b in np.unique(idx):
        sel_ = np.where(idx == b)[0]
        k = len(sel_)
        spread = min(0.40, 0.028 * np.sqrt(k))
        yj[sel_] = np.linspace(-spread, spread, k) if k > 1 else 0.0
    ax.scatter(v, row + yj, c=cval, cmap="viridis", s=1.1, lw=0, alpha=0.65,
               rasterized=True, vmin=0, vmax=1)
ax.axvline(0, color="black", lw=0.6, ls=":")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([names[feats[i]] for i in order], fontsize=8.3)
ax.set_ylim(-0.6, len(order) - 0.4)
ax.set_xlabel("SHAP value for summer LST (°C)")
sm_ = plt.cm.ScalarMappable(cmap="viridis")
sm_.set_array([])
cb2 = fig.colorbar(sm_, ax=ax, pad=0.012, shrink=0.86, ticks=[0, 1])
cb2.ax.set_yticklabels(["low", "high"], fontsize=8.3)
cb2.set_label("Predictor value", fontsize=8.3)
ax.set_title(f"c  SHAP beeswarm (R² = {R['gbm_r2_test']:.2f})", loc="left",
             fontweight="bold", fontsize=9.5, pad=4)
ax.spines[["top", "right"]].set_visible(False)

# constrained layout sizes each panel around its own decorations, which leaves
# b and c a few millimetres off (a). Freeze the layout, then match the boxes.
fig.canvas.draw()
fig.set_layout_engine("none")
for src, dst in ((axA, axB), (axDer, axC)):
    ps, pd = src.get_position(), dst.get_position()
    dst.set_position([pd.x0, ps.y0, pd.width, ps.height])

fig.savefig(f"{F}/F2_buffering.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F2_buffering.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F2_buffering v11")
