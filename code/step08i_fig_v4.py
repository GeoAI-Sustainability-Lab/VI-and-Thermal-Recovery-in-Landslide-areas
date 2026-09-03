"""Step 8i (v4 figures):
F1 study area redrawn in WGS84 lon/lat axes (computation stays TWD97):
ocean cut to white with the coastline kept, grey hillshade only on land with
a legend entry, event patches from the unified 2004-2025 database.
F2 buffering: panel-a legend moved clear of the curves.
F5 validation: panel-b slope histogram made legible, panel-c legend moved.
Also aggregates event-group statistics into outputs/results2.json
("event_groups") for the event-group table.
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
from matplotlib.patches import Patch
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from pyproj import Transformer

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, apply_mpl_standards, WONG, X0, Y1, RES, W, H

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
DS = 3

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


# ---------------- load & warp to WGS84 ----------------
wc = read_grid(f"{D}/worldcover2021.tif")
land = ((wc != 0) & (wc != 80)).astype(np.float32)
dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan
lst = read_grid(f"{D}/lst_c2425.tif"); lst[lst == -9999] = np.nan
chm = read_grid(f"{D}/chm_eth30.tif"); chm[chm == -9999] = np.nan
lossyear = read_grid(f"{D}/hansen_lossyear30.tif").astype(np.float32)

gy, gx = np.gradient(np.nan_to_num(dem), RES)
az, alt = np.radians(315), np.radians(45)
sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
hs = np.clip(np.sin(alt) * np.cos(sl) + np.cos(alt) * np.sin(sl) * np.cos(az - asp),
             0, 1).astype(np.float32)

src_tr = from_origin(X0, Y1, RES * DS, RES * DS)
lon0, lat0, lon1, lat1 = transform_bounds(3826, 4326, X0, Y1 - H * RES, X0 + W * RES, Y1)
dres = 0.001
dW = int((lon1 - lon0) / dres); dH = int((lat1 - lat0) / dres)
dst_tr = from_origin(lon0, lat1, dres, dres)
ext = [lon0, lon1, lat0, lat1]


def warp(a, categorical=False, fill=np.nan):
    srcd = np.ascontiguousarray(a[::DS, ::DS])
    out = np.full((dH, dW), fill, np.float32)
    reproject(srcd, out, src_transform=src_tr, src_crs="EPSG:3826",
              dst_transform=dst_tr, dst_crs="EPSG:4326",
              resampling=Resampling.nearest if categorical else Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=fill)
    return out


from scipy.ndimage import binary_fill_holes, binary_closing
land_w = warp(land, categorical=True, fill=0.0)
hs_w = warp(hs)
lst_w = warp(lst)
chm_w = warp(chm)
ly_w = warp(lossyear, categorical=True, fill=0.0)
island = binary_fill_holes(binary_closing(land_w > 0.5, iterations=2))
onland = island
print("warped", dW, dH, flush=True)

patches = pd.read_parquet(f"{D}/patches_raw2.parquet",
                          columns=["src", "agent", "row", "col", "n_px"])
tf = Transformer.from_crs(3826, 4326, always_xy=True)
px = X0 + patches.col.values * RES
py = Y1 - patches.row.values * RES
plon, plat = tf.transform(px, py)
patches["lon"], patches["lat"] = plon, plat

# ---------------- F1 study area (WGS84) ----------------
GREYBG = np.where(onland, hs_w, np.nan)
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 92 * MM), constrained_layout=True)
lstm = np.where(onland, lst_w, np.nan)
v1, v2 = np.nanpercentile(lstm, [2, 98])
axes[0].imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=0, vmax=1.3,
               interpolation="nearest")
im = axes[0].imshow(lstm, cmap="inferno", vmin=v1, vmax=v2, extent=ext, alpha=0.82,
                    interpolation="nearest")
cb = fig.colorbar(im, ax=axes[0], shrink=0.55, pad=0.02)
cb.set_label("Summer daytime LST 2024-25 (°C)")
axes[0].set_title("a  Landsat summer LST", loc="left", fontweight="bold", fontsize=8)
axes[0].legend(handles=[Patch(facecolor="#b8b8b8", edgecolor="none",
                              label="terrain, not analysed")],
               loc="lower right", frameon=False, fontsize=5.4, handlelength=1.2)

axes[1].imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=0, vmax=1.3,
               interpolation="nearest")
chmm = np.where(onland, chm_w, np.nan)
im = axes[1].imshow(chmm, cmap="viridis", vmin=0, vmax=35, extent=ext, alpha=0.85,
                    interpolation="nearest")
cb = fig.colorbar(im, ax=axes[1], shrink=0.55, pad=0.02)
cb.set_label("Canopy height 2020 (m)")
axes[1].set_title("b  Canopy height (ETH 10 m)", loc="left", fontweight="bold",
                  fontsize=8)

axes[2].imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=0, vmax=1.3,
               interpolation="nearest")
lym = np.where((ly_w > 0) & (ly_w <= 24) & onland, ly_w, np.nan)
im = axes[2].imshow(lym, cmap="magma", vmin=1, vmax=24, extent=ext,
                    interpolation="nearest")
cb = fig.colorbar(im, ax=axes[2], shrink=0.55, pad=0.02, ticks=[1, 9, 15, 21, 24])
cb.ax.set_yticklabels(["2001", "2009", "2015", "2021", "2024"])
cb.set_label("Forest loss year")
ev = patches[patches.src == "event"]
for agent in AGENT_STYLE:
    s = ev[ev.agent == agent]
    axes[2].scatter(s.lon, s.lat, s=0.7, c=AGENT_STYLE[agent][0], lw=0, alpha=0.5)
fp = patches[patches.src == "fire"].nlargest(1, "n_px").iloc[0]
axes[2].scatter([fp.lon], [fp.lat], marker="*", s=55, c=WONG["purple"],
                edgecolors="black", lw=0.4, zorder=5)
hnd = [Line2D([], [], marker="o", ls="", ms=3.2, color=AGENT_STYLE[a][0],
              label=AGENT_STYLE[a][1]) for a in AGENT_STYLE]
hnd.append(Line2D([], [], marker="*", ls="", ms=7, color=WONG["purple"],
                  markeredgecolor="black", markeredgewidth=0.4,
                  label="Huisun fire 2021"))
axes[2].legend(handles=hnd, loc="upper left", frameon=False, handletextpad=0.1,
               fontsize=5.4)
axes[2].set_title("c  Disturbance events 2004-2025", loc="left", fontweight="bold",
                  fontsize=8)

kmdeg = 50.0 / (111.320 * np.cos(np.radians(23.6)))
for ax in axes:
    ax.contour(island.astype(float), levels=[0.5], colors="black", linewidths=0.4,
               extent=ext, origin="upper")
    ax.set_xlim(119.9, 122.1); ax.set_ylim(21.85, 25.4)
    ax.set_xlabel("Longitude (°E)")
    style_ax(ax)
    ax.plot([120.0, 120.0 + kmdeg], [22.0, 22.0], color="black", lw=1.4)
    ax.text(120.0 + kmdeg / 2, 22.06, "50 km", ha="center", fontsize=6)
    ax.set_aspect(1.0 / np.cos(np.radians(23.6)))
axes[0].set_ylabel("Latitude (°N)")
save(fig, "F1_study_area")

# ---------------- F2 buffering: legend fix ----------------
R = json.load(open(f"{O}/results.json"))
bs = R["band_stats"]
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM), constrained_layout=True)
ax = axes[0]
cols = [WONG["skyblue"], WONG["green"], WONG["yellow"], WONG["orange"],
        WONG["blue"], WONG["purple"]]
for b, c in zip(bs, cols):
    bc = np.array(b["bin_centers"]); med = np.array(b["bin_median"], float)
    q25 = np.array(b["bin_q25"], float); q75 = np.array(b["bin_q75"], float)
    m = np.isfinite(med)
    ax.plot(bc[m], med[m], color=c, lw=1.4,
            label=f"{b['band']} m ({b['slope_per_m']*10:+.2f} °C/10 m)")
    ax.fill_between(bc[m], q25[m], q75[m], color=c, alpha=0.15, lw=0)
ax.set_xlabel("Canopy height (m)")
ax.set_ylabel("Terrain-adjusted LST (°C)")
ax.legend(frameon=False, fontsize=5.0, title="Elevation band", title_fontsize=5.4,
          loc="upper right", borderaxespad=0.1, handlelength=1.4)
ax.set_ylim(None, 41.5)
ax.set_title("a  LST vs canopy height by elevation", loc="left", fontweight="bold",
             fontsize=8)
style_ax(ax)

z = np.load(f"{O}/shap_values.npz", allow_pickle=True)
sv, Xd, feats = z["values"], z["data"], list(z["feats"])
ic, ie = feats.index("chm"), feats.index("elev")
ax = axes[1]
sc = ax.scatter(Xd[:, ic], sv[:, ic], c=Xd[:, ie], cmap="viridis", s=1.5, lw=0,
                alpha=0.6, rasterized=True)
cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
cb.set_label("Elevation (m)")
ax.axhline(0, color="black", lw=0.6, ls=":")
ax.set_xlabel("Canopy height (m)")
ax.set_ylabel("SHAP value for LST (°C)")
ax.set_title("b  SHAP dependence: canopy height", loc="left", fontweight="bold",
             fontsize=8)
style_ax(ax)

ax = axes[2]
sm = R["shap_mean_abs"]
names = {"elev": "Elevation", "chm": "Canopy height", "eastness": "Eastness",
         "slope": "Slope", "cos_i": "Illumination cos(i)", "northness": "Northness"}
keys = sorted(sm, key=lambda k: sm[k])
ax.barh([names[k] for k in keys], [sm[k] for k in keys], color=WONG["blue"])
ax.set_xlabel("Mean |SHAP| (°C)")
ax.set_title(f"c  Attribution (GBM R²={R['gbm_r2_test']:.2f})", loc="left",
             fontweight="bold", fontsize=8)
style_ax(ax)
save(fig, "F2_buffering")

# ---------------- F5 validation: b/c fixes ----------------
R2 = json.load(open(f"{O}/results2.json"))
DDC = R2["did_cohorts"]
pdel = pd.read_parquet(f"{D}/patches_deltas2.parquet")
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 58 * MM), constrained_layout=True)
ax = axes[0]
p24 = pdel[(pdel.src == "event") & (pdel.year == 2024) & (pdel.pre_tag == "c23")]
p25 = pdel[(pdel.src == "event") & (pdel.year == 2025) & (pdel.pre_tag == "c24")]
data, labels, colsb = [], [], []
for series, lab, col in [
    (p24.dlst_c23.dropna(), "2024 events\npre-event", WONG["skyblue"]),
    (p24.dlst_c25.dropna(), "2024 events\npost (+1 yr)", WONG["vermillion"]),
    (p25.dlst_c24.dropna(), "2025 events\npre-event", WONG["skyblue"]),
    (p25.dlst_c26.dropna(), "2025 events\npost (+1 yr)", WONG["vermillion"])]:
    if len(series) > 10:
        data.append(series.values); labels.append(lab); colsb.append(col)
bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
                medianprops=dict(color="black", lw=1), widths=0.55)
for patch, col in zip(bp["boxes"], colsb):
    patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor("black")
ax.axhline(0, color="grey", lw=0.6, ls=":")
ax.set_ylabel("ΔLST patch − control (°C)")
txt = [f"{y} DiD: +{DDC[y]['did']:.2f}±{1.96*DDC[y]['se']:.2f} °C (n={DDC[y]['n']})"
       for y in ["2024", "2025"] if y in DDC]
ax.text(0.02, 0.98, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=5.6)
ax.set_title("a  Pre-event bias & DiD effect", loc="left", fontweight="bold",
             fontsize=8)
ax.tick_params(axis="x", labelsize=5.4)
style_ax(ax)

ax = axes[1]
dq = pdel[pdel.n_ctrl >= 30]
ax.hist(dq.ctrl_elev - dq.elev, bins=np.linspace(-250, 250, 51), color=WONG["blue"],
        alpha=0.7)
ax.set_xlabel("Control − patch elevation (m)", color=WONG["blue"])
ax.tick_params(axis="x", colors=WONG["blue"])
ax.set_ylabel("Patches")
ax2 = ax.twiny()
ax2.hist(dq.ctrl_slope - dq.slope, bins=np.linspace(-15, 15, 51), histtype="step",
         color=WONG["vermillion"], lw=1.8)
ax2.set_xlabel("Control − patch slope (°)", color=WONG["vermillion"])
ax2.tick_params(axis="x", colors=WONG["vermillion"])
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
ax.set_xticks(x); ax.set_xticklabels([n.replace("_", "\n") for n in names],
                                     fontsize=5.6)
ax.axhline(0, color="grey", lw=0.5)
ax.set_ylabel("LST difference (°C)")
ax.set_ylim(None, 3.6)
ax.legend(frameon=False, fontsize=5.6, loc="upper left")
ax.set_title("c  Compositing method check", loc="left", fontweight="bold", fontsize=8)
style_ax(ax)
save(fig, "F5_validation")

# ---------------- event-group statistics ----------------
ev = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                     columns=["src", "event", "agent", "year", "area_ha", "elev"])
ev = ev[ev.src == "event"].copy()
ev["event"] = ev.event.fillna("").astype(str)
ev.loc[ev.event.str.strip() == "", "event"] = "（事件欄空白）"
g = ev.groupby(["event", "year"]).agg(n=("event", "size"), area=("area_ha", "sum"),
                                      elev=("elev", "median"),
                                      agent=("agent", lambda s: s.mode().iat[0] if len(s) else ""))
g = g.reset_index().set_index("event")
g["yr0"] = g.year; g["yr1"] = g.year
g = g.sort_values("n", ascending=False)
top = g.head(14)
rest = g.iloc[14:]
groups = [dict(event=i, n=int(r.n), area_ha=float(r.area), year0=int(r.yr0),
               year1=int(r.yr1), elev_med=float(r.elev), agent=str(r.agent))
          for i, r in top.iterrows()]
groups.append(dict(event=f"其他（{len(rest)} 個事件群）", n=int(rest.n.sum()),
                   area_ha=float(rest.area.sum()), year0=int(rest.yr0.min()),
                   year1=int(rest.yr1.max()), elev_med=float(ev.elev.median()),
                   agent=""))
R2j = json.load(open(f"{O}/results2.json"))
R2j["event_groups"] = groups
R2j["event_groups_total"] = dict(n=int(len(ev)), area_ha=float(ev.area_ha.sum()),
                                 n_groups=int(len(g)))
# guard: totals must reconcile
assert sum(x["n"] for x in groups) == len(ev)
json.dump(R2j, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
for gg in groups[:6]:
    print(gg)
print("groups total:", R2j["event_groups_total"])
print("STEP8I COMPLETE")
