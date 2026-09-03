"""Step 8k (v6 figures):
F1 study area as 3 rows x 2 cols: (a) Asia locator; (b) Copernicus GLO-30 elevation;
(c) ETH canopy height; (d) SLSTR diurnal amplitude (forest-masked, inferno);
(e) catalogue event centroids by agent; (f) Hansen loss year. One dataset per
panel (v16). F4 case trajectories (7 panels). F6 is drawn by make_f6_chips4.py.
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
from rasterio.vrt import WarpedVRT
from pyproj import Transformer
from scipy.ndimage import binary_fill_holes, binary_closing, grey_dilation
import geopandas as gpd

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, apply_mpl_standards, WONG, X0, Y1, RES, W, H

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
DS = 3
from grid_utils import H as H0, W as W0

# one agent palette for every figure that splits by trigger (Figs 2, 4, 9):
# blue / orange / purple are separated in hue and stay distinct for readers
# with colour-vision deficiency
AGENT_STYLE = {
    "typhoon_rain": (WONG["blue"],   "Typhoon"),
    "rainfall":     (WONG["orange"], "Heavy rain"),
    "earthquake":   (WONG["purple"], "Earthquake"),
}
# hillshade backdrop: as light as possible, it is context, not data
HS_VMIN, HS_VMAX = -2.2, 1.25


def save(fig, name):
    fig.savefig(f"{F}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{F}/{name}.png", bbox_inches="tight", dpi=500)
    plt.close(fig)
    print("saved", name, flush=True)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)


# ---------------- warp rasters to WGS84 (memory-lean: downsample at read) ----------------
def load_ds(path, nodata=-9999.0):
    a = read_grid(path)
    b = np.ascontiguousarray(a[::DS, ::DS]).astype(np.float32)
    del a
    b[b == nodata] = np.nan
    return b

wc = load_ds(f"{D}/worldcover2021.tif", nodata=np.nan)
land = ((wc != 0) & (wc != 80) & np.isfinite(wc)).astype(np.float32)
del wc
dem = load_ds(f"{D}/dem30.tif")
lst = load_ds(f"{D}/lst_c2425.tif")
chm = load_ds(f"{D}/chm_eth30.tif")
_ly = read_grid(f"{D}/hansen_lossyear30.tif").astype(np.uint8)
lossyear = _ly[:H0 // DS * DS, :W0 // DS * DS].reshape(
    H0 // DS, DS, W0 // DS, DS).max(axis=(1, 3)).astype(np.float32)
del _ly

gy, gx = np.gradient(np.nan_to_num(dem), RES * DS)
az, alt = np.radians(315), np.radians(45)
sl = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(-gx, gy)
hs = np.clip(np.sin(alt) * np.cos(sl) + np.cos(alt) * np.sin(sl) * np.cos(az - asp),
             0, 1).astype(np.float32)
del gy, gx, sl, asp

src_tr = from_origin(X0, Y1, RES * DS, RES * DS)
lon0, lat0, lon1, lat1 = transform_bounds(3826, 4326, X0, Y1 - H * RES, X0 + W * RES, Y1)
dres = 0.001
dW = int((lon1 - lon0) / dres); dH = int((lat1 - lat0) / dres)
dst_tr = from_origin(lon0, lat1, dres, dres)
ext = [lon0, lon1, lat0, lat1]


def warp(a, categorical=False, fill=np.nan):
    srcd = a  # already downsampled
    out = np.full((dH, dW), fill, np.float32)
    reproject(srcd, out, src_transform=src_tr, src_crs="EPSG:3826",
              dst_transform=dst_tr, dst_crs="EPSG:4326",
              resampling=Resampling.nearest if categorical else Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=fill)
    return out


land_w = warp(land, categorical=True, fill=0.0)
hs_w = warp(hs); lst_w = warp(lst); chm_w = warp(chm)
ly_w = warp(lossyear, categorical=True, fill=0.0)
ly_w = grey_dilation(ly_w, size=(5, 5))   # readability: loss pixels are sparse at map scale
island = binary_fill_holes(binary_closing(land_w > 0.5, iterations=2))
onland = island
GREYBG = np.where(onland, hs_w, np.nan)
print("warped", flush=True)

patches = pd.read_parquet(f"{D}/patches_raw2.parquet",
                          columns=["src", "agent", "row", "col", "n_px"])
tf = Transformer.from_crs(3826, 4326, always_xy=True)
plon, plat = tf.transform(X0 + patches.col.values * RES, Y1 - patches.row.values * RES)
patches["lon"], patches["lat"] = plon, plat

# ------- F1 3x2: a locator | b elevation / c canopy | d amplitude / e events | f Hansen -------
# One dataset per panel (v16: events and Hansen no longer share a map); the island
# LST composite panel is replaced by the SLSTR diurnal-amplitude map (was F7c).
ne_land = gpd.read_file(f"{D}/ne/ne_50m_land.shp")
ne_ctry = gpd.read_file(f"{D}/ne/ne_50m_admin_0_countries.shp")
tw_poly = ne_ctry[ne_ctry.NAME == "Taiwan"]

# SLSTR day-night amplitude on its own lat-lon grid, forest-masked as in F7
z3 = np.load(f"{O}/s3_daynight.npz")
s3day, s3night = z3["day"], z3["night"]
S3LON0, S3LAT1, S3GR, S3NX, S3NY = z3["grid"]
extS3 = [S3LON0, S3LON0 + S3GR * S3NX, S3LAT1 - S3GR * S3NY, S3LAT1]
with rasterio.open(f"{D}/forest4_30.tif") as _s:
    _f30 = (_s.read(1) > 0).astype(np.float32)
    _tr30, _crs30 = _s.transform, _s.crs
_ffrac = np.zeros_like(s3day, dtype=np.float32)
reproject(_f30, _ffrac, src_transform=_tr30, src_crs=_crs30,
          dst_transform=from_origin(S3LON0, S3LAT1, S3GR, S3GR), dst_crs="EPSG:4326",
          resampling=Resampling.average, src_nodata=None, dst_nodata=0.0)
amp3 = np.where(_ffrac >= 0.5, s3day - s3night, np.nan)
del _f30, _ffrac, s3day, s3night

# v18: 2 columns x 3 rows so every map is ~80 mm wide instead of ~52 mm
fig = plt.figure(figsize=(183 * MM, 252 * MM))
gs = fig.add_gridspec(3, 4, width_ratios=[1, 0.045, 1, 0.045],
                      wspace=0.20, hspace=0.22,
                      left=0.070, right=0.975, top=0.980, bottom=0.062)
axA = fig.add_subplot(gs[0, 0]); caxA = fig.add_subplot(gs[0, 1]); caxA.axis("off")
axB = fig.add_subplot(gs[0, 2]); caxB = fig.add_subplot(gs[0, 3])
axC = fig.add_subplot(gs[1, 0]); caxC = fig.add_subplot(gs[1, 1])
axD = fig.add_subplot(gs[1, 2]); caxD = fig.add_subplot(gs[1, 3])
axE = fig.add_subplot(gs[2, 0]); caxE = fig.add_subplot(gs[2, 1]); caxE.axis("off")
axF = fig.add_subplot(gs[2, 2]); caxF = fig.add_subplot(gs[2, 3])
kmdeg = 50.0 / (111.320 * np.cos(np.radians(23.6)))
# Taiwan panels: widen lon so each panel is square at true aspect
LATS = (21.85, 25.4)
lonspan = (LATS[1] - LATS[0]) / np.cos(np.radians(23.6))
LONC = 120.97
LONS = (LONC - lonspan / 2, LONC + lonspan / 2)

ax = axA
ne_land.boundary.plot(ax=ax, color="#333333", linewidth=0.45)
ne_ctry.boundary.plot(ax=ax, color="#999999", linewidth=0.25)
tw_poly.plot(ax=ax, facecolor=WONG["green"], edgecolor="#1b5e20", linewidth=0.6)
lon_span_a = 44.0
lat_span_a = lon_span_a * np.cos(np.radians(24.0))
ax.set_xlim(99, 99 + lon_span_a); ax.set_ylim(23.7 - lat_span_a / 2, 23.7 + lat_span_a / 2)
ax.set_xticks([100, 110, 120, 130, 140])
ax.set_yticks([10, 20, 30, 40])
ax.annotate("Taiwan", (121.0, 23.7), xytext=(128.5, 17.5), fontsize=8.3,
            arrowprops=dict(arrowstyle="-", lw=0.6, color="#1b5e20"),
            color="#1b5e20", fontweight="bold")
ax.set_ylabel("Latitude (°N)")
ax.set_aspect(1.0 / np.cos(np.radians(24.0)))
ax.set_title("a  Location of Taiwan", loc="left", fontweight="bold", fontsize=9.5)
style_ax(ax)

ax = axB
demm = np.where(onland, warp(dem), np.nan)
ax.imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=HS_VMIN, vmax=HS_VMAX,
              interpolation="nearest")
im = ax.imshow(demm, cmap="terrain", vmin=0, vmax=3900, extent=ext, alpha=0.80,
               interpolation="nearest")
cb = fig.colorbar(im, cax=caxB, ticks=[0, 1000, 2000, 3000])
cb.set_label("Elevation (m)", labelpad=11)
ax.set_title("b  Elevation (Copernicus GLO-30)", loc="left", fontweight="bold", fontsize=9.5)

ax = axC
chmm = np.where(onland, chm_w, np.nan)
ax.imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=HS_VMIN, vmax=HS_VMAX,
              interpolation="nearest")
im = ax.imshow(chmm, cmap="viridis", vmin=0, vmax=55, extent=ext, alpha=0.85,
               interpolation="nearest")
cb = fig.colorbar(im, cax=caxC, ticks=[0, 10, 20, 30, 40, 50])
cb.set_label("Canopy height 2020 (m)")
ax.set_title("c  Canopy height (ETH 10 m)", loc="left", fontweight="bold", fontsize=9.5)

ax = axD
ax.imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=HS_VMIN, vmax=HS_VMAX,
              interpolation="nearest")
v1, v2 = np.nanpercentile(amp3, [2, 98])
im = ax.imshow(amp3, cmap="inferno", vmin=v1, vmax=v2, extent=extS3,
               interpolation="nearest")
cb = fig.colorbar(im, cax=caxD)
cb.set_label("Day − night LST (°C)")
ax.set_title("d  Diurnal amplitude (SLSTR 2025)", loc="left", fontweight="bold", fontsize=9.5)

ax = axE
ax.imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=HS_VMIN, vmax=HS_VMAX,
              interpolation="nearest")
ev = patches[patches.src == "event"]
for agent in AGENT_STYLE:
    sub_ = ev[ev.agent == agent]
    ax.scatter(sub_.lon, sub_.lat, s=1.4, c=AGENT_STYLE[agent][0], lw=0, alpha=0.45,
               rasterized=True)
hnd = [Line2D([], [], marker="o", ls="", ms=3.2, color=AGENT_STYLE[a][0],
              label=AGENT_STYLE[a][1]) for a in AGENT_STYLE]
# below the panel, so nothing sits on the map itself
ax.legend(handles=hnd, loc="upper center", bbox_to_anchor=(0.5, -0.115), ncol=3,
          frameon=False, handletextpad=0.2, columnspacing=1.1, fontsize=8.3,
          borderaxespad=0.0)
ax.set_title("e  Catalogue events 2004-2025", loc="left", fontweight="bold", fontsize=9.5)

ax = axF
ax.imshow(GREYBG, cmap="Greys_r", extent=ext, vmin=HS_VMIN, vmax=HS_VMAX,
              interpolation="nearest")
lym = np.where((ly_w > 0) & (ly_w <= 24) & onland, ly_w, np.nan)
# truncate magma so the earliest loss years are purple, not black-on-grey
from matplotlib.colors import LinearSegmentedColormap as _LSC
_magma_hi = _LSC.from_list("magma_hi", plt.get_cmap("magma")(np.linspace(0.28, 1.0, 256)))
im = ax.imshow(lym, cmap=_magma_hi, vmin=1, vmax=24, extent=ext, interpolation="nearest")
cb = fig.colorbar(im, cax=caxF, ticks=[1, 9, 15, 21, 24])
cb.ax.set_yticklabels(["2001", "2009", "2015", "2021", "2024"])
cb.set_label("Forest loss year")
ax.set_title("f  Hansen forest loss", loc="left", fontweight="bold", fontsize=9.5)

from matplotlib.ticker import FormatStrFormatter
DEG = FormatStrFormatter("%.1f")
for ax in [axB, axC, axD, axE, axF]:
    ax.contour(island.astype(float), levels=[0.5], colors="black", linewidths=0.4,
               extent=ext, origin="upper")
    ax.set_xlim(*LONS); ax.set_ylim(*LATS)
    # whole-degree graticule only, drawn faintly over the map
    ax.set_xticks([120, 121, 122]); ax.set_yticks([22, 23, 24, 25])
    ax.grid(True, color="#8C8C8C", lw=0.35, alpha=0.55, zorder=6)
    ax.set_axisbelow(False)
    style_ax(ax)
    ax.set_aspect(1.0 / np.cos(np.radians(23.6)))
for ax in [axA, axB, axC, axD, axE, axF]:
    ax.xaxis.set_major_formatter(DEG); ax.yaxis.set_major_formatter(DEG)
axA.grid(True, color="#8C8C8C", lw=0.35, alpha=0.45, zorder=6)
axA.set_axisbelow(False)
for ax in [axA, axE, axF]:
    ax.set_xlabel("Longitude (°E)")
for ax in [axB, axC, axD]:
    ax.tick_params(labelbottom=False)
for ax in [axB, axD, axF]:
    ax.tick_params(labelleft=False)
axB.plot([LONS[0] + 0.15, LONS[0] + 0.15 + kmdeg], [22.0, 22.0], color="black", lw=1.4)
axB.text(LONS[0] + 0.15 + kmdeg / 2, 22.06, "50 km", ha="center", fontsize=8.3)
axC.set_ylabel("Latitude (°N)")
axE.set_ylabel("Latitude (°N)")
# align each colourbar to its map's drawn (square) box
fig.canvas.draw()
for mapax, cax in [(axB, caxB), (axC, caxC), (axD, caxD), (axF, caxF)]:
    pos = mapax.get_position()
    cax.set_position([pos.x1 + 0.006, pos.y0, 0.013, pos.height])
save(fig, "F1_study_area")

# ---------------- F4 without fire ----------------
tr = pd.read_parquet(f"{D}/case_trajectories.parquet")
tr["dlst"] = tr.lst_p - tr.lst_c
tr["dndvi"] = tr.ndvi_p - tr.ndvi_c
CASE_LAB = {
    "hualien_eq_2024": "Hualien EQ slide, Apr 2024",
    "gaemi_typhoon_2024": "Typhoon Gaemi slide, Jul 2024",
    "hansen_2004": "2004 loss patch", "hansen_2009_morakot": "2009 loss (Morakot)",
    "hansen_2015": "2015 loss patch",
    "high_elev_event": "Morakot high-elev slide, 2009",
    "small_morakot_event": "Morakot small slide, 2009"}
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
    ax.plot(ann.index, ann.dlst, "-o", ms=2.2, lw=0.9, color=WONG["vermillion"])
    ax.set_ylabel("ΔLST (°C)", color=WONG["vermillion"])
    ax.tick_params(axis="y", colors=WONG["vermillion"])
    ax2 = ax.twinx()
    ax2.plot(ann.index, ann.dndvi, "-s", ms=2.0, lw=0.9, color=WONG["green"])
    ax2.set_ylabel("ΔNDVI", color=WONG["green"])
    ax2.tick_params(axis="y", colors=WONG["green"])
    ax2.spines[["top"]].set_visible(False)
    pre = ann[ann.index < int(t_ev)]
    if len(pre) >= 3:
        ax.axhspan(pre.dlst.mean() - pre.dlst.std(), pre.dlst.mean() + pre.dlst.std(),
                   color=WONG["skyblue"], alpha=0.18, lw=0)
    ax.set_title(f"{chr(97+k)}  {CASE_LAB[case]}\n{d.area_ha.iloc[0]:.1f} ha · "
                 f"{d.elev.iloc[0]:.0f} m", loc="left", fontweight="bold", fontsize=8.3)
    ax.set_xlim(1999.5, 2027); ax.spines[["top"]].set_visible(False)
    ax.set_xlabel("Year")
for k in range(n, nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
save(fig, "F4_trajectories")

# F6 (Sentinel-2 case chips) is now drawn by make_f6_chips4.py (2x4, four
# agents); the old 2-case version was removed so this script can no longer
# clobber it.
print("STEP8K COMPLETE")
