"""Step 8c: 補充圖 S8 — Sentinel-3 SLSTR day/night LST and structure-dependent
diurnal amplitude statistics (real Jul-Aug 2025 swath composites); the
amplitude MAP itself is main-text 圖 2d (drawn in step08k)."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4

z = np.load(f"{O}/s3_daynight.npz")
day, night = z["day"], z["night"]
LON0, LAT1, GR, NX, NY = z["grid"]
ext = [LON0, LON0 + GR * NX, LAT1 - GR * NY, LAT1]
res = json.load(open(f"{O}/s3_results.json"))

# forest-domain mask on the S3 grid: urban and agricultural signal distracts
# from the canopy story, so non-forest cells are greyed out
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
with rasterio.open(f"{_TROOT}/data/forest4_30.tif") as _s:
    _f30 = (_s.read(1) > 0).astype(np.float32)
    _tr30, _crs30 = _s.transform, _s.crs
_ffrac = np.zeros_like(day, dtype=np.float32)
reproject(_f30, _ffrac, src_transform=_tr30, src_crs=_crs30,
          dst_transform=from_origin(LON0, LAT1, GR, GR), dst_crs="EPSG:4326",
          resampling=Resampling.average, src_nodata=None, dst_nodata=0.0)
FOREST = _ffrac >= 0.5
del _f30, _ffrac
day = np.where(FOREST, day, np.nan)
night = np.where(FOREST, night, np.nan)

# v16: 3 panels — the diurnal-amplitude map was promoted to main-text 圖 2d,
# this figure keeps day, night and the structure statistics as 補充圖 S8
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 80 * MM), constrained_layout=True)
BG = np.where(np.isfinite(day) | np.isfinite(night) | True, 0.92, np.nan)
for ax, arr, ttl in [(axes[0], day, "a  SLSTR day LST (~10:00)"),
                     (axes[1], night, "b  SLSTR night LST (~22:00)")]:
    ax.imshow(np.full_like(arr, 0.93), cmap="Greys_r", vmin=0, vmax=1,
              extent=ext, interpolation="nearest")
    v1, v2 = np.nanpercentile(arr, [2, 98])
    im = ax.imshow(arr, cmap="inferno", vmin=v1, vmax=v2, extent=ext,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
    cb.set_label("LST Jul-Aug 2025 (°C)")
    ax.set_title(ttl, loc="left", fontweight="bold", fontsize=9.5)
    ax.set_xlabel("Longitude (°E)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Latitude (°N)")

ax = axes[2]
st = res["stats"]
bands = sorted({s["elev_band"] for s in st}, key=lambda b: int(b.split("-")[0]))
xb = np.arange(len(bands))
for cls, col, off in [("short (<15 m)", WONG["orange"], -0.17),
                      ("tall (>=22 m)", WONG["green"], 0.17)]:
    vals = []
    for b in bands:
        row = [s for s in st if s["elev_band"] == b and s["chm_class"] == cls]
        vals.append((row[0]["amp_mean"], row[0]["amp_se"]) if row else (np.nan, 0))
    v = np.array(vals)
    ax.errorbar(xb + off, v[:, 0], yerr=1.96 * v[:, 1], fmt="o", ms=3.6, capsize=2.5,
                lw=1.0, color=col, label=cls.replace(">=", "≥"))
ax.set_xticks(xb); ax.set_xticklabels([b + " m" for b in bands], fontsize=8.3, rotation=18)
ax.set_xlabel("Elevation band")
ax.set_ylabel("Day − night LST amplitude (°C)")
ax.legend(frameon=False, fontsize=8.3, title="Canopy height", title_fontsize=8.3)
ax.set_title("c  Amplitude by structure", loc="left", fontweight="bold", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

fig.savefig(f"{F}/F7_s3_daynight.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F7_s3_daynight.png", bbox_inches="tight", dpi=500)
print("saved F7")
