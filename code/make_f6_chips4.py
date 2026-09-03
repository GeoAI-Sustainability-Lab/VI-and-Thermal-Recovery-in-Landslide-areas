"""F6 v23: Sentinel-2 pre/post chips for all four agent types. The four cases
are the SAME agent-year combinations as the population panels of Fig. 7
(typhoon 2024, heavy rain 2025, earthquake 2024, annual loss 2023), and in the
same column order, so the case and the population read as one story.
2 rows (pre/post) x 4 columns; acquisition dates are written inside each chip.
Outline: event polygons (yellow) for the three landslide cases; annual-loss
pixels of the case year for the Hansen case.
Inputs: outputs/s2_chips.npz, s2_chips2.npz, s2_chips3.npz
Output: figures/F6_s2_cases.{pdf,png}
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.transform import from_origin
from pyproj import Transformer

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4

Z1 = np.load(f"{O}/s2_chips.npz", allow_pickle=True)
Z2 = np.load(f"{O}/s2_chips2.npz", allow_pickle=True)
Z3 = np.load(f"{O}/s2_chips3.npz", allow_pickle=True)

# same order and same years as the cohorts of Fig. 7
CASES = [("gaemi_typhoon_2024", Z1, "Typhoon 2024", "event"),
         ("rainfall_case", Z3, "Heavy rain 2025", "event"),
         ("hualien_eq_2024", Z1, "Earthquake 2024", "event"),
         ("hansen_case", Z3, "Annual loss 2023", "loss23")]
STROKE = [pe.withStroke(linewidth=1.7, foreground="black")]
# 四個案例之輪廓遮罩快取。完整柵格在時重算並寫入；公開重現版未附崩塌目錄柵格，
# 此時自快取讀入，快取僅涵蓋本圖已呈現之四個案例範圍。
OVL = f"{O}/case_overlays.npz"
HAVE_RASTER = (os.path.exists(f"{D}/event_any30.tif")
               and os.path.exists(f"{D}/hansen_lossyear30.tif"))
_ovl = {} if HAVE_RASTER else dict(np.load(OVL))
TO_WGS = Transformer.from_crs(3826, 4326, always_xy=True)


def centre_label(x0c, y1c, npx=300, res=10):
    """Chip-centre coordinate, one decimal, same convention as Fig. 2."""
    lon, lat = TO_WGS.transform(x0c + npx * res / 2, y1c - npx * res / 2)
    return f"{lat:.1f}°N  {lon:.1f}°E"

fig, axes = plt.subplots(2, 4, figsize=(183 * MM, 104 * MM), constrained_layout=True)
for j, (case, z, title, om) in enumerate(CASES):
    geom = z[f"{case}_geom"]
    x0c, y1c = geom[0], geom[1]
    tr10 = from_origin(x0c, y1c, 10, 10)
    if not HAVE_RASTER:
        overlay = _ovl[case]
    elif om == "event":
        with rasterio.open(f"{D}/event_any30.tif") as src:
            with WarpedVRT(src, crs="EPSG:3826", transform=tr10, width=300,
                           height=300, resampling=Resampling.nearest) as vrt:
                overlay = vrt.read(1) > 0
        _ovl[case] = overlay
    else:
        with rasterio.open(f"{D}/hansen_lossyear30.tif") as src:
            with WarpedVRT(src, crs="EPSG:3826", transform=tr10, width=300,
                           height=300, resampling=Resampling.nearest) as vrt:
                ly = vrt.read(1)
        overlay = ly == 23
        _ovl[case] = overlay
    for i, phase in enumerate(["pre", "post"]):
        ax = axes[i][j]
        key = f"{case}_{phase}"
        rgb = np.moveaxis(z[key], 0, -1)
        ax.imshow(rgb, interpolation="nearest")
        if overlay.any():
            ax.contour(overlay.astype(float), levels=[0.5],
                       colors=[WONG["yellow"]], linewidths=0.65)
        meta = z[f"{key}_meta"]
        # panel letter + case on the title line; provenance inside the chip
        ax.set_title(f"{chr(97 + i*4 + j)}  {title}", loc="left",
                     fontweight="bold", fontsize=9.5)
        ax.text(8, 22, f"{phase}  {meta[1]}", color="white", fontsize=8.3,
                va="top", ha="left", path_effects=STROKE)
        ax.text(8, 52, centre_label(float(x0c), float(y1c)), color="white",
                fontsize=8.3, va="top", ha="left", path_effects=STROKE)
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axes[1][j].plot([15, 115], [285, 285], color="white", lw=1.8,
                    path_effects=[pe.withStroke(linewidth=3.0, foreground="black")])
    axes[1][j].text(65, 270, "1 km", color="white", ha="center", fontsize=8.3,
                    va="bottom", path_effects=STROKE)

if HAVE_RASTER:
    np.savez_compressed(OVL, **_ovl)
fig.savefig(f"{F}/F6_s2_cases.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F6_s2_cases.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F6_s2_cases v23 (four agents, years matched to Fig. 7)")
