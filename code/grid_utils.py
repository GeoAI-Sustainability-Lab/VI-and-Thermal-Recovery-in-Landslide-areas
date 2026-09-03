"""Common grid, palette, and helpers for the thermal buffering project.
Analysis grid: EPSG:3826 (TWD97 / TM2 zone 121), 30 m, aligned to the
user's taiwan_foresttype_20m raster footprint (main island).
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

# ---- analysis grid (EPSG:3826, 30 m) ----
CRS = "EPSG:3826"
X0, Y1 = 146620.0, 2804960.0          # upper-left
W, H = 7064, 12923                     # cols, rows
RES = 30.0
X1, Y0 = X0 + W * RES, Y1 - H * RES
TRANSFORM = from_origin(X0, Y1, RES, RES)
BBOX_4326 = [119.95, 21.85, 122.06, 25.35]   # STAC query bbox (lon/lat)

# Wong 2011 colorblind-safe palette (user figure standard)
WONG = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}

# One trigger-agent palette and label set for every figure that splits by
# agent (Figs 2, 4, 9). The four hues are far apart and stay distinct under
# colour-vision deficiency; typhoon and rainfall in particular must not be
# neighbouring oranges.
AGENT_COL = {"typhoon_rain": WONG["blue"], "rainfall": WONG["orange"],
             "earthquake": WONG["purple"], "hansen": WONG["green"]}
AGENT_LAB = {"typhoon_rain": "Typhoon", "rainfall": "Heavy rain",
             "earthquake": "Earthquake", "hansen": "Annual loss"}

# Patch-area classes are ordinal, so they get one sequential ramp, used by both
# the stratified figure and the Morakot figure. The light end is a deep gold
# rather than Wong's yellow, which was too pale to read on white.
AREA_COL = ["#E0A100", "#E1590A", "#8E2800"]

PROFILE = dict(driver="GTiff", crs=CRS, transform=TRANSFORM, width=W, height=H,
               count=1, compress="deflate", predictor=2, tiled=True,
               blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")


def write_grid(path, arr, dtype, nodata):
    prof = PROFILE.copy()
    prof.update(dtype=dtype, nodata=nodata)
    if np.issubdtype(np.dtype(dtype), np.floating):
        prof["predictor"] = 3
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def read_grid(path):
    with rasterio.open(path) as src:
        return src.read(1)


def apply_mpl_standards():
    """Figure defaults per user's figure standards.
    v20: fonts sized so that AFTER the figure is scaled to the 166 mm text
    width, every glyph is >= 7.5 pt on the page. A 183 mm figure is scaled
    by 166/183 = 0.907, so the in-figure minimum is 8.3 pt."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.3, "axes.labelsize": 8.3, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.3, "ytick.labelsize": 8.3, "legend.fontsize": 8.3,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": False, "savefig.dpi": 450, "figure.dpi": 120,
    })

