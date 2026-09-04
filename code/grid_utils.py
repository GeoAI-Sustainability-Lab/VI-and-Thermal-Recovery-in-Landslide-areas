"""Common analysis grid and raster helpers for the tier-B (raster) steps.
Analysis grid: EPSG:3826 (TWD97 / TM2 zone 121), 30 m, aligned to the
island-wide 20 m forest-type raster footprint (main island of Taiwan).
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
