"""Step 0: build common 30 m grid layers — DEM/slope/aspect (Copernicus DEM GLO-30),
ESA WorldCover 2021, forest-type (user's 20 m raster) — all on the EPSG:3826 grid."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.vrt import WarpedVRT
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, TRANSFORM, W, H, BBOX_4326, write_grid, PROFILE

OUT = f"{_TROOT}/data"
U = "/mnt/user-data/uploads/文章發想與實踐"
t0 = time.time()

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def mosaic_to_grid(items, asset, resampling, dtype, nodata_out):
    dst = np.full((H, W), nodata_out, dtype=dtype)
    for it in items:
        href = it.assets[asset].href
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=W, height=H,
                           resampling=resampling) as vrt:
                a = vrt.read(1)
                src_nd = src.nodata
        m = np.ones(a.shape, bool)
        if src_nd is not None:
            m &= (a != src_nd)
        m &= (a != 0) if asset == "map" else m  # worldcover 0 = nodata
        dst[m] = a[m]
    return dst


# ---- Copernicus DEM ----
items = list(cat.search(collections=["cop-dem-glo-30"], bbox=BBOX_4326).items())
print("DEM items:", len(items), flush=True)
dem = np.full((H, W), np.nan, dtype=np.float32)
for it in items:
    with rasterio.open(it.assets["data"].href) as src:
        with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=W, height=H,
                       resampling=Resampling.bilinear) as vrt:
            a = vrt.read(1).astype(np.float32)
    m = a > -100
    dem[m] = a[m]
print("DEM done", np.nanmin(dem), np.nanmax(dem), f"{time.time()-t0:.0f}s", flush=True)
write_grid(f"{OUT}/dem30.tif", np.nan_to_num(dem, nan=-9999), "float32", -9999)

# slope/aspect on 30 m grid
gy, gx = np.gradient(np.where(np.isnan(dem), 0, dem), 30.0)
slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
aspect = np.degrees(np.arctan2(-gx, gy)).astype(np.float32)  # 0=N, 90=E
aspect = np.where(aspect < 0, aspect + 360, aspect)
northness = np.cos(np.radians(aspect)).astype(np.float32)
eastness = np.sin(np.radians(aspect)).astype(np.float32)
bad = np.isnan(dem)
for name, arr in [("slope30", slope), ("northness30", northness), ("eastness30", eastness)]:
    arr[bad] = -9999
    write_grid(f"{OUT}/{name}.tif", arr, "float32", -9999)
print("terrain done", f"{time.time()-t0:.0f}s", flush=True)

# ---- ESA WorldCover 2021 v200 ----
items = [it for it in cat.search(collections=["esa-worldcover"], bbox=BBOX_4326).items()
         if "2021" in it.id]
print("WC items:", [it.id for it in items], flush=True)
wc = mosaic_to_grid(items, "map", Resampling.mode, np.uint8, 0)
write_grid(f"{OUT}/worldcover2021.tif", wc, "uint8", 0)
print("worldcover done", dict(zip(*[x.tolist() for x in np.unique(wc, return_counts=True)])),
      f"{time.time()-t0:.0f}s", flush=True)

# ---- user's forest type 20 m -> 30 m (nearest) ----
with rasterio.open(f"{U}/Dataset/01_SOURCE/Earth_Observation/Taiwan_forest_local_rasters/taiwan_foresttype_20m.tif") as src:
    with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=W, height=H,
                   resampling=Resampling.nearest) as vrt:
        ft = vrt.read(1)
write_grid(f"{OUT}/foresttype30.tif", ft, "uint8", 255)
print("foresttype done", f"{time.time()-t0:.0f}s", flush=True)
print("STEP0 COMPLETE")
