"""Step 0b: FIX DEM mosaic — WarpedVRT fill (0) had overwritten earlier tiles.
Use running fmax (terrain >= 0 in Taiwan; coastal 0 unaffected by fill)."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import rasterio
from rasterio.warp import Resampling
from rasterio.vrt import WarpedVRT
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, TRANSFORM, W, H, BBOX_4326, write_grid

OUT = f"{_TROOT}/data"
t0 = time.time()
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

items = list(cat.search(collections=["cop-dem-glo-30"], bbox=BBOX_4326).items())
print("DEM items:", len(items), flush=True)
dem = np.full((H, W), np.nan, dtype=np.float32)
for it in items:
    with rasterio.open(it.assets["data"].href) as src:
        with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=W, height=H,
                       resampling=Resampling.bilinear) as vrt:
            a = vrt.read(1).astype(np.float32)
    dem = np.fmax(dem, a)          # fill=0 never overrides real terrain > 0
    print("  ", it.id, "tile max", float(np.nanmax(a)), flush=True)
print("DEM mosaic max:", float(np.nanmax(dem)), f"{time.time()-t0:.0f}s", flush=True)
assert np.nanmax(dem) > 3800, "Yushan missing — mosaic still wrong"
write_grid(f"{OUT}/dem30.tif", np.nan_to_num(dem, nan=-9999), "float32", -9999)

gy, gx = np.gradient(np.where(np.isnan(dem), 0, dem), 30.0)
slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
aspect = np.degrees(np.arctan2(-gx, gy)).astype(np.float32)
aspect = np.where(aspect < 0, aspect + 360, aspect)
northness = np.cos(np.radians(aspect)).astype(np.float32)
eastness = np.sin(np.radians(aspect)).astype(np.float32)
bad = np.isnan(dem)
for name, arr in [("slope30", slope), ("northness30", northness), ("eastness30", eastness)]:
    arr[bad] = -9999
    write_grid(f"{OUT}/{name}.tif", arr, "float32", -9999)
print("STEP0B COMPLETE", f"{time.time()-t0:.0f}s")
