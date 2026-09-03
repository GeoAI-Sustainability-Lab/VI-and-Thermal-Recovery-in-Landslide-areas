"""Step 2: ETH 10 m global canopy height (Lang et al. 2023) -> 30 m grid (average)."""
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
sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, TRANSFORM, W, H, write_grid

RAW = f"{_TROOT}/data/raw"
OUT = f"{_TROOT}/data"
t0 = time.time()

chm = np.full((H, W), np.nan, dtype=np.float32)
for tile in ["ETH_GlobalCanopyHeight_10m_2020_N21E120_Map.tif",
             "ETH_GlobalCanopyHeight_10m_2020_N24E120_Map.tif"]:
    with rasterio.open(f"{RAW}/{tile}") as src:
        print(tile, src.shape, src.dtypes, src.nodata, flush=True)
        with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=W, height=H,
                       resampling=Resampling.average, src_nodata=255, nodata=np.nan,
                       dtype="float32") as vrt:
            a = vrt.read(1)
    chm = np.fmin(chm, a) if False else np.where(np.isnan(chm), a, chm)
    print("  merged, valid:", int(np.isfinite(chm).sum()), f"{time.time()-t0:.0f}s", flush=True)

print("CHM stats: max", np.nanmax(chm), "mean(valid>0)",
      float(np.nanmean(chm[np.isfinite(chm) & (chm > 0)])), flush=True)
write_grid(f"{OUT}/chm_eth30.tif", np.nan_to_num(chm, nan=-9999), "float32", -9999)
print("STEP2 COMPLETE", f"{time.time()-t0:.0f}s")
