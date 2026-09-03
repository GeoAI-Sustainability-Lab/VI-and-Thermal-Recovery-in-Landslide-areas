"""Step 4: island-wide buffering sample — intact-forest pixels with
LST (c2425), NDVI, canopy height, terrain, type. Output: buffering_sample.parquet
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, RES, X0, Y1

OUT = f"{_TROOT}/data"
t0 = time.time()
rng = np.random.default_rng(42)

lst = read_grid(f"{OUT}/lst_c2425.tif"); lst[lst == -9999] = np.nan
ndvi = read_grid(f"{OUT}/ndvi_c2425.tif"); ndvi[ndvi == -9999] = np.nan
nclear = read_grid(f"{OUT}/nclear_lst_c2425.tif")
chm = read_grid(f"{OUT}/chm_eth30.tif"); chm[chm == -9999] = np.nan
dem = read_grid(f"{OUT}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{OUT}/slope30.tif"); slope[slope == -9999] = np.nan
north = read_grid(f"{OUT}/northness30.tif"); north[north == -9999] = np.nan
east = read_grid(f"{OUT}/eastness30.tif"); east[east == -9999] = np.nan
intact = read_grid(f"{OUT}/intact30.tif") == 1
ftype = read_grid(f"{OUT}/foresttype30.tif")

valid = (intact & np.isfinite(lst) & np.isfinite(ndvi) & np.isfinite(chm)
         & np.isfinite(dem) & np.isfinite(slope) & (nclear >= 8)
         & (dem > 0) & (dem < 3700) & (slope < 55))
idx = np.flatnonzero(valid.ravel())
print("valid intact px:", len(idx), f"{time.time()-t0:.0f}s", flush=True)

n_samp = min(400_000, len(idx))
sel = rng.choice(idx, size=n_samp, replace=False)
r, c = np.unravel_index(sel, lst.shape)

# illumination proxy at Landsat summer overpass (sun az ~107.5 deg, alt ~62.5 deg)
az, alt = np.radians(107.5), np.radians(62.5)
slope_r = np.radians(slope[r, c])
aspect = np.arctan2(east[r, c], north[r, c])           # radians, 0=N
cos_i = (np.cos(slope_r) * np.sin(alt)
         + np.sin(slope_r) * np.cos(alt) * np.cos(az - aspect))

df = pd.DataFrame(dict(
    row=r, col=c,
    x=X0 + (c + 0.5) * RES, y=Y1 - (r + 0.5) * RES,
    lst=lst[r, c], ndvi=ndvi[r, c], chm=chm[r, c], elev=dem[r, c],
    slope=slope[r, c], northness=north[r, c], eastness=east[r, c],
    cos_i=cos_i.astype(np.float32), nclear=nclear[r, c], ftype=ftype[r, c],
))
df.to_parquet(f"{OUT}/buffering_sample.parquet")
print(df.describe().round(2).to_string())
print("STEP4 COMPLETE", f"{time.time()-t0:.0f}s")
