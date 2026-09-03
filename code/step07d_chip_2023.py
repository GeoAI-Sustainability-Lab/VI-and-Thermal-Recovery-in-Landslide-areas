"""Step 7d: Sentinel-2 pre/post chips for a 2023 typhoon-landslide case with a
large temporal separation from the earthquakes (0918 EQ was Sep 2022, 0403 EQ
Apr 2024). Pre window ends before Typhoon Doksuri (2023-07-26); post window
ends before the 0403 earthquake. Case = largest clean 2023 typhoon patch.
Saves outputs/s2_chips2.npz (same key structure as s2_chips.npz).
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MULTIPLEX": "YES", "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "30000000", "GDAL_CACHEMAX": "256",
})
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from rasterio.transform import from_origin
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, RES as GRID_RES, X0, Y1

D = f"{_TROOT}/data"
RESD = f"{_TROOT}/outputs"
t0 = time.time()
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

raw = pd.read_parquet(f"{D}/patches_raw2.parquet")
dd = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                     columns=["src", "agent", "year", "event", "forest2000_frac"])
raw = raw.join(dd[["agent", "forest2000_frac"]].rename(
    columns={"agent": "agent2", "forest2000_frac": "f2000"}))
cand = raw[(raw.src == "event") & (dd.agent == "typhoon_rain") & (raw.year == 2023)
           & (raw.n_px >= 40) & (dd.forest2000_frac > 0.7)]
prow = cand.nlargest(1, "n_px").iloc[0]
evc = json.load(open(f"{D}/event_codes2.json"))
print("case:", evc[str(int(prow.event_code))]["event"], "n_px", prow.n_px,
      "t_event", round(prow.t_event, 3), flush=True)

name = "typhoon_2023"
CHIP = 300
cx = X0 + prow.col * GRID_RES; cy = Y1 - prow.row * GRID_RES
x0, y1 = cx - CHIP * 5, cy + CHIP * 5
tr = from_origin(x0, y1, 10, 10)
bbox = transform_bounds(CRS, "EPSG:4326", x0, y1 - CHIP * 10, x0 + CHIP * 10, y1)
print("bbox lonlat:", [round(b, 3) for b in bbox], flush=True)

res = {}
for phase, rng in [("pre", "2023-03-01/2023-07-20"),
                   ("post", "2023-09-20/2024-03-25")]:
    items = sorted(cat.search(collections=["sentinel-2-l2a"], bbox=list(bbox),
                              datetime=rng).items(),
                   key=lambda i: i.properties["eo:cloud_cover"])
    best, best_arr, best_valid = None, None, -1
    for it in items[:10]:
        try:
            with rasterio.open(it.assets["visual"].href) as src:
                with WarpedVRT(src, crs=CRS, transform=tr, width=CHIP, height=CHIP,
                               resampling=Resampling.bilinear) as vrt:
                    a = vrt.read()
            grey = a.mean(axis=0)
            valid = float(((grey > 8) & (grey < 235)).mean())
            if valid > best_valid:
                best_valid, best, best_arr = valid, it, a
            if valid > 0.985:
                break
        except Exception:
            continue
    assert best is not None, phase
    res[f"{name}_{phase}"] = best_arr
    res[f"{name}_{phase}_meta"] = np.array(
        [best.id, str(best.datetime.date()), f"{best.properties['eo:cloud_cover']:.1f}",
         f"{best_valid:.3f}"], dtype=object)
    print(name, phase, best.id, best.datetime.date(), "valid", round(best_valid, 3),
          flush=True)
res[f"{name}_geom"] = np.array([x0, y1, prow.r0, prow.c0, prow.r1, prow.c1,
                                prow.row, prow.col], dtype=np.float64)
np.savez_compressed(f"{RESD}/s2_chips2.npz", **res)
print("STEP7D COMPLETE", f"{time.time()-t0:.0f}s")
