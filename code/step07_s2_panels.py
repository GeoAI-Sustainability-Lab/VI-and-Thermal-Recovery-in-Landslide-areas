"""Step 7: Sentinel-2 L2A true-colour pre/post chips for flagship cases (real scenes).
Saves outputs/s2_chips.npz with RGB arrays + scene ids/dates."""
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

OUT = f"{_TROOT}/data"
RESD = f"{_TROOT}/outputs"
t0 = time.time()
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

patches = pd.read_parquet(f"{OUT}/patches_raw.parquet")
ev = {int(k): v for k, v in json.load(open(f"{OUT}/event_codes.json")).items()}

CASES = []
fire = patches[patches.src == "fire2021"].nlargest(1, "n_px").iloc[0]
CASES.append(("huisun_fire_2021", fire, "2021-01-01/2021-05-05", "2021-05-25/2021-09-30"))
ard = patches[patches.src == "ardswc"].copy()
ard["evn"] = ard.event_code.map(lambda c: ev[c]["event"])
eq = ard[(ard.evn == "0403花蓮地震") & (ard.forest2000_frac > 0.7)].nlargest(1, "n_px").iloc[0]
CASES.append(("hualien_eq_2024", eq, "2024-01-01/2024-04-01", "2024-04-05/2024-08-31"))
km = ard[(ard.evn == "凱米颱風") & (ard.forest2000_frac > 0.7)].nlargest(1, "n_px").iloc[0]
CASES.append(("gaemi_typhoon_2024", km, "2024-03-01/2024-07-20", "2024-07-28/2024-12-31"))

CHIP = 300  # 300 px at 10 m = 3 km
res = {}
for name, prow, pre_rng, post_rng in CASES:
    cx = X0 + prow.col * GRID_RES; cy = Y1 - prow.row * GRID_RES
    x0, y1 = cx - CHIP * 5, cy + CHIP * 5
    tr = from_origin(x0, y1, 10, 10)
    bbox = transform_bounds(CRS, "EPSG:4326", x0, y1 - CHIP * 10, x0 + CHIP * 10, y1)
    for phase, rng in [("pre", pre_rng), ("post", post_rng)]:
        items = sorted(cat.search(collections=["sentinel-2-l2a"], bbox=list(bbox),
                                  datetime=rng).items(),
                       key=lambda i: i.properties["eo:cloud_cover"])
        best, best_arr, best_valid = None, None, -1
        for it in items[:8]:
            try:
                with rasterio.open(it.assets["visual"].href) as src:
                    with WarpedVRT(src, crs=CRS, transform=tr, width=CHIP, height=CHIP,
                                   resampling=Resampling.bilinear) as vrt:
                        a = vrt.read()
                # validity: not black, not white-cloud
                grey = a.mean(axis=0)
                valid = float(((grey > 8) & (grey < 235)).mean())
                if valid > best_valid:
                    best_valid, best, best_arr = valid, it, a
                if valid > 0.985:
                    break
            except Exception:
                continue
        if best is not None:
            res[f"{name}_{phase}"] = best_arr
            res[f"{name}_{phase}_meta"] = np.array(
                [best.id, str(best.datetime.date()), f"{best.properties['eo:cloud_cover']:.1f}",
                 f"{best_valid:.3f}"], dtype=object)
            print(name, phase, best.id, best.datetime.date(), "valid", round(best_valid, 3), flush=True)
    res[f"{name}_geom"] = np.array([x0, y1, prow.r0, prow.c0, prow.r1, prow.c1,
                                    prow.row, prow.col], dtype=np.float64)

np.savez_compressed(f"{RESD}/s2_chips.npz", **res)
print("STEP7 COMPLETE", f"{time.time()-t0:.0f}s")
