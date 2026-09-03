"""Step 7e: Sentinel-2 pre/post chips for the two missing agent types of the
four-agent case figure: (1) largest clean recent rainfall-triggered landslide,
(2) largest clean recent Hansen annual-loss (harvest) patch. Windows are set
from each case's own event time. Saves outputs/s2_chips3.npz.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, warnings, time, json, datetime as dt
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
evc = json.load(open(f"{D}/event_codes2.json"))


def frac_to_date(t):
    y = int(t)
    return dt.date(y, 1, 1) + dt.timedelta(days=float(t - y) * 365.0)


def fetch(name, prow, rng_pre, rng_post, res):
    CHIP = 300
    cx = X0 + prow.col * GRID_RES; cy = Y1 - prow.row * GRID_RES
    x0, y1 = cx - CHIP * 5, cy + CHIP * 5
    tr = from_origin(x0, y1, 10, 10)
    bbox = transform_bounds(CRS, "EPSG:4326", x0, y1 - CHIP * 10, x0 + CHIP * 10, y1)
    for phase, rng in [("pre", rng_pre), ("post", rng_post)]:
        items = sorted(cat.search(collections=["sentinel-2-l2a"], bbox=list(bbox),
                                  datetime=rng).items(),
                       key=lambda i: i.properties["eo:cloud_cover"])
        best, best_arr, best_valid = None, None, -1
        for it in items[:12]:
            try:
                with rasterio.open(it.assets["visual"].href) as src:
                    with WarpedVRT(src, crs=CRS, transform=tr, width=CHIP,
                                   height=CHIP,
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
        assert best is not None, (name, phase)
        res[f"{name}_{phase}"] = best_arr
        res[f"{name}_{phase}_meta"] = np.array(
            [best.id, str(best.datetime.date()),
             f"{best.properties['eo:cloud_cover']:.1f}", f"{best_valid:.3f}"],
            dtype=object)
        print(name, phase, best.datetime.date(), "valid", round(best_valid, 3),
              flush=True)
    res[f"{name}_geom"] = np.array([x0, y1, prow.r0, prow.c0, prow.r1, prow.c1,
                                    prow.row, prow.col], dtype=np.float64)


res = {}
# ---- rainfall case ----
cand = raw[(raw.src == "event") & (dd.agent == "rainfall") & (raw.year >= 2021)
           & (raw.n_px >= 30) & (dd.forest2000_frac > 0.7)]
prow = cand.nlargest(1, "n_px").iloc[0]
tev = frac_to_date(prow.t_event)
print("rainfall case:", evc[str(int(prow.event_code))]["event"], "n_px",
      prow.n_px, "t_event", tev, flush=True)
pre_rng = f"{tev - dt.timedelta(days=150)}/{tev - dt.timedelta(days=3)}"
post_rng = f"{tev + dt.timedelta(days=40)}/{tev + dt.timedelta(days=260)}"
fetch("rainfall_case", prow, pre_rng, post_rng, res)

# ---- hansen annual-loss case ----
handc = raw[(raw.src == "hansen") & (raw.year >= 2019) & (raw.year <= 2023)
            & (raw.n_px >= 30) & (dd.forest2000_frac > 0.7)]
hrow = handc.nlargest(1, "n_px").iloc[0]
Y = int(hrow.year)
print("hansen case: loss year", Y, "n_px", hrow.n_px, flush=True)
fetch("hansen_case", hrow, f"{Y-1}-02-01/{Y-1}-12-15",
      f"{Y+1}-01-15/{Y+1}-12-15", res)

np.savez_compressed(f"{RESD}/s2_chips3.npz", **res)
print("STEP7E COMPLETE", f"{time.time()-t0:.0f}s")
