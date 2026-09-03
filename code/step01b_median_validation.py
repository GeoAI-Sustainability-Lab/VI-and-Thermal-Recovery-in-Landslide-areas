"""Step 1b QA: exact per-pixel MEDIAN vs streaming MEAN on three 256px subtiles
(summer 2025). Quantifies compositing-method bias for the QA appendix."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2", "GDAL_HTTP_MAX_RETRY": "4",
    "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "30000000", "GDAL_CACHEMAX": "256",
})
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds
from rasterio.transform import from_origin
from concurrent.futures import ThreadPoolExecutor
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, RES, X0, Y1

OUT = f"{_TROOT}/data"
RESD = f"{_TROOT}/outputs"
t0 = time.time()
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# subtiles (row0, col0) on the 30 m grid — west lowland, central mid-elev, east steep
SUBS = {"west_lowland": (23.70, 120.55), "central_mid": (23.50, 120.80),
        "east_steep": (24.30, 121.40)}
N = 256
report = {}
for name, (lat, lon) in SUBS.items():
    xg, yg = rasterio.warp.transform("EPSG:4326", CRS, [lon], [lat])
    c0 = int((xg[0] - X0) / RES) - N // 2
    r0 = int((Y1 - yg[0]) / RES) - N // 2
    tr = from_origin(X0 + c0 * RES, Y1 - r0 * RES, RES, RES)
    bbox = transform_bounds(CRS, "EPSG:4326", X0 + c0 * RES, Y1 - (r0 + N) * RES,
                            X0 + (c0 + N) * RES, Y1 - r0 * RES)
    items = list(cat.search(collections=["landsat-c2-l2"], bbox=list(bbox),
                            datetime="2025-06-01/2025-09-30",
                            query={"eo:cloud_cover": {"lt": 70},
                                   "platform": {"in": ["landsat-8", "landsat-9"]}}).items())
    stack = []
    def one(it):
        try:
            with rasterio.open(it.assets["qa_pixel"].href) as src:
                with WarpedVRT(src, crs=CRS, transform=tr, width=N, height=N,
                               resampling=Resampling.nearest, dtype="uint16", nodata=0) as v:
                    qa = v.read(1)
            clear = (qa != 0) & ((qa & 0b111110) == 0)
            if clear.sum() < 200:
                return None
            with rasterio.open(it.assets["lwir11"].href) as src:
                with WarpedVRT(src, crs=CRS, transform=tr, width=N, height=N,
                               resampling=Resampling.bilinear, dtype="uint16", nodata=0) as v:
                    st = v.read(1)
            lst = st.astype(np.float32) * 0.00341802 + 149.0 - 273.15
            lst[~(clear & (st != 0) & (lst > -5) & (lst < 60))] = np.nan
            return lst
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(one, items):
            if r is not None:
                stack.append(r)
    if not stack:
        continue
    S = np.stack(stack)
    n = np.isfinite(S).sum(axis=0)
    med = np.nanmedian(S, axis=0)
    mean = np.nanmean(S, axis=0)
    ok = n >= 3
    diff = (mean - med)[ok]
    # compare with the production composite (c25)
    with rasterio.open(f"{OUT}/lst_c25.tif") as src:
        prod = src.read(1, window=((r0, r0 + N), (c0, c0 + N)))
    prod = np.where(prod == -9999, np.nan, prod)
    dprod = (prod - med)[ok & np.isfinite(prod)]
    report[name] = dict(n_scenes=len(stack), n_px=int(ok.sum()),
                        mean_minus_median_bias=float(np.mean(diff)),
                        mean_minus_median_rmse=float(np.sqrt(np.mean(diff ** 2))),
                        prod_minus_median_bias=float(np.mean(dprod)),
                        prod_minus_median_rmse=float(np.sqrt(np.mean(dprod ** 2))),
                        median_mean_r=float(np.corrcoef(mean[ok], med[ok])[0, 1]))
    np.savez_compressed(f"{RESD}/medval_{name}.npz", median=med, mean=mean, n=n, prod=prod)
    print(name, report[name], flush=True)

json.dump(report, open(f"{RESD}/median_validation.json", "w"), indent=1)
print("STEP1B COMPLETE", f"{time.time()-t0:.0f}s")
