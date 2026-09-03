"""Step 1: Landsat C2 L2 summer composites (LST + NDVI) on the 30 m grid.

Memory-safe streaming design (7 GB RAM / 2 cores):
per item -> warp only its intersecting subwindow -> QA-mask -> accumulate
count / sum / sum2 per pixel. Composite = clear-sky mean (validated against
exact medians on subtiles in step01b).

Summers accumulated separately:  2024, 2025, 2026(Jun1-Aug10)
Composites derived:  C2425 = 2024+2025 (chronosequence for losses <=2023)
                     C25   = 2025      (for 2024 event inventory)
                     C26   = 2026      (for 2025 event inventory, current state)
LST  = ST_B10 * 0.00341802 + 149.0 K -> degC     (asset lwir11)
NDVI = (nir08 - red) / (nir08 + red), SR scale 2.75e-5, offset -0.2
QA_PIXEL clear: bits 1,2,3,4 (dilated, cirrus, cloud, shadow) == 0, bit0 fill==0,
bit5 snow==0.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, warnings, time, json, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np

# GDAL /vsicurl tuning — critical for throughput
os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIPLEX": "YES", "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "30000000",
    "GDAL_CACHEMAX": "256",
    "CPL_VSIL_CURL_CHUNK_SIZE": "2097152",
})
import rasterio
from rasterio.warp import Resampling, transform_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, TRANSFORM, W, H, RES, X0, Y1, BBOX_4326, write_grid

OUT = f"{_TROOT}/data"
os.makedirs(f"{OUT}/acc", exist_ok=True)
t0 = time.time()

SUMMERS = {
    "s2024": "2024-06-01/2024-09-30",
    "s2025": "2025-06-01/2025-09-30",
    "s2026": "2026-06-01/2026-08-10",
    "s2023": "2023-06-01/2023-09-30",   # placebo epoch for 2024 events
    "s2022": "2022-06-01/2022-09-30",   # placebo for 2023 events
    "s2021": "2021-06-01/2021-09-30",   # placebo for 2022 events
    "s2020": "2020-06-01/2020-09-30",   # placebo for 2021 events
}

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def item_subwindow(item):
    """Grid-aligned subwindow (row0,col0,h,w) covering the item footprint."""
    b = transform_bounds("EPSG:4326", CRS, *item.bbox, densify_pts=21)
    col0 = int(max(0, np.floor((b[0] - X0) / RES)))
    row0 = int(max(0, np.floor((Y1 - b[3]) / RES)))
    col1 = int(min(W, np.ceil((b[2] - X0) / RES)))
    row1 = int(min(H, np.ceil((Y1 - b[1]) / RES)))
    if col1 <= col0 or row1 <= row0:
        return None
    return row0, col0, row1 - row0, col1 - col0


def warp_asset(href, sub, resampling):
    from rasterio.transform import from_origin
    row0, col0, h, w = sub
    tr = from_origin(X0 + col0 * RES, Y1 - row0 * RES, RES, RES)
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=CRS, transform=tr, width=w, height=h,
                       resampling=resampling, dtype="uint16", nodata=0) as vrt:
            return vrt.read(1)


class Acc:
    def __init__(self):
        self.n = np.zeros((H, W), np.uint16)
        self.s = np.zeros((H, W), np.float32)
        self.s2 = np.zeros((H, W), np.float32)

    def add(self, sub, vals, valid):
        r, c, h, w = sub
        nv = self.n[r:r+h, c:c+w]; sv = self.s[r:r+h, c:c+w]; s2v = self.s2[r:r+h, c:c+w]
        nv += valid.astype(np.uint16)
        v = np.where(valid, vals, 0).astype(np.float32)
        sv += v
        s2v += v * v

    def save(self, path):
        np.savez_compressed(path, n=self.n, s=self.s, s2=self.s2)


ACC_LOCK = threading.Lock()


def warp_asset_resigned(it, asset, sub, resampling):
    """warp with one retry after re-signing (SAS tokens expire mid-run)."""
    try:
        return warp_asset(it.assets[asset].href, sub, resampling)
    except Exception as e:
        if "403" not in repr(e):
            raise
        it2 = pc.sign(it)
        it.assets = it2.assets
        return warp_asset(it.assets[asset].href, sub, resampling)


def process_item(it, lst_acc, ndvi_acc):
    """Memory-disciplined: aggressive del / in-place ops (large scene windows)."""
    sub = item_subwindow(it)
    if sub is None:
        return {"id": it.id, "used": False, "reason": "no-overlap"}
    qa = warp_asset_resigned(it, "qa_pixel", sub, Resampling.nearest)
    clear = (qa != 0) & ((qa & 0b0000000000111110) == 0)
    del qa
    nclear = int(clear.sum())
    if nclear < 500:
        return {"id": it.id, "used": False, "clear_px": nclear}

    st = warp_asset_resigned(it, "lwir11", sub, Resampling.bilinear)
    lst = st.astype(np.float32)
    lst *= 0.00341802
    lst += (149.0 - 273.15)
    v_lst = clear & (st != 0)
    del st
    v_lst &= (lst > -5) & (lst < 60)
    lst[~v_lst] = 0.0
    with ACC_LOCK:
        lst_acc.add(sub, lst, v_lst)
    del lst, v_lst

    red = warp_asset_resigned(it, "red", sub, Resampling.bilinear)
    nir = warp_asset_resigned(it, "nir08", sub, Resampling.bilinear)
    valid0 = clear & (red != 0) & (nir != 0)
    del clear
    r_sr = red.astype(np.float32)
    del red
    r_sr *= 2.75e-5; r_sr -= 0.2
    n_sr = nir.astype(np.float32)
    del nir
    n_sr *= 2.75e-5; n_sr -= 0.2
    den = n_sr + r_sr
    np.subtract(n_sr, r_sr, out=n_sr)          # n_sr becomes (nir-red)
    del r_sr
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(n_sr, den, out=n_sr, where=np.abs(den) > 1e-6)
    ndvi = n_sr
    v_nd = valid0 & (np.abs(den) > 1e-6) & np.isfinite(ndvi) & (ndvi >= -1) & (ndvi <= 1)
    del den, valid0
    ndvi[~v_nd] = 0.0
    with ACC_LOCK:
        ndvi_acc.add(sub, ndvi, v_nd)
    del ndvi, v_nd
    return {"id": it.id, "used": True, "clear_px": nclear}


def run_summer(key, dtrange):
    if os.path.exists(f"{OUT}/acc/{key}_lst.npz"):
        print(key, "already done, skip", flush=True)
        return
    search = cat.search(collections=["landsat-c2-l2"], bbox=BBOX_4326,
                        datetime=dtrange,
                        query={"eo:cloud_cover": {"lt": 70},
                               "platform": {"in": ["landsat-8", "landsat-9"]}})
    items = sorted(search.items(), key=lambda i: i.datetime)
    print(f"[{key}] {len(items)} items", flush=True)
    lst_acc, ndvi_acc = Acc(), Acc()
    meta = []
    done = 0
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(process_item, it, lst_acc, ndvi_acc): it for it in items}
        for f in as_completed(futs):
            it = futs[f]
            try:
                meta.append(f.result())
            except Exception as e:
                meta.append({"id": it.id, "used": False, "error": repr(e)[:150]})
            done += 1
            if done % 10 == 0:
                print(f"  [{key}] {done}/{len(items)}  {time.time()-t0:.0f}s", flush=True)
    lst_acc.save(f"{OUT}/acc/{key}_lst.npz")
    ndvi_acc.save(f"{OUT}/acc/{key}_ndvi.npz")
    json.dump(meta, open(f"{OUT}/acc/{key}_items.json", "w"))
    print(f"[{key}] saved. used={sum(1 for m in meta if m.get('used'))}", flush=True)


for key, rng in SUMMERS.items():
    run_summer(key, rng)


def compose(keys, tag, band):
    n = np.zeros((H, W), np.float64); s = np.zeros((H, W), np.float64)
    for k in keys:
        z = np.load(f"{OUT}/acc/{k}_{band}.npz")
        n += z["n"]; s += z["s"]
    mean = np.where(n >= 3, s / np.maximum(n, 1), np.nan).astype(np.float32)
    write_grid(f"{OUT}/{band}_{tag}.tif", np.nan_to_num(mean, nan=-9999), "float32", -9999)
    write_grid(f"{OUT}/nclear_{band}_{tag}.tif", np.minimum(n, 65535), "uint16", 0)
    print(f"composed {band}_{tag}: valid px {(n>=3).sum()}", flush=True)


for tag, keys in [("c2425", ["s2024", "s2025"]), ("c25", ["s2025"]), ("c26", ["s2026"]),
                  ("c23", ["s2023"]), ("c24", ["s2024"]),
                  ("c22", ["s2022"]), ("c21", ["s2021"]), ("c20", ["s2020"])]:
    for band in ["lst", "ndvi"]:
        compose(keys, tag, band)
print("STEP1 COMPLETE", f"{time.time()-t0:.0f}s")
