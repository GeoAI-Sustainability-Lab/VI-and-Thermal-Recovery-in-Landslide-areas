"""Step 23: Landsat 5/7 per-sensor summer composites for the cross-sensor
transfer pilot.

Purpose: the main 2013-2026 stack is TIRS-only by design (2.3). To observe the
Morakot cohort at ages 1-3 the record must reach 2010-2012, which only TM
(120 m thermal) and ETM+ (60 m thermal, SLC-off) cover. Composites are built
PER SENSOR (suffix l5/l7) so that the transfer can be estimated, not assumed:
L7-only composites for 2013-2014 overlap the existing Landsat 8 epochs, and the
same patches' patch-minus-control deltas from the two sensors calibrate the
footprint-dilution bias as a function of patch size.

Same grid, QA logic and scaling as step19 (C2 L2 scale factors are identical
across TM/ETM+/TIRS); thermal asset is "lwir" (ST_B6) instead of "lwir11".
Item metadata goes to data/acc57/ so the main scene counter, which
reads data/acc/, never sees these sensors.

Usage: python3 step23_l57_composites.py landsat-7 2013 2014 2010 2011 2012
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, warnings, time, json, os, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np

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
from rasterio.transform import from_origin
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, W, H, RES, X0, Y1, BBOX_4326, write_grid

OUT = f"{_TROOT}/data"
NWORK = 2
PLATFORM = sys.argv[1]
SUF = {"landsat-5": "l5", "landsat-7": "l7"}[PLATFORM]
t0 = time.time()

# NOTE: the catalogue is opened WITHOUT sign_inplace. Signing every item at
# search time is what silently destroyed the 2020 composite in the first build
# (28 of 46 scenes died with HTTP 403 once the SAS tokens aged past an hour and
# only 12.4 M of 91 M pixels ended up with >=3 clear observations). Each item is
# signed immediately before it is read instead, so a token is never older than
# the few seconds it takes to warp one scene.
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1")


def item_subwindow(item):
    b = transform_bounds("EPSG:4326", CRS, *item.bbox, densify_pts=21)
    col0 = int(max(0, np.floor((b[0] - X0) / RES)))
    row0 = int(max(0, np.floor((Y1 - b[3]) / RES)))
    col1 = int(min(W, np.ceil((b[2] - X0) / RES)))
    row1 = int(min(H, np.ceil((Y1 - b[1]) / RES)))
    if col1 <= col0 or row1 <= row0:
        return None
    return row0, col0, row1 - row0, col1 - col0


def warp_asset(href, sub, resampling):
    row0, col0, h, w = sub
    tr = from_origin(X0 + col0 * RES, Y1 - row0 * RES, RES, RES)
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=CRS, transform=tr, width=w, height=h,
                       resampling=resampling, dtype="uint16", nodata=0) as vrt:
            return vrt.read(1)


def _write(path, arr, dtype, nodata):
    """write without the extra astype() copy write_grid would make."""
    from grid_utils import PROFILE
    prof = PROFILE.copy()
    prof.update(dtype=dtype, nodata=nodata)
    if np.issubdtype(np.dtype(dtype), np.floating):
        prof["predictor"] = 3
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr, 1)


class Acc:
    """count + sum only (variance is never used downstream)."""
    def __init__(self):
        self.n = np.zeros((H, W), np.uint16)
        self.s = np.zeros((H, W), np.float32)

    def add(self, sub, vals, valid):
        r, c, h, w = sub
        self.n[r:r+h, c:c+w] += valid.astype(np.uint16)
        self.s[r:r+h, c:c+w] += np.where(valid, vals, 0).astype(np.float32)


ACC_LOCK = threading.Lock()


def warp_resigned(it, asset, sub, resampling):
    try:
        return warp_asset(it.assets[asset].href, sub, resampling)
    except Exception as e:
        if "403" not in repr(e) and "401" not in repr(e):
            raise
        pc.sas.TOKEN_CACHE.clear()
        it.assets = pc.sign(it).assets
        return warp_asset(it.assets[asset].href, sub, resampling)


def process_item(it_raw, lst_acc, ndvi_acc):
    it = pc.sign(it_raw)                 # sign at read time, never at search time
    sub = item_subwindow(it)
    if sub is None:
        return {"id": it.id, "used": False, "reason": "no-overlap"}
    qa = warp_resigned(it, "qa_pixel", sub, Resampling.nearest)
    clear = (qa != 0) & ((qa & 0b0000000000111110) == 0)
    del qa
    nclear = int(clear.sum())
    if nclear < 500:
        return {"id": it.id, "used": False, "clear_px": nclear}

    st = warp_resigned(it, "lwir", sub, Resampling.bilinear)
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

    red = warp_resigned(it, "red", sub, Resampling.bilinear)
    nir = warp_resigned(it, "nir08", sub, Resampling.bilinear)
    valid0 = clear & (red != 0) & (nir != 0)
    del clear
    r_sr = red.astype(np.float32)
    del red
    r_sr *= 2.75e-5; r_sr -= 0.2
    n_sr = nir.astype(np.float32)
    del nir
    n_sr *= 2.75e-5; n_sr -= 0.2
    den = n_sr + r_sr
    np.subtract(n_sr, r_sr, out=n_sr)
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


def run_summer(year):
    tag = f"c{year-2000:02d}{SUF}"
    if os.path.exists(f"{OUT}/lst_{tag}.tif") and os.path.exists(f"{OUT}/ndvi_{tag}.tif"):
        print(tag, "already composed, skip", flush=True)
        return
    for attempt in range(5):
        try:
            search = cat.search(collections=["landsat-c2-l2"], bbox=BBOX_4326,
                                datetime=f"{year}-06-01/{year}-09-30",
                                query={"eo:cloud_cover": {"lt": 70},
                                       "platform": {"in": [PLATFORM]}})
            items = sorted(search.items(), key=lambda i: i.datetime)
            break
        except Exception as e:
            print("  search retry", attempt + 1, repr(e)[:70], flush=True)
            time.sleep(20)
    else:
        sys.exit(f"{tag}: search failed")
    print(f"[{tag}] {len(items)} items  {time.time()-t0:.0f}s", flush=True)
    lst_acc, ndvi_acc = Acc(), Acc()
    meta = {}
    todo = list(items)
    for rnd in range(4):
        if not todo:
            break
        done = 0
        with ThreadPoolExecutor(max_workers=NWORK) as ex:
            futs = {ex.submit(process_item, it, lst_acc, ndvi_acc): it for it in todo}
            for f in as_completed(futs):
                it = futs[f]
                try:
                    meta[it.id] = f.result()
                except Exception as e:
                    meta[it.id] = {"id": it.id, "used": False,
                                   "error": repr(e)[:150]}
                done += 1
                if done % 10 == 0:
                    print(f"  [{tag}] r{rnd} {done}/{len(todo)} "
                          f"{time.time()-t0:.0f}s", flush=True)
        # a scene that errored is retried with a freshly signed href; a scene
        # legitimately dropped (no overlap / too cloudy) is not
        todo = [it for it in todo if "error" in meta.get(it.id, {})]
        if todo:
            print(f"  [{tag}] retry round {rnd+1}: {len(todo)} scenes", flush=True)
            time.sleep(20)
    meta = list(meta.values())
    used = sum(1 for m in meta if m.get("used"))
    nerr = sum(1 for m in meta if "error" in m)
    if nerr:
        print(f"  [{tag}] WARNING {nerr} scenes unrecovered", flush=True)
    # Compose in place and release each accumulator as soon as it is written.
    # The first version built a float32 copy of the counts, a masked mean and an
    # astype() copy while BOTH accumulators were still alive; with two extractors
    # running that peak was enough to get the process OOM-killed mid-write.
    for band in ("lst", "ndvi"):
        acc = lst_acc if band == "lst" else ndvi_acc
        cnt = acc.n
        mean = acc.s                       # reuse the sum buffer, no new array
        np.divide(mean, np.maximum(cnt, 1).astype(np.float32), out=mean)
        mean[cnt < 3] = -9999.0
        nval = int((cnt >= 3).sum())
        _write(f"{OUT}/{band}_{tag}.tif", mean, "float32", -9999)
        _write(f"{OUT}/nclear_{band}_{tag}.tif",
               np.minimum(cnt, 65535).astype(np.uint16), "uint16", 0)
        print(f"  composed {band}_{tag}: valid px {nval}", flush=True)
        acc.n = None; acc.s = None
        if band == "lst":
            lst_acc = None
        else:
            ndvi_acc = None
        del acc, cnt, mean
    os.makedirs(f"{OUT}/acc57", exist_ok=True)
    json.dump(meta, open(f"{OUT}/acc57/{tag}_items.json", "w"))
    print(f"[{tag}] DONE used={used}/{len(items)}  {time.time()-t0:.0f}s", flush=True)


for y in sys.argv[2:]:
    run_summer(int(y))
print("STEP19 COMPLETE", f"{time.time()-t0:.0f}s")
