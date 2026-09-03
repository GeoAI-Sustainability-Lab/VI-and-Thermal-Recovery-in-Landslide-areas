"""Step 13h: FAST per-date Sentinel-1 extraction (replaces step13d for the v9
re-run). Same sample, masks, matching and Δγ0 definition; the difference is
that each item's 511 patch windows are read by N worker threads, each holding
its OWN rasterio dataset + WarpedVRT handles (rasterio datasets are not safe
for concurrent reads). Cuts a frame from ~370 s to ~60 s, so a whole date
finishes in ~2 min and no process lives long enough to lose its SAS token.
Frees the 30 m rasters after mask construction (container is 7 GB).
Usage: python3 step13h_fast.py 2023-06-15
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, time, threading, warnings
warnings.filterwarnings("ignore")
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                  GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2",
                  GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                  CPL_VSIL_CURL_CACHE_SIZE="100000000", GDAL_CACHEMAX="256")
import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds, Resampling
from scipy import ndimage as ndi
import pystac_client, planetary_computer as pc
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, H, W, X0, Y1

DATE = sys.argv[1]
YR = int(DATE[:4])
NWORK = 6
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
t0 = time.time()

# ---------- sample + masks (identical to step13/13d, seed 7) ----------
raw = pd.read_parquet(f"{D}/patches_raw2.parquet")
dd = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                     columns=["src", "agent", "year", "n_ctrl", "redist_frac",
                              "forest2000_frac", "t_event", "elev", "area_ha"])
ev = dd[(dd.src == "event") & (dd.n_ctrl >= 30) & (dd.redist_frac < 0.05)
        & (dd.forest2000_frac > 0.6)].copy()
ev["age25"] = 2025.62 - ev.t_event
bins = [(0.7, 3), (3, 6), (6, 10), (10, 15), (15, 21.5)]
idx = []
rng = np.random.default_rng(7)
for lo, hi in bins:
    g = ev[(ev.age25 >= lo) & (ev.age25 < hi)]
    idx += list(rng.choice(g.index.values, min(120, len(g)), replace=False))
sample = raw.loc[idx]
samp_meta = dd.loc[idx]

dem = read_grid(f"{D}/dem30.tif").astype(np.float32); dem[dem == -9999] = np.nan
slope = read_grid(f"{D}/slope30.tif").astype(np.float32); slope[slope == -9999] = np.nan
intact = read_grid(f"{D}/intact2_30.tif") == 1
PAD = 40
P10 = []
for pid, row in sample.iterrows():
    r0 = max(0, row.r0 - PAD); c0 = max(0, row.c0 - PAD)
    r1 = min(H, row.r1 + PAD); c1 = min(W, row.c1 + PAD)
    h_, w_ = r1 - r0, c1 - c0
    pm = np.zeros((h_, w_), bool)
    pm[np.asarray(row.px_rows) - r0, np.asarray(row.px_cols) - c0] = True
    dem_w = dem[r0:r1, c0:c1]; slope_w = slope[r0:r1, c0:c1]
    p_elev = float(np.nanmean(dem_w[pm])); p_slope = float(np.nanmean(slope_w[pm]))
    dist = ndi.distance_transform_edt(~pm)
    base_ok = intact[r0:r1, c0:c1] & np.isfinite(dem_w) & np.isfinite(slope_w)
    cm = (base_ok & (dist >= 4) & (dist <= 30)
          & (np.abs(dem_w - p_elev) <= 150) & (np.abs(slope_w - p_slope) <= 10))
    if cm.sum() < 30:
        cm = (base_ok & (dist >= 4) & (dist <= 50)
              & (np.abs(dem_w - p_elev) <= 250) & (np.abs(slope_w - p_slope) <= 15))
    if cm.sum() < 20:
        continue
    P10.append((pid, r0 * 3, c0 * 3,
                np.kron(pm, np.ones((3, 3), bool)),
                np.kron(cm, np.ones((3, 3), bool))))
del dem, slope, intact, raw
print(DATE, "masks ready:", len(P10), f"{time.time()-t0:.0f}s", flush=True)

TR10 = from_origin(X0, Y1, 10.0, 10.0)
W10, H10 = W * 3, H * 3
SOLO = "--solo" in sys.argv
CKPT = f"{O}/s1_acc_{DATE}.npz" if SOLO else f"{O}/s1_acc_ckpt.npz"
if not SOLO and os.path.exists(CKPT):
    z = np.load(CKPT)
    acc = {(int(k[0]), int(k[1])): v.copy() for k, v in zip(z["keys"], z["vals"])}
else:
    acc = {}
lock = threading.Lock()


def open_vrts(href_vh, href_vv):
    srcs = {b: rasterio.open(h) for b, h in [("vh", href_vh), ("vv", href_vv)]}
    vrts = {b: WarpedVRT(srcs[b], crs="EPSG:3826", transform=TR10,
                         width=W10, height=H10, resampling=Resampling.bilinear,
                         src_nodata=-32768, nodata=np.nan)
            for b in ("vh", "vv")}
    return srcs, vrts


def worker(chunk, hrefs, bx):
    srcs, vrts = open_vrts(*hrefs)
    local = {}
    nin = 0
    try:
        for pid, r0_10, c0_10, pm10, cm10 in chunk:
            hh, ww = pm10.shape
            x0p = X0 + c0_10 * 10; y1p = Y1 - r0_10 * 10
            x1p = x0p + ww * 10; y0p = y1p - hh * 10
            if not (x0p >= bx[0] and x1p <= bx[2] and y0p >= bx[1] and y1p <= bx[3]):
                continue
            win = ((r0_10, r0_10 + hh), (c0_10, c0_10 + ww))
            a8 = np.zeros(8)
            ok = False
            for bi, b in enumerate(("vh", "vv")):
                arr = vrts[b].read(1, window=win)
                arr[arr <= 0] = np.nan
                pv = arr[pm10]; cv = arr[cm10]
                np_, nc_ = np.isfinite(pv).sum(), np.isfinite(cv).sum()
                if np_ >= 0.3 * pm10.sum() and nc_ >= 60:
                    a8[bi*4 + 0] = np.nansum(pv); a8[bi*4 + 1] = np_
                    a8[bi*4 + 2] = np.nansum(cv); a8[bi*4 + 3] = nc_
                    ok = True
            if ok:
                local[pid] = a8
                nin += 1
    finally:
        for v in vrts.values():
            v.close()
        for s_ in srcs.values():
            s_.close()
    with lock:
        for pid, a8 in local.items():
            acc.setdefault((pid, YR), np.zeros(8))[:] += a8
    return nin


cat = None
for k in range(6):
    try:
        cat = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1")
        items = [i for i in cat.search(collections=["sentinel-1-rtc"],
                                       bbox=[119.9, 21.8, 122.1, 25.4],
                                       datetime=f"{DATE}/{DATE}").items()
                 if i.properties.get("sat:orbit_state") == "descending"
                 and i.properties.get("sat:relative_orbit") == 105]
        break
    except Exception as e:
        print("  search retry", k + 1, repr(e)[:70], flush=True)
        time.sleep(30)
else:
    sys.exit(f"{DATE}: search failed")
print(DATE, "items:", len(items), flush=True)

nfail = 0
for it in items:
    done = False
    for attempt in (1, 2):
        try:
            pc.sas.TOKEN_CACHE.clear()
            it_s = pc.sign(it)
            hrefs = (it_s.assets["vh"].href, it_s.assets["vv"].href)
            with rasterio.open(hrefs[0]) as s0:
                bx = transform_bounds(s0.crs, "EPSG:3826", *s0.bounds)
            chunks = [P10[i::NWORK] for i in range(NWORK)]
            with ThreadPoolExecutor(max_workers=NWORK) as ex:
                nin = sum(ex.map(lambda c: worker(c, hrefs, bx), chunks))
            print(f"  {DATE} {it.id[-10:]} patches:{nin} {time.time()-t0:.0f}s",
                  flush=True)
            done = True
            break
        except Exception as e:
            print(f"  attempt{attempt} fail {it.id[-14:]} {repr(e)[:70]}",
                  flush=True)
            time.sleep(15)
    if not done:
        nfail += 1
        print(f"  GIVEN UP {it.id[-14:]}", flush=True)

ks = np.array([[k[0], k[1]] for k in acc], dtype=np.int64)
vs = np.stack(list(acc.values()))
np.savez(CKPT, keys=ks, vals=vs)
print(f"{DATE} DONE fails:{nfail} {time.time()-t0:.0f}s", flush=True)
