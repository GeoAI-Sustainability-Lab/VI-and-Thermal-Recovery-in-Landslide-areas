"""Step 13d: mop up remaining S1 items for ONE date (CLI arg), in a fresh
process so the planetary_computer SAS token is newly issued (the long 13c run
hit 403s once its in-process token went stale ~45 min in). Loads the shared
checkpoint, processes this date's missing items, saves the checkpoint.
Usage: python3 step13d_mopup.py 2025-07-28 [--skip-small]
--skip-small: the small (~20-patch) frame of this date is already in the
checkpoint (used for 2025-07-10), so skip the item with min expected coverage.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, time, warnings
warnings.filterwarnings("ignore")
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                  GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2",
                  GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                  CPL_VSIL_CURL_CACHE_SIZE="200000000", GDAL_CACHEMAX="512")
import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds, Resampling
from scipy import ndimage as ndi
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, H, W, X0, Y1

DATE = sys.argv[1]
SKIP_SMALL = "--skip-small" in sys.argv
YR = int(DATE[:4])
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
t0 = time.time()

# ---------- sample + masks (identical to step13c, seed 7) ----------
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
    take = min(120, len(g))
    idx += list(rng.choice(g.index.values, take, replace=False))
sample = raw.loc[idx]
samp_meta = dd.loc[idx]
dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{D}/slope30.tif"); slope[slope == -9999] = np.nan
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
    pm10 = np.kron(pm, np.ones((3, 3), bool))
    cm10 = np.kron(cm, np.ones((3, 3), bool))
    P10.append((pid, r0 * 3, c0 * 3, pm10, cm10))
del dem, slope, intact
print(DATE, "masks ready:", len(P10), f"{time.time()-t0:.0f}s", flush=True)

TR10 = from_origin(X0, Y1, 10.0, 10.0)
W10, H10 = W * 3, H * 3
import os as _os
if _os.path.exists(f"{O}/s1_acc_ckpt.npz"):
    z = np.load(f"{O}/s1_acc_ckpt.npz")
    acc = {(int(k[0]), int(k[1])): v.copy() for k, v in zip(z["keys"], z["vals"])}
else:
    acc = {}


def n_expected(item):
    w_, s_, e_, n_ = item.bbox
    bx = transform_bounds("EPSG:4326", "EPSG:3826", w_, s_, e_, n_)
    n = 0
    for pid, r0_10, c0_10, pm10, cm10 in P10:
        hh, ww = pm10.shape
        x0p = X0 + c0_10 * 10; y1p = Y1 - r0_10 * 10
        x1p = x0p + ww * 10; y0p = y1p - hh * 10
        if x0p >= bx[0] and x1p <= bx[2] and y0p >= bx[1] and y1p <= bx[3]:
            n += 1
    return n


def process_item(it):
    pc.sas.TOKEN_CACHE.clear()
    it_s = pc.sign(it)
    srcs = {b: rasterio.open(it_s.assets[b].href) for b in ["vh", "vv"]}
    bx = transform_bounds(srcs["vh"].crs, "EPSG:3826", *srcs["vh"].bounds)
    vrts = {b: WarpedVRT(srcs[b], crs="EPSG:3826", transform=TR10,
                         width=W10, height=H10,
                         resampling=Resampling.bilinear,
                         src_nodata=-32768, nodata=np.nan)
            for b in ["vh", "vv"]}
    nin = 0
    for pid, r0_10, c0_10, pm10, cm10 in P10:
        hh, ww = pm10.shape
        x0p = X0 + c0_10 * 10; y1p = Y1 - r0_10 * 10
        x1p = x0p + ww * 10; y0p = y1p - hh * 10
        if not (x0p >= bx[0] and x1p <= bx[2] and y0p >= bx[1] and y1p <= bx[3]):
            continue
        win = ((r0_10, r0_10 + hh), (c0_10, c0_10 + ww))
        a = acc.setdefault((pid, YR), np.zeros(8))
        ok = False
        for bi, b in enumerate(["vh", "vv"]):
            arr = vrts[b].read(1, window=win)
            arr[arr <= 0] = np.nan
            pv = arr[pm10]; cv = arr[cm10]
            np_, nc_ = np.isfinite(pv).sum(), np.isfinite(cv).sum()
            if np_ >= 0.3 * pm10.sum() and nc_ >= 60:
                a[bi*4 + 0] += np.nansum(pv); a[bi*4 + 1] += np_
                a[bi*4 + 2] += np.nansum(cv); a[bi*4 + 3] += nc_
                ok = True
        if ok:
            nin += 1
    for v in vrts.values():
        v.close()
    for s_ in srcs.values():
        s_.close()
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
        print("  search retry", k + 1, repr(e)[:80], flush=True)
        time.sleep(45)
else:
    sys.exit(f"{DATE}: search failed")
print(DATE, "items:", [i.id[-14:] for i in items], flush=True)

if SKIP_SMALL and len(items) > 1:
    nexp = [n_expected(it) for it in items]
    skip_i = int(np.argmin(nexp))
    print(f"  skipping done small frame n_exp={nexp[skip_i]} of {nexp}", flush=True)
    items = [it for i, it in enumerate(items) if i != skip_i]

nfail = 0
for it in items:
    done = False
    for attempt in (1, 2, 3):
        try:
            nin = process_item(it)
            print(f"  {DATE} {it.id[-14:]} patches:{nin} {time.time()-t0:.0f}s",
                  flush=True)
            done = True
            break
        except Exception as e:
            print(f"  attempt{attempt} fail {it.id[-14:]} {repr(e)[:70]}", flush=True)
            time.sleep(30 * attempt)
    if not done:
        nfail += 1
        print(f"  GIVEN UP {it.id[-14:]}", flush=True)

ks = np.array([[k[0], k[1]] for k in acc], dtype=np.int64)
vs = np.stack(list(acc.values()))
np.savez(f"{O}/s1_acc_ckpt.npz", keys=ks, vals=vs)
print(f"{DATE} DONE fails:{nfail} {time.time()-t0:.0f}s", flush=True)
