"""Step 7b: Sentinel-3 SLSTR LST day/night composites over Taiwan (Jul-Aug 2025)
and day-night amplitude by canopy-structure class. Real swath NetCDFs from
Planetary Computer; cloud-masked via flags_in confidence summary_cloud bit.
Output: outputs/s3_daynight.npz + stats in outputs/s3_results.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, time, warnings, tempfile, urllib.request
warnings.filterwarnings("ignore")
import numpy as np
import netCDF4
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, X0, Y1, RES, W, H

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
t0 = time.time()

# 0.02 deg grid over Taiwan
LON0, LON1, LAT0, LAT1 = 119.9, 122.1, 21.8, 25.4
GR = 0.02
NX = int(round((LON1 - LON0) / GR)); NY = int(round((LAT1 - LAT0) / GR))

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
items = list(cat.search(collections=["sentinel-3-slstr-lst-l2-netcdf"],
                        bbox=[120.2, 22.0, 121.9, 25.3],
                        datetime="2025-06-01/2025-09-30").items())
print("S3 items:", len(items), flush=True)


def fetch(url, dest):
    urllib.request.urlretrieve(url, dest)
    return dest


def accumulate(item, acc):
    """acc: dict with 'sum','n' (NY,NX) per orbit-state key."""
    state = item.properties["sat:orbit_state"]        # descending=day (~10:00), ascending=night (~22:00)
    key = "day" if state == "descending" else "night"
    with tempfile.TemporaryDirectory() as td:
        try:
            f_lst = fetch(item.assets["lst-in"].href, f"{td}/lst.nc")
            f_geo = fetch(item.assets["slstr-geodetic-in"].href, f"{td}/geo.nc")
            f_flg = fetch(item.assets["slstr-flags-in"].href, f"{td}/flags.nc")
        except Exception as e:
            print("  dl fail", item.id[:60], repr(e)[:80], flush=True)
            return 0
        with netCDF4.Dataset(f_lst) as ds:
            lst = ds["LST"][:].filled(np.nan)
            if lst.ndim == 3:
                lst = lst[0]
        with netCDF4.Dataset(f_geo) as ds:
            lat = ds["latitude_in"][:].filled(np.nan)
            lon = ds["longitude_in"][:].filled(np.nan)
        with netCDF4.Dataset(f_flg) as ds:
            conf = ds["confidence_in"][:].filled(0)
        cloud = (conf.astype(np.uint32) & 16384) > 0          # summary_cloud bit 14
        ok = (np.isfinite(lst) & ~cloud & (lat >= LAT0) & (lat < LAT1)
              & (lon >= LON0) & (lon < LON1) & (lst > 250) & (lst < 340))
        if ok.sum() < 100:
            return 0
        ix = ((lon[ok] - LON0) / GR).astype(np.int32)
        iy = ((LAT1 - lat[ok]) / GR).astype(np.int32)
        flat = iy * NX + ix
        np.add.at(acc[key]["sum"].ravel(), flat, lst[ok] - 273.15)
        np.add.at(acc[key]["n"].ravel(), flat, 1)
        return int(ok.sum())


acc = {k: {"sum": np.zeros((NY, NX), np.float64), "n": np.zeros((NY, NX), np.int32)}
       for k in ["day", "night"]}
used = {"day": 0, "night": 0}
for it in items:
    state = "day" if it.properties["sat:orbit_state"] == "descending" else "night"
    if used[state] >= 60:
        continue
    got = accumulate(it, acc)
    if got:
        used[state] += 1
        print(f"  {state} {used['day']}+{used['night']}  {it.id[:48]}  px={got}  "
              f"{time.time()-t0:.0f}s", flush=True)
    if used["day"] >= 60 and used["night"] >= 60:
        break

day = np.where(acc["day"]["n"] >= 3, acc["day"]["sum"] / np.maximum(acc["day"]["n"], 1), np.nan)
night = np.where(acc["night"]["n"] >= 3, acc["night"]["sum"] / np.maximum(acc["night"]["n"], 1), np.nan)

# aggregate CHM / DEM / forest to the S3 grid
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.transform import from_origin
tr = from_origin(LON0, LAT1, GR, GR)


def to_s3grid(path, resampling, nodata_in=None):
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs="EPSG:4326", transform=tr, width=NX, height=NY,
                       resampling=resampling, dtype="float32",
                       src_nodata=nodata_in, nodata=np.nan) as vrt:
            return vrt.read(1)


chm = to_s3grid(f"{D}/chm_eth30.tif", Resampling.average, -9999)
dem = to_s3grid(f"{D}/dem30.tif", Resampling.average, -9999)
treefrac = to_s3grid(f"{D}/intact30.tif", Resampling.average, 255)

amp = day - night
stats = []
for lo, hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2800)]:
    band = (dem >= lo) & (dem < hi) & (treefrac > 0.35) & np.isfinite(amp) & np.isfinite(chm)
    for cl, (clo, chi_) in {"short (<15 m)": (0, 15), "tall (>=22 m)": (22, 60)}.items():
        m = band & (chm >= clo) & (chm < chi_)
        if m.sum() >= 12:
            stats.append(dict(elev_band=f"{lo}-{hi}", chm_class=cl, n=int(m.sum()),
                              day=float(np.nanmean(day[m])), night=float(np.nanmean(night[m])),
                              amp_mean=float(np.nanmean(amp[m])),
                              amp_se=float(np.nanstd(amp[m]) / np.sqrt(m.sum()))))

np.savez_compressed(f"{O}/s3_daynight.npz", day=day.astype(np.float32),
                    night=night.astype(np.float32), chm=chm, dem=dem, treefrac=treefrac,
                    n_day=acc["day"]["n"], n_night=acc["night"]["n"],
                    grid=np.array([LON0, LAT1, GR, NX, NY]))
json.dump(dict(used=used, stats=stats), open(f"{O}/s3_results.json", "w"), indent=1)
print(json.dumps(stats, indent=1)[:900])
print("STEP7B COMPLETE", f"{time.time()-t0:.0f}s")
