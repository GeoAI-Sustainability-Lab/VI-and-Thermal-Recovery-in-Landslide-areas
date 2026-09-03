"""Step 11: sample Meta/WRI CHMv2 (1 m, DINOv3, EPSG:3857 COGs on public S3)
at a stratified subset of the intact-forest buffering sample, aggregate the
1 m pixels within each 30 m analysis cell, and compare against the ETH 10 m
canopy height model: agreement stats and the elevation-band LST-height slope
re-estimated with CHMv2 (same residualization as step08a).
Outputs: data/chm2_sample.parquet, outputs/chm2_compare.json
Reads COGs directly from the public S3 bucket (CC BY 4.0) — same tiles the
user archived in Dataset/01_SOURCE/Terrain_Canopy/CHMv2_Meta_WRI_2026.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import os, json, time, warnings
warnings.filterwarnings("ignore")
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                  CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                  GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_VERSION="2",
                  CPL_VSIL_CURL_CACHE_SIZE="200000000", GDAL_CACHEMAX="512",
                  GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from scipy import stats

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
URL = ("https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/"
       "dinov3_global_chm_v2_ml3/chm/{tile}.tif")
BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3600)]
PER_BAND = 5000
t0 = time.time()

df = pd.read_parquet(f"{D}/buffering_sample.parquet")
rng = np.random.default_rng(42)
parts = []
for lo, hi in BANDS:
    d = df[(df.elev >= lo) & (df.elev < hi)]
    take = min(PER_BAND, len(d))
    parts.append(d.sample(take, random_state=42))
s = pd.concat(parts, ignore_index=True)
print("sample:", len(s), flush=True)

to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
to_merc = Transformer.from_crs(3826, 3857, always_xy=True)
lon, lat = to_ll.transform(s.x.values, s.y.values)
mx, my = to_merc.transform(s.x.values, s.y.values)
s["lon"], s["lat"], s["mx"], s["my"] = lon, lat, mx, my

tiles = pd.read_csv(f"{D}/chm2_tiles.csv")
s["tile"] = ""
for r in tiles.itertuples():
    m = (s.lon >= r.west) & (s.lon < r.east) & (s.lat >= r.south) & (s.lat < r.north)
    s.loc[m, "tile"] = str(r.tile)
print("unassigned:", int((s.tile == "").sum()), flush=True)
s = s[s.tile != ""].reset_index(drop=True)

out_chm2, out_nv = np.full(len(s), np.nan), np.zeros(len(s), int)
for tk, (tile, g) in enumerate(s.groupby("tile")):
    url = "/vsicurl/" + URL.format(tile=tile)
    try:
        with rasterio.open(url) as src:
            nod = src.nodata
            res = src.res[0]
            half = max(2, int(round(15.0 / (res * np.cos(np.radians(g.lat.mean()))))))
            idx = g.sort_values("my", ascending=False).index
            for i in idx:
                r_, c_ = src.index(s.mx[i], s.my[i])
                r0, c0 = max(0, r_ - half), max(0, c_ - half)
                w = Window(c0, r0, min(2*half+1, src.width - c0),
                           min(2*half+1, src.height - r0))
                a = src.read(1, window=w).astype(np.float32)
                if nod is not None:
                    a[a == nod] = np.nan
                a[(a < 0) | (a > 150)] = np.nan
                nv = int(np.isfinite(a).sum())
                if nv >= 0.4 * (2*half+1)**2:
                    out_chm2[i] = float(np.nanmean(a))
                    out_nv[i] = nv
    except Exception as e:
        print("tile fail", tile, repr(e)[:80], flush=True)
    print(f"tile {tk+1}/{s.tile.nunique()} {tile} done {time.time()-t0:.0f}s", flush=True)

s["chm2"] = out_chm2; s["chm2_nvalid"] = out_nv
s.to_parquet(f"{D}/chm2_sample.parquet")
v = s[np.isfinite(s.chm2)]
print("valid:", len(v), flush=True)

# ---------------- agreement & slope sensitivity ----------------
res = {"n": int(len(v)), "r": float(np.corrcoef(v.chm, v.chm2)[0, 1]),
       "bias_mean": float((v.chm2 - v.chm).mean()),
       "rmsd": float(np.sqrt(((v.chm2 - v.chm) ** 2).mean()))}
by_h = {}
for lo, hi in [(0, 5), (5, 15), (15, 25), (25, 35), (35, 60)]:
    g = v[(v.chm >= lo) & (v.chm < hi)]
    if len(g) >= 100:
        by_h[f"{lo}-{hi}"] = dict(n=int(len(g)), bias=float((g.chm2 - g.chm).mean()),
                                  sd=float((g.chm2 - g.chm).std()))
res["bias_by_eth_class"] = by_h

slopes = {}
for lo, hi in BANDS:
    d = v[(v.elev >= lo) & (v.elev < hi)]
    if len(d) < 800:
        continue
    Xc = np.column_stack([d.cos_i, d.slope, d.northness, np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(Xc, d.lst, rcond=None)
    lst_res = d.lst - Xc @ beta + d.lst.mean()
    sl2, _, r2_, p2, se2 = stats.linregress(d.chm2, lst_res)
    sl1, _, r1_, p1, se1 = stats.linregress(d.chm, lst_res)
    slopes[f"{lo}-{hi}"] = dict(n=int(len(d)),
                                eth=dict(slope_per_m=float(sl1), se=float(se1)),
                                chm2=dict(slope_per_m=float(sl2), se=float(se2)))
res["band_slopes"] = slopes
by_e = {}
for lo, hi in BANDS:
    g = v[(v.elev >= lo) & (v.elev < hi)]
    if len(g) >= 500:
        by_e[f"{lo}-{hi}"] = dict(n=int(len(g)), bias=float((g.chm2 - g.chm).mean()))
res["bias_by_elev"] = by_e
json.dump(res, open(f"{O}/chm2_compare.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ["n", "r", "bias_mean", "rmsd"]}))
print("STEP11 COMPLETE", f"{time.time()-t0:.0f}s")
