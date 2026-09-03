"""Step 6c: two additional 26-year case trajectories for the v9 case figure —
(1) high_elev_event: largest clean event landslide above 2,000 m with a long
trajectory (year <= 2015); (2) small_morakot_event: a small (<2 ha) clean
Morakot-cohort event landslide as the area contrast to the giant Morakot
annual-loss case. Patch masks come directly from stored pixel lists.
Appends to data/case_trajectories.parquet (replacing same-named cases).
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, warnings, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

os.environ.update({
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIPLEX": "YES", "GDAL_HTTP_VERSION": "2",
    "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": "30000000", "GDAL_CACHEMAX": "256",
})
import rasterio
from rasterio.warp import Resampling, transform_bounds
from rasterio.vrt import WarpedVRT
from rasterio.transform import from_origin
from scipy import ndimage as ndi
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, RES, X0, Y1, H, W, read_grid

OUT = f"{_TROOT}/data"
t0 = time.time()
dem = read_grid(f"{OUT}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{OUT}/slope30.tif"); slope[slope == -9999] = np.nan
intact = read_grid(f"{OUT}/intact2_30.tif") == 1

raw = pd.read_parquet(f"{OUT}/patches_raw2.parquet")
dd = pd.read_parquet(f"{OUT}/patches_deltas2.parquet",
                     columns=["src", "agent", "year", "n_ctrl", "redist_frac",
                              "forest2000_frac", "t_event", "elev", "area_ha"])
ok = (dd.src == "event") & (dd.n_ctrl >= 30) & (dd.redist_frac < 0.05) \
     & (dd.forest2000_frac > 0.7)

hi = raw[ok & (dd.elev > 2000) & (raw.year <= 2015) & (raw.year >= 2008)]
hrow = hi.loc[dd.loc[hi.index].area_ha.idxmax()]
mor = raw[ok & (dd.t_event >= 2009.5) & (dd.t_event <= 2009.75)
          & (dd.area_ha < 2) & (dd.elev > 800)]
srow = mor.loc[dd.loc[mor.index].area_ha.idxmax()]
CASES = [("high_elev_event", hrow, float(dd.loc[hrow.name, "t_event"])),
         ("small_morakot_event", srow, float(dd.loc[srow.name, "t_event"]))]
for nm, r, te in CASES:
    print(nm, "n_px", int(r.n_px), "elev", round(float(dd.loc[r.name, 'elev'])),
          "area", round(float(dd.loc[r.name, 'area_ha']), 2), "t_event",
          round(te, 2), flush=True)

cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)


def warp_small(href, tr, w_, h_, resampling, dtype="uint16"):
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=CRS, transform=tr, width=w_, height=h_,
                       resampling=resampling, dtype=dtype, nodata=0) as vrt:
            return vrt.read(1)


def run_case(name, prow, t_event):
    PAD = 50
    r0 = max(0, int(prow.r0) - PAD); c0 = max(0, int(prow.c0) - PAD)
    r1 = min(H, int(prow.r1) + PAD); c1 = min(W, int(prow.c1) + PAD)
    h_, w_ = r1 - r0, c1 - c0
    tr = from_origin(X0 + c0 * RES, Y1 - r0 * RES, RES, RES)
    pm = np.zeros((h_, w_), bool)
    pm[np.asarray(prow.px_rows) - r0, np.asarray(prow.px_cols) - c0] = True
    p_elev = float(dd.loc[prow.name, "elev"])
    dem_w = dem[r0:r1, c0:c1]; slope_w = slope[r0:r1, c0:c1]
    p_slope = float(np.nanmean(slope_w[pm]))
    dist = ndi.distance_transform_edt(~pm)
    cm = (intact[r0:r1, c0:c1] & (dist >= 4) & (dist <= 30)
          & (np.abs(dem_w - p_elev) <= 150) & (np.abs(slope_w - p_slope) <= 10))
    if cm.sum() < 30:
        cm = (intact[r0:r1, c0:c1] & (dist >= 4) & (dist <= 50)
              & (np.abs(dem_w - p_elev) <= 250) & (np.abs(slope_w - p_slope) <= 15))
    print(name, "patch px", int(pm.sum()), "ctrl px", int(cm.sum()), flush=True)
    bbox4326 = transform_bounds(CRS, "EPSG:4326", X0 + c0 * RES, Y1 - r1 * RES,
                                X0 + c1 * RES, Y1 - r0 * RES)
    rows = []
    lock = threading.Lock()

    def one_item(it):
        akeys = it.assets.keys()
        st_key = "lwir11" if "lwir11" in akeys else ("lwir" if "lwir" in akeys else None)
        if st_key is None:
            return
        qa = warp_small(it.assets["qa_pixel"].href, tr, w_, h_, Resampling.nearest)
        clear = (qa != 0) & ((qa & 0b0000000000111110) == 0)
        if clear[pm].sum() < max(3, 0.2 * pm.sum()) or clear[cm].sum() < 15:
            return
        st = warp_small(it.assets[st_key].href, tr, w_, h_, Resampling.bilinear)
        lst = st.astype(np.float32) * 0.00341802 + 149.0 - 273.15
        v = clear & (st != 0) & (lst > -5) & (lst < 60)
        red = warp_small(it.assets["red"].href, tr, w_, h_, Resampling.bilinear)
        nir = warp_small(it.assets["nir08"].href, tr, w_, h_, Resampling.bilinear)
        r_sr = red.astype(np.float32) * 2.75e-5 - 0.2
        n_sr = nir.astype(np.float32) * 2.75e-5 - 0.2
        den = r_sr + n_sr
        with np.errstate(all="ignore"):
            ndvi = np.where(np.abs(den) > 1e-6, (n_sr - r_sr) / den, np.nan)
        vn = clear & (red != 0) & (nir != 0) & np.isfinite(ndvi)

        def mstat(a, valid, m):
            vv = a[valid & m]
            return (float(np.mean(vv)), int(len(vv))) if len(vv) >= 3 else (np.nan, int(len(vv)))
        lp, nlp = mstat(lst, v, pm); lc, nlc = mstat(lst, v, cm)
        np_, nnp = mstat(ndvi, vn, pm); nc, nnc = mstat(ndvi, vn, cm)
        with lock:
            rows.append(dict(date=str(it.datetime.date()), year=it.datetime.year,
                             platform=it.properties.get("platform"),
                             lst_p=lp, lst_c=lc, n_lst_p=nlp, n_lst_c=nlc,
                             ndvi_p=np_, ndvi_c=nc))

    for yr in range(2000, 2027):
        end = "2026-08-10" if yr == 2026 else f"{yr}-09-30"
        for attempt in (1, 2, 3):
            try:
                items = list(cat.search(collections=["landsat-c2-l2"],
                                        bbox=list(bbox4326),
                                        datetime=f"{yr}-06-01/{end}",
                                        query={"eo:cloud_cover": {"lt": 85}}).items())
                break
            except Exception as e:
                print(name, yr, "search retry", repr(e)[:60], flush=True)
                time.sleep(20 * attempt)
                items = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(one_item, it) for it in items]
            for f in as_completed(futs):
                try:
                    f.result()
                except Exception:
                    pass
    df = pd.DataFrame(rows)
    df["case"] = name; df["t_event"] = t_event
    df["patch_px"] = int(pm.sum()); df["ctrl_px"] = int(cm.sum())
    df["elev"] = p_elev; df["area_ha"] = float(dd.loc[prow.name, "area_ha"])
    print(name, "obs rows:", len(df), f"{time.time()-t0:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    new = []
    for name, prow, t_ev in CASES:
        d = run_case(name, prow, t_ev)
        if d is not None and len(d):
            new.append(d)
    old = pd.read_parquet(f"{OUT}/case_trajectories.parquet")
    names = [c[0] for c in CASES]
    old = old[~old.case.isin(names)]
    pd.concat([old] + new).to_parquet(f"{OUT}/case_trajectories.parquet")
    print("STEP6C COMPLETE", f"{time.time()-t0:.0f}s")
