"""Step 6: 26-year summer LST/NDVI event-time trajectories for selected cases.

Cases: Huisun 2021 fire; largest 0403-earthquake & 凱米 typhoon landslides (2024);
largest Hansen patches of 2004 / 2009 (Morakot) / 2015; and a placebo intact site.
Per summer (Jun-Sep 2000..2026): clear-sky mean LST & NDVI for patch vs matched
control annulus -> annual ΔLST(t), ΔNDVI(t).
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, warnings, time, json, threading
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
from rasterio.warp import Resampling
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
intact = read_grid(f"{OUT}/intact30.tif") == 1
patches = pd.read_parquet(f"{OUT}/patches_raw.parquet")
event_codes = {int(k): v for k, v in json.load(open(f"{OUT}/event_codes.json")).items()}

# ---------- case selection ----------
cases = []
fire = patches[patches.src == "fire2021"].nlargest(1, "n_px").iloc[0]
cases.append(("huisun_fire_2021", fire, 2021.37))

ard = patches[patches.src == "ardswc"].copy()
ard["agent"] = ard.event_code.map(lambda c: event_codes[c]["agent"])
ard["ev"] = ard.event_code.map(lambda c: event_codes[c]["event"])
eq = ard[(ard.ev == "0403花蓮地震") & (ard.forest2000_frac > 0.7)].nlargest(1, "n_px").iloc[0]
cases.append(("hualien_eq_2024", eq, 2024.25))
km = ard[(ard.ev == "凱米颱風") & (ard.forest2000_frac > 0.7)].nlargest(1, "n_px").iloc[0]
cases.append(("gaemi_typhoon_2024", km, 2024.56))

h = patches[patches.src == "hansen"]
for yr, tag in [(2004, "hansen_2004"), (2009, "hansen_2009_morakot"), (2015, "hansen_2015")]:
    hh = h[(h.year == yr) & (h.elev > 500)]
    if len(hh):
        cases.append((tag, hh.nlargest(1, "n_px").iloc[0], yr + 0.5))

print("cases:", [(c[0], int(c[1].n_px), round(float(c[1].elev)) ) for c in cases], flush=True)

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

    # rebuild patch mask locally (same labelling as step03 within window: recompute
    # via source masks) — safest: use bbox of the stored patch and events/lossyear
    ly = read_grid(f"{OUT}/hansen_lossyear30.tif")[r0:r1, c0:c1]
    ev = read_grid(f"{OUT}/events30.tif")[r0:r1, c0:c1]
    tc = read_grid(f"{OUT}/treecover2000_30.tif")[r0:r1, c0:c1]
    if prow.src == "hansen":
        mask_src = (ly == prow.year - 2000) & (tc >= 60) & (ev == 0)
    elif prow.src == "ardswc":
        mask_src = ev == prow.event_code
    else:
        mask_src = ev == max(event_codes)
    lab, _ = ndi.label(mask_src, np.ones((3, 3), bool))
    # component nearest to stored centroid
    cy, cx = prow.row - r0, prow.col - c0
    rr, cc = np.nonzero(mask_src)
    if len(rr) == 0:
        print(name, "EMPTY mask"); return None
    lid = lab[rr[np.argmin((rr - cy) ** 2 + (cc - cx) ** 2)], cc[np.argmin((rr - cy) ** 2 + (cc - cx) ** 2)]]
    pm = lab == lid
    dist = ndi.distance_transform_edt(~pm)
    dem_w = dem[r0:r1, c0:c1]; slope_w = slope[r0:r1, c0:c1]
    intact_w = intact[r0:r1, c0:c1]
    cm = (intact_w & (dist >= 4) & (dist <= 30)
          & (np.abs(dem_w - prow.elev) <= 150) & (np.abs(slope_w - prow.slope) <= 10))
    if cm.sum() < 30:
        cm = (intact_w & (dist >= 4) & (dist <= 50)
              & (np.abs(dem_w - prow.elev) <= 250) & (np.abs(slope_w - prow.slope) <= 15))
    print(name, "patch px", int(pm.sum()), "ctrl px", int(cm.sum()), flush=True)

    bbox4326 = rasterio.warp.transform_bounds(CRS, "EPSG:4326",
        X0 + c0 * RES, Y1 - r1 * RES, X0 + c1 * RES, Y1 - r0 * RES)
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
        try:
            items = list(cat.search(collections=["landsat-c2-l2"], bbox=list(bbox4326),
                                    datetime=f"{yr}-06-01/{end}",
                                    query={"eo:cloud_cover": {"lt": 85}}).items())
        except Exception as e:
            print(name, yr, "search fail", repr(e)[:80]); items = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(one_item, it) for it in items]
            for f in as_completed(futs):
                try: f.result()
                except Exception: pass
    df = pd.DataFrame(rows)
    df["case"] = name; df["t_event"] = t_event
    df["patch_px"] = int(pm.sum()); df["ctrl_px"] = int(cm.sum())
    df["elev"] = prow.elev; df["area_ha"] = prow.area_ha
    print(name, "obs rows:", len(df), f"{time.time()-t0:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    all_df = []
    for name, prow, t_ev in cases:
        d = run_case(name, prow, t_ev)
        if d is not None and len(d):
            all_df.append(d)
    pd.concat(all_df).to_parquet(f"{OUT}/case_trajectories.parquet")
    print("STEP6 COMPLETE", f"{time.time()-t0:.0f}s")
