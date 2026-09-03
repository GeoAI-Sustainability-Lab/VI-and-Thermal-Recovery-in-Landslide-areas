"""Step 13: Sentinel-1 RTC pilot — structural recovery clock from radar.
600 clean landslide patches stratified by age; descending orbit 105 only
(fixed geometry); summers 2023-2025, up to 6 dates each; VH & VV gamma0.
Delta = 10log10(patch mean power) - 10log10(control mean power), same
annulus/terrain-matched controls as the main design. Chronosequence fit
gives tau_VH / tau_VV to compare with tau_NDVI and tau_LST.
Outputs: outputs/s1_pilot.json, data/s1_pilot_rows.parquet
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, time, warnings
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
from scipy.optimize import curve_fit
import pystac_client, planetary_computer as pc

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, H, W, X0, Y1

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
t0 = time.time()
SUMMERS = {2023: "2023-06-01/2023-09-30", 2024: "2024-06-01/2024-09-30",
           2025: "2025-06-01/2025-09-30"}
SUM_MID = {2023: 2023.62, 2024: 2024.62, 2025: 2025.62}
MAX_DATES = 6

# ---------- sample patches ----------
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
print("sample:", len(sample), flush=True)

# ---------- masks at 10 m ----------
dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{D}/slope30.tif"); slope[slope == -9999] = np.nan
intact = read_grid(f"{D}/intact2_30.tif") == 1
PAD = 40
P10 = []   # (pid, r0_10, c0_10, pm10, cm10, t_event)
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
    P10.append((pid, r0 * 3, c0 * 3, pm10, cm10, float(samp_meta.loc[pid, "t_event"])))
del dem, slope, intact
print("masks ready:", len(P10), f"{time.time()-t0:.0f}s", flush=True)

TR10 = from_origin(X0, Y1, 10.0, 10.0)
W10, H10 = W * 3, H * 3

# accumulators: {(pid, year): [p_sum_vh, p_cnt_vh, c_sum_vh, c_cnt_vh, ...vv]}
acc = {}
cat = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

for yr, rng_ in SUMMERS.items():
    items = [i for i in cat.search(collections=["sentinel-1-rtc"],
                                   bbox=[119.9, 21.8, 122.1, 25.4],
                                   datetime=rng_).items()
             if i.properties.get("sat:orbit_state") == "descending"
             and i.properties.get("sat:relative_orbit") == 105]
    bydate = {}
    for it in items:
        bydate.setdefault(str(it.datetime.date()), []).append(it)
    dates = sorted(bydate)
    sel = dates[:: max(1, len(dates) // MAX_DATES)][:MAX_DATES]
    print(yr, "dates:", sel, flush=True)
    for dt_ in sel:
        for it in bydate[dt_]:
            try:
                it_s = pc.sign(it)
                srcs = {b: rasterio.open(it_s.assets[b].href) for b in ["vh", "vv"]}
                bx = transform_bounds(srcs["vh"].crs, "EPSG:3826", *srcs["vh"].bounds)
                vrts = {b: WarpedVRT(srcs[b], crs="EPSG:3826", transform=TR10,
                                     width=W10, height=H10,
                                     resampling=Resampling.bilinear,
                                     src_nodata=-32768, nodata=np.nan)
                        for b in ["vh", "vv"]}
                nin = 0
                for pid, r0_10, c0_10, pm10, cm10, tev in P10:
                    hh, ww = pm10.shape
                    x0p = X0 + c0_10 * 10; y1p = Y1 - r0_10 * 10
                    x1p = x0p + ww * 10; y0p = y1p - hh * 10
                    if not (x0p >= bx[0] and x1p <= bx[2] and y0p >= bx[1] and y1p <= bx[3]):
                        continue
                    win = ((r0_10, r0_10 + hh), (c0_10, c0_10 + ww))
                    k = (pid, yr)
                    a = acc.setdefault(k, np.zeros(8))
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
                print(f"  {dt_} {it.id[-10:]} patches:{nin} {time.time()-t0:.0f}s",
                      flush=True)
            except Exception as e:
                print("  item fail", it.id[-14:], repr(e)[:70], flush=True)
        ks = np.array([[k[0], k[1]] for k in acc], dtype=np.int64)
        vs = np.stack(list(acc.values())) if acc else np.zeros((0, 8))
        np.savez(f"{O}/s1_acc_ckpt.npz", keys=ks, vals=vs)

# ---------- rows & fits ----------
tev_map = {pid: tev for pid, _, _, _, _, tev in P10}
rows = []
for (pid, yr), a in acc.items():
    rec = {"pid": int(pid), "year_obs": yr, "age": SUM_MID[yr] - tev_map[pid]}
    for bi, b in enumerate(["vh", "vv"]):
        if a[bi*4 + 1] > 0 and a[bi*4 + 3] > 0:
            pmean = a[bi*4 + 0] / a[bi*4 + 1]
            cmean = a[bi*4 + 2] / a[bi*4 + 3]
            rec[f"d{b}_db"] = float(10*np.log10(pmean) - 10*np.log10(cmean))
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_parquet(f"{D}/s1_pilot_rows.parquet")
print("rows:", len(df), flush=True)


def fit_rec(d, val, tmax=21, min_bin=10):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    bins_ = np.arange(0.6, tmax + 1, 1.0)
    bc, bm, bs, bn = [], [], [], []
    for a_, b_ in zip(bins_[:-1], bins_[1:]):
        g = d[(d.age >= a_) & (d.age < b_)][val]
        if len(g) >= min_bin:
            bc.append(0.5*(a_+b_)); bm.append(g.mean())
            bs.append(g.std()/np.sqrt(len(g))); bn.append(len(g))
    if len(bc) < 5:
        return None
    bc_, bm_, bs_ = map(np.array, (bc, bm, bs))
    popt, pcov = curve_fit(lambda t, A, tau: A*np.exp(-t/tau), bc_, bm_,
                           p0=[bm_[0], 8.0], sigma=np.maximum(bs_, 1e-3),
                           absolute_sigma=True, maxfev=20000,
                           bounds=([-30, 0.3], [30, 60]))
    perr = np.sqrt(np.diag(pcov))
    return dict(A=float(popt[0]), tau=float(popt[1]), tau_se=float(perr[1]),
                n=int(len(d)),
                bins=dict(center=bc_.tolist(), mean=bm_.tolist(),
                          se=bs_.tolist(), n=bn))


out = {"n_rows": int(len(df)), "n_patches": int(df.pid.nunique())}
for b in ["vh", "vv"]:
    f_ = fit_rec(df, f"d{b}_db")
    out[f"fit_{b}"] = f_
    if f_:
        print(b, f"tau={f_['tau']:.1f}±{f_['tau_se']:.1f}  A={f_['A']:.2f} dB",
              flush=True)
json.dump(out, open(f"{O}/s1_pilot.json", "w"), indent=1)
print("STEP13 COMPLETE", f"{time.time()-t0:.0f}s")
