"""Step 12b: population absolute-series EXTENSION for the v9 figure — paired
patch−control differences per epoch (for the difference strips) plus the same
differences stratified by elevation band and by official area class.
Same cohorts and matching as step12. Output: outputs/abs_series2.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, H, W

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
t0 = time.time()
EPOCH_MID = {"c13": 2013.62, "c14": 2014.62, "c15": 2015.62,
             "c16": 2016.62, "c17": 2017.62, "c18": 2018.62,
             "c19": 2019.62,
             "c20": 2020.62, "c21": 2021.62, "c22": 2022.62, "c23": 2023.62,
             "c24": 2024.62, "c25": 2025.62, "c26": 2026.55}
EPOCHS = list(EPOCH_MID)

dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{D}/slope30.tif"); slope[slope == -9999] = np.nan
intact = read_grid(f"{D}/intact2_30.tif") == 1
srcs = {}
for tag in EPOCHS:
    for band in ["lst", "ndvi"]:
        p = f"{D}/{band}_{tag}.tif"
        if os.path.exists(p):
            srcs[(band, tag)] = rasterio.open(p)

raw = pd.read_parquet(f"{D}/patches_raw2.parquet")
deltas = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                         columns=["src", "agent", "year", "n_ctrl", "elev",
                                  "area_ha"])
raw = raw.join(deltas[["n_ctrl", "elev"]])
COHORTS = {
    "typhoon_2024": (raw.src == "event") & (raw.agent == "typhoon_rain") & (raw.year == 2024),
    "rainfall_2025": (raw.src == "event") & (raw.agent == "rainfall") & (raw.year == 2025),
    "earthquake_2024": (raw.src == "event") & (raw.agent == "earthquake") & (raw.year == 2024),
    "hansen_2023": (raw.src == "hansen") & (raw.year == 2023),
}
PAD = 50
ELEV_B = [("lt1000", 0, 1000), ("1000_2000", 1000, 2000), ("gt2000", 2000, 9999)]
AREA_B = [("lt2", 0, 2), ("2_10", 2, 10), ("ge10", 10, 1e9)]

res = {}
for name, m in COHORTS.items():
    sub = raw[m & (raw.n_ctrl >= 30)]
    if len(sub) > 900:
        sub = sub.sample(900, random_state=42)
    rows = []
    tev = []
    for row in sub.itertuples():
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
        tev.append(float(row.t_event))
        rec = {"elev": p_elev, "area": float(row.area_ha)}  # area_ha from raw
        for tag in EPOCHS:
            for band in ["lst", "ndvi"]:
                if (band, tag) not in srcs:
                    continue
                a = srcs[(band, tag)].read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
                a[a == -9999] = np.nan
                pv, cv = a[pm], a[cm]
                if np.isfinite(pv).sum() >= max(3, 0.3 * row.n_px) and np.isfinite(cv).sum() >= 20:
                    rec[f"d{band}_{tag}"] = float(np.nanmean(pv) - np.nanmean(cv))
        rows.append(rec)
    df = pd.DataFrame(rows)

    def series(d, band):
        out = {}
        for tag in EPOCHS:
            c = f"d{band}_{tag}"
            if c in d.columns:
                g = d[c].dropna()
                if len(g) >= 12:
                    out[tag] = dict(mid=EPOCH_MID[tag], n=int(len(g)),
                                    diff=float(g.mean()),
                                    se=float(g.std() / np.sqrt(len(g))))
        return out

    entry = {"t_event_med": float(np.median(tev)), "n_patches": len(df),
             "diff": {b: series(df, b) for b in ["lst", "ndvi"]},
             "elev_strata": {}, "area_strata": {}}
    for nm, lo, hi in ELEV_B:
        d = df[(df.elev >= lo) & (df.elev < hi)]
        s = series(d, "lst")
        if s:
            entry["elev_strata"][nm] = dict(n=int(len(d)), series=s)
    for nm, lo, hi in AREA_B:
        d = df[(df.area >= lo) & (df.area < hi)]
        s = series(d, "lst")
        if s:
            entry["area_strata"][nm] = dict(n=int(len(d)), series=s)
    res[name] = entry
    print(name, "n =", len(df), f"{time.time()-t0:.0f}s", flush=True)

json.dump(res, open(f"{O}/abs_series2.json", "w"), indent=1)
print("STEP12B COMPLETE", f"{time.time()-t0:.0f}s")
