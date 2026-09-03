"""Step 12: population-level ABSOLUTE LST/NDVI series, event patches vs their
matched intact-forest controls, per calendar epoch (2013-2026), for four
recent cohorts (typhoon 2024, rainfall 2025, earthquake 2024, Hansen loss
2022). Same annulus/terrain matching as step05c, but stores the absolute
patch mean and control mean per epoch instead of only the difference.
Both series share inter-annual climate variation, so their gap is the event
effect and recovery is defined as the gap returning to zero.
Output: outputs/abs_series.json
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

# v18: full 2013-2026 stack (the script predated the epoch extension and
# only read c20-c26, so the absolute panels started years after the diff
# strips; both now cover the same 14 epochs; c2425 excluded as a 2-yr stack)
EPOCH_MID = {"c13": 2013.62, "c14": 2014.62, "c15": 2015.62, "c16": 2016.62,
             "c17": 2017.62, "c18": 2018.62, "c19": 2019.62,
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
                         columns=["src", "agent", "year", "n_ctrl"])
raw = raw.join(deltas[["n_ctrl"]])
COHORTS = {
    "typhoon_2024": (raw.src == "event") & (raw.agent == "typhoon_rain") & (raw.year == 2024),
    "rainfall_2025": (raw.src == "event") & (raw.agent == "rainfall") & (raw.year == 2025),
    "earthquake_2024": (raw.src == "event") & (raw.agent == "earthquake") & (raw.year == 2024),
    "hansen_2023": (raw.src == "hansen") & (raw.year == 2023),
}
raw["agent2"] = deltas.agent
res = {}
PAD = 50
for name, m in COHORTS.items():
    sub = raw[m & (raw.n_ctrl >= 30)]
    if len(sub) > 900:
        sub = sub.sample(900, random_state=42)
    acc = {t: {b: {"p": [], "c": []} for b in ["lst", "ndvi"]} for t in EPOCHS}
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
        for tag in EPOCHS:
            for band in ["lst", "ndvi"]:
                if (band, tag) not in srcs:
                    continue
                a = srcs[(band, tag)].read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
                a[a == -9999] = np.nan
                pv, cv = a[pm], a[cm]
                if np.isfinite(pv).sum() >= max(3, 0.3 * row.n_px) and np.isfinite(cv).sum() >= 20:
                    acc[tag][band]["p"].append(float(np.nanmean(pv)))
                    acc[tag][band]["c"].append(float(np.nanmean(cv)))
    out = {"t_event_med": float(np.median(tev)), "n_patches": len(tev), "epochs": {}}
    for tag in EPOCHS:
        e = {}
        for band in ["lst", "ndvi"]:
            pv = np.array(acc[tag][band]["p"]); cv = np.array(acc[tag][band]["c"])
            if len(pv) >= 15:
                e[band] = dict(n=int(len(pv)),
                               patch=float(pv.mean()), patch_se=float(pv.std()/np.sqrt(len(pv))),
                               ctrl=float(cv.mean()), ctrl_se=float(cv.std()/np.sqrt(len(cv))))
        out["epochs"][tag] = dict(mid=EPOCH_MID[tag], **e)
    res[name] = out
    print(name, "n =", out["n_patches"], f"{time.time()-t0:.0f}s", flush=True)

json.dump(res, open(f"{O}/abs_series.json", "w"), indent=1)
print("STEP12 COMPLETE", f"{time.time()-t0:.0f}s")
