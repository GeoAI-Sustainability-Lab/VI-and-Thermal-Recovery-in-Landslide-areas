"""Step 5c: patch-vs-control ΔLST/ΔNDVI for the unified multi-year patch DB,
at every valid epoch (c20..c26, c2425). Controls = intact2 annulus, matched
elevation ±150 m / slope ±10° (widened fallback). Placebo epoch = last full
summer strictly before the event (2021-2025 cohorts).
Output: data/patches_deltas2.parquet
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
t0 = time.time()

EPOCH_MID = {"c13": 2013.62, "c14": 2014.62, "c15": 2015.62,
             "c16": 2016.62, "c17": 2017.62, "c18": 2018.62,
             "c19": 2019.62,
             "c20": 2020.62, "c21": 2021.62, "c22": 2022.62, "c23": 2023.62,
             "c24": 2024.62, "c25": 2025.62, "c26": 2026.55, "c2425": 2025.12}
EPOCHS = list(EPOCH_MID)

dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{D}/slope30.tif"); slope[slope == -9999] = np.nan
north = read_grid(f"{D}/northness30.tif"); north[north == -9999] = np.nan
intact = read_grid(f"{D}/intact2_30.tif") == 1

srcs = {}
for tag in EPOCHS:
    for band in ["lst", "ndvi"]:
        p = f"{D}/{band}_{tag}.tif"
        if os.path.exists(p):
            srcs[(band, tag)] = rasterio.open(p)
print("epochs available:", sorted({t for _, t in srcs}), flush=True)

patches = pd.read_parquet(f"{D}/patches_raw2.parquet")
print("patches:", len(patches), flush=True)

PAD = 50
recs = []
for i, row in enumerate(patches.itertuples()):
    r0 = max(0, row.r0 - PAD); c0 = max(0, row.c0 - PAD)
    r1 = min(H, row.r1 + PAD); c1 = min(W, row.c1 + PAD)
    h_, w_ = r1 - r0, c1 - c0
    pm = np.zeros((h_, w_), bool)
    pm[np.asarray(row.px_rows) - r0, np.asarray(row.px_cols) - c0] = True
    dem_w = dem[r0:r1, c0:c1]; slope_w = slope[r0:r1, c0:c1]
    north_w = north[r0:r1, c0:c1]
    p_elev = float(np.nanmean(dem_w[pm])); p_slope = float(np.nanmean(slope_w[pm]))
    p_north = float(np.nanmean(north_w[pm]))
    dist = ndi.distance_transform_edt(~pm)
    base_ok = intact[r0:r1, c0:c1] & np.isfinite(dem_w) & np.isfinite(slope_w)

    cm = (base_ok & (dist >= 4) & (dist <= 30)
          & (np.abs(dem_w - p_elev) <= 150) & (np.abs(slope_w - p_slope) <= 10))
    if cm.sum() < 30:
        cm = (base_ok & (dist >= 4) & (dist <= 50)
              & (np.abs(dem_w - p_elev) <= 250) & (np.abs(slope_w - p_slope) <= 15))

    t_ev = float(row.t_event)
    yr = int(row.year)
    pre_tag = f"c{yr-1-2000:02d}" if 2014 <= yr <= 2025 else None
    if pre_tag is not None and (pre_tag, ) and ("lst", pre_tag) not in srcs:
        pre_tag = None
    # pre epoch must lie fully before the event
    if pre_tag is not None and EPOCH_MID[pre_tag] + 0.21 > t_ev:
        pre_tag = None

    rec = dict(pid=int(row.Index), src=row.src, year=yr, event_code=int(row.event_code),
               event=getattr(row, "event", ""), agent=getattr(row, "agent", ""),
               era=getattr(row, "era", ""), dq=getattr(row, "dq", ""),
               t_event=t_ev, n_px=int(row.n_px), area_ha=float(row.area_ha),
               redist_frac=float(row.redist_frac),
               forest2000_frac=float(row.forest2000_frac),
               elev=p_elev, slope=p_slope, northness=p_north,
               row=float(row.row), col=float(row.col),
               n_ctrl=int(cm.sum()), pre_tag=pre_tag or "",
               ctrl_elev=float(np.nanmean(dem_w[cm])) if cm.any() else np.nan,
               ctrl_slope=float(np.nanmean(slope_w[cm])) if cm.any() else np.nan)

    for tag in EPOCHS:
        if ("lst", tag) not in srcs:
            continue
        age = EPOCH_MID[tag] - t_ev
        is_pre = (tag == pre_tag)
        # keep every epoch that is unambiguously before (lead) or after (lag)
        # the event; drop only the summer that straddles it. Leads make the
        # parallel-trends test possible now that the stack reaches 2013.
        if not is_pre and -0.30 < age < 0.7:
            continue
        for band in ["lst", "ndvi"]:
            ds = srcs[(band, tag)]
            a = ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
            a[a == -9999] = np.nan
            pv = a[pm]; cv = a[cm]
            npv = int(np.isfinite(pv).sum()); ncv = int(np.isfinite(cv).sum())
            if npv >= max(3, 0.3 * row.n_px) and ncv >= 20:
                rec[f"d{band}_{tag}"] = float(np.nanmean(pv) - np.nanmean(cv))
    recs.append(rec)
    if (i + 1) % 4000 == 0:
        print(f"  {i+1}/{len(patches)}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(recs)
df.to_parquet(f"{D}/patches_deltas2.parquet")
cov = {t: int(df[f"dlst_{t}"].notna().sum()) for t in EPOCHS if f"dlst_{t}" in df.columns}
print("saved", len(df), "coverage:", cov)
print("STEP5C COMPLETE", f"{time.time()-t0:.0f}s")
