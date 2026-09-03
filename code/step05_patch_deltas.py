"""Step 5: patch-vs-matched-control LST/NDVI anomalies (ΔLST, ΔNDVI).

Control = intact forest annulus (120-900 m from patch edge; widened to 1500 m
if <30 valid px), matched on elevation (±150 m, widened ±250 m) and slope (±10°).
Composites used per patch (post-event only) + placebo epochs where defined:
  hansen (2001-2023) : post c2425, c26          (chronosequence, 2 epochs)
  ardswc 2024 events : placebo c23, post c25, c26
  ardswc 2025 events : placebo c24, post c26
  fire2021 (Huisun)  : post c2425, c26
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, warnings, time, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, H, W

OUT = f"{_TROOT}/data"
t0 = time.time()

# in-RAM context
dem = read_grid(f"{OUT}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{OUT}/slope30.tif"); slope[slope == -9999] = np.nan
intact = read_grid(f"{OUT}/intact30.tif") == 1
lossyear = read_grid(f"{OUT}/hansen_lossyear30.tif")
tc2000 = read_grid(f"{OUT}/treecover2000_30.tif")
events = read_grid(f"{OUT}/events30.tif")
event_codes = {int(k): v for k, v in json.load(open(f"{OUT}/event_codes.json")).items()}
FIRE_CODE = max(event_codes)
forest2000 = tc2000 >= 60

patches = pd.read_parquet(f"{OUT}/patches_raw.parquet")
print("patches:", len(patches), patches.groupby("src").size().to_dict(), flush=True)

COMPOSITES = ["c2425", "c25", "c26", "c23", "c24"]
srcs = {}
for tag in COMPOSITES:
    srcs[("lst", tag)] = rasterio.open(f"{OUT}/lst_{tag}.tif")
    srcs[("ndvi", tag)] = rasterio.open(f"{OUT}/ndvi_{tag}.tif")

def read_win(ds, r0, c0, r1, c1):
    a = ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
    a[a == -9999] = np.nan
    return a

def epochs_for(row):
    if row.src == "hansen":
        return [], ["c2425", "c26"]
    if row.src == "fire2021":
        return [], ["c2425", "c26"]
    if row.src == "ardswc":
        return (["c23"], ["c25", "c26"]) if row.year == 2024 else (["c24"], ["c26"])
    return [], []

S3x3 = np.ones((3, 3), bool)
SRC_MASKS = {
    "hansen": lambda: (lossyear >= 1) & (lossyear <= 23) & forest2000 & (events == 0),
    "ardswc": lambda: (events > 0) & (events < FIRE_CODE),
    "fire2021": lambda: events == FIRE_CODE,
}

PAD = 50
recs = []
i = -1
for src_name, mask_fn in SRC_MASKS.items():
    lab = ndi.label(mask_fn(), S3x3)[0]          # one source at a time (RAM)
    sub_patches = patches[patches.src == src_name]
    print(f"labels {src_name}: {len(sub_patches)} patches", f"{time.time()-t0:.0f}s", flush=True)
    for row in sub_patches.itertuples():
        i += 1
        r0 = max(0, row.r0 - PAD); c0 = max(0, row.c0 - PAD)
        r1 = min(H, row.r1 + PAD); c1 = min(W, row.c1 + PAD)
        lab_w = lab[r0:r1, c0:c1]
        pm = lab_w == row.label_id
        if not pm.any():
            continue
        dist = ndi.distance_transform_edt(~pm)
        dem_w = dem[r0:r1, c0:c1]; slope_w = slope[r0:r1, c0:c1]
        intact_w = intact[r0:r1, c0:c1]
        base_ok = intact_w & np.isfinite(dem_w) & np.isfinite(slope_w)

        def ctrl_mask(dmax, e_tol, s_tol):
            return (base_ok & (dist >= 4) & (dist <= dmax)
                    & (np.abs(dem_w - row.elev) <= e_tol)
                    & (np.abs(slope_w - row.slope) <= s_tol))

        cm = ctrl_mask(30, 150, 10)
        if cm.sum() < 30:
            cm = ctrl_mask(50, 250, 15)
        rec = dict(src=row.src, label_id=row.label_id, year=int(row.year),
                   event_code=int(row.event_code), n_px=row.n_px, area_ha=row.area_ha,
                   elev=row.elev, slope=row.slope, northness=row.northness,
                   ftype=row.ftype, forest2000_frac=row.forest2000_frac,
                   row=row.row, col=row.col,
                   n_ctrl=int(cm.sum()),
                   ctrl_elev=float(np.nanmean(dem_w[cm])) if cm.any() else np.nan,
                   ctrl_slope=float(np.nanmean(slope_w[cm])) if cm.any() else np.nan)
        placebo, post = epochs_for(row)
        for tag in placebo + post:
            for band in ["lst", "ndvi"]:
                a = read_win(srcs[(band, tag)], r0, c0, r1, c1)
                pv = a[pm]; cv = a[cm]
                npv, ncv = int(np.isfinite(pv).sum()), int(np.isfinite(cv).sum())
                if npv >= max(3, 0.3 * row.n_px) and ncv >= 20:
                    rec[f"d{band}_{tag}"] = float(np.nanmean(pv) - np.nanmean(cv))
                    rec[f"{band}_ctrl_{tag}"] = float(np.nanmean(cv))
                    rec[f"n_{band}_{tag}"] = npv
        recs.append(rec)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(patches)}  {time.time()-t0:.0f}s", flush=True)

    del lab

df = pd.DataFrame(recs)
df.to_parquet(f"{OUT}/patches_deltas.parquet")
print("saved", len(df), "rows;",
      {c: int(df[c].notna().sum()) for c in df.columns if c.startswith("dlst")},
      flush=True)
print("STEP5 COMPLETE", f"{time.time()-t0:.0f}s")
