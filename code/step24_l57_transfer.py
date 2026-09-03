"""Step 24: cross-sensor transfer check, L5/L7 versus the Landsat 8 stack.

Question: can TM/ETM+ composites carry the SAME patch-minus-control estimator
without a footprint bias? Test on the overlap summers 2013-2014, where an
L7-only composite and the production L8 composite see the SAME patches in the
SAME summer: regress the per-patch delta from L7 on the delta from L8, overall
and by patch-size class. A slope near 1 with small intercept means the delta
design absorbs the sensor change; attenuation concentrated in small patches
would dictate a patch-size threshold for any pre-2013 extension. The L5-L7
pairing on 2010-2011 chains TM in. Also reports per-sensor coverage.

Output: data/l57_deltas.parquet, outputs/l57_transfer.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi
import sys
sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, H, W

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
t0 = time.time()

TAGS = [t for t in ["c13", "c13l7", "c14", "c14l7", "c10l5", "c10l7",
                    "c11l5", "c11l7", "c12l7"]
        if os.path.exists(f"{D}/lst_{t}.tif")]
print("tags:", TAGS, flush=True)

dem = read_grid(f"{D}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{D}/slope30.tif"); slope[slope == -9999] = np.nan
intact = read_grid(f"{D}/intact2_30.tif") == 1
srcs = {(b, t): rasterio.open(f"{D}/{b}_{t}.tif")
        for t in TAGS for b in ("lst", "ndvi") if os.path.exists(f"{D}/{b}_{t}.tif")}

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
    rec = dict(pid=int(row.Index), src=row.src, year=int(row.year),
               event=getattr(row, "event", ""), agent=getattr(row, "agent", ""),
               t_event=float(row.t_event), n_px=int(row.n_px),
               area_ha=float(row.area_ha), elev=p_elev,
               redist_frac=float(row.redist_frac),
               forest2000_frac=float(row.forest2000_frac), n_ctrl=int(cm.sum()))
    for (b, tg), ds in srcs.items():
        a = ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float32)
        a[a == -9999] = np.nan
        pv = a[pm]; cv = a[cm]
        npv = int(np.isfinite(pv).sum()); ncv = int(np.isfinite(cv).sum())
        if npv >= max(3, 0.3 * row.n_px) and ncv >= 20:
            rec[f"d{b}_{tg}"] = float(np.nanmean(pv) - np.nanmean(cv))
            rec[f"cov{b}_{tg}"] = npv / row.n_px
    recs.append(rec)
    if (i + 1) % 4000 == 0:
        print(f"  {i+1}/{len(patches)}  {time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(recs)
df.to_parquet(f"{D}/l57_deltas.parquet")
print("saved", len(df), flush=True)

R = {"tags": TAGS}
def compare(a, b, name, band="lst"):
    ca, cb = f"d{band}_{a}", f"d{band}_{b}"
    if ca not in df.columns or cb not in df.columns:
        return
    s = df[df[ca].notna() & df[cb].notna()]
    if len(s) < 50:
        return
    ent = {}
    for lab, g in [("all", s), ("lt2", s[s.area_ha < 2]),
                   ("2_10", s[(s.area_ha >= 2) & (s.area_ha < 10)]),
                   ("ge10", s[s.area_ha >= 10])]:
        if len(g) < 30:
            continue
        x, y = g[cb].values, g[ca].values     # x = reference (L8 or L7)
        b1, b0 = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        ent[lab] = dict(n=int(len(g)), slope=float(b1), intercept=float(b0),
                        r=r, bias=float((y - x).mean()),
                        sd_diff=float((y - x).std()),
                        mean_ref=float(x.mean()))
    R[f"{name}_{band}"] = ent
    print(name, band, {k: (v["n"], round(v["slope"], 3), round(v["bias"], 3))
                       for k, v in ent.items()}, flush=True)

for band in ("lst", "ndvi"):
    compare("c13l7", "c13", "L7_vs_L8_2013", band)
    compare("c14l7", "c14", "L7_vs_L8_2014", band)
    compare("c10l5", "c10l7", "L5_vs_L7_2010", band)
    compare("c11l5", "c11l7", "L5_vs_L7_2011", band)

# coverage per tag (share of patches measurable)
R["coverage"] = {t: int(df[f"dlst_{t}"].notna().sum())
                 for t in TAGS if f"dlst_{t}" in df.columns}
print("coverage:", R["coverage"], flush=True)
json.dump(R, open(f"{O}/l57_transfer.json", "w"), indent=1)
print("STEP24 COMPLETE", f"{time.time()-t0:.0f}s")
