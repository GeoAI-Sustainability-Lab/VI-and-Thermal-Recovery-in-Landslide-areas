"""Step 29: the intact-forest buffering baseline from the shipped pixel table.

Re-derives, from data/buffering_sample.parquet alone, every buffering-baseline
quantity that the text quotes and that used to live in the raster-side step
(step08a) or in a figure script (the local slopes of the binned curve):

  * per-band linear canopy-height slopes after terrain residualisation
    (results.json -> band_stats),
  * the gradient-boosting model of LST on canopy height and terrain, its
    held-out R2 and the mean |SHAP| of each feature
    (results.json -> gbm_r2_test, shap_mean_abs; outputs/shap_buffering.parquet),
  * the counterfactual 5 m -> 25 m canopy contrast at the band's median terrain
    (results.json -> counterfactual),
  * the local slope of the pooled binned curve at 15, 20, ... 40 m canopy height
    (outputs/baseline_slope_by_height.json).

The other keys of results.json belong to the earlier single-epoch database
and are left untouched. Settings are those of the original run: 75/25 split
with random_state 42, HistGradientBoostingRegressor(max_iter=400,
learning_rate=0.08, min_samples_leaf=50, random_state=42), TreeExplainer on
the first 6,000 test rows.
Reads : data/buffering_sample.parquet
Writes: outputs/results.json (buffering keys), outputs/shap_buffering.parquet,
        outputs/baseline_slope_by_height.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import shap

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
t0 = time.time()
RES_PATH = f"{O}/results.json"
R = json.load(open(RES_PATH, encoding="utf-8")) if _os.path.exists(RES_PATH) else {}

BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3600)]
FEATS = ["chm", "elev", "slope", "cos_i", "northness", "eastness"]

raw = pd.read_parquet(f"{D}/buffering_sample.parquet")

# ---------------- band slopes, GBM, SHAP, counterfactual (step08a settings) ----------------
df = raw[(raw.chm >= 0) & (raw.chm <= 45) & raw.lst.notna()].copy()
R["buffering_n"] = int(len(df))

band_stats = []
for lo, hi in BANDS:
    d = df[(df.elev >= lo) & (df.elev < hi)]
    if len(d) < 2000:
        continue
    Xc = np.column_stack([d.cos_i, d.slope, d.northness, np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(Xc, d.lst, rcond=None)
    lst_res = d.lst - Xc @ beta + d.lst.mean()
    sl, itc, r, p, se = stats.linregress(d.chm, lst_res)
    bins = np.arange(0, 40, 2.5)
    bc = 0.5 * (bins[:-1] + bins[1:])
    cells = [(d.chm >= a) & (d.chm < b) for a, b in zip(bins[:-1], bins[1:])]
    med = [float(np.median(lst_res[m])) if m.sum() > 50 else np.nan for m in cells]
    q25 = [float(np.percentile(lst_res[m], 25)) if m.sum() > 50 else np.nan for m in cells]
    q75 = [float(np.percentile(lst_res[m], 75)) if m.sum() > 50 else np.nan for m in cells]
    band_stats.append(dict(band=f"{lo}-{hi}", lo=lo, hi=hi, n=int(len(d)),
                           slope_per_m=float(sl), slope_se=float(se), p=float(p),
                           bin_centers=bc.tolist(), bin_median=med, bin_q25=q25, bin_q75=q75))
R["band_stats"] = band_stats
print("band slopes (°C per 10 m):",
      {b["band"]: round(b["slope_per_m"] * 10, 3) for b in band_stats}, flush=True)

X = df[FEATS].values.astype(np.float32)
y = df.lst.values.astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_depth=None,
                                    min_samples_leaf=50, random_state=42)
gbm.fit(Xtr, ytr)
R["gbm_r2_test"] = float(gbm.score(Xte, yte))
print("GBM test R2:", round(R["gbm_r2_test"], 4), f"{time.time() - t0:.0f}s", flush=True)

sv = shap.TreeExplainer(gbm)(Xte[:6000])
R["shap_mean_abs"] = {f: float(np.abs(sv.values[:, i]).mean()) for i, f in enumerate(FEATS)}
sh = pd.DataFrame(Xte[:6000], columns=FEATS)
for i, f in enumerate(FEATS):
    sh[f"shap_{f}"] = sv.values[:, i]
sh.to_parquet(f"{O}/shap_buffering.parquet")
print("mean |SHAP|:", {k: round(v, 3) for k, v in R["shap_mean_abs"].items()}, flush=True)

cf = []
for lo, hi in BANDS:
    d = df[(df.elev >= lo) & (df.elev < hi)]
    if len(d) < 2000:
        continue
    base = d[FEATS].median().values.astype(np.float32)
    lo_v = base.copy(); lo_v[0] = 5.0
    hi_v = base.copy(); hi_v[0] = 25.0
    cf.append(dict(band=f"{lo}-{hi}",
                   lst_at_5m=float(gbm.predict([lo_v])[0]),
                   lst_at_25m=float(gbm.predict([hi_v])[0])))
R["counterfactual"] = cf

json.dump(R, open(RES_PATH, "w"), ensure_ascii=False, indent=1)

# ---------------- local slope of the pooled binned curve ----------------
# Anomaly relative to a 25 m canopy, built inside 250 m elevation slices with
# terrain (cos i, slope, northness) removed by a linear model, exactly as the
# descriptive step (step28) and the former figure script did.
REF_LO, REF_HI = 24.0, 26.0
d0 = raw.dropna(subset=["lst", "chm", "elev", "slope", "northness", "cos_i"])
d0 = d0.assign(slice=(d0.elev // 250).astype(int))
parts = []
for _, g in d0.groupby("slice"):
    if len(g) < 3000 or ((g.chm >= REF_LO) & (g.chm < REF_HI)).sum() < 100:
        continue
    Xs = np.column_stack([g.cos_i, g.slope, g.northness, np.ones(len(g))])
    beta, *_ = np.linalg.lstsq(Xs, g.lst, rcond=None)
    res = g.lst.values - Xs @ beta
    parts.append(g.assign(anom=res - np.median(res[(g.chm >= REF_LO) & (g.chm < REF_HI)])))
d = pd.concat(parts)
d = d[(d.elev >= BANDS[0][0]) & (d.elev < BANDS[-1][1])]
sel = (d.chm >= 8) & (d.chm <= 50) & (d.anom > -6.2) & (d.anom < 6.2)
x, yv = d.chm[sel].values, d.anom[sel].values
edges = np.arange(8, 50.5, 1.0)
c, m = [], []
for a, b in zip(edges[:-1], edges[1:]):
    s = yv[(x >= a) & (x < b)]
    if len(s) >= 80:
        c.append(0.5 * (a + b)); m.append(float(s.mean()))
c, m = np.array(c), np.array(m)


def local_slope(h, half=3.0):
    """Slope of the binned curve over h ± 3 m, expressed per 10 m of canopy."""
    return (np.interp(h + half, c, m) - np.interp(h - half, c, m)) / (2 * half) * 10


BSH = {int(h): float(local_slope(h)) for h in (15, 20, 25, 30, 35, 40)}
json.dump(BSH, open(f"{O}/baseline_slope_by_height.json", "w"), indent=1)
print("local slopes (°C per 10 m):", {k: round(v, 3) for k, v in BSH.items()}, flush=True)
print("STEP29 COMPLETE", f"{time.time() - t0:.0f}s")
