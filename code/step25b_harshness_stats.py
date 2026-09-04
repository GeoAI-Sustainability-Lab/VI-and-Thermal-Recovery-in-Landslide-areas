"""Step 25b: does regional summer harshness modulate the patch-minus-control
delta? The table-side half of step25.

step25 builds the per-epoch intact-forest LST anomaly from the 30 m composites
(tier B) and stores it in data/epoch_harshness.json; this step takes that index
and the recovery table and re-estimates the elasticity
    dlst_it = age-bin effects + beta * harsh_t
with a patch-cluster bootstrap (200 replicates, seed 7), together with the
linear trend of the index inside the window.
Reads : data/epoch_harshness.json, outputs/chrono2_long.parquet
Writes: outputs/harshness_check.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
EP = {f"c{y-2000:02d}": y for y in range(2013, 2027)}
harsh = json.load(open(f"{D}/epoch_harshness.json", encoding="utf-8"))["harsh"]
harsh = {t: float(harsh[t]) for t in EP if t in harsh}
print("epoch harshness anomaly (°C):", {t: round(v, 2) for t, v in sorted(harsh.items())})
yrs = np.array([EP[t] for t in harsh]); hh = np.array([harsh[t] for t in harsh])
b1, b0 = np.polyfit(yrs, hh, 1)
print(f"linear trend inside window: {b1:+.3f} °C/yr")

L = pd.read_parquet(f"{O}/chrono2_long.parquet")
d = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= 22) & (L.src == "event") & L.dlst.notna()].copy()
d = d[d.epoch.isin(harsh)]
d["harsh"] = d.epoch.map(harsh); d["ab"] = np.clip(d.age.astype(int), 1, 21)
Xb = pd.get_dummies(d.ab, prefix="a").values.astype(float)
X = np.column_stack([Xb, d.harsh.values])
y = d.dlst.values
beta = np.linalg.lstsq(X, y, rcond=None)[0][-1]
rng = np.random.default_rng(7)
pids = d.pid.unique(); gi = {p: np.flatnonzero(d.pid.values == p) for p in pids}
bs = []
for _ in range(200):
    rows = np.concatenate([gi[p] for p in rng.choice(pids, len(pids))])
    bs.append(np.linalg.lstsq(X[rows], y[rows], rcond=None)[0][-1])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"beta (ΔLST per °C of hotter summer) = {beta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(d)}")
json.dump(dict(harsh=harsh, trend_per_yr=float(b1), beta=float(beta),
               beta_lo=float(lo), beta_hi=float(hi), n=int(len(d))),
          open(f"{O}/harshness_check.json", "w"), indent=1)
print("STEP25B COMPLETE")
