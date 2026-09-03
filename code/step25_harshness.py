"""Does regional summer harshness (the warming/interannual signal) leak into
or modulate the patch-minus-control delta?

Index: per-epoch intact-forest LST anomaly computed PIXEL-WISE (each pixel
minus its own cross-epoch mean, pixels with >=10 valid epochs), so coverage
composition cannot fake a warm year.
Test:  dlst_it = age-bin effects + beta * harsh_t, patch-cluster bootstrap.
beta = 0  -> the difference design is immune to hot/cold summers (and to trend)
beta > 0  -> a given deficit expresses as a larger delta in hotter summers
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, pandas as pd, rasterio, json
import os
os.chdir(f"{_TROOT}")
D="data"; O="outputs"
EP = {f"c{y-2000:02d}": y for y in range(2013,2027)}
with rasterio.open(f"{D}/intact2_30.tif") as s: intact = s.read(1)==1
idx = np.flatnonzero(intact.ravel()); del intact
sub = idx[::7]                       # ~2M intact pixels, plenty
S = np.zeros(len(sub), np.float64); N = np.zeros(len(sub), np.int16)
vals = {}
for t in EP:
    with rasterio.open(f"{D}/lst_{t}.tif") as s: a = s.read(1).ravel()[sub]
    a[a==-9999]=np.nan; vals[t]=a.astype(np.float32)
    m=np.isfinite(a); S[m]+=a[m]; N[m]+=1
ok = N>=10
pixmean = S/np.maximum(N,1)
harsh = {t: float(np.nanmean(vals[t][ok]-pixmean[ok])) for t in EP}
print("epoch harshness anomaly (°C):", {t: round(v,2) for t,v in sorted(harsh.items())})
yrs = np.array([EP[t] for t in harsh]); hh = np.array([harsh[t] for t in harsh])
b1,b0 = np.polyfit(yrs, hh, 1)
print(f"linear trend inside window: {b1:+.3f} °C/yr")
del vals, S, N, pixmean

L = pd.read_parquet(f"{O}/chrono2_long.parquet")
d = L[(~L.is_pre)&L.clean&(L.age>0.6)&(L.age<=22)&(L.src=="event")&L.dlst.notna()].copy()
d = d[d.epoch.isin(harsh)]
d["harsh"]=d.epoch.map(harsh); d["ab"]=np.clip(d.age.astype(int),1,21)
Xb = pd.get_dummies(d.ab, prefix="a").values.astype(float)
X = np.column_stack([Xb, d.harsh.values])
y = d.dlst.values
beta = np.linalg.lstsq(X, y, rcond=None)[0][-1]
rng=np.random.default_rng(7)
pids=d.pid.unique(); gi={p:np.flatnonzero(d.pid.values==p) for p in pids}
bs=[]
for _ in range(200):
    rows=np.concatenate([gi[p] for p in rng.choice(pids,len(pids))])
    bs.append(np.linalg.lstsq(X[rows], y[rows], rcond=None)[0][-1])
lo,hi=np.percentile(bs,[2.5,97.5])
print(f"beta (ΔLST per °C of hotter summer) = {beta:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(d)}")
json.dump(dict(harsh=harsh, trend_per_yr=float(b1), beta=float(beta),
               beta_lo=float(lo), beta_hi=float(hi), n=int(len(d))),
          open(f"{O}/harshness_check.json","w"), indent=1)
