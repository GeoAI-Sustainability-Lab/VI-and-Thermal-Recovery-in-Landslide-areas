"""Step 15: is the apparent elevation gradient in canopy-height buffering real,
or a canopy-height-distribution artifact?

The published slope is a linear fit of terrain-residualised LST on canopy height
over EACH BAND'S OWN canopy-height support. Because the LST-height relationship
saturates and because band supports differ (lowland p50 ~21 m, montane ~32 m),
that estimator can produce a gradient with no change in the underlying
sensitivity. This step recomputes marginal sensitivity with four estimators that
control for support, plus counterfactuals restricted to observed support, and
stores everything for the revised Fig. 3 and Table 3.

Outputs: outputs/gradient_check.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500),
         (2500, 3600)]
FEATS = ["chm", "elev", "slope", "cos_i", "northness", "eastness"]
WIN_LO, WIN_MID, WIN_HI = 20.0, 25.0, 30.0     # common comparison window
rng = np.random.default_rng(11)

df = pd.read_parquet(f"{D}/buffering_sample.parquet").dropna(
    subset=["lst", "chm", "elev", "slope", "northness", "eastness", "cos_i"])
print("sample:", len(df), flush=True)

gbm = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.1,
                                    random_state=0).fit(df[FEATS], df.lst)
print("GBM in-sample R2:", round(gbm.score(df[FEATS], df.lst), 3), flush=True)

out = {"window": [WIN_LO, WIN_MID, WIN_HI], "bands": {}}
for lo, hi in BANDS:
    d = df[(df.elev >= lo) & (df.elev < hi)]
    if len(d) < 2000:
        continue
    key = f"{lo}-{hi}"
    # ---- terrain residualisation ----
    # NOTE: within-band elevation MUST be included. Canopy height correlates
    # with elevation inside every band (r = -0.20 to +0.41), so leaving the
    # residual lapse rate in confounds the canopy signal: in the 2,500-3,600 m
    # band tall canopy sits ~170 m lower than short canopy and is therefore
    # spuriously warm, which bends the tall-canopy tail upwards.
    Xc = np.column_stack([d.elev, d.cos_i, d.slope, d.northness, np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(Xc, d.lst, rcond=None)
    res = d.lst - Xc @ beta + d.lst.mean()
    d = d.assign(res=res)

    # (1) published estimator: linear over the band's own full support
    s_full = stats.linregress(d.chm, d.res)

    # (2) same estimator restricted to a common support window
    m = (d.chm >= WIN_LO) & (d.chm <= 35.0)
    s_com = stats.linregress(d.chm[m], d.res[m]) if m.sum() > 500 else None

    # (3) non-parametric: mean residual difference between two fixed windows
    a = d.res[(d.chm >= WIN_LO) & (d.chm < WIN_MID)]
    c = d.res[(d.chm >= WIN_MID) & (d.chm < WIN_HI)]
    if len(a) > 200 and len(c) > 200:
        diff = float(c.mean() - a.mean())
        se = float(np.sqrt(a.var() / len(a) + c.var() / len(c)))
        np_slope, np_se = diff * 2, se * 2          # per 10 m
    else:
        np_slope = np_se = None

    # (4) GBM local slope at the common reference height, band-median terrain
    base = {f: float(d[f].median()) for f in FEATS}

    def pred(h):
        return float(gbm.predict(pd.DataFrame([{**base, "chm": h}])[FEATS])[0])

    gbm_local = (pred(WIN_MID + 2.5) - pred(WIN_MID - 2.5)) * 2   # per 10 m

    # counterfactuals
    p10, p50, p90 = [float(d.chm.quantile(q)) for q in (0.10, 0.50, 0.90)]
    cf_extrap = pred(25.0) - pred(5.0)              # published, extrapolated
    cf_support = pred(p90) - pred(p10)              # observed support
    cf_common = pred(WIN_HI) - pred(WIN_LO)         # common 20 -> 30 m

    # support diagnostics
    n_at5 = int(((d.chm >= 4) & (d.chm <= 6)).sum())
    n_at25 = int(((d.chm >= 24) & (d.chm <= 26)).sum())

    # partial-dependence curve for plotting
    hs = np.arange(14, 46.5, 1.0)
    pd_curve = [pred(float(h)) for h in hs]

    # canopy-height histogram for the support rug
    hist, edges = np.histogram(d.chm, bins=np.arange(5, 55, 1.0))
    # binned median of the elevation-adjusted residual (for Fig. S1a)
    be = np.arange(8, 52, 2.0)
    bcs, bmed, bq25, bq75 = [], [], [], []
    for lo_, hi_ in zip(be[:-1], be[1:]):
        v = d.res[(d.chm >= lo_) & (d.chm < hi_)]
        if len(v) >= 50:
            bcs.append(float(0.5 * (lo_ + hi_))); bmed.append(float(v.median()))
            bq25.append(float(v.quantile(.25))); bq75.append(float(v.quantile(.75)))

    out["bands"][key] = dict(
        n=int(len(d)),
        chm_p1=float(d.chm.quantile(0.01)), chm_p10=p10, chm_p50=p50,
        chm_p90=p90, chm_p99=float(d.chm.quantile(0.99)),
        n_near5m=n_at5, n_near25m=n_at25,
        slope_full=float(s_full.slope * 10), slope_full_se=float(s_full.stderr * 10),
        slope_common=(float(s_com.slope * 10) if s_com else None),
        slope_common_se=(float(s_com.stderr * 10) if s_com else None),
        slope_np=np_slope, slope_np_se=np_se,
        slope_gbm_local=float(gbm_local),
        cf_extrap_5_25=float(cf_extrap), cf_support_p10_p90=float(cf_support),
        cf_common_20_30=float(cf_common),
        pd_h=hs.tolist(), pd_lst=pd_curve,
        hist_edges=edges.tolist(), hist_n=hist.tolist(),
        bin_centers=bcs, bin_median=bmed, bin_q25=bq25, bin_q75=bq75,
    )
    print(f"{key:11s} full {s_full.slope*10:+.2f} | common {s_com.slope*10 if s_com else float('nan'):+.2f}"
          f" | np {np_slope:+.2f}±{1.96*np_se:.2f} | gbm@25 {gbm_local:+.2f}"
          f" | cf 5→25 {cf_extrap:+.2f} vs p10→p90 {cf_support:+.2f}"
          f" (n@5m={n_at5})", flush=True)

# ---- does the gradient survive? test slope-vs-elevation across bands ----
mid = np.array([0.5 * (lo + hi) for lo, hi in BANDS if f"{lo}-{hi}" in out["bands"]])
for est in ["slope_full", "slope_common", "slope_np", "slope_gbm_local"]:
    v = np.array([out["bands"][k][est] for k in out["bands"]], float)
    ok = np.isfinite(v)
    r = stats.linregress(mid[ok], v[ok])
    out.setdefault("gradient_test", {})[est] = dict(
        slope_per_1000m=float(r.slope * 1000), p=float(r.pvalue),
        r2=float(r.rvalue ** 2),
        range=[float(np.nanmin(v)), float(np.nanmax(v))])
    print(f"gradient test {est:18s}: {r.slope*1000:+.3f} °C/10m per 1000 m, "
          f"p={r.pvalue:.3f}, R2={r.rvalue**2:.2f}", flush=True)

json.dump(out, open(f"{O}/gradient_check.json", "w"), indent=1)
print("STEP15 COMPLETE")

# ---- primary estimator: 250 m narrow elevation slices ----
# Within a 250 m slice the residual lapse rate spans <1.5 C, so canopy height
# can be compared without a linear elevation adjustment. This avoids both the
# within-band lapse confounding of wide bands and any risk of over-correcting
# for elevation where canopy height and elevation are collinear.
df2 = df.copy()
df2["slice"] = (df2.elev // 250).astype(int)
rows_nb = []
for sl_, g in df2.groupby("slice"):
    if len(g) < 3000:
        continue
    X = np.column_stack([g.cos_i, g.slope, g.northness, np.ones(len(g))])
    b_, *_ = np.linalg.lstsq(X, g.lst, rcond=None)
    r = g.lst.values - X @ b_
    a_ = r[(g.chm >= WIN_LO) & (g.chm < WIN_MID)]
    c_ = r[(g.chm >= WIN_MID) & (g.chm < WIN_HI)]
    if len(a_) > 150 and len(c_) > 150:
        rows_nb.append(dict(elev_lo=int(sl_ * 250), n=int(len(g)),
                            slope=float((c_.mean() - a_.mean()) * 2),
                            se=float(np.sqrt(a_.var()/len(a_) + c_.var()/len(c_)) * 2)))
vn = np.array([r["slope"] for r in rows_nb]); sn = np.array([r["se"] for r in rows_nb])
wn = 1 / sn ** 2
nb_slope = float((vn * wn).sum() / wn.sum()); nb_se = float(np.sqrt(1 / wn.sum()))
midn = np.array([r["elev_lo"] + 125 for r in rows_nb])
lrn = stats.linregress(midn, vn)
out2 = dict(slices=rows_nb, slope=nb_slope, se=nb_se,
            trend_per_1000m=float(lrn.slope * 1000), trend_p=float(lrn.pvalue),
            n_slices=len(rows_nb))
G0 = json.load(open(f"{O}/gradient_check.json")); G0["narrow_band"] = out2
json.dump(G0, open(f"{O}/gradient_check.json", "w"), indent=1)
print(f"NARROW-BAND primary: {nb_slope:+.2f} ± {1.96*nb_se:.2f} °C/10 m "
      f"({len(rows_nb)} slices); trend p = {lrn.pvalue:.3f}", flush=True)

# ---- support-controlled sensitivity, pooled as a band-weighted mean ----
# NOTE: pooling the raw residuals across bands is biased, because the band
# composition of the 20-25 m and 25-30 m groups differs (Simpson's paradox).
# The defensible pooled figure is the inverse-variance weighted mean of the
# within-band estimates.
G = json.load(open(f"{O}/gradient_check.json"))
v = np.array([b["slope_np"] for b in G["bands"].values()], float)
se = np.array([b["slope_np_se"] for b in G["bands"].values()], float)
w = 1.0 / se ** 2
pooled = float((v * w).sum() / w.sum())
pooled_se = float(np.sqrt(1.0 / w.sum()))
# heterogeneity across bands (Cochran's Q)
Q = float((w * (v - pooled) ** 2).sum())
from scipy.stats import chi2
pQ = float(1 - chi2.cdf(Q, len(v) - 1))

# marginal sensitivity as a function of canopy height (island-wide GBM,
# terrain at overall median): reported instead of a single "saturation point"
hs = np.arange(12, 45.5, 0.5)
base = {f: float(df[f].median()) for f in FEATS}
pv = np.array([float(gbm.predict(pd.DataFrame([{**base, "chm": h}])[FEATS])[0])
               for h in hs])
loc = np.gradient(pv, hs) * 10
curve = {int(h): float(loc[np.argmin(abs(hs - h))]) for h in (15, 20, 25, 30, 35, 40)}
below03 = hs[np.where(loc > -0.3)[0]]
sat = float(below03[0]) if len(below03) else None

G["pooled"] = dict(np_slope=pooled, np_se=pooled_se,
                   band_range=[float(v.min()), float(v.max())],
                   Q=Q, p_heterogeneity=pQ, saturation_m=sat, slope_by_height=curve,
                   cf_support_range=[min(b["cf_support_p10_p90"] for b in G["bands"].values()),
                                     max(b["cf_support_p10_p90"] for b in G["bands"].values())],
                   cf_extrap_range=[min(b["cf_extrap_5_25"] for b in G["bands"].values()),
                                    max(b["cf_extrap_5_25"] for b in G["bands"].values())])
json.dump(G, open(f"{O}/gradient_check.json", "w"), indent=1)
print(f"band-weighted sensitivity: {pooled:+.2f} ± {1.96*pooled_se:.2f} °C/10 m"
      f"  (bands {v.min():+.2f} to {v.max():+.2f})")
print(f"heterogeneity Q={Q:.1f}, p={pQ:.3f}")
print("local slope by canopy height (°C/10 m):", {k: round(v,2) for k,v in curve.items()},
      f"| |slope|<0.3 above {sat:.0f} m")
print("counterfactual: extrapolated", [round(x,2) for x in G["pooled"]["cf_extrap_range"]],
      "vs support-limited", [round(x,2) for x in G["pooled"]["cf_support_range"]])
