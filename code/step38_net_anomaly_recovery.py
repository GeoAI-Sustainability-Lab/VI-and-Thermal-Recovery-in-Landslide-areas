# -*- coding: utf-8 -*-
"""Recovery of the NET anomaly, with the pre-event level of each patch removed.

The main recovery curves describe the patch-minus-control difference itself, which for
landslide-prone sites includes warmth that existed before the event. A study that subtracts a
pre-disturbance baseline (Su et al., 2026) measures the dissipation of the event-induced part
only. This step builds that quantity where the record allows it: for every landslide patch
with a pre-event summer (cohorts 2014-2025), Delta_net(t) = Delta(t) - Delta_pre, binned by
age; the ten-year fraction is read from the observed bins and, for comparison, from an
exponential fit. Only the 2014-2016 cohorts reach ten years, so that horizon rests on them.
Output: outputs/net_anomaly_recovery.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

O = f"{_TROOT}/outputs"
RNG = np.random.default_rng(7)
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
ev = L[(L.src == "event") & L.clean].copy()
pre = ev[ev.is_pre].groupby("pid").agg(dlst_pre=("dlst", "mean"), dndvi_pre=("dndvi", "mean"))
post = ev[(~ev.is_pre) & (ev.age > 0.6) & (ev.age <= 22)].merge(pre, on="pid", how="inner")
post["dlst_net"] = post.dlst - post.dlst_pre
post["dndvi_net"] = post.dndvi - post.dndvi_pre
post["bin"] = np.floor(post.age - 0.6).astype(int)


def binned(d, col, min_bin=12):
    g = d.groupby("bin")[col].agg(["mean", "std", "count"])
    g = g[g["count"] >= min_bin]
    return g.index.values + 1.1, g["mean"].values, (g["std"] / np.sqrt(g["count"])).values, g["count"].values


def ten_year(d, col):
    bc, bm, bs, bn = binned(d, col)
    if len(bc) < 3 or 0 not in (bc - 1.1).astype(int):
        return None
    i1 = int(np.argmin(np.abs(bc - 1.1))); j = int(np.argmin(np.abs(bc - 10.1)))
    if abs(bc[j] - 10.1) > 1.0:
        return None
    rem = bm[j] / bm[i1]
    return dict(mean_ref=float(bm[i1]), mean_10=float(bm[j]), n_ref=int(bn[i1]), n_10=int(bn[j]),
                remaining_frac=float(rem), dissipated_pct=float(100 * (1 - rem)))


def fit_exp(d, col):
    bc, bm, bs, bn = binned(d, col)
    try:
        p, c = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), bc, bm, p0=[bm[0], 6.0],
                         sigma=np.maximum(bs, 1e-3), absolute_sigma=True,
                         bounds=([-25, .3], [25, 60]), maxfev=200000)
        return dict(A=float(p[0]), tau=float(p[1]), tau_se=float(np.sqrt(c[1, 1])),
                    dissipated_pct_10=float(100 * (1 - np.exp(-10 / p[1]))),
                    n_bins=int(len(bc)), age_max=float(bc[-1]))
    except Exception as e:
        return dict(error=repr(e)[:80])


R = {"n_patches_with_pre": int(post.pid.nunique()), "n_rows": int(len(post)),
     "cohorts": sorted(int(y) for y in post.year.unique())}
for col, name in (("dlst_net", "thermal"), ("dndvi_net", "greenness"), ("dlst", "thermal_raw")):
    R[name] = dict(ten_year_observed=ten_year(post, col), exp_fit_all_cohorts=fit_exp(post, col))

# bootstrap over patches for the observed ten-year fraction of the net thermal anomaly
pids = post.pid.unique(); acc = []
for _ in range(400):
    take = RNG.choice(pids, size=len(pids), replace=True)
    d = post.set_index("pid").loc[take].reset_index()
    t = ten_year(d, "dlst_net")
    if t: acc.append(t["dissipated_pct"])
R["thermal"]["ten_year_observed"]["dissipated_pct_ci"] = [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]
R["thermal"]["ten_year_observed"]["n_boot"] = len(acc)
# per-bin series of the raw and the net anomaly (pooled cohorts) with the number of cohorts per bin,
# plus the two cohorts that carry the record beyond five years, for the supplementary figure
def series(d, col):
    g = d.groupby("bin").agg(mean=(col, "mean"), n=(col, "count"), ncoh=("year", "nunique"))
    g = g[g["n"] >= 12]
    return {"age": [float(b + 1.1) for b in g.index], "mean": [float(v) for v in g["mean"]],
            "n": [int(v) for v in g["n"]], "n_cohorts": [int(v) for v in g["ncoh"]]}
R["series"] = {"pooled": {c: series(post, c) for c in ("dlst", "dlst_net", "dndvi", "dndvi_net")}}
for yr in (2016, 2017):
    R["series"][str(yr)] = {c: series(post[post.year == yr], c) for c in ("dlst", "dlst_net", "dndvi", "dndvi_net")}
    R["series"][str(yr)]["n_patches"] = int(post[post.year == yr].pid.nunique())
    R["series"][str(yr)]["pre_mean"] = float(pre.loc[post[post.year == yr].pid.unique(), "dlst_pre"].mean())
# patch bootstrap band for the pooled net thermal series
bands = {}
for _ in range(300):
    take = RNG.choice(pids, size=len(pids), replace=True)
    d = post.set_index("pid").loc[take].reset_index()
    g = d.groupby("bin")["dlst_net"].agg(["mean", "count"]); g = g[g["count"] >= 12]
    for b, v in g["mean"].items():
        bands.setdefault(int(b), []).append(float(v))
R["series"]["pooled"]["dlst_net_ci"] = {str(b): [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                        for b, v in bands.items() if len(v) >= 100}
# per-cohort reach: patches, rows and the oldest bin observed, so captions can say which
# cohorts are followed beyond a given age without hand-written lists
R["cohort_summary"] = {str(int(y)): dict(n_patches=int(g.pid.nunique()), n_rows=int(len(g)),
                                         max_age=float(g.age.max()),
                                         max_bin_ge12=int(max([b for b, n in g.bin.value_counts().items() if n >= 12], default=-1)))
                       for y, g in post.groupby("year")}
# which cohorts supply the ten-year bin
ten = post[post.bin == 9]
R["ten_year_bin_cohorts"] = {str(int(y)): int(n) for y, n in ten.year.value_counts().sort_index().items()}
# pre-event share of the first-year anomaly, pooled
first = post[post.bin == 0]
R["pre_share_of_first_year"] = float(first.dlst_pre.mean() / first.dlst.mean())
json.dump(R, open(f"{O}/net_anomaly_recovery.json", "w"), indent=1)

print("patches with a pre-event summer:", R["n_patches_with_pre"], "rows", R["n_rows"], "cohorts", R["cohorts"])
for name in ("thermal", "greenness", "thermal_raw"):
    t = R[name]["ten_year_observed"]; f = R[name]["exp_fit_all_cohorts"]
    print(f"{name:12s} observed 10-yr dissipated {t['dissipated_pct']:.1f}%  (ref {t['mean_ref']:+.2f} n={t['n_ref']}, "
          f"10 yr {t['mean_10']:+.2f} n={t['n_10']})   exp fit tau {f.get('tau', float('nan')):.1f} -> {f.get('dissipated_pct_10', float('nan')):.1f}%")
print("thermal net 95% CI", [round(x, 1) for x in R["thermal"]["ten_year_observed"]["dissipated_pct_ci"]])
print("ten-year bin cohorts:", R["ten_year_bin_cohorts"], " pre share of first-year anomaly:", round(R["pre_share_of_first_year"], 2))
