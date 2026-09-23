# -*- coding: utf-8 -*-
"""Model-free statement of the thermal-versus-greenness lag.

tau is defined only under the single-exponential form, so the lag should also be shown
without any fitted curve. For the same landslide patches this compares, bin by bin, the
fraction of each index's first-year anomaly that is still present, and bootstraps the
difference over patches so the comparison carries an interval.
Output: outputs/modelfree_lag.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

O = f"{_TROOT}/outputs"
RNG = np.random.default_rng(20260915)
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
ev = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.src == "event")]
ev = ev[ev.dlst.notna() & ev.dndvi.notna()].copy()
ev["bin"] = np.floor(ev.age - 0.6).astype(int)          # bin 0 is ages 0.6-1.6

REF, HOR = 0, 9                                          # first bin, and the bin at ~10 yr
COL = {"dlst": "thermal", "dndvi": "greenness"}


def curve(d):
    """Fraction of the first-year anomaly still present, per 1-year bin, per index."""
    out = {}
    for c, name in COL.items():
        g = d.groupby("bin")[c].agg(["mean", "count"])
        g = g[g["count"] >= 12]
        if REF not in g.index:
            return None
        ref = g.loc[REF, "mean"]
        out[name] = {int(b): float(g.loc[b, "mean"] / ref) for b in g.index}
    return out


base = curve(ev)
pids = ev.pid.unique()
BOOT = 400
acc = {"thermal": [], "greenness": [], "diff": []}
percurve = {"thermal": {}, "greenness": {}, "diff": {}}     # per-bin resamples, for the bands
for _ in range(BOOT):
    take = RNG.choice(pids, size=len(pids), replace=True)
    d = ev.set_index("pid").loc[take].reset_index()
    c = curve(d)
    if not c or HOR not in c["thermal"] or HOR not in c["greenness"]:
        continue
    t, g = c["thermal"][HOR], c["greenness"][HOR]
    acc["thermal"].append(t); acc["greenness"].append(g); acc["diff"].append(t - g)
    for b in c["thermal"]:
        if b in c["greenness"]:
            percurve["thermal"].setdefault(b, []).append(c["thermal"][b])
            percurve["greenness"].setdefault(b, []).append(c["greenness"][b])
            percurve["diff"].setdefault(b, []).append(c["thermal"][b] - c["greenness"][b])


def ci(v):
    v = np.array(v)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


RES = {
    "bin_definition": "bin b covers ages b+0.6 to b+1.6 yr; bin 0 is the reference",
    "horizon_bin": HOR, "n_boot": len(acc["diff"]), "n_patches": int(len(pids)),
    "remaining_at_10yr": {
        "thermal": {"point": base["thermal"][HOR], "ci": ci(acc["thermal"])},
        "greenness": {"point": base["greenness"][HOR], "ci": ci(acc["greenness"])},
        "difference_thermal_minus_greenness": {
            "point": base["thermal"][HOR] - base["greenness"][HOR],
            "ci": ci(acc["diff"]),
            "p_gt0": float(np.mean(np.array(acc["diff"]) > 0))},
    },
    "curves": base,
    "curve_ci": {k: {int(b): ci(v) for b, v in d.items() if len(v) >= 100}
                 for k, d in percurve.items()},
    "diff_p_gt0_by_bin": {int(b): float(np.mean(np.array(v) > 0))
                          for b, v in percurve["diff"].items() if len(v) >= 100},
}
json.dump(RES, open(f"{O}/modelfree_lag.json", "w"), indent=1)

print("fraction of the first-year anomaly still present, no curve fitted")
print(f"{'age bin (yr)':>13s} {'thermal':>9s} {'greenness':>10s} {'difference':>11s}")
for b in sorted(base["thermal"]):
    if b in base["greenness"]:
        t, g = base["thermal"][b], base["greenness"][b]
        print(f"{b+0.6:5.1f}-{b+1.6:5.1f} {t:9.3f} {g:10.3f} {t-g:11.3f}")
r = RES["remaining_at_10yr"]
print(f"\nat ~10 yr  thermal {r['thermal']['point']:.3f} "
      f"[{r['thermal']['ci'][0]:.3f}, {r['thermal']['ci'][1]:.3f}]   "
      f"greenness {r['greenness']['point']:.3f} "
      f"[{r['greenness']['ci'][0]:.3f}, {r['greenness']['ci'][1]:.3f}]")
d = r["difference_thermal_minus_greenness"]
print(f"difference {d['point']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}], "
      f"P(thermal remains more) = {d['p_gt0']:.3f}  ({RES['n_boot']} patch resamples)")
