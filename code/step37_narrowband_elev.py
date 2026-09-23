# -*- coding: utf-8 -*-
"""Sensitivity of the 250 m-slice canopy sensitivity to a within-slice elevation term.

The primary estimate residualises LST on illumination, slope and northness within each 250 m
slice and contrasts the 20-25 m and 25-30 m canopy groups, on the argument that the residual
lapse rate inside a slice is too small to matter. That argument is tested here rather than
assumed: the same contrast is recomputed with elevation added to the within-slice model, and
the mean elevation offset between the two canopy groups inside each slice is reported, since
that offset times the lapse rate is the size of the bias the term removes.
Output: outputs/narrowband_elev.json (also merged into gradient_check.json as "narrow_band_elev")
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy import stats

D = f"{_TROOT}/data"; O = f"{_TROOT}/outputs"
WIN_LO, WIN_MID, WIN_HI = 20.0, 25.0, 30.0
df = pd.read_parquet(f"{D}/buffering_sample.parquet").dropna(
    subset=["lst", "chm", "elev", "slope", "northness", "eastness", "cos_i"])
df["slice"] = (df.elev // 250).astype(int)


def contrast(g, with_elev):
    cols = [g.cos_i, g.slope, g.northness] + ([g.elev] if with_elev else []) + [np.ones(len(g))]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, g.lst, rcond=None)
    r = g.lst.values - X @ b
    a = r[(g.chm >= WIN_LO) & (g.chm < WIN_MID)]
    c = r[(g.chm >= WIN_MID) & (g.chm < WIN_HI)]
    if len(a) <= 150 or len(c) <= 150:
        return None
    return dict(slope=float((c.mean() - a.mean()) * 2),
                se=float(np.sqrt(a.var() / len(a) + c.var() / len(c)) * 2),
                lapse=(float(b[3]) * 1000 if with_elev else None))       # degC per km, within slice


rows = []
for sl, g in df.groupby("slice"):
    if len(g) < 3000:
        continue
    r0, r1 = contrast(g, False), contrast(g, True)
    if r0 is None or r1 is None:
        continue
    ga = g[(g.chm >= WIN_LO) & (g.chm < WIN_MID)]; gc = g[(g.chm >= WIN_MID) & (g.chm < WIN_HI)]
    rows.append(dict(elev_lo=int(sl * 250), n=int(len(g)),
                     slope_noelev=r0["slope"], se_noelev=r0["se"],
                     slope_elev=r1["slope"], se_elev=r1["se"],
                     lapse_within=r1["lapse"],
                     r_chm_elev=float(np.corrcoef(g.chm, g.elev)[0, 1]),
                     delev_tall_minus_short=float(gc.elev.mean() - ga.elev.mean())))


def pool(key, sekey):
    v = np.array([r[key] for r in rows]); s = np.array([r[sekey] for r in rows]); w = 1 / s ** 2
    m = float((v * w).sum() / w.sum()); se = float(np.sqrt(1 / w.sum()))
    mid = np.array([r["elev_lo"] + 125 for r in rows]); lr = stats.linregress(mid, v)
    return dict(slope=m, se=se, trend_per_1000m=float(lr.slope * 1000), trend_p=float(lr.pvalue))


out = dict(slices=rows, n_slices=len(rows),
           without_elevation=pool("slope_noelev", "se_noelev"),
           with_elevation=pool("slope_elev", "se_elev"))
out["change_pct"] = 100 * (out["with_elevation"]["slope"] - out["without_elevation"]["slope"]) / out["without_elevation"]["slope"]
out["delev_range_m"] = [min(r["delev_tall_minus_short"] for r in rows), max(r["delev_tall_minus_short"] for r in rows)]
out["delev_median_m"] = float(np.median([r["delev_tall_minus_short"] for r in rows]))
out["r_chm_elev_range"] = [min(r["r_chm_elev"] for r in rows), max(r["r_chm_elev"] for r in rows)]
out["lapse_within_median"] = float(np.median([r["lapse_within"] for r in rows]))
json.dump(out, open(f"{O}/narrowband_elev.json", "w"), indent=1)
G = json.load(open(f"{O}/gradient_check.json")); G["narrow_band_elev"] = out
json.dump(G, open(f"{O}/gradient_check.json", "w"), indent=1)

print(f"{'slice':>6s} {'n':>7s} {'no elev':>9s} {'with elev':>10s} {'lapse':>7s} {'r(chm,elev)':>12s} {'dElev(m)':>9s}")
for r in rows:
    print(f"{r['elev_lo']:6d} {r['n']:7d} {r['slope_noelev']:+9.3f} {r['slope_elev']:+10.3f} "
          f"{r['lapse_within']:+7.2f} {r['r_chm_elev']:+12.3f} {r['delev_tall_minus_short']:+9.1f}")
a, b = out["without_elevation"], out["with_elevation"]
print(f"\npooled without elevation {a['slope']:+.3f} ± {1.96*a['se']:.3f} (trend p {a['trend_p']:.2f})")
print(f"pooled with elevation    {b['slope']:+.3f} ± {1.96*b['se']:.3f} (trend p {b['trend_p']:.2f})   change {out['change_pct']:+.1f}%")
print(f"tall group sits {out['delev_median_m']:+.0f} m (median) relative to the short group inside a slice; "
      f"within-slice lapse median {out['lapse_within_median']:+.2f} degC/km")
