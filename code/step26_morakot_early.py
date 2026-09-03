"""Step 26: Morakot ages 1-3 through the cross-sensor bridge (display only).

Combines the clean-Morakot per-sensor deltas (step 24) with the same-patch
overlap calibrations to express ages 1-3 on the Landsat-8 scale. Both overlap
years' calibrations are carried as an explicit systematic band because the
L7->L8 transfer differs between 2013 and 2014; L5 is chained L5->L7->L8. These
values never enter any fit. Output: outputs/morakot_early.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

import os
os.chdir(f"{_TROOT}")
d = pd.read_parquet("data/l57_deltas.parquet")
T = json.load(open("outputs/l57_transfer.json"))
mk = d[(d.src == "event") & d.event.fillna("").str.contains("莫拉克")
       & (d.redist_frac < 0.05) & (d.n_ctrl >= 30) & (d.forest2000_frac > 0.6)]

c13, c14 = T["L7_vs_L8_2013_lst"]["all"], T["L7_vs_L8_2014_lst"]["all"]
c13n, c14n = T["L7_vs_L8_2013_ndvi"]["all"], T["L7_vs_L8_2014_ndvi"]["all"]
l510, l511 = T["L5_vs_L7_2010_lst"]["all"], T["L5_vs_L7_2011_lst"]["all"]
l510n, l511n = T["L5_vs_L7_2010_ndvi"]["all"], T["L5_vs_L7_2011_ndvi"]["all"]


def inv(y, cal):                     # L7 value -> L8 scale
    return (y - cal["intercept"]) / cal["slope"]


R = {"cal_l7_lst": {"2013": c13, "2014": c14},
     "cal_l5_lst": {"2010": l510, "2011": l511}, "points": []}
for tag, age, sensor in [("c10l5", 1.0, "L5"), ("c11l7", 2.0, "L7"),
                         ("c12l7", 3.0, "L7")]:
    cl, cn = f"dlst_{tag}", f"dndvi_{tag}"
    g = mk[mk[cl].notna()]
    if len(g) < 20:
        continue
    raw = float(g[cl].mean()); se = float(g[cl].std() / np.sqrt(len(g)))
    if sensor == "L7":
        eq = [inv(raw, c13), inv(raw, c14)]
        se_eq = se / min(c13["slope"], c14["slope"])
    else:                             # chain L5 -> L7 (year-matched) -> L8
        l7v = [inv(raw, l510), inv(raw, l511)]
        eq = [inv(v, c) for v in l7v for c in (c13, c14)]
        se_eq = se / (min(l510["slope"], l511["slope"])
                      * min(c13["slope"], c14["slope"]))
    gn = g[cn].dropna() if cn in g.columns else pd.Series(dtype=float)
    nd_raw = float(gn.mean()) if len(gn) > 20 else None
    nd_se = float(gn.std() / np.sqrt(len(gn))) if len(gn) > 20 else None
    if nd_raw is not None:
        if sensor == "L7":            # NDVI L7->L8 is near-identity
            nd_eq = [(nd_raw - c["intercept"]) / c["slope"] for c in (c13n, c14n)]
        else:
            l7n = [(nd_raw - c["intercept"]) / c["slope"] for c in (l510n, l511n)]
            nd_eq = [(v - c["intercept"]) / c["slope"]
                     for v in l7n for c in (c13n, c14n)]
    else:
        nd_eq = None
    R["points"].append(dict(
        tag=tag, age=age, sensor=sensor, n=int(len(g)),
        lst_raw=raw, lst_se=se, lst_eq_lo=float(min(eq)),
        lst_eq_hi=float(max(eq)), lst_eq_mid=float(np.mean(eq)),
        lst_se_eq=float(se_eq),
        ndvi_raw=nd_raw, ndvi_se=nd_se,
        ndvi_eq_lo=float(min(nd_eq)) if nd_eq else None,
        ndvi_eq_hi=float(max(nd_eq)) if nd_eq else None,
        ndvi_eq_mid=float(np.mean(nd_eq)) if nd_eq else None))
    print(R["points"][-1])
json.dump(R, open("outputs/morakot_early.json", "w"), indent=1)
print("STEP26 COMPLETE")
