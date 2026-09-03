"""Step 8o: official-boundary area strata (<2 / 2-10 / >=10 ha) recovery fits
on the rebuilt database -> results2["area_strata_official"]. Boundaries: 10 ha
is the official large-scale-landslide area threshold (ARDSWC), 2 ha is the
integer boundary at the sample median. Same fit as step08m.
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
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6)]
ev = post[post.src == "event"]
print("median patch area (ha):",
      round(float(ev.drop_duplicates(["event", "year", "elev", "area_ha"])
                  .area_ha.median()), 2))


def fit_rec(d, val="dlst", tmax=22, min_bin=10):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    if len(d) < 80:
        return None
    bins = np.arange(0.6, tmax + 1, 1.0)
    bc, bm, bs = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        g = d[(d.age >= a) & (d.age < b)][val]
        if len(g) >= min_bin:
            bc.append(0.5 * (a + b)); bm.append(g.mean())
            bs.append(g.std() / np.sqrt(len(g)))
    if len(bc) < 5:
        return None
    bc_, bm_, bs_ = map(np.array, (bc, bm, bs))
    popt, pcov = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), bc_, bm_,
                           p0=[bm_[0], 6.0], sigma=np.maximum(bs_, 1e-3),
                           absolute_sigma=True, maxfev=20000,
                           bounds=([-25, 0.3], [25, 60]))
    perr = np.sqrt(np.diag(pcov))
    return dict(A=float(popt[0]), tau=float(popt[1]), tau_se=float(perr[1]),
                n=int(len(d)))


classes = {"small_lt2": ev[ev.area_ha < 2],
           "mid_2_10": ev[(ev.area_ha >= 2) & (ev.area_ha < 10)],
           "large_ge10": ev[ev.area_ha >= 10]}
out = {}
for name, d in classes.items():
    ft = fit_rec(d, "dlst"); fg = fit_rec(d, "dndvi")
    out[name] = dict(
        n_rows=int(len(d)), med_area=float(d.area_ha.median()),
        med_elev=float(d.elev.median()),
        t=ft["tau"] if ft else None, t_se=ft["tau_se"] if ft else None,
        g=fg["tau"] if fg else None, g_se=fg["tau_se"] if fg else None,
        ratio=(ft["tau"] / fg["tau"]) if ft and fg else None)
    print(name, {k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in out[name].items()})

R = json.load(open(f"{O}/results2.json"))
R["area_strata_official"] = out
json.dump(R, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
print("STEP8O COMPLETE")
