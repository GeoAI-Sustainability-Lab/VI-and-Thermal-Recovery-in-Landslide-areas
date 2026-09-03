"""Step 8m: two mechanism analyses from the existing unified database.
(1) Patch-area-stratified recovery fits (does landslide size control recovery
    speed?) -> results2["area_strata"].
(2) ΔLST-ΔNDVI hysteresis: paired age-binned means of both anomalies on the
    SAME patch-epoch rows -> results2["hysteresis"]. If thermal recovery waits
    for full structural closure, the trajectory returns along a path where
    ΔNDVI is near zero while ΔLST remains positive.
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
print("area quantiles (ha):", ev.area_ha.quantile([0.33, 0.66, 0.9]).round(2).to_dict())


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


# ---- (1) area strata (terciles of event patch area) ----
q1, q2 = ev.area_ha.quantile([1/3, 2/3])
classes = {
    f"small (<{q1:.1f} ha)": ev[ev.area_ha < q1],
    f"medium ({q1:.1f}-{q2:.1f} ha)": ev[(ev.area_ha >= q1) & (ev.area_ha < q2)],
    f"large (>{q2:.1f} ha)": ev[ev.area_ha >= q2],
}
area_strata = {}
for name, d in classes.items():
    ft = fit_rec(d, "dlst"); fg = fit_rec(d, "dndvi")
    med_elev = float(d.elev.median())
    area_strata[name] = dict(
        n_patches=int(d.drop_duplicates(["event", "year", "elev", "area_ha"]).shape[0]),
        med_area=float(d.area_ha.median()), med_elev=med_elev,
        t=ft["tau"] if ft else None, t_se=ft["tau_se"] if ft else None,
        g=fg["tau"] if fg else None, g_se=fg["tau_se"] if fg else None,
        ratio=(ft["tau"] / fg["tau"]) if ft and fg else None)
    print(name, area_strata[name])
# year-1 shock by size (does size control impact magnitude?)
y1 = ev[(ev.age > 0.6) & (ev.age < 2.0)]
shock = {}
for name, d in classes.items():
    g = y1[y1.area_ha.isin(d.area_ha)]  # same class filter via area bounds instead:
shock = {name: dict(n=int(len(y1[(y1.area_ha >= d.area_ha.min()) & (y1.area_ha <= d.area_ha.max())])),
                    dlst=float(y1[(y1.area_ha >= d.area_ha.min()) & (y1.area_ha <= d.area_ha.max())].dlst.mean()))
         for name, d in classes.items()}
for k, v in shock.items():
    area_strata[k]["year1_dlst"] = v["dlst"]

# ---- (2) hysteresis: paired age-binned means ----
pair = ev[ev.dlst.notna() & ev.dndvi.notna()]
bins = np.arange(0.6, 23, 1.0)
hy = []
for a, b in zip(bins[:-1], bins[1:]):
    g = pair[(pair.age >= a) & (pair.age < b)]
    if len(g) >= 30:
        hy.append(dict(age=float(0.5 * (a + b)), n=int(len(g)),
                       dndvi=float(g.dndvi.mean()),
                       dndvi_se=float(g.dndvi.std() / np.sqrt(len(g))),
                       dlst=float(g.dlst.mean()),
                       dlst_se=float(g.dlst.std() / np.sqrt(len(g)))))
print("hysteresis tail (oldest 4):", [(round(h['age'],1), round(h['dndvi'],3), round(h['dlst'],2)) for h in hy[-4:]])

R = json.load(open(f"{O}/results2.json"))
R["area_strata"] = area_strata
R["hysteresis"] = hy
json.dump(R, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
print("STEP8M COMPLETE")
