"""Step 13g: cohort-aware analysis of the S1 pilot rows.
(1) Pooled cross-section fit is unconstrained because the 3-summer window
    confounds age with cohort (Morakot dominates ages 14-16). Refit excluding
    the Morakot cohort (t_event 2009.5-2009.75).
(2) Within-patch slopes d(dgamma)/dt from the 3 yearly observations per patch:
    cohort-composition-free closure rate; implied tau = -mean(delta)/mean(slope).
Adds keys fit_vh_xmor, fit_vv_xmor, within_patch, morakot_s1 to s1_pilot.json.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
df = pd.read_parquet(f"{D}/s1_pilot_rows.parquet")
dd = pd.read_parquet(f"{D}/patches_deltas2.parquet", columns=["t_event", "area_ha"])
df = df.merge(dd, left_on="pid", right_index=True)
mor = (df.t_event >= 2009.5) & (df.t_event <= 2009.75)
print("rows:", len(df), "morakot rows:", int(mor.sum()),
      "morakot patches:", df[mor].pid.nunique())


def fit_rec(d, val, tmax=21, min_bin=10):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    bins_ = np.arange(0.6, tmax + 1, 1.0)
    bc, bm, bs, bn = [], [], [], []
    for a_, b_ in zip(bins_[:-1], bins_[1:]):
        g = d[(d.age >= a_) & (d.age < b_)][val]
        if len(g) >= min_bin:
            bc.append(0.5*(a_+b_)); bm.append(g.mean())
            bs.append(g.std()/np.sqrt(len(g))); bn.append(len(g))
    if len(bc) < 5:
        return None
    bc_, bm_, bs_ = map(np.array, (bc, bm, bs))
    popt, pcov = curve_fit(lambda t, A, tau: A*np.exp(-t/tau), bc_, bm_,
                           p0=[bm_[0], 8.0], sigma=np.maximum(bs_, 1e-3),
                           absolute_sigma=True, maxfev=20000,
                           bounds=([-30, 0.3], [30, 60]))
    perr = np.sqrt(np.diag(pcov))
    return dict(A=float(popt[0]), tau=float(popt[1]), tau_se=float(perr[1]),
                n=int(len(d)),
                bins=dict(center=bc_.tolist(), mean=bm_.tolist(),
                          se=bs_.tolist(), n=bn))


S = json.load(open(f"{O}/s1_pilot.json"))
for b in ["vh", "vv"]:
    f_ = fit_rec(df[~mor], f"d{b}_db")
    S[f"fit_{b}_xmor"] = f_
    if f_:
        print(f"{b} xmor tau={f_['tau']:.1f}±{f_['tau_se']:.1f} A={f_['A']:.2f}")

# Morakot cohort mean deficit at ages 14-16
g = df[mor & df.dvh_db.notna()]
S["morakot_s1"] = dict(n_rows=int(len(g)), n_patches=int(g.pid.nunique()),
                       vh_mean=float(g.dvh_db.mean()),
                       vh_se=float(g.dvh_db.std()/np.sqrt(len(g))),
                       age_min=float(g.age.min()), age_max=float(g.age.max()))
print("morakot:", {k: round(v, 2) if isinstance(v, float) else v
                   for k, v in S["morakot_s1"].items()})

# within-patch slopes (>=2 valid years, use all 3 where present)
wp = {}
for b in ["vh", "vv"]:
    rows = []
    for pid, g in df[df[f"d{b}_db"].notna()].groupby("pid"):
        if len(g) < 2:
            continue
        x = g.age.values; y = g[f"d{b}_db"].values
        sl = np.polyfit(x, y, 1)[0]
        rows.append((pid, float(g.age.mean()), float(g[f"d{b}_db"].mean()),
                     float(sl), bool((g.t_event.iloc[0] >= 2009.5)
                                     and (g.t_event.iloc[0] <= 2009.75))))
    w = pd.DataFrame(rows, columns=["pid", "age", "dmean", "slope", "mor"])
    res = {}
    for name, d in [("all", w), ("xmor", w[~w.mor]),
                    ("young_lt8", w[w.age < 8]), ("old_ge8", w[(w.age >= 8) & ~w.mor]),
                    ("morakot", w[w.mor])]:
        if len(d) < 10:
            continue
        msl = d.slope.mean(); ssl = d.slope.std()/np.sqrt(len(d))
        mdm = d.dmean.mean()
        tau_imp = -mdm/msl if msl > 0 else None
        # delta-method se for tau = -m/s
        if tau_imp:
            sdm = d.dmean.std()/np.sqrt(len(d))
            tau_se = abs(tau_imp)*np.sqrt((sdm/mdm)**2 + (ssl/msl)**2)
        else:
            tau_se = None
        res[name] = dict(n=int(len(d)), slope=float(msl), slope_se=float(ssl),
                         dmean=float(mdm),
                         tau_implied=(float(tau_imp) if tau_imp else None),
                         tau_se=(float(tau_se) if tau_se else None))
        print(b, name, {k: (round(v, 3) if isinstance(v, float) else v)
                        for k, v in res[name].items()})
    wp[b] = res
S["within_patch"] = wp
json.dump(S, open(f"{O}/s1_pilot.json", "w"), indent=1)
print("STEP13G COMPLETE")
