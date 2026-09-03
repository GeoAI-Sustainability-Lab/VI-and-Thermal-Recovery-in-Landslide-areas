"""Step 8p: rebuild the results2 keys that earlier lived only as inline
session computations — tau_ratios (all strata), the loose-binned earthquake
greenness fit, event_groups (+total), n_redist_ok — from the rebuilt
chrono2_long / patches_deltas2. Run after step08e/g/m/o/n.
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
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6)]
ev = post[post.src == "event"]
hz = post[post.src == "hansen"]
R = json.load(open(f"{O}/results2.json"))
F2 = R["fits2"]


def fit_rec(d, val="dlst", tmax=22, min_bin=10, step=1.0, p0tau=6.0):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    if len(d) < 60:
        return None
    bins = np.arange(0.6, tmax + step, step)
    bc, bm, bs = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        g = d[(d.age >= a) & (d.age < b)][val]
        if len(g) >= min_bin:
            bc.append(0.5 * (a + b)); bm.append(g.mean())
            bs.append(g.std() / np.sqrt(len(g)))
    if len(bc) < 4:
        return None
    bc_, bm_, bs_ = map(np.array, (bc, bm, bs))
    popt, pcov = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), bc_, bm_,
                           p0=[bm_[0], p0tau], sigma=np.maximum(bs_, 1e-3),
                           absolute_sigma=True, maxfev=20000,
                           bounds=([-25, 0.3], [25, 60]))
    perr = np.sqrt(np.diag(pcov))
    return dict(A=float(popt[0]), tau=float(popt[1]), tau_se=float(perr[1]),
                n=int(len(d)))


def pair(key_t, key_g):
    t = F2[key_t]["tau"]; g = F2[key_g]["tau"]
    return dict(t=float(t), g=float(g), ratio=float(t / g),
                t_se=float(F2[key_t]["tau_se"]), g_se=float(F2[key_g]["tau_se"]))


TRR = {
    "event_pooled": pair("event_dlst", "event_dndvi"),
    "event_low": pair("ev_elev_low_dlst", "ev_elev_low_dndvi"),
    "event_mid": pair("ev_elev_mid_dlst", "ev_elev_mid_dndvi"),
    "event_high": pair("ev_elev_high_dlst", "ev_elev_high_dndvi"),
    "typhoon": pair("agent_typhoon_rain_dlst", "agent_typhoon_rain_dndvi"),
    "rainfall": pair("agent_rainfall_dlst", "agent_rainfall_dndvi"),
}
hr_t = fit_rec(hz[(hz.year >= 2015) & (hz.year <= 2023)], "dlst")
hr_g = fit_rec(hz[(hz.year >= 2015) & (hz.year <= 2023)], "dndvi")
TRR["hansen_recent"] = dict(t=hr_t["tau"], g=hr_g["tau"],
                            ratio=hr_t["tau"] / hr_g["tau"],
                            t_se=hr_t["tau_se"], g_se=hr_g["tau_se"])
yr_t = fit_rec(ev[ev.year >= 2018], "dlst", tmax=9)
yr_g = fit_rec(ev[ev.year >= 2018], "dndvi", tmax=9)
TRR["event_recent"] = dict(t=yr_t["tau"], g=yr_g["tau"],
                           ratio=yr_t["tau"] / yr_g["tau"],
                           t_se=yr_t["tau_se"], g_se=yr_g["tau_se"])
R["tau_ratios"] = TRR
for k, v in TRR.items():
    print(k, round(v["t"], 1), round(v["g"], 1), round(v["ratio"], 2))

F2["agent_earthquake_dndvi_loose"] = fit_rec(
    ev[ev.agent == "earthquake"], "dndvi", tmax=22, min_bin=6, step=2.0)
print("eq loose:", F2["agent_earthquake_dndvi_loose"])
R["fits2"] = F2

dd = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                     columns=["src", "agent", "year", "event", "elev",
                              "area_ha", "redist_frac"])
evp = dd[dd.src == "event"]
groups = []
for (name, agent, yr), g in evp.groupby(["event", "agent", "year"]):
    groups.append(dict(event=name, agent=agent, year0=int(yr),
                       year1=int(yr), n=int(len(g)),
                       area_ha=float(g.area_ha.sum()),
                       elev_med=float(g.elev.median())))
groups.sort(key=lambda x: -x["n"])
R["event_groups"] = groups
R["event_groups_total"] = dict(n_groups=len(groups), n=int(len(evp)),
                               area_ha=float(evp.area_ha.sum()))
R["n_redist_ok"] = int((dd.redist_frac < 0.05).sum())
print("groups:", len(groups), "n_redist_ok:", R["n_redist_ok"])

json.dump(R, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
print("STEP8P COMPLETE")
