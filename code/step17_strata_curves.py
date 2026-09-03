"""Step 17: stratified recovery curves and patch-level bootstrap tau intervals
for the stratified recovery figure. Strata use one common standard across the whole
2004-2025 database: trigger agent (four classes), elevation band, and the
official area classes. Every interval is resampled at the patch level.
Output: outputs/strata_curves.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

O = f"{_TROOT}/outputs"
NBOOT = 120
TMAX = 22.0
rng = np.random.default_rng(23)

L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= TMAX)].copy()
if "pid" not in post.columns:           # fallback for pre-v11 long tables
    post["pid"] = post.groupby(["src", "event", "year", "elev",
                                "area_ha"]).ngroup()
ev = post[post.src == "event"]
hz = post[post.src == "hansen"]

STRATA = [
    ("agent", "typhoon", "颱風崩塌", ev[ev.agent == "typhoon_rain"]),
    ("agent", "rainfall", "豪雨崩塌", ev[ev.agent == "rainfall"]),
    ("agent", "earthquake", "地震崩塌", ev[ev.agent == "earthquake"]),
    ("agent", "hansen", "年損失 2015–2023",
     hz[(hz.year >= 2015) & (hz.year <= 2023)]),
    ("elev", "lt1000", "<1,000 m", ev[ev.elev < 1000]),
    ("elev", "1000_2000", "1,000–2,000 m", ev[(ev.elev >= 1000) & (ev.elev < 2000)]),
    ("elev", "gt2000", ">2,000 m", ev[ev.elev >= 2000]),
    ("area", "lt2", "<2 ha", ev[ev.area_ha < 2]),
    ("area", "2_10", "2–10 ha", ev[(ev.area_ha >= 2) & (ev.area_ha < 10)]),
    ("area", "ge10", "≥10 ha (大規模崩塌)", ev[ev.area_ha >= 10]),
]
EDGES = np.arange(0.6, TMAX + 1, 1.0)


def binned(d, val):
    c, m, se, n = [], [], [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        g = d[(d.age >= a) & (d.age < b)][val].dropna()
        if len(g) >= 10:
            c.append(0.5 * (a + b)); m.append(float(g.mean()))
            se.append(float(g.std() / np.sqrt(len(g)))); n.append(int(len(g)))
    return np.array(c), np.array(m), np.array(se), n


def fit(d, val):
    c, m, se, _ = binned(d, val)
    if len(c) < 5:
        return None
    try:
        popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), c, m,
                            p0=[m[0], 8.0], sigma=np.maximum(se, 1e-3),
                            absolute_sigma=True, maxfev=20000,
                            bounds=([-25, 0.3], [25, 60]))
        return float(popt[0]), float(popt[1])
    except Exception:
        return None


out = {}
for kind, key, label, d in STRATA:
    if len(d) < 100:
        print("skip", key, len(d)); continue
    entry = dict(kind=kind, label=label, n_rows=int(len(d)),
                 n_patches=int(d.pid.nunique()))
    for val, nm in [("dlst", "thermal"), ("dndvi", "greenness")]:
        c, m, se, n = binned(d, val)
        f = fit(d, val)
        entry[nm] = dict(bin_center=c.tolist(), bin_mean=m.tolist(),
                         bin_se=se.tolist(), bin_n=n,
                         A=(f[0] if f else None), tau=(f[1] if f else None))
    # patch bootstrap for tau and the ratio
    pids = d.pid.unique()
    ix = {p: g.index.values for p, g in d.groupby("pid")}
    tl, gl, rl = [], [], []
    for _ in range(NBOOT):
        samp = rng.choice(pids, len(pids), replace=True)
        b = d.loc[np.concatenate([ix[p] for p in samp])]
        ft, fg = fit(b, "dlst"), fit(b, "dndvi")
        if ft:
            tl.append(ft[1])
        if fg:
            gl.append(fg[1])
        if ft and fg:
            rl.append(ft[1] / fg[1])

    def ci(a):
        a = np.array(a)
        return ([float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
                if len(a) > 20 else None)
    entry["thermal"]["tau_ci"] = ci(tl)
    entry["greenness"]["tau_ci"] = ci(gl)
    entry["ratio"] = dict(mean=(float(np.mean(rl)) if rl else None), ci=ci(rl),
                          p_gt1=(float((np.array(rl) > 1).mean()) if rl else None),
                          n_boot=len(rl))
    out[key] = entry
    tt = entry["thermal"]["tau"]; gg = entry["greenness"]["tau"]
    print(f"{key:12s} n={entry['n_patches']:5d}  tau_T="
          f"{tt if tt is None else round(tt,1)}  tau_G="
          f"{gg if gg is None else round(gg,1)}  ratio="
          f"{entry['ratio']['mean'] and round(entry['ratio']['mean'],2)}"
          f"  P(>1)={entry['ratio']['p_gt1']}", flush=True)

json.dump(out, open(f"{O}/strata_curves.json", "w"), ensure_ascii=False, indent=1)
print("STEP17 COMPLETE")
