# -*- coding: utf-8 -*-
"""Recovery expressed as the fraction of the initial thermal anomaly that has
dissipated by a given age, which is the quantity Su et al. (2026, Nature
Geoscience) report for Europe (>82% within 10 years). Same binned weighted
exponential fit and patch-level block bootstrap as the published step16/step17,
so every number here reconciles with the deposited results.
Writes: outputs/class_dissipation.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

TMAX, NBOOT = 22.0, 200
EDGES = np.arange(0.6, TMAX + 1, 1.0)
rng = np.random.default_rng(17)
os.makedirs("outputs", exist_ok=True)

L = pd.read_parquet(f"{_TROOT}/outputs/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= TMAX)]
ev = post[post.src == "event"]
hz = post[post.src == "hansen"]

CLASSES = [
    ("landslide_all",  "Landslide, all triggers",      ev),
    ("typhoon",        "Landslide, typhoon",           ev[ev.agent == "typhoon_rain"]),
    ("rainfall",       "Landslide, heavy rainfall",    ev[ev.agent == "rainfall"]),
    ("earthquake",     "Landslide, earthquake",        ev[ev.agent == "earthquake"]),
    ("ls_lt2",         "Landslide, < 2 ha",            ev[ev.area_ha < 2]),
    ("ls_2_10",        "Landslide, 2-10 ha",           ev[(ev.area_ha >= 2) & (ev.area_ha < 10)]),
    ("ls_ge10",        "Landslide, >= 10 ha",          ev[ev.area_ha >= 10]),
    ("annual_loss",    "Annual tree-cover loss, all",    hz[(hz.year >= 2015) & (hz.year <= 2023)]),
    ("al_lt2",         "Annual tree-cover loss, < 2 ha",          hz[(hz.year >= 2015) & (hz.year <= 2023) & (hz.area_ha < 2)]),
    ("al_ge2",         "Annual tree-cover loss, >= 2 ha",         hz[(hz.year >= 2015) & (hz.year <= 2023) & (hz.area_ha >= 2)]),
]
AGES = [5.0, 10.0, 15.0, 20.0]


def binned(d, val):
    c, m, se = [], [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        g = d[(d.age >= a) & (d.age < b)][val].dropna()
        if len(g) >= 10:
            c.append(0.5 * (a + b)); m.append(float(g.mean()))
            se.append(float(g.std() / np.sqrt(len(g))))
    return np.array(c), np.array(m), np.array(se)


def fit(d, val):
    c, m, se = binned(d, val)
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


def dissipated(tau, t):
    """Fraction of the initial anomaly that has gone by age t, for A*exp(-t/tau)."""
    return 1.0 - np.exp(-t / tau)


out = {}
hdr = f"{'class':30s} {'patches':>8s} {'A (°C)':>8s} {'tau_LST':>8s} {'tau_NDVI':>9s} {'ratio':>6s}  " + \
      "  ".join(f"{'d'+str(int(a))+'yr':>12s}" for a in AGES)
print(hdr); print("-" * len(hdr))
for key, label, d in CLASSES:
    if len(d) < 200:
        print(f"{label:30s}  (too few rows: {len(d)})"); continue
    ft, fg = fit(d, "dlst"), fit(d, "dndvi")
    if not ft:
        print(f"{label:30s}  (fit failed)"); continue
    A, tau = ft
    pids = d.pid.unique()
    idx = {p: g.index.values for p, g in d.groupby("pid")}
    bt = {a: [] for a in AGES}; bt_tau = []; bt_ratio = []
    for _ in range(NBOOT):
        s = rng.choice(pids, len(pids), replace=True)
        dd = d.loc[np.concatenate([idx[p] for p in s])]
        f2 = fit(dd, "dlst")
        if not f2:
            continue
        bt_tau.append(f2[1])
        for a in AGES:
            bt[a].append(dissipated(f2[1], a) * 100)
        g2 = fit(dd, "dndvi")
        if g2:
            bt_ratio.append(f2[1] / g2[1])
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else None
    _c, _m, _se = binned(d, "dlst")
    e = dict(label=label, n_rows=int(len(d)), n_patches=int(d.pid.nunique()),
             age_min=float(_c.min()), age_max=float(_c.max()),
             A=A, tau=tau, tau_ci=ci(bt_tau),
             tau_ndvi=(fg[1] if fg else None),
             ratio=(tau / fg[1] if fg else None), ratio_ci=ci(bt_ratio),
             dissipated={str(int(a)): dissipated(tau, a) * 100 for a in AGES},
             dissipated_ci={str(int(a)): ci(bt[a]) for a in AGES})
    out[key] = e
    cells = "  ".join(f"{e['dissipated'][str(int(a))]:5.0f}% [{e['dissipated_ci'][str(int(a))][0]:.0f}-{e['dissipated_ci'][str(int(a))][1]:.0f}]" for a in AGES)
    rr = f"{e['ratio']:6.2f}" if e['ratio'] else "     -"
    tn = f"{e['tau_ndvi']:9.2f}" if e['tau_ndvi'] else "        -"
    print(f"{label:30s} {e['n_patches']:8d} {A:8.2f} {tau:8.2f} {tn} {rr}  {cells}")

json.dump(out, open(f"{_TROOT}/outputs/class_dissipation.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved outputs/class_dissipation.json")
