"""Step 22: does the sparser 2013-2019 coverage change the answer?

The 2013-2019 epochs carry Landsat 8 only, so fewer pixels reach three clear
looks and fewer patches are measurable. That is a coverage difference, not a
measurement bias (patch and control are always read from the same composite),
but it could still shift which patches populate which age bins. This step
refits the pooled recovery separately on the two epoch eras and compares the
patch populations they sample.

Output: outputs/epoch_era.json
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
TMAX = 22.0
NBOOT = 200
rng = np.random.default_rng(31)

L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= TMAX)
         & (L.src == "event")].copy()
if "pid" not in post.columns:
    post["pid"] = post.groupby(["event", "year", "elev", "area_ha"]).ngroup()
post["eyear"] = post.epoch.str[1:].astype(int) + 2000


def fit_tau(d, val="dlst", tmax=TMAX):
    edges = np.arange(0.6, tmax + 1, 1.0)
    c, m, se = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        g = d[(d.age >= a) & (d.age < b)][val].dropna()
        if len(g) >= 10:
            c.append(0.5 * (a + b)); m.append(g.mean())
            se.append(g.std() / np.sqrt(len(g)))
    if len(c) < 5:
        return None
    c, m, se = np.array(c), np.array(m), np.array(se)
    try:
        popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), c, m,
                            p0=[m[0], 8.0], sigma=np.maximum(se, 1e-3),
                            absolute_sigma=True, maxfev=20000,
                            bounds=([-25, 0.3], [25, 60]))
        return float(popt[0]), float(popt[1]), len(c)
    except Exception:
        return None


def boot_tau(d, val="dlst", tmax=TMAX):
    pids = d.pid.unique()
    ix = {p: g.index.values for p, g in d.groupby("pid")}
    out = []
    for _ in range(NBOOT):
        pick = rng.choice(pids, size=len(pids), replace=True)
        r = fit_tau(d.loc[np.concatenate([ix[p] for p in pick])], val, tmax)
        if r:
            out.append(r[1])
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) \
        if len(out) > 20 else (np.nan, np.nan)


early = post[post.eyear <= 2019]
late = post[post.eyear >= 2020]
# The two eras do not observe the same ages: an event can only be seen at a
# young age in an early epoch, so the early era tops out at ~15 yr while the
# late era reaches 22 yr. An exponential fitted over a younger window returns a
# shorter tau whenever the true decay is not a single exponential, so the eras
# must be compared over the age range they share, not over their own ranges.
AGE_LO = max(early.age.min(), late.age.min())
AGE_HI = min(early.age.max(), late.age.max())
print(f"common age window: {AGE_LO:.1f}-{AGE_HI:.1f} yr", flush=True)
R = {"n_boot": NBOOT, "common_age_window": [float(AGE_LO), float(AGE_HI)]}
sets = [("all", post, False), ("early_2013_2019", early, False),
        ("late_2020_2026", late, False),
        ("all_common", post, True), ("early_common", early, True),
        ("late_common", late, True)]
for name, d0, common in sets:
    d = d0[(d0.age >= AGE_LO) & (d0.age <= AGE_HI)] if common else d0
    if len(d) < 300:
        print(name, "too few rows", len(d), flush=True)
        continue
    ent = dict(n_rows=int(len(d)), n_patches=int(d.pid.nunique()),
               elev_med=float(d.elev.median()), area_med=float(d.area_ha.median()),
               age_span=[float(d.age.min()), float(d.age.max())], common=common)
    for val, key in [("dlst", "thermal"), ("dndvi", "greenness")]:
        r = fit_tau(d, val, tmax=AGE_HI if common else TMAX)
        if r:
            lo, hi = boot_tau(d, val, tmax=AGE_HI if common else TMAX)
            ent[key] = dict(A=r[0], tau=r[1], n_bins=r[2], tau_lo=lo, tau_hi=hi)
    R[name] = ent
    print(name, "rows", ent["n_rows"],
          {k: round(ent[k]["tau"], 2) for k in ("thermal", "greenness") if k in ent},
          flush=True)

# --- composition-free reference: one event, one date, one trigger ---
# Morakot (Aug 2009) is now observed continuously from about age 4 to age 17, so
# an exponential fitted to this cohort alone contains no between-cohort
# composition at all. It is the cleanest single-event recovery curve available
# and the natural yardstick for the pooled fit.
mk = post[post.event.fillna("").str.contains("莫拉克")]
if len(mk) > 1000:
    ent = dict(n_rows=int(len(mk)), n_patches=int(mk.pid.nunique()),
               elev_med=float(mk.elev.median()), area_med=float(mk.area_ha.median()),
               age_span=[float(mk.age.min()), float(mk.age.max())])
    for val, key in [("dlst", "thermal"), ("dndvi", "greenness")]:
        r = fit_tau(mk, val)
        if r:
            lo, hi = boot_tau(mk, val)
            ent[key] = dict(A=r[0], tau=r[1], n_bins=r[2], tau_lo=lo, tau_hi=hi)
    R["morakot_only"] = ent
    print("morakot_only rows", ent["n_rows"],
          {k: round(ent[k]["tau"], 2) for k in ("thermal", "greenness") if k in ent},
          flush=True)

# --- which cohorts populate the long-age bins in each era? ---
# this is the mechanism behind any era difference: at a given age the two eras
# necessarily draw on different event years
occ = {}
for nm, d in [("early_2013_2019", early), ("late_2020_2026", late)]:
    tail = d[d.age >= 10]
    occ[nm] = {str(int(y)): int(n) for y, n in
               tail.year.value_counts().sort_index().items() if n >= 50}
R["tail_cohort_mix"] = occ
print("tail cohort mix:", json.dumps(occ), flush=True)

json.dump(R, open(f"{O}/epoch_era.json", "w"), indent=1)
print("STEP22 COMPLETE")
