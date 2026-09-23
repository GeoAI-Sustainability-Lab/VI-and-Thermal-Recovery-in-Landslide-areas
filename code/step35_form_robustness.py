# -*- coding: utf-8 -*-
"""Is the paper's conclusion an artefact of the exponential form?

Two checks on the same binned data as step34:
 (1) the tau ratio (thermal / greenness) refitted under exponential-with-offset;
 (2) a model-free version of the ten-year indicator, one minus the observed binned
     mean near ten years divided by the observed binned mean in the first year,
     which uses no functional form at all. The two bins are nine years apart, so
     the exponential is also evaluated over that same span, 1 - exp(-9/tau), for a
     like-for-like comparison, next to the main-text value 1 - exp(-10/tau) that
     starts from the fitted initial anomaly at age zero.
Output: outputs/form_robustness.json
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


def binned(d, val, tmax=22, min_bin=12):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    bins = np.arange(0.6, tmax + 1, 1.0)
    bc, bm, bs, bn = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        g = d[(d.age >= a) & (d.age < b)][val]
        if len(g) >= min_bin:
            bc.append(0.5 * (a + b)); bm.append(g.mean())
            bs.append(g.std() / np.sqrt(len(g))); bn.append(len(g))
    return np.array(bc), np.array(bm), np.maximum(np.array(bs), 1e-3), np.array(bn)


def fit_exp(bc, bm, bs):
    p, c = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), bc, bm, p0=[bm[0], 6.0],
                     sigma=bs, absolute_sigma=True, bounds=([-25, .3], [25, 60]), maxfev=200000)
    return float(p[0]), float(p[1])


def fit_exp_off(bc, bm, bs):
    p, c = curve_fit(lambda t, A, tau, k: A * np.exp(-t / tau) + k, bc, bm,
                     p0=[bm[0], 6.0, 0.0], sigma=bs, absolute_sigma=True,
                     bounds=([-25, .3, -10], [25, 60, 10]), maxfev=200000)
    return float(p[0]), float(p[1]), float(p[2])


ev = post[post.src == "event"]
han = post[(post.src == "hansen") & post.year.between(2015, 2023)]
CLS = {
    "landslide_all":  ev,
    "ls_lt2":         ev[ev.area_ha < 2],
    "ls_2_10":        ev[(ev.area_ha >= 2) & (ev.area_ha < 10)],
    "ls_ge10":        ev[ev.area_ha >= 10],
    "annual_loss":    han,
    "al_ge2":         han[han.area_ha >= 2],
}
R = {"tau_ratio": {}, "model_free_10yr": {}, "classes": {}}

# ---- (1) tau ratio under both forms, pooled landslides ----
bcT, bmT, bsT, _ = binned(ev, "dlst")
bcN, bmN, bsN, _ = binned(ev, "dndvi")
aT, tT = fit_exp(bcT, bmT, bsT); aN, tN = fit_exp(bcN, bmN, bsN)
aT2, tT2, kT2 = fit_exp_off(bcT, bmT, bsT); aN2, tN2, kN2 = fit_exp_off(bcN, bmN, bsN)
R["tau_ratio"] = {
    "exponential": {"tau_thermal": tT, "tau_greenness": tN, "ratio": tT / tN},
    "exponential_offset": {"tau_thermal": tT2, "tau_greenness": tN2, "ratio": tT2 / tN2,
                           "offset_thermal": kT2, "offset_greenness": kN2},
}

# ---- (2) model-free ten-year indicator: observed bins only ----
def model_free(d, val="dlst"):
    bc, bm, bs, bn = binned(d, val)
    if len(bc) < 3:
        return None
    i1 = int(np.argmin(np.abs(bc - 1.1)))            # first observed bin
    j = np.argmin(np.abs(bc - 10.1))                 # bin nearest ten years
    if abs(bc[j] - 10.1) > 1.0:
        return None
    rem = bm[j] / bm[i1]
    # delta-method SE on the ratio of two independent bin means
    se = abs(rem) * np.sqrt((bs[j] / bm[j]) ** 2 + (bs[i1] / bm[i1]) ** 2)
    return {"age_ref": float(bc[i1]), "age_10": float(bc[j]),
            "mean_ref": float(bm[i1]), "mean_10": float(bm[j]),
            "remaining_frac": float(rem), "dissipated_pct": float(100 * (1 - rem)),
            "dissipated_pct_ci": [float(100 * (1 - rem - 1.96 * se)),
                                  float(100 * (1 - rem + 1.96 * se))],
            "n_ref": int(bn[i1]), "n_10": int(bn[j])}

CD = json.load(open(f"{O}/class_dissipation.json"))
for k, d in CLS.items():
    mf = model_free(d)
    bc, bm, bs, _ = binned(d, "dlst")
    a, t = fit_exp(bc, bm, bs)
    row = {"tau_exp": t, "A_exp": a,
           "dissipated_pct_exp": float(100 * (1 - np.exp(-10.0 / t))),
           "dissipated_pct_paper": CD[k]["dissipated"]["10"] if k in CD else None,
           "model_free": mf}
    if mf:
        # the exponential over the same span as the observed bins (first bin to the
        # bin nearest ten years), so both quantities start from the first observed year
        span = mf["age_10"] - mf["age_ref"]
        d_span = float(100 * (1 - np.exp(-span / t)))
        row.update(span_yr=float(span), dissipated_pct_exp_span=d_span,
                   exp_span_inside_ci=bool(mf["dissipated_pct_ci"][0] <= d_span <= mf["dissipated_pct_ci"][1]),
                   exp_span_minus_observed=float(d_span - mf["dissipated_pct"]))
    try:
        a2, t2, k2 = fit_exp_off(bc, bm, bs)
        row["tau_exp_offset"] = t2; row["offset"] = k2
    except Exception as e:
        row["tau_exp_offset"] = None
    R["classes"][k] = row

# ---- (3) sensitivity to the truncated 2026 summer (1 June to 10 August): the pooled
#          landslide clocks refitted with every c26 observation removed ----
ev26 = ev[ev.epoch != "c26"]
bcT6, bmT6, bsT6, _ = binned(ev26, "dlst")
bcN6, bmN6, bsN6, _ = binned(ev26, "dndvi")
aT6, tT6 = fit_exp(bcT6, bmT6, bsT6); aN6, tN6 = fit_exp(bcN6, bmN6, bsN6)
R["without_2026"] = {"n_rows_removed": int((ev.epoch == "c26").sum()), "n_rows_kept": int(len(ev26)),
                     "tau_thermal": tT6, "tau_greenness": tN6, "ratio": tT6 / tN6,
                     "dissipated_pct_exp_10": float(100 * (1 - np.exp(-10.0 / tT6)))}

json.dump(R, open(f"{O}/form_robustness.json", "w"), indent=1)

t = R["tau_ratio"]
w = R["without_2026"]
print(f"without the 2026 summer ({w['n_rows_removed']} rows removed): tau thermal {w['tau_thermal']:.2f}, "
      f"greenness {w['tau_greenness']:.2f}, ratio {w['ratio']:.3f}, 10-yr dissipated {w['dissipated_pct_exp_10']:.1f}%")
print("tau ratio, pooled landslides")
print(f"  exponential          thermal {t['exponential']['tau_thermal']:6.2f}  "
      f"greenness {t['exponential']['tau_greenness']:6.2f}  ratio {t['exponential']['ratio']:.3f}")
e = t["exponential_offset"]
print(f"  exponential + offset thermal {e['tau_thermal']:6.2f}  greenness "
      f"{e['tau_greenness']:6.2f}  ratio {e['ratio']:.3f}   "
      f"(offsets {e['offset_thermal']:+.2f} degC, {e['offset_greenness']:+.3f} NDVI)")
print(f"\n{'class':14s} {'tau_exp':>8s} {'tau_off':>8s} {'d10_exp':>8s} {'d9_exp':>8s} {'d10_paper':>10s} "
      f"{'d_modelfree':>12s} {'95% CI':>16s} {'inside':>7s}")
for k, v in R["classes"].items():
    mf = v["model_free"]
    s = (f"{mf['dissipated_pct']:12.1f} [{mf['dissipated_pct_ci'][0]:5.1f},"
         f"{mf['dissipated_pct_ci'][1]:5.1f}] {str(v['exp_span_inside_ci']):>7s}") if mf else f"{'n/a':>12s}{'':25s}"
    to = f"{v['tau_exp_offset']:8.2f}" if v['tau_exp_offset'] else f"{'n/a':>8s}"
    ds = f"{v['dissipated_pct_exp_span']:8.1f}" if mf else f"{'n/a':>8s}"
    print(f"{k:14s} {v['tau_exp']:8.2f} {to} {v['dissipated_pct_exp']:8.1f} {ds} "
          f"{v['dissipated_pct_paper']:10.1f} {s}")
