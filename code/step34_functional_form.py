# -*- coding: utf-8 -*-
"""Functional-form comparison for the recovery trajectory.

The paper fits a single exponential relaxation, Delta(t) = A exp(-t/tau), to the 1-year
binned means. A reviewer will reasonably ask why that form and not a polynomial, a power
law or a curve that relaxes to a non-zero level. This step refits the same binned data
with a set of candidate forms under identical weights and reports weighted R-squared,
reduced chi-square and AICc, plus the fitted offset of the exponential-with-offset model,
which is the form that would matter if recovery were incomplete.

Output: outputs/functional_form.json
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
    return (np.array(bc), np.array(bm), np.maximum(np.array(bs), 1e-3), np.array(bn))


# candidate forms. p0 and bounds are written for a positive-amplitude decay and are
# sign-flipped automatically for the greenness series, which starts negative.
MODELS = {
    "exponential":        (lambda t, A, tau: A * np.exp(-t / tau), 2),
    "exponential+offset": (lambda t, A, tau, c: A * np.exp(-t / tau) + c, 3),
    "two-term exponential": (lambda t, A1, tau1, A2, tau2:
                             A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2), 4),
    "power law":          (lambda t, A, b: A * np.power(t, -b), 2),
    "linear":             (lambda t, a, b: a + b * t, 2),
    "quadratic":          (lambda t, a, b, c: a + b * t + c * t * t, 3),
    "logarithmic":        (lambda t, a, b: a + b * np.log(t), 2),
}


def start(name, y0):
    s = np.sign(y0) if y0 != 0 else 1.0
    return {
        "exponential":          ([y0, 6.0], ([-25, 0.3], [25, 60])),
        "exponential+offset":   ([y0, 6.0, 0.0], ([-25, 0.3, -10], [25, 60, 10])),
        "two-term exponential": ([y0 * 0.5, 2.0, y0 * 0.5, 20.0],
                                 ([-25, 0.2, -25, 0.2], [25, 60, 25, 200])),
        "power law":            ([y0, 0.3], ([-25, -2], [25, 5])),
        "linear":               ([y0, -0.1 * s], ([-25, -25], [25, 25])),
        "quadratic":            ([y0, -0.1 * s, 0.0], ([-25, -25, -25], [25, 25, 25])),
        "logarithmic":          ([y0, -0.3 * s], ([-25, -25], [25, 25])),
    }[name]


def compare(bc, bm, bs):
    """Weighted fits of every candidate on identical data and weights."""
    out, n = {}, len(bc)
    wmean = np.sum(bm / bs ** 2) / np.sum(1 / bs ** 2)
    sst = float(np.sum(((bm - wmean) / bs) ** 2))          # weighted total sum of squares
    for name, (fn, k) in MODELS.items():
        p0, bd = start(name, float(bm[0]))
        try:
            popt, pcov = curve_fit(fn, bc, bm, p0=p0, sigma=bs, absolute_sigma=True,
                                   bounds=bd, maxfev=200000)
        except Exception as e:
            out[name] = {"error": repr(e)[:120]}
            continue
        resid = (bm - fn(bc, *popt)) / bs
        chi2 = float(np.sum(resid ** 2))
        # Gaussian log-likelihood with known sigma: -2 lnL = chi2 + const, const shared
        # across models on the same data, so AIC differences are exact.
        aic = chi2 + 2 * k
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        out[name] = {"k": k, "params": [float(v) for v in popt],
                     "param_se": [float(v) for v in np.sqrt(np.diag(pcov))],
                     "chi2": chi2, "chi2_red": chi2 / max(n - k, 1),
                     "r2_w": 1.0 - chi2 / sst, "aicc": aicc,
                     "max_abs_resid_sigma": float(np.max(np.abs(resid)))}
    best = min((v["aicc"] for v in out.values() if "aicc" in v), default=None)
    for v in out.values():
        if "aicc" in v:
            v["d_aicc"] = v["aicc"] - best
    # The bin standard errors treat patch x epoch rows as independent, although one patch
    # contributes to many bins, so chi-square is inflated. Following Burnham and Anderson
    # (2002), a variance inflation factor c-hat is taken from the most parameterised model
    # and the quasi-likelihood QAICc is reported, with one extra parameter for c-hat.
    glob = out.get("two-term exponential", {})
    chat = max(1.0, glob.get("chi2_red", 1.0)) if "chi2_red" in glob else 1.0
    for v in out.values():
        if "chi2" in v:
            kq = v["k"] + 1
            v["qaicc"] = v["chi2"] / chat + 2 * kq + (2 * kq * (kq + 1)) / max(n - kq - 1, 1)
    qbest = min((v["qaicc"] for v in out.values() if "qaicc" in v), default=None)
    for v in out.values():
        if "qaicc" in v:
            v["d_qaicc"] = v["qaicc"] - qbest
    eo = out.get("exponential+offset", {})
    if "params" in eo:
        c, se = eo["params"][2], eo["param_se"][2]
        # naive standard error scaled by sqrt(c-hat), the same inflation applied to the AICc
        eo["offset"] = c
        eo["offset_halfwidth95_scaled"] = 1.96 * se * np.sqrt(chat)
        eo["offset_nonzero"] = bool(abs(c) > 1.96 * se * np.sqrt(chat))
    return {"n_bins": n, "sst_w": sst, "c_hat": chat, "models": out}


RES = {}
ev = post[post.src == "event"]
han = post[(post.src == "hansen") & post.year.between(2015, 2023)]
CASES = {
    "landslide_dlst": (ev, "dlst"),
    "landslide_dndvi": (ev, "dndvi"),
    "annual_loss_dlst": (han, "dlst"),
}
for tag, (d, val) in CASES.items():
    bc, bm, bs, bn = binned(d, val)
    RES[tag] = compare(bc, bm, bs)
    RES[tag]["bins"] = {"center": bc.tolist(), "mean": bm.tolist(),
                        "se": bs.tolist(), "n": bn.tolist()}

json.dump(RES, open(f"{O}/functional_form.json", "w"), indent=1)

for tag, r in RES.items():
    print(f"\n=== {tag}  ({r['n_bins']} bins)")
    rows = sorted((v for v in r["models"].items() if "aicc" in v[1]),
                  key=lambda x: x[1]["aicc"])
    print(f"  c-hat = {r['c_hat']:.2f}")
    print(f"{'model':22s} {'k':>2s} {'R2_w':>7s} {'chi2/df':>8s} {'dAICc':>8s} {'dQAICc':>8s}")
    for name, v in rows:
        print(f"{name:22s} {v['k']:2d} {v['r2_w']:7.3f} {v['chi2_red']:8.2f} "
              f"{v['d_aicc']:8.1f} {v['d_qaicc']:8.1f}")
    eo = r["models"].get("exponential+offset", {})
    if "offset" in eo:
        print(f"  offset c = {eo['offset']:+.3f} +/- {eo['offset_halfwidth95_scaled']:.3f} "
              f"(95%, scaled by sqrt c-hat), "
              f"{'DIFFERENT FROM ZERO' if eo['offset_nonzero'] else 'consistent with zero'}")
