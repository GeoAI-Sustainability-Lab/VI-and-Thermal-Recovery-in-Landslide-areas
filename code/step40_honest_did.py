# -*- coding: utf-8 -*-
"""Step 40: honest confidence sets for the first post-event effect (Rambachan and Roth, 2023).

The event study of step20 is normalised at the last pre-event summer (k = -1). The first
post-event summer is k = +1, so two annual steps separate the reference period from the
first estimate (the event falls between them and no k = 0 composite exists). Under the
relative-magnitude restriction Delta^RM(Mbar) the post-event departure from parallel
trends over one step may be at most Mbar times the largest step among the pre-event
coefficients. Applied to the two-step interval, the departure may accumulate over both
steps, which is the restriction with Mbar = 2 in one-step notation; Mbar = 1 would allow
only one step's worth of departure and is reported for comparison only.

The conditional least-favourable hybrid confidence sets are computed with the honestdid
Python implementation of the R package HonestDiD, on the joint bootstrap covariance of
the six pre-event and two post-event coefficients saved by step20. A self-check compares
the implementation's identified set at a near-zero covariance with the analytical set
beta_1 -/+ Mbar * (largest pre-event step), so the result files carry their own
verification of the moment construction.

Output: outputs/honest_did.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import honestdid as hd

O = f"{_TROOT}/outputs"
ES = json.load(open(f"{O}/eventstudy.json"))
ALPHA = 0.05
MBARS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
HYBRID = "C-LF"                    # conditional least-favourable hybrid, the R package default
GRID_POINTS = 120                  # theta grid of the test inversion; resolution is reported


def _rm_cs(beta, sigma, n_pre, n_post, l_vec, mbar, max_step, method=HYBRID):
    """robust confidence set for the first post-event effect under Delta^RM(mbar)."""
    se1 = float(np.sqrt(sigma[n_pre, n_pre]))
    span = (max(MBARS) + 1.0) * max_step + 6.0 * se1
    lb, ub = float(beta[n_pre] - span), float(beta[n_pre] + span)
    res = hd.createSensitivityResults_relativeMagnitudes(
        betahat=beta, sigma=sigma, numPrePeriods=n_pre, numPostPeriods=n_post,
        l_vec=l_vec, Mbarvec=[float(mbar)], alpha=ALPHA, method=method,
        gridPoints=GRID_POINTS, grid_lb=lb, grid_ub=ub)
    lo, hi = _lb_ub(res)
    return lo, hi, (ub - lb) / (GRID_POINTS - 1)


def _lb_ub(res):
    """lower/upper bound from the package's return value (DataFrame-like or dict)."""
    try:
        return float(res["lb"].iloc[0]), float(res["ub"].iloc[0])
    except Exception:
        return float(np.asarray(res["lb"]).ravel()[0]), float(np.asarray(res["ub"]).ravel()[0])


def analyse(band):
    P = ES[f"{band}_pretrend"]
    periods = P["es_periods"]
    beta = np.array(P["es_betahat"], float)
    sigma = np.array(P["es_sigma"], float)
    n_pre = sum(1 for k in periods if k < 0)
    n_post = len(periods) - n_pre
    assert periods[:n_pre] == sorted(periods[:n_pre]) and periods[n_pre] == 1
    l_vec = hd.basis_vector(index=1, size=n_post)
    # largest step among the pre-event coefficients, reference summer (zero) included
    pre = np.append(beta[:n_pre], 0.0)
    max_step = float(np.abs(np.diff(pre)).max())
    out = dict(periods=periods, betahat=beta.tolist(), se=np.sqrt(np.diag(sigma)).tolist(),
               n_pre=n_pre, n_post=n_post, max_pre_step=max_step,
               steps_reference_to_first_post=2, alpha=ALPHA, hybrid=HYBRID)
    # original confidence set (parallel trends assumed to hold exactly)
    orig = hd.constructOriginalCS(betahat=beta, sigma=sigma, numPrePeriods=n_pre,
                                  numPostPeriods=n_post, l_vec=l_vec, alpha=ALPHA)
    out["original_cs"] = list(_lb_ub(orig))
    # robust confidence sets and identified sets over Mbar
    rows = []
    for mb in MBARS:
        lo, hi, step = _rm_cs(beta, sigma, n_pre, n_post, l_vec, mb, max_step)
        rows.append(dict(Mbar=mb, id_set=[float(beta[n_pre] - mb * max_step), float(beta[n_pre] + mb * max_step)],
                         robust_cs=[lo, hi], excludes_zero=bool(lo > 0 or hi < 0)))
        print(f"   {band} Mbar {mb:.1f}: robust CS [{lo:.3f}, {hi:.3f}]", flush=True)
    out["by_Mbar"] = rows
    out["theta_grid_resolution"] = step
    # the conditional set without the hybrid, for comparison at the two-step value
    clo2, chi2, _ = _rm_cs(beta, sigma, n_pre, n_post, l_vec, 2.0, max_step, method="Conditional")
    out["conditional_cs_Mbar2"] = [clo2, chi2]
    # breakdown value: the largest Mbar at which the robust set still excludes zero,
    # found by bisection on [0, 6] to a tolerance of 0.05
    def excludes(mb):
        lo, hi, _ = _rm_cs(beta, sigma, n_pre, n_post, l_vec, mb, max_step)
        return lo > 0 or hi < 0
    a, b = 0.0, 6.0
    if not excludes(b):
        while b - a > 0.05:
            m = 0.5 * (a + b)
            if excludes(m):
                a = m
            else:
                b = m
        breakdown = a
    else:
        breakdown = b
    out["breakdown_Mbar"] = float(breakdown)
    out["breakdown_tolerance"] = 0.05
    # self-check of the moment construction: with a near-zero covariance the robust set
    # must collapse onto the analytical identified set (up to the theta grid resolution)
    tiny = np.eye(len(beta)) * 1e-10
    clo, chi, step_chk = _rm_cs(beta, tiny, n_pre, n_post, l_vec, 2.0, max_step)
    ana = [beta[n_pre] - 2.0 * max_step, beta[n_pre] + 2.0 * max_step]
    out["self_check"] = dict(Mbar=2.0, near_zero_variance_cs=[clo, chi], analytical_id_set=[float(a) for a in ana],
                             max_abs_diff=float(max(abs(clo - ana[0]), abs(chi - ana[1]))),
                             theta_grid_resolution=step_chk)
    return out


R = {"method": "Rambachan and Roth (2023) relative-magnitude restriction, conditional least-favourable hybrid",
     "software": "honestdid 0.1.1 (Python implementation of HonestDiD), scalar conversions patched for NumPy 2.4",
     "Mbar_values": MBARS, "grid_points": GRID_POINTS}
for band in ("lst", "ndvi"):
    R[band] = analyse(band)
    b = R[band]
    print(band, "original CS", np.round(b["original_cs"], 3), "max pre step", round(b["max_pre_step"], 3))
    for r in b["by_Mbar"]:
        print(f"   Mbar {r['Mbar']:.1f}: id set {np.round(r['id_set'], 3)}  robust CS {np.round(r['robust_cs'], 3)}"
              f"  excludes 0: {r['excludes_zero']}")
    print("   breakdown Mbar:", b["breakdown_Mbar"], " self-check max diff:", f"{b['self_check']['max_abs_diff']:.2e}")
json.dump(R, open(f"{O}/honest_did.json", "w"), indent=1)
print("STEP40 COMPLETE")
