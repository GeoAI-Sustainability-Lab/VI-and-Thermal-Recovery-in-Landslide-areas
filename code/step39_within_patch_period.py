# -*- coding: utf-8 -*-
"""Sensitivity of the within-patch coefficient to calendar-period effects.

The main within-patch model regresses Delta on the pooled fitted shape g(age) with patch fixed
effects, so beta/A is the share of the pooled decline reproduced inside patches. Age equals
observation year minus event year, so with patch effects (which absorb the event year) a set of
calendar-epoch effects is nearly collinear with age, and beta is then identified only through
the curvature of g. This step adds epoch fixed effects (two-way within transformation, iterated)
and reports beta/A, so the reader can see how much of the identification rests on the exclusion
of period effects. Output: outputs/within_patch_period.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

O = f"{_TROOT}/outputs"
TMAX = 22.0; MIN_EP = 4; MIN_SPAN = 4.0
RC = json.load(open(f"{O}/recovery_clocks.json"))
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= TMAX) & (L.src == "event")].copy()


def prep(val):
    d = post[post[val].notna()][["pid", "age", "epoch", val]].rename(columns={val: "y"})
    g = d.groupby("pid").age
    keep = (g.count() >= MIN_EP) & ((g.max() - g.min()) >= MIN_SPAN)
    return d[d.pid.isin(keep[keep].index)].reset_index(drop=True)


def demean_two_way(x, inv_p, inv_t, n_p, n_t, iters=60):
    """Alternating projection onto the two fixed-effect sets (converges for balanced-ish panels)."""
    x = x.astype(float).copy()
    for _ in range(iters):
        x -= (np.bincount(inv_p, weights=x, minlength=n_p) / np.bincount(inv_p, minlength=n_p))[inv_p]
        x -= (np.bincount(inv_t, weights=x, minlength=n_t) / np.bincount(inv_t, minlength=n_t))[inv_t]
    return x


R = {}
for val, name in (("dlst", "thermal"), ("dndvi", "greenness")):
    d = prep(val)
    tau_p, A_p = RC[name]["tau"], RC[name]["A"]
    pid_u, inv_p = np.unique(d.pid.values, return_inverse=True)
    ep_u, inv_t = np.unique(d.epoch.values, return_inverse=True)
    n_p, n_t = len(pid_u), len(ep_u)
    y, g = d.y.values, np.exp(-d.age.values / tau_p)
    # (1) patch effects only, as in the main text
    yw = y - (np.bincount(inv_p, weights=y, minlength=n_p) / np.bincount(inv_p, minlength=n_p))[inv_p]
    gw = g - (np.bincount(inv_p, weights=g, minlength=n_p) / np.bincount(inv_p, minlength=n_p))[inv_p]
    beta1 = float(yw @ gw / (gw @ gw))
    # (2) patch and epoch effects
    y2, g2 = demean_two_way(y, inv_p, inv_t, n_p, n_t), demean_two_way(g, inv_p, inv_t, n_p, n_t)
    beta2 = float(y2 @ g2 / (g2 @ g2))
    # how much of g survives the two-way projection: the identifying variation
    share_var = float((g2 @ g2) / (gw @ gw))
    R[name] = dict(n_rows=int(len(d)), n_patches=n_p, n_epochs=n_t, A_pooled=A_p, tau_pooled=tau_p,
                   beta_patch_fe=beta1, frac_patch_fe=beta1 / A_p,
                   beta_patch_epoch_fe=beta2, frac_patch_epoch_fe=beta2 / A_p,
                   g_variation_left_after_epoch_fe=share_var)
    print(f"{name:10s} beta patch-FE {beta1:+.3f} (frac {beta1/A_p:.2f}) | patch+epoch FE {beta2:+.3f} "
          f"(frac {beta2/A_p:.2f}) | share of within-patch variation of g left after epoch effects {share_var:.2f}")
json.dump(R, open(f"{O}/within_patch_period.json", "w"), indent=1)
