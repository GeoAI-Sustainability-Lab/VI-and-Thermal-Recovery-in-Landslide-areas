"""Step 21: is the recovery curve longitudinal, or an artefact of which events
happened when?

With only the 2020-2026 epochs, patch age and event year were almost collinear:
"older scars are cooler" and "scars from older events are cooler" could not be
told apart. The 2013-2026 stack breaks that collinearity, so the question can be
put directly to the data with patch fixed effects.

Primary test (scale test). Take the SHAPE of the pooled fit, g(t) = exp(-t/tau),
tau from step 16, and fit

    dX_{i,t} = alpha_i + beta * g(t_{i,t}) + e

Alpha_i absorbs everything time-invariant about a patch (its size, its aspect,
its trigger, its event year), so beta is identified purely from how a patch
changes as it ages. Under the pooled model beta equals the pooled amplitude A;
if the pooled decline were entirely composition, beta would be 0. The reported
statistic is the recovered fraction beta / A. Being linear in beta, the within
estimator has no incidental-parameter problem.

Secondary (descriptive). A within-patch linear slope in age, compared with the
average slope of the pooled curve over the same age distribution.

Tertiary (reported with a caveat). Concentrated nonlinear least squares for tau
itself with a free per-patch amplitude. Each patch spans a limited age range, so
the residual sum of squares is very flat in tau above ~30 yr and this estimator
is biased towards long tau whenever the per-observation noise is large relative
to the within-patch decline; it is reported for completeness, not as the headline.

Output: outputs/within_patch.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

O = f"{_TROOT}/outputs"
NBOOT = 400
TMAX = 22.0
MIN_EP = 4          # epochs per patch
MIN_SPAN = 4.0      # yr between the youngest and oldest age of a patch
rng = np.random.default_rng(29)

RC = json.load(open(f"{O}/recovery_clocks.json"))
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= TMAX)
         & (L.src == "event")].copy()
if "pid" not in post.columns:
    post["pid"] = post.groupby(["event", "year", "elev", "area_ha"]).ngroup()


def prep(val):
    d = post[post[val].notna()][["pid", "age", val]].rename(columns={val: "y"})
    g = d.groupby("pid").age
    keep = (g.count() >= MIN_EP) & ((g.max() - g.min()) >= MIN_SPAN)
    return d[d.pid.isin(keep[keep].index)].reset_index(drop=True)


def within(x, inv, npat):
    """subtract each patch's own mean (the within transformation)."""
    m = np.bincount(inv, weights=x, minlength=npat) / np.bincount(inv, minlength=npat)
    return x - m[inv]


def fe_beta(y, g, inv, npat):
    yw, gw = within(y, inv, npat), within(g, inv, npat)
    den = float(gw @ gw)
    return float(yw @ gw / den) if den > 0 else np.nan


def tau_fe_arr(age, y, inv, grid):
    npat = int(inv.max()) + 1
    best, best_ssr = np.nan, np.inf
    for tau in grid:
        w = np.exp(-age / tau)
        num = np.bincount(inv, weights=y * w, minlength=npat)
        den = np.bincount(inv, weights=w * w, minlength=npat)
        A = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
        r = y - A[inv] * w
        ssr = float(r @ r)
        if ssr < best_ssr:
            best_ssr, best = ssr, float(tau)
    return best


R = {"min_epochs": MIN_EP, "min_span_yr": MIN_SPAN, "n_boot": NBOOT}
for val, name in [("dlst", "thermal"), ("dndvi", "greenness")]:
    d = prep(val)
    if len(d) < 200:
        print(name, "insufficient", len(d), flush=True)
        continue
    tau_p, A_p = RC[name]["tau"], RC[name]["A"]
    pid_u, inv = np.unique(d.pid.values, return_inverse=True)
    npat = len(pid_u)
    age, y = d.age.values, d.y.values
    g = np.exp(-age / tau_p)
    beta = fe_beta(y, g, inv, npat)
    # within-patch linear slope in age, and the pooled curve's mean slope
    slope = fe_beta(y, age, inv, npat)
    pooled_slope = float(np.mean(-A_p / tau_p * np.exp(-age / tau_p)))
    tau_fe = tau_fe_arr(age, y, inv, np.arange(1.0, 60.01, 0.25))

    rows_by_pid = [np.flatnonzero(inv == j) for j in range(npat)]
    lens = np.array([len(a) for a in rows_by_pid])
    bb, bs, bt = [], [], []
    for b in range(NBOOT):
        pick = rng.integers(0, npat, size=npat)
        rows = np.concatenate([rows_by_pid[j] for j in pick])
        iv = np.repeat(np.arange(npat), lens[pick])
        bb.append(fe_beta(y[rows], g[rows], iv, npat))
        bs.append(fe_beta(y[rows], age[rows], iv, npat))
        if b < 150:                        # tau grid search is the slow part
            bt.append(tau_fe_arr(age[rows], y[rows], iv, np.arange(1.0, 60.01, 0.5)))
        if (b + 1) % 100 == 0:
            print(f"  {name} boot {b+1}/{NBOOT}", flush=True)
    bb, bs, bt = np.array(bb), np.array(bs), np.array(bt)
    R[name] = dict(
        n_rows=int(len(d)), n_patches=int(npat),
        epochs_per_patch=float(len(d) / npat),
        age_span=[float(age.min()), float(age.max())],
        tau_pooled=tau_p, A_pooled=A_p,
        beta=beta, beta_lo=float(np.percentile(bb, 2.5)),
        beta_hi=float(np.percentile(bb, 97.5)),
        frac=float(beta / A_p), frac_lo=float(np.percentile(bb, 2.5) / A_p),
        frac_hi=float(np.percentile(bb, 97.5) / A_p),
        p_beta_gt0=float((bb > 0).mean() if A_p > 0 else (bb < 0).mean()),
        slope=slope, slope_lo=float(np.percentile(bs, 2.5)),
        slope_hi=float(np.percentile(bs, 97.5)), pooled_slope=pooled_slope,
        tau_fe=tau_fe, tau_fe_lo=float(np.percentile(bt, 2.5)),
        tau_fe_hi=float(np.percentile(bt, 97.5)), tau_fe_nboot=len(bt))
    print(f"{name}: beta={beta:.3f} [{R[name]['beta_lo']:.3f},{R[name]['beta_hi']:.3f}] "
          f"A_pooled={A_p:.3f} frac={R[name]['frac']:.2f} "
          f"within-slope={slope:.4f}/yr (pooled {pooled_slope:.4f}) "
          f"tau_FE={tau_fe:.1f}  n={len(d)} patches={npat}", flush=True)

json.dump(R, open(f"{O}/within_patch.json", "w"), indent=1)
print("STEP21 COMPLETE")
