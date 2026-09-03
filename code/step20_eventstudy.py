"""Step 20: event-study (leads and lags) around the disturbance date.

Now that the epoch stack reaches back to 2013, most cohorts have several
summers of pre-event observation, so the parallel-trends assumption that the
difference-in-differences estimate rests on can be tested rather than asserted.

For every clean event patch with a defined reference summer (the last full
summer before the event) we take the patch-minus-control anomaly at every
epoch and subtract the anomaly in the reference summer:

    beta_{i,k} = dX_{i,k} - dX_{i,ref},   k = round(epoch_mid - t_event)

k < 0 are leads (should be flat and near zero if patches and their controls
were on parallel paths), k >= 1 are lags (the treatment path). Uncertainty is
a patch-level block bootstrap because one patch supplies many epochs.

Output: outputs/eventstudy.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
NBOOT = 400
KMIN, KMAX = -7, 13
NMIN = 25
rng = np.random.default_rng(23)

EPOCH_MID = {"c13": 2013.62, "c14": 2014.62, "c15": 2015.62, "c16": 2016.62,
             "c17": 2017.62, "c18": 2018.62, "c19": 2019.62, "c20": 2020.62,
             "c21": 2021.62, "c22": 2022.62, "c23": 2023.62, "c24": 2024.62,
             "c25": 2025.62, "c26": 2026.55}          # c2425 excluded: 2-yr stack

p = pd.read_parquet(f"{D}/patches_deltas2.parquet")
p = p[(p.src == "event") & (p.redist_frac < 0.05) & (p.n_ctrl >= 30)
      & (p.forest2000_frac > 0.6) & (p.pre_tag != "")].copy()
print("clean event patches with a reference summer:", len(p), flush=True)

recs = []
for band in ("lst", "ndvi"):
    # the reference value is the same for every epoch, so build it once:
    # pick d{band}_{pre_tag} row-wise with a vectorised lookup
    ref = pd.Series(np.nan, index=p.index)
    for tg in p.pre_tag.unique():
        c = f"d{band}_{tg}"
        if c in p.columns:
            m = p.pre_tag.values == tg
            ref[m] = p[c].values[m]
    for tag, mid in EPOCH_MID.items():
        col = f"d{band}_{tag}"
        if col not in p.columns:
            continue
        k = np.round(mid - p.t_event).astype(float)
        ok = p[col].notna() & ref.notna() & (k >= KMIN) & (k <= KMAX)
        if not ok.any():
            continue
        recs.append(pd.DataFrame(dict(
            pid=p.index.values[ok.values], band=band, k=k[ok.values].astype(int),
            beta=(p[col] - ref)[ok.values].values,
            agent=p.agent.values[ok.values], year=p.year.values[ok.values])))
E = pd.concat(recs, ignore_index=True)
# the reference summer itself is identically zero and carries no information
E = E[E.k != -1]
print("event-study rows:", len(E), " patches:", E.pid.nunique(), flush=True)


def _patch_tables(d, ks):
    """per-patch sum and count of beta for each k -> O(n_patch) bootstrap."""
    pid_u, inv = np.unique(d.pid.values, return_inverse=True)
    S, N = {}, {}
    for k in ks:
        m = d.k.values == k
        S[k] = np.bincount(inv[m], weights=d.beta.values[m], minlength=len(pid_u))
        N[k] = np.bincount(inv[m], minlength=len(pid_u)).astype(float)
    return pid_u, S, N


def curve(d, nmin=None):
    """mean beta by k with a patch-level block bootstrap CI.

    Resampling patches with replacement is equivalent to drawing a multinomial
    weight vector over patches, so each replicate is two dot products per k
    rather than a rebuild of the table.
    """
    ks = sorted(k for k, g in d.groupby("k") if len(g) >= (nmin or NMIN))
    if not ks:
        return []
    pid_u, S, N = _patch_tables(d, ks)
    npat = len(pid_u)
    Wm = rng.multinomial(npat, np.full(npat, 1.0 / npat), size=NBOOT).astype(float)
    out = []
    for k in ks:
        num = Wm @ S[k]
        den = Wm @ N[k]
        bb = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        bb = bb[np.isfinite(bb)]
        g = d[d.k == k]
        out.append(dict(k=int(k), n=int(len(g)), n_patch=int(g.pid.nunique()),
                        beta=float(g.beta.mean()),
                        lo=float(np.percentile(bb, 2.5)),
                        hi=float(np.percentile(bb, 97.5)),
                        p_gt0=float((bb > 0).mean())))
    return out


def pooled_boot(d):
    """bootstrap the mean of beta over a subset, resampling patches."""
    pid_u, inv = np.unique(d.pid.values, return_inverse=True)
    S = np.bincount(inv, weights=d.beta.values, minlength=len(pid_u))
    N = np.bincount(inv, minlength=len(pid_u)).astype(float)
    npat = len(pid_u)
    Wm = rng.multinomial(npat, np.full(npat, 1.0 / npat), size=NBOOT).astype(float)
    bb = (Wm @ S) / np.maximum(Wm @ N, 1e-12)
    return bb


R = {"n_boot": NBOOT, "k_ref": -1, "n_min_per_k": NMIN}
for band in ("lst", "ndvi"):
    d = E[E.band == band]
    R[band] = curve(d)
    leads = [r for r in R[band] if r["k"] <= -2]
    lags = [r for r in R[band] if r["k"] >= 1]
    # parallel-trends summary: pooled lead mean and the largest single lead
    dl = d[d.k <= -2]
    if len(dl):
        bb = pooled_boot(dl)
        R[f"{band}_lead_pooled"] = dict(
            n=int(len(dl)), n_patch=int(dl.pid.nunique()),
            mean=float(dl.beta.mean()), lo=float(np.percentile(bb, 2.5)),
            hi=float(np.percentile(bb, 97.5)),
            max_abs_lead=float(max((abs(r["beta"]) for r in leads), default=np.nan)),
            n_leads=len(leads))
    R[f"{band}_n_lags"] = len(lags)
    print(band, "leads:", [(r["k"], round(r["beta"], 3)) for r in R[band] if r["k"] <= -2],
          flush=True)
    print(band, "lags:", [(r["k"], round(r["beta"], 3)) for r in R[band] if r["k"] >= 1],
          flush=True)

# by agent, thermal only (identification check: does any single trigger drive it?)
R["lst_by_agent"] = {}
for ag, g in E[E.band == "lst"].groupby("agent"):
    if g.pid.nunique() >= 150:
        R["lst_by_agent"][ag] = curve(g)
        print("agent", ag, "ks:", [r["k"] for r in R["lst_by_agent"][ag]], flush=True)

# by cohort year, thermal only: each cohort's jump should land on its own event
# year. This is the cohort-level counterpart of the pooled test, feasible for
# cohorts with >= 20 reference-summer patches (display threshold; the
# pooled fit is the inferential object).
R["lst_by_year"] = {}
for yr, g in E[E.band == "lst"].groupby("year"):
    if g.pid.nunique() >= 20:          # readability cap handled at plot time
        # per-k floor 10 (not NMIN): this panel is a qualitative timing check
        # with bootstrap CIs shown; the pooled curve is the inferential object
        R["lst_by_year"][str(int(yr))] = curve(g, nmin=10)
        print("year", yr, "patches", g.pid.nunique(),
              "ks:", [r["k"] for r in R["lst_by_year"][str(int(yr))]], flush=True)

json.dump(R, open(f"{O}/eventstudy.json", "w"), indent=1)
print("STEP20 COMPLETE")
