"""Step 16: pooled recovery clocks with patch-level uncertainty, for the new
the recovery-clock figure. Each patch contributes up to 8 epochs, so binned standard
errors treat 25,371 correlated rows as independent. Everything here is
resampled at the PATCH level (block bootstrap), which is the unit of
independence, and the normalised recovery curves are reported on a single axis
so that the thermal-versus-greenness comparison cannot be manufactured by the
choice of twin-axis scaling.
Output: outputs/recovery_clocks.json
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
NBOOT = 300
TMAX = 22.0
rng = np.random.default_rng(17)

L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= TMAX)]
ev = post[post.src == "event"].copy()
if "pid" not in ev.columns:            # fallback for pre-v11 long tables
    ev["pid"] = ev.groupby(["event", "year", "elev", "area_ha"]).ngroup()
print("rows:", len(ev), "patches:", ev.pid.nunique(), flush=True)


def binned(d, val, edges):
    c, m, se, n = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        g = d[(d.age >= a) & (d.age < b)][val].dropna()
        if len(g) >= 10:
            c.append(0.5 * (a + b)); m.append(float(g.mean()))
            se.append(float(g.std() / np.sqrt(len(g)))); n.append(int(len(g)))
    return np.array(c), np.array(m), np.array(se), n


def fit(d, val):
    c, m, se, _ = binned(d, val, np.arange(0.6, TMAX + 1, 1.0))
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
for val, key in [("dlst", "thermal"), ("dndvi", "greenness")]:
    c, m, se, n = binned(ev, val, np.arange(0.6, TMAX + 1, 1.0))
    A, tau = fit(ev, val)
    out[key] = dict(bin_center=c.tolist(), bin_mean=m.tolist(),
                    bin_se=se.tolist(), bin_n=n, A=A, tau=tau,
                    t50=tau * np.log(2), t90=tau * np.log(10))
    print(f"{key}: A={A:+.3f} tau={tau:.2f} t50={tau*np.log(2):.1f}", flush=True)

# ---- patch-level block bootstrap ----
pids = ev.pid.unique()
idx_by_pid = {p: g.index.values for p, g in ev.groupby("pid")}
bt = {"thermal": {"tau": [], "A": []}, "greenness": {"tau": [], "A": []},
      "ratio": []}
for b in range(NBOOT):
    samp = rng.choice(pids, len(pids), replace=True)
    d = ev.loc[np.concatenate([idx_by_pid[p] for p in samp])]
    ft, fg = fit(d, "dlst"), fit(d, "dndvi")
    if ft and fg:
        bt["thermal"]["A"].append(ft[0]); bt["thermal"]["tau"].append(ft[1])
        bt["greenness"]["A"].append(fg[0]); bt["greenness"]["tau"].append(fg[1])
        bt["ratio"].append(ft[1] / fg[1])
    if (b + 1) % 100 == 0:
        print("  boot", b + 1, flush=True)

tt = np.linspace(0.6, TMAX, 120)
for key in ["thermal", "greenness"]:
    taus = np.array(bt[key]["tau"]); As = np.array(bt[key]["A"])
    curves = np.array([np.exp(-tt / t) for t in taus])          # normalised
    out[key].update(
        tau_boot_mean=float(taus.mean()),
        tau_ci=[float(np.percentile(taus, 2.5)), float(np.percentile(taus, 97.5))],
        t50_ci=[float(np.percentile(taus * np.log(2), 2.5)),
                float(np.percentile(taus * np.log(2), 97.5))],
        norm_t=tt.tolist(),
        norm_lo=np.percentile(curves, 2.5, axis=0).tolist(),
        norm_hi=np.percentile(curves, 97.5, axis=0).tolist())
    print(f"{key}: tau {taus.mean():.2f} CI [{np.percentile(taus,2.5):.2f}, "
          f"{np.percentile(taus,97.5):.2f}]", flush=True)
r = np.array(bt["ratio"])
out["ratio"] = dict(mean=float(r.mean()),
                    ci=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                    p_gt1=float((r > 1).mean()), samples=r.tolist(),
                    n_boot=len(r))
print(f"ratio {r.mean():.3f} CI [{np.percentile(r,2.5):.3f}, "
      f"{np.percentile(r,97.5):.3f}]  P(>1)={float((r>1).mean()):.3f}")

# lag at 50% recovery, in years, with CI
lag = np.array(bt["thermal"]["tau"]) * np.log(2) - np.array(bt["greenness"]["tau"]) * np.log(2)
out["lag_t50"] = dict(mean=float(lag.mean()),
                      ci=[float(np.percentile(lag, 2.5)), float(np.percentile(lag, 97.5))])
print(f"lag at 50% recovery: {lag.mean():.2f} yr "
      f"CI [{np.percentile(lag,2.5):.2f}, {np.percentile(lag,97.5):.2f}]")
out["n_rows"] = int(len(ev)); out["n_patches"] = int(ev.pid.nunique())
json.dump(out, open(f"{O}/recovery_clocks.json", "w"), indent=1)
print("STEP16 COMPLETE")
