# -*- coding: utf-8 -*-
"""Does disturbance TYPE still matter once severity is held fixed?

Severity is measured per patch as the mean DNDVI over the first post-event
observations (ages 0.6-3 yr). Using greenness to define severity and temperature
to measure recovery keeps the two measurement noises independent, so the bins do
not create a spurious decline through regression to the mean.
Writes: outputs/severity_match.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

TMAX, NBOOT = 22.0, 200
EDGES = np.arange(0.6, TMAX + 1, 1.0)
rng = np.random.default_rng(23)

L = pd.read_parquet(f"{_TROOT}/outputs/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= TMAX)].copy()
post = post[post.src.isin(["event", "hansen"])]
post["cls"] = np.where(post.src == "event", "landslide", "annual_loss")

# age coverage per class
print("age coverage (post-event, clean):")
for c, g in post.groupby("cls"):
    print(f"  {c:12s} n={len(g):7d} patches={g.pid.nunique():6d} "
          f"age {g.age.min():.1f}-{g.age.max():.1f}, 95th pct {g.age.quantile(.95):.1f}")

# ---- per-patch severity from early greenness loss ----
early = post[(post.age >= 0.6) & (post.age < 3.0)]
sev = early.groupby("pid").agg(sev_ndvi=("dndvi", "mean"), cls=("cls", "first"),
                               area_ha=("area_ha", "first")).dropna(subset=["sev_ndvi"])
print(f"\npatches with an early-severity value: {len(sev):,}")
print(sev.groupby("cls").sev_ndvi.describe()[["count", "25%", "50%", "75%"]].round(3))

post = post.join(sev[["sev_ndvi"]], on="pid")
d = post[post.sev_ndvi.notna()]

BINS = [(-1.00, -0.30, "severe (DNDVI < -0.30)"),
        (-0.30, -0.20, "high (-0.30 to -0.20)"),
        (-0.20, -0.12, "moderate (-0.20 to -0.12)"),
        (-0.12,  0.00, "light (-0.12 to 0)")]


def fit(x, val):
    c, m, se = [], [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        g = x[(x.age >= a) & (x.age < b)][val].dropna()
        if len(g) >= 10:
            c.append(0.5 * (a + b)); m.append(float(g.mean()))
            se.append(float(g.std() / np.sqrt(len(g))))
    if len(c) < 5:
        return None
    try:
        p, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), np.array(c), np.array(m),
                         p0=[m[0], 8.0], sigma=np.maximum(se, 1e-3), absolute_sigma=True,
                         maxfev=20000, bounds=([-25, 0.3], [25, 60]))
        return float(p[0]), float(p[1])
    except Exception:
        return None


def boot_tau(x):
    pids = x.pid.unique(); idx = {p: g.index.values for p, g in x.groupby("pid")}
    v = []
    for _ in range(NBOOT):
        s = rng.choice(pids, len(pids), replace=True)
        f = fit(x.loc[np.concatenate([idx[p] for p in s])], "dlst")
        if f: v.append(f[1])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else None


out = {}
print(f"\n{'severity bin':28s} {'class':12s} {'patches':>8s} {'A (°C)':>7s} {'tau_LST':>8s} "
      f"{'95% CI':>14s} {'% gone @10yr':>13s}")
print("-" * 96)
for lo, hi, lab in BINS:
    sel = d[(d.sev_ndvi >= lo) & (d.sev_ndvi < hi)]
    row = {}
    for cls in ["landslide", "annual_loss"]:
        x = sel[sel.cls == cls]
        if x.pid.nunique() < 60:
            print(f"{lab:28s} {cls:12s} {x.pid.nunique():8d}   (too few patches)"); continue
        f = fit(x, "dlst")
        if not f:
            print(f"{lab:28s} {cls:12s} {x.pid.nunique():8d}   (fit failed)"); continue
        A, tau = f
        ci = boot_tau(x)
        g10 = (1 - np.exp(-10.0 / tau)) * 100
        row[cls] = dict(n_patches=int(x.pid.nunique()), A=A, tau=tau, tau_ci=ci, d10=g10)
        cis = f"[{ci[0]:5.1f},{ci[1]:5.1f}]" if ci else "            -"
        print(f"{lab:28s} {cls:12s} {x.pid.nunique():8d} {A:7.2f} {tau:8.2f} {cis:>14s} {g10:12.0f}%")
    if len(row) == 2:
        row["tau_ratio_ls_over_al"] = row["landslide"]["tau"] / row["annual_loss"]["tau"]
        print(f"{'':28s} {'--> ratio':12s} {'':8s} {'':7s} "
              f"{row['tau_ratio_ls_over_al']:8.2f}  (landslide tau / annual-loss tau)")
    out[lab] = row

# Severity domains of the two populations, and the size of the thinnest matched cell, so that the
# manuscript can quote both without a hand-written number.
_summary = {}
for cls, g in sev.groupby("cls"):
    _summary[cls] = dict(n_patches=int(len(g)),
                         q25=float(g.sev_ndvi.quantile(0.25)),
                         median=float(g.sev_ndvi.median()),
                         q75=float(g.sev_ndvi.quantile(0.75)))
out["severity_summary"] = _summary
_sev_lab = BINS[0][2]
out["severe_bin_annual_loss_patches"] = int(out[_sev_lab]["annual_loss"]["n_patches"]) \
    if "annual_loss" in out.get(_sev_lab, {}) else None
print("\nseverity domains:", {k: round(v["median"], 3) for k, v in _summary.items()},
      "| severe-bin annual-loss patches:", out["severe_bin_annual_loss_patches"])

json.dump(out, open(f"{_TROOT}/outputs/severity_match.json", "w"), indent=1, ensure_ascii=False)
print("\nsaved outputs/severity_match.json")
