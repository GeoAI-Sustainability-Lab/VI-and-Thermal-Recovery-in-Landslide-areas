"""Step 8e: statistics for the unified multi-year event database.
Outputs outputs/results2.json + outputs/chrono2_long.parquet
Key upgrades vs v1: day-precision event ages 0.7-22 yr, agent-resolved fits
(incl. earthquake), era robustness (annual_swcb vs event_ardswc), cohort
placebo/DiD for every cohort with a pre- and post-event summer, Morakot direct multi-epoch trajectory.
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

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
EPOCH_MID = {"c13": 2013.62, "c14": 2014.62, "c15": 2015.62,
             "c16": 2016.62, "c17": 2017.62, "c18": 2018.62,
             "c19": 2019.62,
             "c20": 2020.62, "c21": 2021.62, "c22": 2022.62, "c23": 2023.62,
             "c24": 2024.62, "c25": 2025.62, "c26": 2026.55}
# The pooled two-summer composite c2425 is NOT an epoch of the age axis: its summers already enter
# through c24 and c25, so including it would count 2024-2025 twice per patch (corrected in 1.3.0).
# It remains available in patches_deltas2 for cross-sectional uses only.

p = pd.read_parquet(f"{D}/patches_deltas2.parquet")
R = {"n_patches": int(len(p)), "n_event": int((p.src == "event").sum()),
     "n_hansen": int((p.src == "hansen").sum())}

# quality: clean (no later re-disturbance), matched controls, forest context
p["clean"] = (p.redist_frac < 0.05) & (p.n_ctrl >= 30) & (p.forest2000_frac > 0.6)
R["n_clean"] = int(p.clean.sum())

# ---------------- long table ----------------
rows = []
for tag, mid in EPOCH_MID.items():
    cl, cn = f"dlst_{tag}", f"dndvi_{tag}"
    if cl not in p.columns:
        continue
    sub = p[p[cl].notna()].copy()
    sub["age"] = mid - sub.t_event
    sub["epoch"] = tag
    sub["dlst"] = sub[cl]
    sub["dndvi"] = sub[cn] if cn in sub.columns else np.nan
    sub["is_pre"] = sub.pre_tag == tag
    rows.append(sub[["pid", "src", "agent", "era", "dq", "event", "year", "age",
                     "dlst", "dndvi", "elev", "slope", "area_ha", "clean",
                     "is_pre", "epoch", "n_px", "redist_frac"]])
L = pd.concat(rows, ignore_index=True)
L.to_parquet(f"{O}/chrono2_long.parquet")
R["long_rows"] = int(len(L))
post = L[(~L.is_pre) & L.clean & (L.age > 0.6)]

def fit_rec(d, val="dlst", tmax=22, min_bin=12):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    if len(d) < 80:
        return None
    bins = np.arange(0.6, tmax + 1, 1.0)
    bc, bm, bs, bn = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        g = d[(d.age >= a) & (d.age < b)][val]
        if len(g) >= min_bin:
            bc.append(0.5 * (a + b)); bm.append(g.mean())
            bs.append(g.std() / np.sqrt(len(g))); bn.append(len(g))
    if len(bc) < 5:
        return None
    bc_, bm_, bs_ = np.array(bc), np.array(bm), np.array(bs)
    try:
        popt, pcov = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), bc_, bm_,
                               p0=[bm_[0], 6.0], sigma=np.maximum(bs_, 1e-3),
                               absolute_sigma=True, maxfev=20000,
                               bounds=([-25, 0.3], [25, 60]))
        perr = np.sqrt(np.diag(pcov))
        return dict(A=float(popt[0]), tau=float(popt[1]), A_se=float(perr[0]),
                    tau_se=float(perr[1]),
                    bins=dict(center=bc_.tolist(), mean=bm_.tolist(),
                              se=bs_.tolist(), n=bn))
    except Exception as e:
        return dict(error=repr(e)[:100],
                    bins=dict(center=bc_.tolist(), mean=bm_.tolist(),
                              se=bs_.tolist(), n=bn))

ev = post[post.src == "event"]
F = {}
F["event_dlst"] = fit_rec(ev)
F["event_dndvi"] = fit_rec(ev, "dndvi")
for ag in ["typhoon_rain", "rainfall", "earthquake"]:
    F[f"agent_{ag}_dlst"] = fit_rec(ev[ev.agent == ag], min_bin=8)
    F[f"agent_{ag}_dndvi"] = fit_rec(ev[ev.agent == ag], "dndvi", min_bin=8)
for band, (lo, hi) in {"low": (0, 1000), "mid": (1000, 2000), "high": (2000, 3600)}.items():
    F[f"ev_elev_{band}_dlst"] = fit_rec(ev[(ev.elev >= lo) & (ev.elev < hi)])
    F[f"ev_elev_{band}_dndvi"] = fit_rec(ev[(ev.elev >= lo) & (ev.elev < hi)], "dndvi")
for era in ["annual_swcb", "event_ardswc"]:
    F[f"era_{era}_dlst"] = fit_rec(ev[ev.era == era])
    F[f"era_{era}_dndvi"] = fit_rec(ev[ev.era == era], "dndvi")
F["hansen_dlst"] = fit_rec(post[post.src == "hansen"])
F["hansen_dndvi"] = fit_rec(post[post.src == "hansen"], "dndvi")
R["fits2"] = F

# ---------------- cohort placebo & DiD (every cohort with pre+post) ------
did = {}
for yr in range(2014, 2026):
    pre = f"c{yr-1-2000:02d}"
    post_tag = f"c{yr+1-2000:02d}"
    cpre, cpost = f"dlst_{pre}", f"dlst_{post_tag}"
    if cpre not in p.columns or cpost not in p.columns:
        continue
    s = p[(p.src == "event") & (p.year == yr) & (p.pre_tag == pre)
          & p[cpre].notna() & p[cpost].notna()]
    if len(s) < 8:
        continue
    d = s[cpost] - s[cpre]
    did[str(yr)] = dict(n=int(len(s)), pre=float(s[cpre].mean()),
                        post=float(s[cpost].mean()), did=float(d.mean()),
                        se=float(d.std() / np.sqrt(len(d))))
R["did_cohorts"] = did

# ---------------- Morakot direct multi-epoch trajectory ----------------
mk = p[(p.src == "event") & (p.year == 2009) & p.clean
       & p.event.str.contains("莫拉克")]
traj = {}
for tag, mid in EPOCH_MID.items():
    cl = f"dlst_{tag}"
    if cl in mk.columns and tag != "c2425":
        g = mk[mk[cl].notna()]
        gn = g[f"dndvi_{tag}"].dropna() if f"dndvi_{tag}" in g.columns else pd.Series(dtype=float)
        if len(g) >= 30:
            traj[tag] = dict(age=round(mid - 2009.6, 1), n=int(len(g)),
                             dlst=float(g[cl].mean()),
                             dlst_se=float(g[cl].std() / np.sqrt(len(g))),
                             dndvi=float(gn.mean()) if len(gn) else None)
R["morakot_traj"] = traj
R["morakot_n_clean"] = int(len(mk))

# year-1 shock by agent (day-precision cohorts only)
y1 = ev[(ev.age > 0.6) & (ev.age < 2.0) & (ev.dq.isin(["canonical", "mmdd"]))]
R["year1_by_agent"] = {a: dict(n=int(len(g)), mean=float(g.dlst.mean()),
                               se=float(g.dlst.std() / np.sqrt(len(g))))
                       for a, g in y1.groupby("agent") if len(g) >= 8}

# Later steps (08g, 08m, 08n, 08o, 08p) add their own keys to results2.json; keep them on a rerun.
if _os.path.exists(f"{O}/results2.json"):
    _old = json.load(open(f"{O}/results2.json", encoding="utf-8"))
    _old.update(R)
    R = _old
json.dump(R, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
key = {k: (f"tau={v['tau']:.1f}±{v['tau_se']:.1f}" if v and "tau" in v else "n/a")
       for k, v in F.items()}
print(json.dumps(key, indent=0))
print("did:", json.dumps(did, indent=0))
print("morakot:", json.dumps(traj, indent=0)[:400])
print("STEP8E COMPLETE")
