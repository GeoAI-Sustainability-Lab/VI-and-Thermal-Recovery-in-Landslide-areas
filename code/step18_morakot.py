"""Step 18: the Typhoon Morakot deep dive. Morakot (August 2009)
is the largest single event in the catalogue. With the epoch stack extended
back to 2013 its record now runs continuously from about four to seventeen years
after the event, so it is treated separately rather than folded into the pooled
fits.
Output: outputs/morakot.json
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
D = f"{_TROOT}/data"
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6)]
mask = post.event.fillna("").str.contains("莫拉克") & (post.src == "event")
mk = post[mask].copy()
oth = post[(post.src == "event") & ~mask]
if "pid" not in mk.columns:            # fallback for pre-v11 long tables
    mk["pid"] = mk.groupby(["event", "year", "elev", "area_ha"]).ngroup()
print("Morakot rows:", len(mk), "patches:", mk.pid.nunique(),
      "events:", mk.event.nunique())

out = {"n_rows": int(len(mk)), "n_patches": int(mk.pid.nunique()),
       "area_med": float(mk.area_ha.median()),
       "elev_med": float(mk.elev.median())}

# yearly means of the cohort, against the rest of the event database
def series(d):
    r = []
    for a, b in zip(np.arange(0.6, 22, 1.0), np.arange(1.6, 23, 1.0)):
        g = d[(d.age >= a) & (d.age < b)]
        if len(g) >= 20:
            r.append(dict(age=float(0.5 * (a + b)), n=int(len(g)),
                          dlst=float(g.dlst.mean()),
                          dlst_se=float(g.dlst.std() / np.sqrt(len(g))),
                          dndvi=float(g.dndvi.mean()),
                          dndvi_se=float(g.dndvi.std() / np.sqrt(g.dndvi.notna().sum()))))
    return r
out["cohort"] = series(mk)
out["others"] = series(oth)
print("cohort ages:", [round(r["age"], 1) for r in out["cohort"]])

# within-Morakot strata: area and elevation
for nm, col, cuts, labs in [
        ("area", "area_ha", [0, 2, 10, 1e9], ["<2 ha", "2–10 ha", "≥10 ha"]),
        ("elev", "elev", [0, 1000, 2000, 9999],
         ["<1,000 m", "1,000–2,000 m", ">2,000 m"])]:
    o = {}
    for i, lab in enumerate(labs):
        g = mk[(mk[col] >= cuts[i]) & (mk[col] < cuts[i + 1])]
        if len(g) < 60:
            continue
        o[lab] = dict(n_rows=int(len(g)),
                      n_patches=int(g.pid.nunique()),
                      dlst=float(g.dlst.mean()),
                      dlst_se=float(g.dlst.std() / np.sqrt(len(g))),
                      dndvi=float(g.dndvi.mean()),
                      series=series(g))
        print(f"  {nm} {lab}: n={o[lab]['n_patches']} dLST={o[lab]['dlst']:+.2f}")
    out[nm] = o

# Sentinel-1 structural deficit for the same cohort
s1 = json.load(open(f"{O}/s1_pilot.json"))
out["s1"] = s1.get("morakot_s1")

# 26-year case trajectories that belong to Morakot
tr = pd.read_parquet(f"{D}/case_trajectories.parquet")
tr["dlst"] = tr.lst_p - tr.lst_c
tr["dndvi"] = tr.ndvi_p - tr.ndvi_c
cases = {}
for c in ["hansen_2009_morakot", "high_elev_event", "small_morakot_event"]:
    d = tr[tr.case == c]
    if not len(d):
        continue
    ann = d.groupby("year").agg(dlst=("dlst", "mean"), dndvi=("dndvi", "mean"),
                                n=("dlst", "count"))
    ann = ann[ann.n >= 2]
    cases[c] = dict(year=[int(y) for y in ann.index],
                    dlst=[float(v) for v in ann.dlst],
                    dndvi=[float(v) for v in ann.dndvi],
                    t_event=float(d.t_event.iloc[0]),
                    area_ha=float(d.area_ha.iloc[0]),
                    elev=float(d.elev.iloc[0]))
    print(f"  case {c}: {len(ann)} yrs, {cases[c]['area_ha']:.0f} ha, "
          f"{cases[c]['elev']:.0f} m")
out["cases"] = cases
# catalogue-scale numbers for the event summary
p_all = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                        columns=["src", "event", "area_ha", "elev"])
pe = p_all[p_all.src == "event"]
mkp = pe[pe.event.fillna("").str.contains("莫拉克")]
out["catalog"] = dict(
    n_patch=int(len(mkp)), area_ha=float(mkp.area_ha.sum()),
    share_area=float(mkp.area_ha.sum() / pe.area_ha.sum()),
    share_n=float(len(mkp) / len(pe)),
    max_scar_ha=float(mkp.area_ha.max()),
    n_ge10=int((mkp.area_ha >= 10).sum()),
    n_ge10_all=int((pe.area_ha >= 10).sum()),
    elev_p10=float(mkp.elev.quantile(0.10)),
    elev_p90=float(mkp.elev.quantile(0.90)))
print("catalog:", out["catalog"])
json.dump(out, open(f"{O}/morakot.json", "w"), ensure_ascii=False, indent=1)
print("STEP18 COMPLETE")
