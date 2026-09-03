"""Step 8n: uniform-standard DiD for Table 4. Pools every cohort that has
both a pre- and a post-event summer (2014-2025 with the 2013-2026 stack)
(each patch uses its own last-full-pre-event summer and first full post-event
summer) and breaks the net thermal shock by: agent (earthquake / typhoon_rain /
rainfall, plus hansen annual-loss cohorts 2021-2023), elevation band, and
official area class. Adds results2["did_uniform"].
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
POST_TAG = {yr: f"c{yr+1-2000:02d}" for yr in range(2014, 2026)}   # 2013-2026 stack: every cohort with a pre- and a post-event summer

p = pd.read_parquet(f"{D}/patches_deltas2.parquet")
R = json.load(open(f"{O}/results2.json"))

rows = []
for src in ["event", "hansen"]:
    for yr, post_tag in POST_TAG.items():
        pre = f"c{yr-1-2000:02d}"
        cpre, cpost = f"dlst_{pre}", f"dlst_{post_tag}"
        if cpre not in p.columns or cpost not in p.columns:
            continue
        s = p[(p.src == src) & (p.year == yr) & (p.pre_tag == pre)
              & p[cpre].notna() & p[cpost].notna()].copy()
        if not len(s):
            continue
        s["pre_v"], s["post_v"] = s[cpre], s[cpost]
        rows.append(s[["src", "agent", "year", "elev", "area_ha",
                       "pre_v", "post_v"]])
u = pd.concat(rows, ignore_index=True)
u["did_v"] = u.post_v - u.pre_v
print("pooled DiD sample:", len(u), "by src:",
      u.src.value_counts().to_dict())


def stat(g):
    return dict(n=int(len(g)), pre=float(g.pre_v.mean()),
                post=float(g.post_v.mean()), did=float(g.did_v.mean()),
                se=float(g.did_v.std() / np.sqrt(len(g))),
                years=f"{int(g.year.min())}–{int(g.year.max())}")


ev = u[u.src == "event"]
out = {"pooled_event": stat(ev)}
ag = {}
for a, g in ev.groupby("agent"):
    if len(g) >= 8 and a:
        ag[a] = stat(g)
hz = u[u.src == "hansen"]
if len(hz) >= 8:
    ag["hansen"] = stat(hz)
out["agents"] = ag

elev = {}
for name, lo, hi in [("lt1000", 0, 1000), ("1000_2000", 1000, 2000),
                     ("gt2000", 2000, 9999)]:
    g = ev[(ev.elev >= lo) & (ev.elev < hi)]
    if len(g) >= 8:
        elev[name] = stat(g)
out["elev"] = elev

area = {}
for name, lo, hi in [("lt2", 0, 2), ("2_10", 2, 10), ("ge10", 10, 1e9)]:
    g = ev[(ev.area_ha >= lo) & (ev.area_ha < hi)]
    if len(g) >= 8:
        area[name] = stat(g)
out["area"] = area

R["did_uniform"] = out
json.dump(R, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
for k, v in out.items():
    print(k, json.dumps(v, ensure_ascii=False)[:400])
print("STEP8N COMPLETE")
