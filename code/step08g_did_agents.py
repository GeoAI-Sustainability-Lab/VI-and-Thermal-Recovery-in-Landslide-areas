"""Step 8g: agent-resolved cohort DiD + pooled placebo-by-agent from the
unified 2004-2025 patch database. Merges results into outputs/results2.json
(keys: did_agents, placebo_by_agent, text_ranges). Same sample rule as step8e
did_cohorts: src=event, pre_tag == last full pre-event summer, both epochs valid.
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

p = pd.read_parquet(f"{D}/patches_deltas2.parquet")
R = json.load(open(f"{O}/results2.json"))

POST_TAG = {yr: f"c{yr+1-2000:02d}" for yr in range(2014, 2026)}   # 2013-2026 stack: every cohort with a pre- and a post-event summer

did_agents = {}
pre_pool = []
for yr, post_tag in POST_TAG.items():
    pre = f"c{yr-1-2000:02d}"
    cpre, cpost = f"dlst_{pre}", f"dlst_{post_tag}"
    if cpre not in p.columns or cpost not in p.columns:
        continue
    s = p[(p.src == "event") & (p.year == yr) & (p.pre_tag == pre)
          & p[cpre].notna() & p[cpost].notna()].copy()
    if len(s) < 8:
        continue
    s["pre_v"], s["post_v"] = s[cpre], s[cpost]
    pre_pool.append(s[["agent", "pre_v"]])
    grp = {}
    for name, g in [("all", s)] + list(s.groupby("agent")):
        if len(g) < 8:
            continue
        d = g["post_v"] - g["pre_v"]
        grp[name] = dict(n=int(len(g)), pre=float(g["pre_v"].mean()),
                         post=float(g["post_v"].mean()), did=float(d.mean()),
                         se=float(d.std() / np.sqrt(len(d))))
    did_agents[str(yr)] = grp

pp = pd.concat(pre_pool, ignore_index=True)
placebo_by_agent = {}
for ag, g in pp.groupby("agent"):
    if len(g) >= 8 and ag:
        placebo_by_agent[ag] = dict(n=int(len(g)), pre=float(g.pre_v.mean()),
                                    se=float(g.pre_v.std() / np.sqrt(len(g))))

# NDVI cohort DiD (pooled agents, same sample rule)
did_ndvi = {}
for yr, post_tag in POST_TAG.items():
    pre = f"c{yr-1-2000:02d}"
    cpre, cpost = f"dndvi_{pre}", f"dndvi_{post_tag}"
    if cpre not in p.columns or cpost not in p.columns:
        continue
    s = p[(p.src == "event") & (p.year == yr) & (p.pre_tag == pre)
          & p[cpre].notna() & p[cpost].notna()]
    if len(s) < 8:
        continue
    d = s[cpost] - s[cpre]
    did_ndvi[str(yr)] = dict(n=int(len(s)), did=float(d.mean()),
                             se=float(d.std() / np.sqrt(len(d))))

yr_counts = {int(y): int(n) for y, n in
             p[p.src == "event"].year.value_counts().sort_index().items()}

dd = R["did_cohorts"]
R["did_agents"] = did_agents
R["did_ndvi"] = did_ndvi
R["event_patches_by_year"] = yr_counts
R["placebo_by_agent"] = placebo_by_agent
# ranges quoted in the abstract/highlights use cohorts with a usable sample;
# with the 2013-2026 stack there are ~12 cohorts and the smallest are noisy
NMIN_TXT = 50
_dd = {k: v for k, v in dd.items() if v["n"] >= NMIN_TXT} or dd
print("text_range cohorts:", {k: v["n"] for k, v in _dd.items()})
R["text_ranges"] = dict(
    n_min_cohort=NMIN_TXT, n_cohorts_quoted=len(_dd),
    cohorts_quoted=sorted(_dd, key=int),
    pre_min=float(min(v["pre"] for v in _dd.values())),
    pre_max=float(max(v["pre"] for v in _dd.values())),
    did_min=float(min(v["did"] for v in _dd.values())),
    did_max=float(max(v["did"] for v in _dd.values())),
)
json.dump(R, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(did_agents, indent=1))
print("placebo_by_agent:", json.dumps(placebo_by_agent, indent=1))
print("ranges:", R["text_ranges"])
print("STEP8G COMPLETE")
