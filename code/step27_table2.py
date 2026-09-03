"""Step 27: the per-agent summary behind Table 2 — how many patches each
trigger contributes, and the single most extensive event within it.

For the three catalogue agents the largest event is the single named event
(combined annual-inventory records, whose names are joined by a Chinese comma,
are excluded from the ranking because they are not one event). For the Hansen
annual-loss layer the equivalent unit is the loss year.
Output: outputs/table2_agents.json
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

d = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                    columns=["src", "year", "event", "agent", "t_event", "area_ha", "dq"])
AGENTS = [("typhoon_rain", "颱風"), ("rainfall", "豪雨"), ("earthquake", "地震")]
out = {"agents": [], "hansen": {}, "residual": {}}


def _date(sub):
    """Median catalogue date of a set of patches, as a calendar date."""
    t = float(np.median(sub.t_event))
    y = int(np.floor(t))
    day = int(round((t - y) * 365.25)) + 1
    return (pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=day - 1)).strftime("%Y-%m-%d")


ev = d[d.src == "event"]
for key, lab in AGENTS:
    sub = ev[ev.agent == key]
    singles = sub[~sub.event.fillna("").str.contains("、")]
    top = None
    if len(singles):
        g = singles.groupby("event").agg(n=("area_ha", "size"), area=("area_ha", "sum"))
        name = g.area.idxmax()
        s = singles[singles.event == name]
        top = {"event": name, "n": int(len(s)), "area_ha": float(s.area_ha.sum()),
               "date": _date(s), "dq": sorted(set(s.dq.astype(str)))}
    out["agents"].append({
        "key": key, "label": lab, "n": int(len(sub)), "area_ha": float(sub.area_ha.sum()),
        "year0": int(sub.year.min()), "year1": int(sub.year.max()), "top": top})

hs = d[d.src == "hansen"]
gy = hs.groupby("year").agg(n=("area_ha", "size"), area=("area_ha", "sum"))
ytop = int(gy.area.idxmax())
out["hansen"] = {"label": "年損失", "n": int(len(hs)), "area_ha": float(hs.area_ha.sum()),
                 "year0": int(hs.year.min()), "year1": int(hs.year.max()),
                 "top": {"event": f"{ytop} 年損失", "n": int(gy.loc[ytop, "n"]),
                         "area_ha": float(gy.loc[ytop, "area"]), "date": f"{ytop}"}}

res = ev[~ev.agent.isin([k for k, _ in AGENTS])]
out["residual"] = {"n": int(len(res)), "area_ha": float(res.area_ha.sum()),
                   "agents": sorted(set(res.agent.astype(str)))}

json.dump(out, open(f"{O}/table2_agents.json", "w"), ensure_ascii=False, indent=1)
for a in out["agents"]:
    print(a["label"], a["n"], round(a["area_ha"]), "| top:", a["top"]["event"] if a["top"] else None,
          a["top"]["n"] if a["top"] else "", round(a["top"]["area_ha"]) if a["top"] else "",
          a["top"]["date"] if a["top"] else "")
print("hansen", out["hansen"]["n"], round(out["hansen"]["area_ha"]), "| top:", out["hansen"]["top"])
print("residual", out["residual"])
