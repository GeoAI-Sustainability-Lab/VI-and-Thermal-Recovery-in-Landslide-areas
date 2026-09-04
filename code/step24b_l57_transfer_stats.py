"""Step 24b: cross-sensor transfer statistics, the table-side half of step24.

step24 builds data/l57_deltas.parquet from the Landsat 5/7 per-sensor composites
(tier B). This step regresses, patch by patch, the delta seen by one sensor on
the delta seen by the other in the same summer (L7 on L8 for 2013 and 2014,
L5 on L7 for 2010 and 2011), overall and by patch-size class, and reports the
per-sensor coverage.
Reads : data/l57_deltas.parquet
Writes: outputs/l57_transfer.json
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
df = pd.read_parquet(f"{D}/l57_deltas.parquet")
TAGS = [t for t in ["c13", "c13l7", "c14", "c14l7", "c10l5", "c10l7", "c11l5", "c11l7", "c12l7"]
        if f"dlst_{t}" in df.columns]
print("tags:", TAGS, flush=True)

R = {"tags": TAGS}


def compare(a, b, name, band="lst"):
    ca, cb = f"d{band}_{a}", f"d{band}_{b}"
    if ca not in df.columns or cb not in df.columns:
        return
    s = df[df[ca].notna() & df[cb].notna()]
    if len(s) < 50:
        return
    ent = {}
    for lab, g in [("all", s), ("lt2", s[s.area_ha < 2]),
                   ("2_10", s[(s.area_ha >= 2) & (s.area_ha < 10)]),
                   ("ge10", s[s.area_ha >= 10])]:
        if len(g) < 30:
            continue
        x, y = g[cb].values, g[ca].values     # x = reference (L8 or L7)
        b1, b0 = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        ent[lab] = dict(n=int(len(g)), slope=float(b1), intercept=float(b0),
                        r=r, bias=float((y - x).mean()),
                        sd_diff=float((y - x).std()),
                        mean_ref=float(x.mean()))
    R[f"{name}_{band}"] = ent
    print(name, band, {k: (v["n"], round(v["slope"], 3), round(v["bias"], 3))
                       for k, v in ent.items()}, flush=True)


for band in ("lst", "ndvi"):
    compare("c13l7", "c13", "L7_vs_L8_2013", band)
    compare("c14l7", "c14", "L7_vs_L8_2014", band)
    compare("c10l5", "c10l7", "L5_vs_L7_2010", band)
    compare("c11l5", "c11l7", "L5_vs_L7_2011", band)

R["coverage"] = {t: int(df[f"dlst_{t}"].notna().sum()) for t in TAGS}
print("coverage:", R["coverage"], flush=True)
json.dump(R, open(f"{O}/l57_transfer.json", "w"), indent=1)
print("STEP24B COMPLETE")
