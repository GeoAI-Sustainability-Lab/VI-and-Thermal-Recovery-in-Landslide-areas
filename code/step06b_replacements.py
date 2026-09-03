"""Step 6b: replacement trajectory cases for the two ST-void patches
(hualien_eq_2024, hansen_2004). Pick largest candidates WITH valid dlst
(guarantees Landsat ST coverage), rerun, merge into case_trajectories.parquet."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

sys.path.insert(0, f"{_TROOT}/code")
import step06_trajectories as S   # reuse run_case machinery (executes its case list? no: module-level code runs)

# NOTE: importing step06 executes its module-level case selection but NOT run_case calls
# (the __main__-style loop at bottom DID run on import guard?) -> step06 has no guard;
# to avoid re-running everything we re-implement selection here and call S.run_case.
OUT = f"{_TROOT}/data"

pdel = pd.read_parquet(f"{OUT}/patches_deltas.parquet")
ev = {int(k): v for k, v in json.load(open(f"{OUT}/event_codes.json")).items()}
patches = pd.read_parquet(f"{OUT}/patches_raw.parquet")

ok24 = pdel[(pdel.src == "ardswc") & pdel.dlst_c25.notna()]
ok24 = ok24[ok24.event_code.map(lambda c: ev[int(c)]["event"]) == "0403花蓮地震"]
eq_id = ok24.nlargest(1, "n_px").iloc[0].label_id
eq_row = patches[(patches.src == "ardswc") & (patches.label_id == eq_id)].iloc[0]

okh = pdel[(pdel.src == "hansen") & (pdel.year == 2004) & pdel.dlst_c2425.notna()
           & (pdel.forest2000_frac > 0.8)]
h_id = okh.nlargest(1, "n_px").iloc[0].label_id
h_row = patches[(patches.src == "hansen") & (patches.label_id == h_id)].iloc[0]

print("replacements:", int(eq_row.n_px), "px eq;", int(h_row.n_px), "px 2004", flush=True)
d1 = S.run_case("hualien_eq_2024", eq_row, 2024.25)
d2 = S.run_case("hansen_2004", h_row, 2004.5)

tr = pd.read_parquet(f"{OUT}/case_trajectories.parquet")
tr = tr[~tr.case.isin(["hualien_eq_2024", "hansen_2004"])]
tr = pd.concat([tr] + [d for d in [d1, d2] if d is not None and len(d)])
tr.to_parquet(f"{OUT}/case_trajectories.parquet")
print("merged:", tr.groupby("case").size().to_dict())
print("STEP6B COMPLETE")
