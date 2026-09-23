# -*- coding: utf-8 -*-
"""Elevation distribution of the analysed patches and of the intact-forest baseline
sample, so that the stated scope of the study is a number read from the data rather
than an adjective. Writes: outputs/patch_elevation.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, warnings
warnings.filterwarnings("ignore")
import pandas as pd

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 9000)]
QS = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]


def describe(elev):
    e = elev.dropna()
    q = e.quantile(QS)
    return dict(n=int(len(e)),
                q={str(p): float(q[p]) for p in QS},
                share={f"{lo}_{hi}": float(((e >= lo) & (e < hi)).mean() * 100) for lo, hi in BANDS})


L = pd.read_parquet(f"{_TROOT}/outputs/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean]
out = {
    "patches_all": describe(post.groupby("pid").elev.first()),
    "patches_landslide": describe(post[post.src == "event"].groupby("pid").elev.first()),
    "patches_annual_loss": describe(post[post.src == "hansen"].groupby("pid").elev.first()),
    "baseline_sample": describe(pd.read_parquet(f"{_TROOT}/data/buffering_sample.parquet", columns=["elev"]).elev),
}
json.dump(out, open(f"{_TROOT}/outputs/patch_elevation.json", "w"), indent=1)
for k, v in out.items():
    print(f"{k:22s} n={v['n']:7,}  median={v['q']['0.5']:6.0f} m  "
          f"range {v['q']['0.0']:.0f}-{v['q']['1.0']:.0f} m  "
          f"<500 m {v['share']['0_500']:.0f}%  >1000 m "
          f"{v['share']['1000_2000']+v['share']['2000_9000']:.0f}%")
print("\nsaved outputs/patch_elevation.json")
