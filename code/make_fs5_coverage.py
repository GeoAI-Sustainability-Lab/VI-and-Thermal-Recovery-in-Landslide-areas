"""Supplementary figure S5: what each summer epoch actually contributes.

Scene counts and clear-sky coverage are not uniform across 2013-2026: the
2013-2019 epochs have Landsat 8 only, the 2022-2026 epochs have Landsat 8 and 9.
Coverage differences change how many patches are measurable in an epoch but not
what is measured, because a patch and its control are always read from the same
composite. This figure states the difference rather than leaving it implicit.

Output: figures/FS5_coverage.{pdf,png}
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json, glob, os
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards, WONG

apply_mpl_standards()
D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
MM = 1 / 25.4
EPOCH_YEAR = {f"c{y-2000:02d}": y for y in range(2013, 2027)}
TAGS = [t for t in EPOCH_YEAR if os.path.exists(f"{D}/nclear_lst_{t}.tif")]
TAGS.sort(key=lambda t: EPOCH_YEAR[t])


def _epoch_key(b):
    t = b.split("_items")[0]
    return "c" + t[-2:] if t.startswith("s20") else t


files = {}
for p in sorted(glob.glob(f"{D}/acc/*_items.json")):
    b = os.path.basename(p)
    if "firstbuild" in b:
        continue
    k = _epoch_key(b)
    if k not in files or b.startswith("c"):
        files[k] = p
nscene = {}
for k, p in files.items():
    try:
        nscene[k] = sum(1 for m in json.load(open(p)) if m.get("used"))
    except Exception:
        nscene[k] = np.nan

with rasterio.open(f"{D}/forest4_30.tif") as s:
    dom = s.read(1) > 0
ndom = int(dom.sum())
cov, medn = [], []
for t in TAGS:
    with rasterio.open(f"{D}/nclear_lst_{t}.tif") as s:
        n = s.read(1)
    cov.append(float(((n >= 3) & dom).sum()) / ndom * 100)
    v = n[dom & (n > 0)]
    medn.append(float(np.median(v)) if v.size else np.nan)
    del n
del dom

import pyarrow.parquet as pq
_sch = pq.read_schema(f"{D}/patches_deltas2.parquet").names
_cols = [f"dlst_{t}" for t in TAGS if f"dlst_{t}" in _sch]
p = pd.read_parquet(f"{D}/patches_deltas2.parquet", columns=_cols)
nobs = [int(p[f"dlst_{t}"].notna().sum()) if f"dlst_{t}" in p.columns else 0
        for t in TAGS]

x = np.arange(len(TAGS))
lab = [str(EPOCH_YEAR[t]) for t in TAGS]
col = [WONG["skyblue"] if EPOCH_YEAR[t] <= 2021 else WONG["blue"] for t in TAGS]

fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 72 * MM), constrained_layout=True)
ax = axs[0]
ax.bar(x, [nscene.get(t, np.nan) for t in TAGS], color=col, width=0.72)
ax.set_ylabel("Landsat scenes used")
ax.set_title("a  Scenes per summer", loc="left", fontweight="bold", fontsize=9.5)

ax = axs[1]
ax.bar(x, cov, color=col, width=0.72)
ax.set_ylabel("Forest-domain pixels with ≥ 3 clear looks (%)")
ax.set_title("b  Clear-sky coverage", loc="left", fontweight="bold", fontsize=9.5)
ax.set_ylim(0, 100)

ax = axs[2]
ax.bar(x, nobs, color=col, width=0.72)
ax.set_ylabel("Patch × epoch observations")
ax.set_title("c  Measurable patches", loc="left", fontweight="bold", fontsize=9.5)

for ax in axs:
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=90, fontsize=8.3)
    ax.set_xlabel("Summer epoch")
    ax.spines[["top", "right"]].set_visible(False)
h = [plt.Rectangle((0, 0), 1, 1, color=WONG["skyblue"]),
     plt.Rectangle((0, 0), 1, 1, color=WONG["blue"])]
axs[0].legend(h, ["Landsat 8 only", "Landsat 8 + 9"], frameon=False, fontsize=8.3,
              loc="upper left")

fig.savefig(f"{F}/FS5_coverage.pdf", bbox_inches="tight")
fig.savefig(f"{F}/FS5_coverage.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved FS5_coverage",
      dict(zip(lab, [f"{c:.0f}%" for c in cov])))
