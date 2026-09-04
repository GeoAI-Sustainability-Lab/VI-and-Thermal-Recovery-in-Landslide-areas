"""Step 30: the three validity checks, recomputed from the shipped tables.

  * Compositing method. data/medval_pixels.parquet holds, for three 256 x 256
    subareas of the full 2025 scene stack, the per-pixel median, the clear-sky
    mean, the production composite value and the clear-observation count. The
    mean-minus-median bias and RMSE, the production-minus-median bias and RMSE
    and the median-mean correlation are recomputed on pixels with >= 3 clear
    observations (outputs/median_validation.json). The scene count per subarea
    cannot be derived from the pixel table and is kept from the released file.
  * Day-night amplitude. data/s3_daynight_cells.parquet holds the Sentinel-3
    SLSTR summer-2025 day and night LST on a 0.02 degree grid with canopy
    height, elevation and tree fraction; the amplitude by elevation band and
    canopy class is recomputed (outputs/s3_results.json -> stats). The number
    of scenes used is kept from the released file.
  * Canopy-height cross-check. data/chm2_sample.parquet holds 30,000 intact
    pixels with the ETH 10 m height and the Meta/WRI CHMv2 1 m height sampled
    at the same place; agreement, bias by height class and by elevation, and
    the canopy slope with either height model are recomputed
    (outputs/chm2_compare.json).

Reads : data/medval_pixels.parquet, data/s3_daynight_cells.parquet,
        data/chm2_sample.parquet
Writes: outputs/median_validation.json, outputs/s3_results.json,
        outputs/chm2_compare.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json
import numpy as np
import pandas as pd
from scipy import stats

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"


def _load(path):
    return json.load(open(path, encoding="utf-8")) if _os.path.exists(path) else {}


# ---------------- 1. clear-sky mean against per-pixel median ----------------
mv = pd.read_parquet(f"{D}/medval_pixels.parquet")
prev = _load(f"{O}/median_validation.json")
report = {}
for name in ["west_lowland", "central_mid", "east_steep"]:
    p = mv[(mv.subarea == name) & (mv.n >= 3)]
    diff = (p["mean"] - p["median"]).to_numpy(dtype=np.float32)
    okp = np.isfinite(p["prod"].to_numpy())
    dprod = (p["prod"] - p["median"]).to_numpy(dtype=np.float32)[okp]
    report[name] = dict(
        n_scenes=prev.get(name, {}).get("n_scenes"),
        n_px=int(len(p)),
        mean_minus_median_bias=float(np.mean(diff)),
        mean_minus_median_rmse=float(np.sqrt(np.mean(diff ** 2))),
        prod_minus_median_bias=float(np.mean(dprod)),
        prod_minus_median_rmse=float(np.sqrt(np.mean(dprod ** 2))),
        median_mean_r=float(np.corrcoef(p["mean"].to_numpy(), p["median"].to_numpy())[0, 1]))
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report[name].items()},
          flush=True)
json.dump(report, open(f"{O}/median_validation.json", "w"), indent=1)

# ---------------- 2. Sentinel-3 day-night amplitude by canopy class ----------------
s3 = pd.read_parquet(f"{D}/s3_daynight_cells.parquet").astype(
    {"day": "float64", "night": "float64", "chm": "float64", "dem": "float64", "treefrac": "float64"})
prev = _load(f"{O}/s3_results.json")
amp = s3.day - s3.night
s3stats = []
for lo, hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2800)]:
    band = ((s3.dem >= lo) & (s3.dem < hi) & (s3.treefrac > 0.35)
            & np.isfinite(amp) & np.isfinite(s3.chm))
    for cl, (clo, chi_) in {"short (<15 m)": (0, 15), "tall (>=22 m)": (22, 60)}.items():
        m = band & (s3.chm >= clo) & (s3.chm < chi_)
        if m.sum() >= 12:
            s3stats.append(dict(elev_band=f"{lo}-{hi}", chm_class=cl, n=int(m.sum()),
                                day=float(np.nanmean(s3.day[m])),
                                night=float(np.nanmean(s3.night[m])),
                                amp_mean=float(np.nanmean(amp[m])),
                                amp_se=float(np.nanstd(amp[m]) / np.sqrt(m.sum()))))
json.dump(dict(used=prev.get("used"), stats=s3stats), open(f"{O}/s3_results.json", "w"), indent=1)
print("s3 amplitude:", [(s["elev_band"], s["chm_class"], round(s["amp_mean"], 2)) for s in s3stats],
      flush=True)

# ---------------- 3. ETH canopy height against Meta/WRI CHMv2 ----------------
s = pd.read_parquet(f"{D}/chm2_sample.parquet")
v = s[np.isfinite(s.chm2)]
res = {"n": int(len(v)), "r": float(np.corrcoef(v.chm, v.chm2)[0, 1]),
       "bias_mean": float((v.chm2 - v.chm).mean()),
       "rmsd": float(np.sqrt(((v.chm2 - v.chm) ** 2).mean()))}
by_h = {}
for lo, hi in [(0, 5), (5, 15), (15, 25), (25, 35), (35, 60)]:
    g = v[(v.chm >= lo) & (v.chm < hi)]
    if len(g) >= 100:
        by_h[f"{lo}-{hi}"] = dict(n=int(len(g)), bias=float((g.chm2 - g.chm).mean()),
                                  sd=float((g.chm2 - g.chm).std()))
res["bias_by_eth_class"] = by_h
BANDS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3600)]
slopes = {}
for lo, hi in BANDS:
    d = v[(v.elev >= lo) & (v.elev < hi)]
    if len(d) < 800:
        continue
    Xc = np.column_stack([d.cos_i, d.slope, d.northness, np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(Xc, d.lst, rcond=None)
    lst_res = d.lst - Xc @ beta + d.lst.mean()
    sl2, _, _, _, se2 = stats.linregress(d.chm2, lst_res)
    sl1, _, _, _, se1 = stats.linregress(d.chm, lst_res)
    slopes[f"{lo}-{hi}"] = dict(n=int(len(d)),
                                eth=dict(slope_per_m=float(sl1), se=float(se1)),
                                chm2=dict(slope_per_m=float(sl2), se=float(se2)))
res["band_slopes"] = slopes
by_e = {}
for lo, hi in BANDS:
    g = v[(v.elev >= lo) & (v.elev < hi)]
    if len(g) >= 500:
        by_e[f"{lo}-{hi}"] = dict(n=int(len(g)), bias=float((g.chm2 - g.chm).mean()))
res["bias_by_elev"] = by_e
json.dump(res, open(f"{O}/chm2_compare.json", "w"), indent=1)
print("chm2:", {k: round(res[k], 4) for k in ["n", "r", "bias_mean", "rmsd"]}, flush=True)
print("STEP30 COMPLETE")
