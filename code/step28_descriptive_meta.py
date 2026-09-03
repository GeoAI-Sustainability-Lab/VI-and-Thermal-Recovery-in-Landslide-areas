"""Step 28: derive the descriptive metadata quoted in the text directly from the
source tables, so no count is hand-written. Covers the analysis
grid and its geographic envelope, the harmonised event catalogue's dating-quality
and agent breakdown, the forest inventory polygon count, the four intact-forest
sample populations (drawn / elevation-banded / 250 m-sliced / plotted in Fig. 3a),
and the residual lapse rate inside a 250 m slice.
The analysis grid and the forest-inventory polygon count need the 30 m rasters
(tier B); when they are absent the values already recorded in outputs/grid_meta.json
are kept and the fact is printed. The SHAP display subset is read from the shipped
outputs/shap_buffering.parquet.
Output: outputs/grid_meta.json
"""
import json, struct
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np
import pandas as pd

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
PREV = f"{O}/grid_meta.json"
out = json.load(open(PREV, encoding="utf-8")) if _os.path.exists(PREV) else {}

# ---- analysis grid, forest inventory polygons (tier B rasters) ----
try:
    import rasterio
    from pyproj import Transformer
    # ---- analysis grid and geographic envelope ----
    with rasterio.open(f"{D}/lst_c13.tif") as s:
        W, H, B, CRS, RES = s.width, s.height, s.bounds, str(s.crs), s.res
    tr = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    pts = [tr.transform(x, y) for x in (B.left, B.right) for y in (B.bottom, B.top)]
    lo_x = np.floor(min(p[0] for p in pts) * 100) / 100
    hi_x = np.ceil(max(p[0] for p in pts) * 100) / 100
    lo_y = np.floor(min(p[1] for p in pts) * 100) / 100
    hi_y = np.ceil(max(p[1] for p in pts) * 100) / 100
    out["grid"] = dict(width=W, height=H, crs=CRS, res_m=RES[0],
                       lon=[float(lo_x), float(hi_x)], lat=[float(lo_y), float(hi_y)])
    print("grid", W, "x", H, CRS, f"{lo_x}-{hi_x}E {lo_y}-{hi_y}N", flush=True)

    # ---- forest inventory polygons ----
    out["forest4_polygons"] = struct.unpack(
        "<I", open(f"{D}/forest4/forest4.dbf", "rb").read(8)[4:8])[0]

    print("grid and inventory recomputed from rasters", flush=True)
except Exception as _e:                      # tier A: rasters not shipped
    print(f"rasters not available ({type(_e).__name__}); grid / forest4 values kept from "
          f"the released grid_meta.json", flush=True)

# ---- harmonised event catalogue ----
ec = json.load(open(f"{D}/event_codes2.json", encoding="utf-8"))
dq, ag, era, eq = {}, {}, {}, {}
for v in ec.values():
    dq[v["dq"]] = dq.get(v["dq"], 0) + v["n"]
    ag[v["agent"]] = ag.get(v["agent"], 0) + v["n"]
    era[v["era"]] = era.get(v["era"], 0) + v["n"]
    if v["agent"] == "earthquake":
        eq[v["event"]] = eq.get(v["event"], 0) + v["n"]
top_eq = max(eq.items(), key=lambda x: x[1])
out["catalogue"] = dict(
    total=sum(v["n"] for v in ec.values()), n_codes=len(ec),
    date_quality=dq, agent=ag, era=era,
    earthquake_dates=sorted({e[:4] for e in eq if e[:4].isdigit()}),
    earthquake_top=dict(event=top_eq[0], n=top_eq[1]))
print("catalogue", out["catalogue"]["total"], dq, ag, flush=True)

# ---- intact-forest sample: four populations ----
REF_LO, REF_HI = 24.0, 26.0
BAND_LO, BAND_HI = 0, 3600
COLS = ["lst", "chm", "elev", "slope", "northness", "eastness", "cos_i"]
df = pd.read_parquet(f"{D}/buffering_sample.parquet").dropna(subset=COLS)
n_drawn = len(df)
n_band = int(((df.elev >= BAND_LO) & (df.elev < BAND_HI)).sum())
df = df.assign(slice=(df.elev // 250).astype(int))
parts, lapse = [], []
for _, g in df.groupby("slice"):
    if len(g) < 3000 or ((g.chm >= REF_LO) & (g.chm < REF_HI)).sum() < 100:
        continue
    lapse.append(abs(np.polyfit(g.elev, g.lst, 1)[0]) * 250.0)
    X = np.column_stack([g.cos_i, g.slope, g.northness, np.ones(len(g))])
    beta, *_ = np.linalg.lstsq(X, g.lst, rcond=None)
    res = g.lst.values - X @ beta
    parts.append(g.assign(anom=res - np.median(res[(g.chm >= REF_LO) & (g.chm < REF_HI)])))
d = pd.concat(parts)
sel = (d.chm >= 8) & (d.chm <= 50) & (d.anom > -6.2) & (d.anom < 6.2)
hb = df[(df.elev >= 2500) & (df.elev < BAND_HI)]
h38 = hb[hb.chm >= 38]
lapse = np.array(lapse)
out["intact_sample"] = dict(
    drawn=n_drawn, band_0_3600=n_band, slice_250m=int(len(d)),
    fig3a_plotted=int(sel.sum()), fig3a_frac=float(sel.mean()),
    tall_high_band_n=int(len(h38)),
    tall_high_band_elev_p10=float(np.percentile(h38.elev, 10)),
    tall_high_band_elev_p90=float(np.percentile(h38.elev, 90)),
    slice_lapse_median=float(np.median(lapse)),
    slice_lapse_max=float(lapse.max()), n_slices=int(len(lapse)))
print("intact", out["intact_sample"], flush=True)

# ---- Fig. 3a binned curve: anomaly at 15 m and above 40 m (same bins as the figure) ----
ds = d[sel]
e = np.arange(8, 50.5, 1.0)
cc, mm = [], []
for a, b in zip(e[:-1], e[1:]):
    g = ds.anom[(ds.chm >= a) & (ds.chm < b)]
    if len(g) >= 80:
        cc.append(0.5 * (a + b)); mm.append(float(g.mean()))
cc, mm = np.array(cc), np.array(mm)
out["curve"] = dict(anom_at_15=float(np.interp(15.0, cc, mm)),
                    anom_ge40_mean=float(mm[cc >= 40].mean()),
                    anom_at_40=float(np.interp(40.0, cc, mm)))
print("curve", out["curve"], flush=True)

# ---- SHAP panel: rows plotted vs available ----
sh = pd.read_parquet(f"{O}/shap_buffering.parquet")
out["shap_panel"] = dict(n_rows=int(len(sh)), n_plotted=int(min(4000, len(sh))),
                         n_features=sum(1 for c in sh.columns if c.startswith("shap_")), n_shown=4)

# ---- mixed-source recovery fit (event + annual loss pooled), same binned NLS as step16 ----
from scipy.optimize import curve_fit
L = pd.read_parquet(f"{O}/chrono2_long.parquet")
post = L[(~L.is_pre) & L.clean & (L.age > 0.6) & (L.age <= 22.0)]


def _fit(dd, val):
    c, m, se = [], [], []
    ed = np.arange(0.6, 23.0, 1.0)
    for a, b in zip(ed[:-1], ed[1:]):
        g = dd[(dd.age >= a) & (dd.age < b)][val].dropna()
        if len(g) >= 10:
            c.append(0.5 * (a + b)); m.append(float(g.mean()))
            se.append(float(g.std() / np.sqrt(len(g))))
    popt, _ = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), c, m, p0=[m[0], 8.0],
                        sigma=np.maximum(se, 1e-3), absolute_sigma=True, maxfev=20000,
                        bounds=([-25, 0.3], [25, 60]))
    return float(popt[1])


mix = post[post.src.isin(["event", "hansen"])]
tt, tg = _fit(mix, "dlst"), _fit(mix, "dndvi")
out["mixed_fit"] = dict(tau_thermal=tt, tau_greenness=tg, ratio=tt / tg, rows=int(len(mix)),
                        definition="event landslides and all annual-loss patches pooled; binned exponential fit with the step16 settings")
print("mixed fit", out["mixed_fit"], flush=True)

json.dump(out, open(PREV, "w"), indent=1, ensure_ascii=False)
print("STEP28 COMPLETE")
