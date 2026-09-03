"""Step 8a: statistical results — buffering model + SHAP, chronosequence long
table + recovery fits. Saves outputs/results.json + tables (all real numbers)."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

sys.path.insert(0, f"{_TROOT}/code")
OUT = f"{_TROOT}/data"
RES = f"{_TROOT}/outputs"
os.makedirs(RES, exist_ok=True)
t0 = time.time()
R = {}

# ---------------- buffering model ----------------
df = pd.read_parquet(f"{OUT}/buffering_sample.parquet")
df = df[(df.chm >= 0) & (df.chm <= 45) & df.lst.notna()].copy()
R["buffering_n"] = int(len(df))

# elevation-band binned slopes: LST ~ CHM within band, controlling cos_i & slope via residualization
bands = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3600)]
band_stats = []
for lo, hi in bands:
    d = df[(df.elev >= lo) & (df.elev < hi)]
    if len(d) < 2000:
        continue
    # residualize LST on terrain confounders (linear): cos_i, slope, northness
    Xc = np.column_stack([d.cos_i, d.slope, d.northness, np.ones(len(d))])
    beta, *_ = np.linalg.lstsq(Xc, d.lst, rcond=None)
    lst_res = d.lst - Xc @ beta + d.lst.mean()
    sl, itc, r, p, se = stats.linregress(d.chm, lst_res)
    # binned medians for plotting
    bins = np.arange(0, 40, 2.5)
    bc = 0.5 * (bins[:-1] + bins[1:])
    med = [float(np.median(lst_res[(d.chm >= a) & (d.chm < b)])) if ((d.chm >= a) & (d.chm < b)).sum() > 50 else np.nan
           for a, b in zip(bins[:-1], bins[1:])]
    q25 = [float(np.percentile(lst_res[(d.chm >= a) & (d.chm < b)], 25)) if ((d.chm >= a) & (d.chm < b)).sum() > 50 else np.nan
           for a, b in zip(bins[:-1], bins[1:])]
    q75 = [float(np.percentile(lst_res[(d.chm >= a) & (d.chm < b)], 75)) if ((d.chm >= a) & (d.chm < b)).sum() > 50 else np.nan
           for a, b in zip(bins[:-1], bins[1:])]
    band_stats.append(dict(band=f"{lo}-{hi}", lo=lo, hi=hi, n=int(len(d)),
                           slope_per_m=float(sl), slope_se=float(se), p=float(p),
                           bin_centers=bc.tolist(), bin_median=med, bin_q25=q25, bin_q75=q75))
R["band_stats"] = band_stats

# GBM + SHAP
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
import shap

FEATS = ["chm", "elev", "slope", "cos_i", "northness", "eastness"]
X = df[FEATS].values.astype(np.float32)
y = df.lst.values.astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                    max_depth=None, min_samples_leaf=50,
                                    random_state=42)
gbm.fit(Xtr, ytr)
r2 = float(gbm.score(Xte, yte))
R["gbm_r2_test"] = r2
print("GBM R2:", r2, f"{time.time()-t0:.0f}s", flush=True)

bg = shap.utils.sample(Xtr, 200, random_state=0)
expl = shap.TreeExplainer(gbm) if hasattr(shap, "TreeExplainer") else None
try:
    sv = expl(Xte[:6000])
    np.savez_compressed(f"{RES}/shap_values.npz", values=sv.values, data=Xte[:6000],
                        feats=np.array(FEATS, dtype=object))
    R["shap_mean_abs"] = {f: float(np.abs(sv.values[:, i]).mean()) for i, f in enumerate(FEATS)}
except Exception as e:
    print("shap fail", repr(e)[:200]); R["shap_mean_abs"] = None

# counterfactual buffering magnitude from GBM: chm 5m -> 25m at median terrain, per band
cf = []
for lo, hi in bands:
    d = df[(df.elev >= lo) & (df.elev < hi)]
    if len(d) < 2000:
        continue
    base = d[FEATS].median().values.astype(np.float32)
    lo_v = base.copy(); lo_v[0] = 5.0
    hi_v = base.copy(); hi_v[0] = 25.0
    cf.append(dict(band=f"{lo}-{hi}",
                   lst_at_5m=float(gbm.predict([lo_v])[0]),
                   lst_at_25m=float(gbm.predict([hi_v])[0])))
R["counterfactual"] = cf

# ---------------- chronosequence long table ----------------
pdel = pd.read_parquet(f"{OUT}/patches_deltas.parquet")
ev = {int(k): v for k, v in json.load(open(f"{OUT}/event_codes.json")).items()}
MID = {"c2425": 2025.12, "c25": 2025.62, "c26": 2026.55, "c23": 2023.62, "c24": 2024.62}

def event_time(row):
    if row.src == "hansen":
        return row.year + 0.5
    if row.src == "fire2021":
        return 2021.33
    d = ev[row.event_code]["date"]
    return int(d[:4]) + (int(d[5:7]) - 0.5) / 12

pdel["t_event"] = pdel.apply(event_time, axis=1)
pdel["agent"] = np.where(pdel.src == "hansen", "hansen_loss",
                np.where(pdel.src == "fire2021", "fire",
                         pdel.event_code.map(lambda c: ev.get(int(c), {}).get("agent", "unknown"))))
long = []
for tag in ["c2425", "c25", "c26", "c23", "c24"]:
    cl, cn = f"dlst_{tag}", f"dndvi_{tag}"
    if cl not in pdel.columns:
        continue
    sub = pdel[pdel[cl].notna()].copy()
    sub["age"] = MID[tag] - sub.t_event
    sub["dlst"] = sub[cl]
    sub["dndvi"] = sub[cn] if cn in sub.columns else np.nan
    sub["epoch"] = tag
    long.append(sub[["src", "agent", "event_code", "year", "age", "dlst", "dndvi",
                     "area_ha", "elev", "slope", "northness", "ftype",
                     "forest2000_frac", "n_ctrl", "epoch", "label_id"]])
long = pd.concat(long, ignore_index=True)
# quality: require decent forest context for hansen/fire; events keep all but flag
long["quality"] = (long.n_ctrl >= 30) & ((long.src != "hansen") | (long.forest2000_frac > 0.8))
long.to_parquet(f"{RES}/chronosequence_long.parquet")
R["chrono_n_rows"] = int(len(long))
R["chrono_n_quality"] = int(long.quality.sum())

# placebo (pre-event) check
plc = []
for tag, srcyr in [("c23", 2024), ("c24", 2025)]:
    cl = f"dlst_{tag}"
    s = pdel[(pdel.src == "ardswc") & (pdel.year == srcyr) & pdel[cl].notna()]
    if len(s):
        plc.append(dict(epoch=tag, event_year=srcyr, n=int(len(s)),
                        mean=float(s[cl].mean()), sd=float(s[cl].std()),
                        q50=float(s[cl].median())))
R["placebo_dlst"] = plc

# post-event effect sizes year-1
y1 = long[(long.quality) & (long.age > 0.5) & (long.age < 2.0)]
R["year1_dlst_by_agent"] = {a: dict(n=int(len(g)), mean=float(g.dlst.mean()),
                                    q50=float(g.dlst.median()), sd=float(g.dlst.std()))
                            for a, g in y1.groupby("agent") if len(g) >= 5}

# ---- difference-in-differences (handles pre-event site-selection warmth) ----
def did_block(sub, pre, post, band):
    b = sub[sub[f"d{band}_{pre}"].notna() & sub[f"d{band}_{post}"].notna()]
    if len(b) < 8:
        return None
    d = b[f"d{band}_{post}"] - b[f"d{band}_{pre}"]
    return dict(n=int(len(b)), mean=float(d.mean()),
                se=float(d.std() / np.sqrt(len(d))), q50=float(d.median()),
                pre_mean=float(b[f"d{band}_{pre}"].mean()),
                post_mean=float(b[f"d{band}_{post}"].mean()))

did = {}
for yr_, pre, post in [(2024, "c23", "c25"), (2025, "c24", "c26")]:
    sub = pdel[(pdel.src == "ardswc") & (pdel.year == yr_)]
    did[f"events{yr_}_lst"] = did_block(sub, pre, post, "lst")
    did[f"events{yr_}_ndvi"] = did_block(sub, pre, post, "ndvi")
    sub2 = sub.copy()
    sub2["agent"] = sub2.event_code.map(lambda c: ev[int(c)]["agent"])
    for ag, g in sub2.groupby("agent"):
        blk = did_block(g, pre, post, "lst")
        if blk:
            did[f"events{yr_}_lst_{ag}"] = blk
R["did"] = did

# recovery fits: binned exponential fit ΔT(t) = A * exp(-t/tau) on quality rows
def fit_recovery(d, val="dlst", tmax=24):
    d = d[(d.age > 0.4) & (d.age <= tmax) & d[val].notna()]
    if len(d) < 60:
        return None
    bins = np.arange(0.5, tmax + 1, 1.0)
    bc, bm, bs, bn = [], [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        g = d[(d.age >= a) & (d.age < b)][val]
        if len(g) >= 12:
            bc.append(0.5 * (a + b)); bm.append(g.mean()); bs.append(g.std() / np.sqrt(len(g))); bn.append(len(g))
    if len(bc) < 5:
        return None
    bc, bm, bs = np.array(bc), np.array(bm), np.array(bs)
    try:
        popt, pcov = curve_fit(lambda t, A, tau: A * np.exp(-t / tau), bc, bm,
                               p0=[bm[0], 6.0], sigma=np.maximum(bs, 1e-3),
                               absolute_sigma=True, maxfev=20000,
                               bounds=([-20, 0.3], [20, 60]))
        perr = np.sqrt(np.diag(pcov))
        return dict(A=float(popt[0]), tau=float(popt[1]), A_se=float(perr[0]),
                    tau_se=float(perr[1]), t90=float(popt[1] * np.log(10)),
                    bins=dict(center=bc.tolist(), mean=bm.tolist(), se=bs.tolist(), n=bn))
    except Exception as e:
        return dict(error=repr(e)[:120], bins=dict(center=bc.tolist(), mean=bm.tolist(), se=bs.tolist(), n=bn))

fits = {}
q = long[long.quality]
fits["all_dlst"] = fit_recovery(q, "dlst")
fits["all_dndvi"] = fit_recovery(q, "dndvi")
fits["hansen_dlst"] = fit_recovery(q[q.agent == "hansen_loss"], "dlst")
fits["hansen_dndvi"] = fit_recovery(q[q.agent == "hansen_loss"], "dndvi")
# cohort-aware fits: 2015-2023 losses give a clean early-age curve free of the
# Morakot (2009) mega-scar cohort that dominates ages 11-16
qr = q[(q.agent == "hansen_loss") & (q.year >= 2015)]
fits["recent2015_dlst"] = fit_recovery(qr, "dlst", tmax=11)
fits["recent2015_dndvi"] = fit_recovery(qr, "dndvi", tmax=11)
for band, (lo, hi) in {"low": (0, 1000), "mid": (1000, 2000), "high": (2000, 3600)}.items():
    fits[f"elev_{band}_dlst"] = fit_recovery(q[(q.elev >= lo) & (q.elev < hi)], "dlst")
    fits[f"elev_{band}_dndvi"] = fit_recovery(q[(q.elev >= lo) & (q.elev < hi)], "dndvi")
R["recovery_fits"] = fits

# cohort composition behind the age 11-16 bump (Morakot 2009 etc.)
coh = {}
hq = q[(q.agent == "hansen_loss") & (q.epoch == "c2425")]
for y0, y1_ in [(2001, 2005), (2005, 2009), (2009, 2010), (2010, 2015), (2015, 2020), (2020, 2024)]:
    g = hq[(hq.year >= y0) & (hq.year < y1_)]
    if len(g) >= 20:
        coh[f"{y0}-{y1_-1}"] = dict(n=int(len(g)), dlst=float(g.dlst.mean()),
                                    dndvi=float(g.dndvi.mean()),
                                    area_ha_mean=float(g.area_ha.mean()),
                                    elev_mean=float(g.elev.mean()))
R["hansen_cohorts_c2425"] = coh

# ---------------- per-patch recovery-rate drivers (two epochs) ----------------
try:
    hb = pdel[(pdel.src == "hansen") & pdel.dlst_c2425.notna() & pdel.dlst_c26.notna()
              & (pdel.n_ctrl >= 30) & (pdel.forest2000_frac > 0.8)].copy()
    DT_EPOCH = MID["c26"] - MID["c2425"]
    hb["rate"] = (hb.dlst_c26 - hb.dlst_c2425) / DT_EPOCH     # °C / yr (negative = cooling back)
    hb["age_mid"] = MID["c2425"] - (hb.year + 0.5)
    hb = hb[(hb.age_mid > 0.5) & (hb.age_mid < 24)]
    RF = ["age_mid", "elev", "slope", "northness", "area_ha"]
    if len(hb) >= 300:
        Xr = hb[RF].values.astype(np.float32)
        yr_ = hb.rate.values.astype(np.float32)
        Xtr2, Xte2, ytr2, yte2 = train_test_split(Xr, yr_, test_size=0.25, random_state=1)
        g2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                           min_samples_leaf=40, random_state=1)
        g2.fit(Xtr2, ytr2)
        sv2 = shap.TreeExplainer(g2)(Xte2[:4000])
        np.savez_compressed(f"{RES}/shap_rate.npz", values=sv2.values, data=Xte2[:4000],
                            feats=np.array(RF, dtype=object))
        R["rate_model"] = dict(n=int(len(hb)), r2_test=float(g2.score(Xte2, yte2)),
                               rate_mean=float(hb.rate.mean()),
                               rate_by_elev={f"{lo}-{hi}": dict(
                                   n=int(((hb.elev >= lo) & (hb.elev < hi)).sum()),
                                   mean=float(hb.rate[(hb.elev >= lo) & (hb.elev < hi)].mean()))
                                   for lo, hi in [(0, 1000), (1000, 2000), (2000, 3600)]},
                               shap_mean_abs={f: float(np.abs(sv2.values[:, i]).mean())
                                              for i, f in enumerate(RF)})
        hb[["label_id", "year", "age_mid", "rate", "elev", "slope", "area_ha"]].to_parquet(
            f"{RES}/recovery_rates.parquet")
except Exception as e:
    print("rate model fail:", repr(e)[:200]); R["rate_model"] = None

json.dump(R, open(f"{RES}/results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k in
                  ["buffering_n", "gbm_r2_test", "placebo_dlst", "year1_dlst_by_agent"]},
                 ensure_ascii=False, indent=1)[:1500])
print("STEP8A COMPLETE", f"{time.time()-t0:.0f}s")
