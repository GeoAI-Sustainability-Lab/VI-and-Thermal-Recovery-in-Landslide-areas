"""Step 13e: build rows and fits from the final S1 checkpoint.
Outputs: data/s1_pilot_rows.parquet, outputs/s1_pilot.json
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
SUM_MID = {2023: 2023.62, 2024: 2024.62, 2025: 2025.62}

dd = pd.read_parquet(f"{D}/patches_deltas2.parquet",
                     columns=["src", "n_ctrl", "redist_frac",
                              "forest2000_frac", "t_event"])
z = np.load(f"{O}/s1_acc_ckpt.npz")
rows = []
for k, a in zip(z["keys"], z["vals"]):
    pid, yr = int(k[0]), int(k[1])
    rec = {"pid": pid, "year_obs": yr,
           "age": SUM_MID[yr] - float(dd.loc[pid, "t_event"])}
    for bi, b in enumerate(["vh", "vv"]):
        if a[bi*4 + 1] > 0 and a[bi*4 + 3] > 0:
            pmean = a[bi*4 + 0] / a[bi*4 + 1]
            cmean = a[bi*4 + 2] / a[bi*4 + 3]
            rec[f"d{b}_db"] = float(10*np.log10(pmean) - 10*np.log10(cmean))
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_parquet(f"{D}/s1_pilot_rows.parquet")
print("rows:", len(df), "patches:", df.pid.nunique(), flush=True)


def fit_rec(d, val, tmax=21, min_bin=10):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    bins_ = np.arange(0.6, tmax + 1, 1.0)
    bc, bm, bs, bn = [], [], [], []
    for a_, b_ in zip(bins_[:-1], bins_[1:]):
        g = d[(d.age >= a_) & (d.age < b_)][val]
        if len(g) >= min_bin:
            bc.append(0.5*(a_+b_)); bm.append(g.mean())
            bs.append(g.std()/np.sqrt(len(g))); bn.append(len(g))
    if len(bc) < 5:
        return None
    bc_, bm_, bs_ = map(np.array, (bc, bm, bs))
    popt, pcov = curve_fit(lambda t, A, tau: A*np.exp(-t/tau), bc_, bm_,
                           p0=[bm_[0], 8.0], sigma=np.maximum(bs_, 1e-3),
                           absolute_sigma=True, maxfev=20000,
                           bounds=([-30, 0.3], [30, 60]))
    perr = np.sqrt(np.diag(pcov))
    return dict(A=float(popt[0]), tau=float(popt[1]), tau_se=float(perr[1]),
                n=int(len(d)),
                bins=dict(center=bc_.tolist(), mean=bm_.tolist(),
                          se=bs_.tolist(), n=bn))


out = {"n_rows": int(len(df)), "n_patches": int(df.pid.nunique())}
for b in ["vh", "vv"]:
    f_ = fit_rec(df, f"d{b}_db")
    out[f"fit_{b}"] = f_
    if f_:
        print(b, f"tau={f_['tau']:.1f}±{f_['tau_se']:.1f}  A={f_['A']:.2f} dB",
              flush=True)
json.dump(out, open(f"{O}/s1_pilot.json", "w"), indent=1)
print("STEP13E COMPLETE")
