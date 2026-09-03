"""Step 13i: merge per-date S1 accumulators (s1_acc_<date>.npz, written by
step13h_fast --solo) into the main checkpoint s1_acc_ckpt.npz by summing the
8-vector [p_sum, p_cnt, c_sum, c_cnt] x (vh, vv) per (patch, year)."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import glob, os
import numpy as np

O = f"{_TROOT}/outputs"
acc = {}
if os.path.exists(f"{O}/s1_acc_ckpt.npz"):
    z = np.load(f"{O}/s1_acc_ckpt.npz")
    for k, v in zip(z["keys"], z["vals"]):
        acc[(int(k[0]), int(k[1]))] = v.copy()
    print("base ckpt keys:", len(acc))
n = 0
for f in sorted(glob.glob(f"{O}/s1_acc_20*.npz")):
    z = np.load(f)
    for k, v in zip(z["keys"], z["vals"]):
        key = (int(k[0]), int(k[1]))
        if key in acc:
            acc[key] = acc[key] + v
        else:
            acc[key] = v.copy()
    print("merged", os.path.basename(f), len(z["keys"]))
    n += 1
ks = np.array([[k[0], k[1]] for k in acc], dtype=np.int64)
vs = np.stack(list(acc.values()))
np.savez(f"{O}/s1_acc_ckpt.npz", keys=ks, vals=vs)
import collections
print("merged files:", n, "total keys:", len(acc),
      "by year:", dict(collections.Counter(int(k[1]) for k in acc)))
