"""Step 3d: repair the 2004 inventory's Events strings.
The 2004 DBF stores UTF-8 text but its .cpg declares Big5, so the original
read produced mojibake and agent fell to 'unknown'. Recover the true strings
by byte-level DBF read, patch event_codes2.json and both patch tables
(event + agent -> typhoon_rain), then refresh the agent-dependent statistics
in results2.json (typhoon fits, tau_ratios.typhoon) and event_groups.
Dating quality is unchanged (yearmid).
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import json, struct
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"
P = ("/mnt/user-data/uploads/文章發想與實踐/Projects/森林擾動/main/"
     "phase19_event_catalog_audit/source_archives/93年事件型崩塌目錄/"
     "Event_Inventory_2004_SWCB.dbf")

b = open(P, "rb").read()
n_rec = struct.unpack("<I", b[4:8])[0]
hdr = struct.unpack("<H", b[8:10])[0]
rlen = struct.unpack("<H", b[10:12])[0]
fields, off = [], 32
while b[off] != 0x0D:
    name = b[off:off+11].split(b"\0")[0].decode("ascii", "replace")
    fields.append((name, b[off+16])); off += 32
pos, starts = 1, {}
for name, flen in fields:
    starts[name] = (pos, flen); pos += flen
s, l = starts["Events"]
names = set()
for i in range(n_rec):
    raw = b[hdr + i*rlen: hdr + (i+1)*rlen][s:s+l].rstrip(b" \x00")
    if raw:
        names.add(raw.decode("utf-8"))
names = sorted(names, key=len)
print("recovered:", names)
# map by polygon count: code with n=980 -> shorter list? verify by n
ev = json.load(open(f"{D}/event_codes2.json"))
garbled = {k: v for k, v in ev.items() if v.get("year") == 2004}
assert len(garbled) == 2 and len(names) == 2
# count polygons per recovered name to assign correctly
cnt = {}
for i in range(n_rec):
    raw = b[hdr + i*rlen: hdr + (i+1)*rlen][s:s+l].rstrip(b" \x00")
    if raw:
        cnt[raw.decode("utf-8")] = cnt.get(raw.decode("utf-8"), 0) + 1
print("polygon counts:", cnt)
by_n = {v["n"]: k for k, v in garbled.items()}
mapping = {}
for nm, c in cnt.items():
    code = by_n.get(c)
    assert code is not None, (nm, c, by_n)
    mapping[code] = nm
old_new = {}
for code, nm in mapping.items():
    old_new[ev[code]["event"]] = nm
    ev[code]["event"] = nm
    ev[code]["agent"] = "typhoon_rain"
json.dump(ev, open(f"{D}/event_codes2.json", "w"), ensure_ascii=False, indent=1)
print("codes patched:", mapping)

for pq in ["patches_raw2.parquet", "patches_deltas2.parquet"]:
    df = pd.read_parquet(f"{D}/{pq}")
    m = (df.get("src") == "event") & (df.year == 2004)
    if "event" in df.columns:
        df.loc[m, "event"] = df.loc[m, "event_code"].astype(int).astype(str).map(
            {k: v for k, v in mapping.items()})
        df.loc[m, "agent"] = "typhoon_rain"
        df.to_parquet(f"{D}/{pq}")
        print(pq, "patched rows:", int(m.sum()))

# ---- refresh agent-dependent stats in results2.json ----
p = pd.read_parquet(f"{D}/patches_deltas2.parquet")
p["clean"] = (p.redist_frac < 0.05) & (p.n_ctrl >= 30) & (p.forest2000_frac > 0.6)
EPOCH_MID = {"c20": 2020.62, "c21": 2021.62, "c22": 2022.62, "c23": 2023.62,
             "c24": 2024.62, "c25": 2025.62, "c26": 2026.55, "c2425": 2025.12}
rows = []
for tag, mid in EPOCH_MID.items():
    cl, cn = f"dlst_{tag}", f"dndvi_{tag}"
    if cl not in p.columns:
        continue
    sub = p[p[cl].notna()].copy()
    sub["age"] = mid - sub.t_event
    sub["dlst"] = sub[cl]
    sub["dndvi"] = sub[cn] if cn in sub.columns else np.nan
    sub["is_pre"] = sub.pre_tag == tag
    rows.append(sub[["src", "agent", "age", "dlst", "dndvi", "clean", "is_pre"]])
L = pd.concat(rows, ignore_index=True)
post = L[(~L.is_pre) & L.clean & (L.age > 0.6)]
ev_ = post[post.src == "event"]


def fit_rec(d, val="dlst", tmax=22, min_bin=8):
    d = d[(d.age > 0.6) & (d.age <= tmax) & d[val].notna()]
    if len(d) < 80:
        return None
    bins = np.arange(0.6, tmax + 1, 1.0)
    bc, bm, bs, bn = [], [], [], []
    for a, b_ in zip(bins[:-1], bins[1:]):
        g = d[(d.age >= a) & (d.age < b_)][val]
        if len(g) >= min_bin:
            bc.append(0.5*(a+b_)); bm.append(g.mean())
            bs.append(g.std()/np.sqrt(len(g))); bn.append(len(g))
    if len(bc) < 5:
        return None
    bc_, bm_, bs_ = np.array(bc), np.array(bm), np.array(bs)
    popt, pcov = curve_fit(lambda t, A, tau: A*np.exp(-t/tau), bc_, bm_,
                           p0=[bm_[0], 6.0], sigma=np.maximum(bs_, 1e-3),
                           absolute_sigma=True, maxfev=20000,
                           bounds=([-25, 0.3], [25, 60]))
    perr = np.sqrt(np.diag(pcov))
    return dict(A=float(popt[0]), tau=float(popt[1]), A_se=float(perr[0]),
                tau_se=float(perr[1]),
                bins=dict(center=bc_.tolist(), mean=bm_.tolist(),
                          se=bs_.tolist(), n=bn))


R = json.load(open(f"{O}/results2.json"))
ty = ev_[ev_.agent == "typhoon_rain"]
newT = fit_rec(ty)
newG = fit_rec(ty, "dndvi")
oldT = R["fits2"]["agent_typhoon_rain_dlst"]["tau"]
R["fits2"]["agent_typhoon_rain_dlst"] = newT
R["fits2"]["agent_typhoon_rain_dndvi"] = newG
R["tau_ratios"]["typhoon"] = dict(t=newT["tau"], g=newG["tau"],
                                  ratio=newT["tau"]/newG["tau"])
R["fix2004"] = dict(recovered=names, n_polygons=int(n_rec),
                    note="2004 DBF stores UTF-8 but .cpg says Big5; repaired at byte level")
json.dump(R, open(f"{O}/results2.json", "w"), ensure_ascii=False, indent=1)
print(f"typhoon tau: {oldT:.2f} -> {newT['tau']:.2f} (g {newG['tau']:.2f}, "
      f"ratio {newT['tau']/newG['tau']:.2f})")
print("STEP3D COMPLETE")
