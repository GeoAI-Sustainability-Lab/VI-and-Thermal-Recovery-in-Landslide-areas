"""Step 3c: unified multi-year event patch database (民國93-114 / 2004-2025).

- Harmonizes 22 annual SWCB/ARDSWC event landslide inventories + Huisun fire.
- Per-polygon event date: canonical single-event date when unambiguous (validated
  against the Before/After image bracket), else bracket midpoint; bracket width
  kept as dating uncertainty.
- Agent from event string: 地震->earthquake, 颱風->typhoon_rain, 豪雨/熱帶低壓->
  rainfall, 堰塞湖->other.
- Cross-year censoring: last_dist_year raster (events ∪ Hansen loss); patch
  redist_frac = share of its pixels disturbed again LATER.
- Hansen chronosequence patches rebuilt EXCLUDING all event polygons (no double
  counting).  All patches store their pixel lists for fast delta extraction.
Outputs: data/patches_raw2.parquet, data/event_codes2.json,
         data/event_any30.tif, data/last_dist_year30.tif, data/intact2_30.tif
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json, time, warnings, re, datetime
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import features as rfeat
from scipy import ndimage as ndi

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, TRANSFORM, W, H, write_grid, read_grid

U = "/mnt/user-data/uploads/文章發想與實踐"
B = f"{U}/Projects/森林擾動/main/phase19_event_catalog_audit/source_archives"
D = f"{_TROOT}/data"
t0 = time.time()

YEAR_FILES = {
 2004: f"{B}/93年事件型崩塌目錄/Event_Inventory_2004_SWCB.shp",
 2005: f"{B}/94年事件型崩塌目錄/Event_Inventory_2005_SWCB.shp",
 2006: f"{B}/95年事件型崩塌目錄/95年事件型崩塌目錄_UTF8.shp",
 2007: f"{B}/96年事件型崩塌目錄/96年事件型崩塌目錄_UTF8.shp",
 2008: f"{B}/97年事件型崩塌目錄/97年事件型崩塌目錄.shp",
 2009: f"{B}/98年事件型崩塌目錄/98年事件型崩塌目錄.shp",
 2010: f"{B}/99年事件型崩塌目錄/Event_Inventory_2010_SWCB.shp",
 2011: f"{B}/100年事件型崩塌目錄/Event_Inventory_2011_SWCB.shp",
 2012: f"{B}/101年事件型崩塌目錄/Event_Inventory_2012_SWCB.shp",
 2013: f"{B}/102年事件型崩塌目錄/Event_Inventory_2013_SWCB.shp",
 2014: f"{B}/103年事件型崩塌目錄/Event_Inventory_2014_SWCB.shp",
 2015: f"{B}/104年事件型崩塌目錄/Event_Inventory_2015_SWCB.shp",
 2016: f"{B}/105年事件型崩塌目錄/Event_Inventory_2016_SWCB.shp",
 2017: f"{B}/106年事件型崩塌目錄/Event_Inventory_2017_SWCB.shp",
 2018: f"{B}/107年事件型崩塌目錄_(UTF8)/Event_Inventory_2018_SWCB.shp",
 2019: f"{B}/108年事件型崩塌目錄_UTF8/Event_Inventory_2019_SWCB_UTF8.shp",
 2020: f"{B}/109年事件型崩塌目錄_UTF8/109年事件型崩塌目錄_UTF8.shp",
 2021: f"{B}/110年事件型崩塌目錄/Event_Inventory_2021_SWCB.shp",
 2022: f"{B}/111年事件型崩塌目錄_UTF8/111年事件型崩塌目錄_UTF8.shp",
 2023: f"{B}/112年事件型崩塌目錄_UTF8/Event_Inventory_2023_SWCB.shp",
 2024: f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2024_113/Event_Inventory_2024_ARDSWC_V7.shp",
 2025: f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2025_114/20260526_114年事件型崩塌目錄判釋成果/20260526_114年事件型崩塌目錄判釋成果/Event_Inventory_2025_ARDSWC_V2.shp",
}

# canonical dates for UNAMBIGUOUS single-event strings (landfall/peak, CWA records)
CANON = {
 (2009, "莫拉克颱風"): "2009-08-08", (2012, "蘇拉颱風"): "2012-08-01",
 (2013, "蘇力颱風"): "2013-07-12", (2013, "0602地震"): "2013-06-02",
 (2014, "麥德姆颱風"): "2014-07-23", (2014, "鳳凰颱風"): "2014-09-21",
 (2015, "蘇迪勒颱風"): "2015-08-08", (2015, "杜鵑颱風"): "2015-09-28",
 (2018, "0206花蓮地震"): "2018-02-06", (2018, "0823熱帶低壓"): "2018-08-23",
 (2019, "0418地震"): "2019-04-18", (2019, "利奇馬颱風"): "2019-08-09",
 (2019, "白鹿颱風"): "2019-08-24", (2021, "彩雲颱風"): "2021-06-04",
 (2021, "烟花颱風"): "2021-07-23", (2021, "璨樹颱風"): "2021-09-11",
 (2021, "0731豪雨暨盧碧颱風"): "2021-08-01", (2021, "圓規颱風暨1013豪雨"): "2021-10-13",
 (2022, "0918地震"): "2022-09-18", (2022, "0323地震"): "2022-03-23",
 (2023, "卡努颱風"): "2023-08-03", (2023, "海葵颱風"): "2023-09-03",
 (2023, "杜蘇芮颱風"): "2023-07-26", (2023, "小犬颱風"): "2023-10-04",
 (2024, "凱米颱風"): "2024-07-24", (2024, "0403花蓮地震"): "2024-04-03",
 (2024, "康芮颱風"): "2024-10-31", (2024, "山陀兒颱風"): "2024-10-03",
 (2024, "天兔颱風"): "2024-11-15",
 (2025, "0728豪雨"): "2025-07-28", (2025, "樺加沙颱風"): "2025-09-22",
 (2025, "丹娜絲颱風、薇帕颱風、0721豪雨"): "2025-07-06",
 (2025, "楊柳颱風"): "2025-08-13", (2025, "0121嘉義地震"): "2025-01-21",
 (2025, "鳳凰颱風"): "2025-11-12",
}
MMDD = re.compile(r"^(\d{2})(\d{2})")


def parse_dt(v):
    s = str(v).strip().replace("​", "")
    for fmt in ("%Y%m%d", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.date(*datetime.datetime.strptime(s[:10], fmt).timetuple()[:3])
        except Exception:
            continue
    return None


def dec_year(d):
    return d.year + (d.timetuple().tm_yday - 0.5) / 365.25


def agent_of(ev):
    ev = "" if ev is None else str(ev)
    if "地震" in ev:
        return "earthquake"
    if "颱風" in ev:
        return "typhoon_rain"
    if "豪雨" in ev or "熱帶低壓" in ev:
        return "rainfall"
    if "堰塞湖" in ev:
        return "other"
    return "unknown"


def event_date(year, ev, b_dt, a_dt):
    """(date, quality) — canonical / mmdd / bracket-midpoint."""
    ev = str(ev).strip().replace("​", "")
    d = None; q = "midpoint"
    if (year, ev) in CANON:
        d = datetime.date.fromisoformat(CANON[(year, ev)]); q = "canonical"
    elif "、" not in ev and "暨" not in ev:
        m = MMDD.match(ev)
        if m:
            mo, da = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 1 <= da <= 31:
                try:
                    d = datetime.date(year, mo, da); q = "mmdd"
                except ValueError:
                    d = None
    if d is not None and b_dt and a_dt:
        if not (b_dt - datetime.timedelta(days=45) <= d <= a_dt + datetime.timedelta(days=45)):
            d = None                      # canonical date outside bracket -> distrust
    if d is None:
        if b_dt and a_dt:
            d = b_dt + (a_dt - b_dt) / 2; q = "midpoint"
        else:
            d = datetime.date(year, 7, 15); q = "yearmid"
    return d, q


# ---------------- load & harmonize ----------------
frames = []
for yr, p in sorted(YEAR_FILES.items()):
    g = None
    for enc in ("utf-8", "cp950"):
        try:
            g = gpd.read_file(p, encoding=enc)
            evcol = "Events" if "Events" in g.columns else "Event_s_"
            if g[evcol].astype(str).str.contains("颱風|豪雨|地震|堰塞湖", regex=True).any():
                break
        except Exception:
            g = None
    assert g is not None, f"read fail {yr}"
    evcol = "Events" if "Events" in g.columns else "Event_s_"
    g = g[[evcol, "BeforeDate", "AfterDate", "geometry"]].rename(columns={evcol: "ev"})
    g["year"] = yr
    g = g.set_crs(CRS, allow_override=True)     # all are TWD97 TM2 variants
    frames.append(g)
    print(f"  {yr}: {len(g)}", flush=True)
allg = pd.concat(frames, ignore_index=True)
allg["ev"] = allg.ev.fillna("").astype(str)
print("total polygons:", len(allg), f"{time.time()-t0:.0f}s", flush=True)

# per-polygon dates
b_dt = allg.BeforeDate.map(parse_dt)
a_dt = allg.AfterDate.map(parse_dt)
res = [event_date(y, e, b, a) for y, e, b, a in zip(allg.year, allg.ev.astype(str), b_dt, a_dt)]
allg["t_event"] = [dec_year(d) for d, q in res]
allg["dq"] = [q for d, q in res]
allg["bracket_days"] = [(a - b).days if (a and b) else -1 for a, b in zip(b_dt, a_dt)]
allg["agent"] = allg.ev.astype(str).map(agent_of)
allg["era"] = np.where(allg.year <= 2017, "annual_swcb", "event_ardswc")
print("date quality:", allg.dq.value_counts().to_dict(), flush=True)
print("agents:", allg.agent.value_counts().to_dict(), flush=True)

# ---------------- event-string codes ----------------
key = list(zip(allg.year, allg.ev.astype(str).str.strip()))
uniq = sorted(set(key))
code_of = {k: i + 1 for i, k in enumerate(uniq)}
allg["code"] = [code_of[k] for k in key]
ev_meta = {}
for (yr, ev), c in code_of.items():
    sub = allg[(allg.year == yr) & (allg.ev.astype(str).str.strip() == ev)]
    ev_meta[c] = dict(year=int(yr), event=ev, agent=sub.agent.iloc[0],
                      t_event=float(sub.t_event.median()), n=int(len(sub)),
                      dq=sub.dq.iloc[0], era=sub.era.iloc[0])
json.dump(ev_meta, open(f"{D}/event_codes2.json", "w"), ensure_ascii=False)
print("event codes:", len(ev_meta), flush=True)

# ---------------- rasters: last disturbance year + any-event ----------------
lossyear = read_grid(f"{D}/hansen_lossyear30.tif")
tc2000 = read_grid(f"{D}/treecover2000_30.tif")
forest2000 = tc2000 >= 60
last_dist = np.zeros((H, W), np.int16)
sel = (lossyear >= 1) & (lossyear <= 24)
last_dist[sel] = 2000 + lossyear[sel].astype(np.int16)
event_any = np.zeros((H, W), bool)
year_buf = np.zeros((H, W), np.uint16)          # reusable

for yr in sorted(YEAR_FILES):
    sub = allg[allg.year == yr]
    shp = ((geom, 1) for geom in sub.geometry if geom is not None and not geom.is_empty)
    year_buf[:] = 0
    rfeat.rasterize(shp, out=year_buf, transform=TRANSFORM, default_value=1)
    m = year_buf == 1
    event_any |= m
    np.maximum(last_dist, np.where(m, yr, 0).astype(np.int16), out=last_dist)
    print(f"  raster {yr}: {int(m.sum())} px", flush=True)
# fire (2021-05-12)
fire = gpd.read_file(f"{U}/Dataset/01_SOURCE/Hazard_Events/Huisun_fire_and_Taiwania_range/2021火燒範圍/2021火燒範圍/2021火燒範圍.shp",
                     encoding="utf-8").to_crs(CRS)
fire = fire[fire.Name == "火燒杜鵑嶺"]
year_buf[:] = 0
rfeat.rasterize(((g, 1) for g in fire.geometry), out=year_buf, transform=TRANSFORM,
                default_value=1)
fire_m = year_buf == 1
np.maximum(last_dist, np.where(fire_m, 2021, 0).astype(np.int16), out=last_dist)

write_grid(f"{D}/event_any30.tif", event_any.astype(np.uint8), "uint8", 255)
write_grid(f"{D}/last_dist_year30.tif", last_dist, "int16", 0)

# refined intact control domain
wc = read_grid(f"{D}/worldcover2021.tif")
intact2 = forest2000 & (lossyear == 0) & (~event_any) & (~fire_m) & (wc == 10)
write_grid(f"{D}/intact2_30.tif", intact2.astype(np.uint8), "uint8", 255)
print("intact2 px:", int(intact2.sum()), f"{time.time()-t0:.0f}s", flush=True)

# ---------------- patch extraction with pixel lists ----------------
S8 = np.ones((3, 3), bool)
code_buf = np.zeros((H, W), np.uint16)
rows = []

def harvest(mask, src, year_const, code_arr=None, min_px=10, t_ev_const=None,
            extra=None):
    lab, n = ndi.label(mask, S8)
    if n == 0:
        return 0
    objs = ndi.find_objects(lab)
    sizes = np.bincount(lab.ravel())
    kept = 0
    for pid in range(1, n + 1):
        if sizes[pid] < min_px:
            continue
        sl = objs[pid - 1]
        m = lab[sl] == pid
        rr, cc = np.nonzero(m)
        r0, c0 = sl[0].start, sl[1].start
        rows_px = (rr + r0).astype(np.int32); cols_px = (cc + c0).astype(np.int32)
        code = int(np.bincount(code_arr[sl][m]).argmax()) if code_arr is not None else 0
        ld = last_dist[sl][m]
        f2 = float(forest2000[sl][m].mean())
        yr_ = int(year_const)
        redist = float((ld > yr_).mean())
        rec = dict(src=src, year=yr_, event_code=code, n_px=int(sizes[pid]),
                   area_ha=float(sizes[pid] * 0.09),
                   r0=int(r0), c0=int(c0), r1=int(sl[0].stop), c1=int(sl[1].stop),
                   row=float(rows_px.mean()), col=float(cols_px.mean()),
                   forest2000_frac=f2, redist_frac=redist,
                   px_rows=rows_px, px_cols=cols_px)
        if t_ev_const is not None:
            rec["t_event"] = float(t_ev_const)
        if extra:
            rec.update(extra)
        rows.append(rec)
        kept += 1
    return kept

# events per year (own-year mask; code raster for attribution)
for yr in sorted(YEAR_FILES):
    sub = allg[allg.year == yr]
    code_buf[:] = 0
    rfeat.rasterize(((g, c) for g, c in zip(sub.geometry, sub.code)
                     if g is not None and not g.is_empty),
                    out=code_buf, transform=TRANSFORM)
    k = harvest(code_buf > 0, "event", yr, code_arr=code_buf, min_px=10)
    print(f"  patches {yr}: {k}", flush=True)

# per-patch event fields from code
ev_map = {c: v for c, v in ev_meta.items()}
for r in rows:
    if r["src"] == "event":
        v = ev_map.get(r["event_code"], {})
        r["t_event"] = v.get("t_event", r["year"] + 0.5)
        r["agent"] = v.get("agent", "unknown")
        r["event"] = v.get("event", "?")
        r["dq"] = v.get("dq", "midpoint")
        r["era"] = v.get("era", "?")

# fire patch
harvest(fire_m, "fire", 2021, min_px=5, t_ev_const=2021.36,
        extra=dict(agent="fire", event="惠蓀2021火燒(杜鵑嶺)", dq="canonical",
                   era="event_ardswc"))

# hansen rebuilt, excluding ALL event polygons + fire
hmask = (lossyear >= 1) & (lossyear <= 23) & forest2000 & (~event_any) & (~fire_m)
for ly in range(1, 24):
    k = harvest(hmask & (lossyear == ly), "hansen", 2000 + ly, min_px=10,
                t_ev_const=2000 + ly + 0.5,
                extra=dict(agent="hansen_loss", event=f"loss{2000+ly}",
                           dq="yearmid", era="hansen"))
print("total patches:", len(rows), f"{time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.to_parquet(f"{D}/patches_raw2.parquet")
print(df.groupby("src").size().to_dict())
print("clean (redist<5%):", int((df.redist_frac < 0.05).sum()), "/", len(df))
print("STEP3C COMPLETE", f"{time.time()-t0:.0f}s")
