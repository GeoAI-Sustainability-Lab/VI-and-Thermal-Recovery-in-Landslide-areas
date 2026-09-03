"""Step 3: disturbance patch database on the 30 m grid.

Sources:
  A) Hansen GFC-2024 v1.12 lossyear (2001-2024) & treecover2000  (chronosequence)
  B) ARDSWC event landslide inventories 2024 (V7) & 2025 (V2)    (event-attributed)
  C) Huisun 2021 fire polygons                                    (named case)

Output: data/patches.parquet (one row per patch, with terrain/type/structure
attributes) + grid rasters hansen_lossyear30.tif, treecover2000_30.tif,
events30.tif (0 none / 1 ls2024 / 2 ls2025 / 3 huisun_fire).
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import Resampling
from rasterio.vrt import WarpedVRT
from rasterio import features as rfeat
from scipy import ndimage as ndi

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import CRS, TRANSFORM, W, H, write_grid, read_grid

U = "/mnt/user-data/uploads/文章發想與實踐"
RAW = f"{_TROOT}/data/raw"
OUT = f"{_TROOT}/data"
t0 = time.time()

# ---- Hansen -> grid ----
def hansen_to_grid(name, resampling):
    with rasterio.open(f"{RAW}/{name}") as src:
        with WarpedVRT(src, crs=CRS, transform=TRANSFORM, width=W, height=H,
                       resampling=resampling) as vrt:
            return vrt.read(1)

lossyear = hansen_to_grid("Hansen_lossyear_30N_120E.tif", Resampling.nearest)
tc2000 = hansen_to_grid("Hansen_treecover2000_30N_120E.tif", Resampling.average)
write_grid(f"{OUT}/hansen_lossyear30.tif", lossyear, "uint8", 255)
write_grid(f"{OUT}/treecover2000_30.tif", tc2000, "uint8", 255)
print("hansen gridded; loss px 2001-24:", int(((lossyear > 0) & (lossyear <= 24)).sum()),
      f"{time.time()-t0:.0f}s", flush=True)

# ---- event polygons -> grid ----
ls24 = gpd.read_file(f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2024_113/Event_Inventory_2024_ARDSWC_V7.shp")
ls25 = gpd.read_file(f"{U}/Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2025_114/20260526_114年事件型崩塌目錄判釋成果/20260526_114年事件型崩塌目錄判釋成果/Event_Inventory_2025_ARDSWC_V2.shp",
                     encoding="utf-8")
fire = gpd.read_file(f"{U}/Dataset/01_SOURCE/Hazard_Events/Huisun_fire_and_Taiwania_range/2021火燒範圍/2021火燒範圍/2021火燒範圍.shp",
                     encoding="utf-8").to_crs(CRS)
# keep only the unambiguous fire scar; '50號1/2' may be Taiwania planting compartments
fire = fire[fire.Name == "火燒杜鵑嶺"].copy()
assert len(fire) == 1, "expected exactly one 火燒杜鵑嶺 polygon"
ls24 = ls24.to_crs(CRS); ls25 = ls25.to_crs(CRS)
print("events loaded:", len(ls24), len(ls25), len(fire), flush=True)

# canonical event dates + agent classes (from ARDSWC Events field + CWA records)
EVENT_META = {
    "凱米颱風":   ("2024-07-24", "typhoon_rain"), "0403花蓮地震": ("2024-04-03", "earthquake"),
    "康芮颱風":   ("2024-10-31", "typhoon_rain"), "山陀兒颱風":  ("2024-10-03", "typhoon_rain"),
    "0629豪雨":  ("2024-06-29", "rainfall"),     "1023豪雨":   ("2024-10-23", "rainfall"),
    "1112豪雨":  ("2024-11-12", "rainfall"),     "天兔颱風":    ("2024-11-15", "typhoon_rain"),
    "0728豪雨":  ("2025-07-28", "rainfall"),     "樺加沙颱風":  ("2025-09-22", "typhoon_rain"),
    "丹娜絲颱風、薇帕颱風、0721豪雨": ("2025-07-06", "typhoon_rain"),
    "楊柳颱風":   ("2025-08-13", "typhoon_rain"), "0121嘉義地震": ("2025-01-21", "earthquake"),
    "0613豪雨":  ("2025-06-13", "rainfall"),     "鳳凰颱風":    ("2025-11-12", "typhoon_rain"),
    "1020豪雨":  ("2025-10-20", "rainfall"),
}
import json as _json
event_codes = {}
events = np.zeros((H, W), np.uint8)
code = 0
for gdf, src_tag in [(ls24, "ls2024"), (ls25, "ls2025")]:
    for ev, sub in gdf.groupby("Events"):
        ev_clean = ev.strip().replace("​", "")
        code += 1
        event_codes[code] = dict(event=ev_clean, src=src_tag,
                                 date=EVENT_META.get(ev_clean, (None, "unknown"))[0],
                                 agent=EVENT_META.get(ev_clean, (None, "unknown"))[1],
                                 n_polys=len(sub))
        shp = ((g, code) for g in sub.geometry if g is not None and not g.is_empty)
        rfeat.rasterize(shp, out=events, transform=TRANSFORM, default_value=code)
code += 1
event_codes[code] = dict(event="惠蓀2021火燒(杜鵑嶺)", src="fire", date="2021-05-12",
                         agent="fire", n_polys=len(fire))
shp = ((g, code) for g in fire.geometry if g is not None and not g.is_empty)
rfeat.rasterize(shp, out=events, transform=TRANSFORM, default_value=code)
FIRE_CODE = code
_json.dump(event_codes, open(f"{OUT}/event_codes.json", "w"), ensure_ascii=False, indent=1)
write_grid(f"{OUT}/events30.tif", events, "uint8", 0)
print("events rasterized:", {event_codes[c]['event']: int((events == c).sum())
      for c in event_codes}, f"{time.time()-t0:.0f}s", flush=True)

# ---- context layers ----
dem = read_grid(f"{OUT}/dem30.tif"); dem[dem == -9999] = np.nan
slope = read_grid(f"{OUT}/slope30.tif"); slope[slope == -9999] = np.nan
north = read_grid(f"{OUT}/northness30.tif"); north[north == -9999] = np.nan
wc = read_grid(f"{OUT}/worldcover2021.tif")
ftype = read_grid(f"{OUT}/foresttype30.tif")
chm = read_grid(f"{OUT}/chm_eth30.tif"); chm[chm == -9999] = np.nan

# domain: land & was-forest-in-2000-ish
forest2000 = tc2000 >= 60
intact = forest2000 & (lossyear == 0) & (events == 0) & (wc == 10)

# ---- patch labelling ----
STRUCT = np.ones((3, 3), bool)
rows = []

def add_patches(mask, src_tag, year_arr=None, year_const=None, min_px=10,
                event_arr=None):
    lab, n = ndi.label(mask, structure=STRUCT)
    if n == 0:
        return
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
        yr = int(year_const) if year_const is not None else int(np.bincount(year_arr[sl][m]).argmax())
        ecode = int(np.bincount(event_arr[sl][m]).argmax()) if event_arr is not None else 0
        d = dem[sl][m]; s = slope[sl][m]; no = north[sl][m]
        ft_m = ftype[sl][m]
        fcov = float(forest2000[sl][m].mean())
        rows.append(dict(
            src=src_tag, label_id=pid, year=yr, event_code=ecode,
            n_px=int(sizes[pid]), area_ha=float(sizes[pid] * 0.09),
            row=float(r0 + rr.mean()), col=float(c0 + cc.mean()),
            r0=int(r0), c0=int(c0), r1=int(sl[0].stop), c1=int(sl[1].stop),
            elev=float(np.nanmean(d)), slope=float(np.nanmean(s)),
            northness=float(np.nanmean(no)), forest2000_frac=fcov,
            ftype=int(np.bincount(ft_m[ft_m < 250]).argmax()) if (ft_m < 250).any() else -1,
        ))
        kept += 1
    print(f"  {src_tag}: {kept} patches >= {min_px}px (of {n} components)", flush=True)

# A) Hansen chronosequence patches (2001-2023 losses inside 2000-forest, outside event polys)
hmask_all = (lossyear >= 1) & (lossyear <= 23) & forest2000 & (events == 0)
add_patches(hmask_all, "hansen", year_arr=lossyear.astype(np.int32))

# B) event inventories: one patch set per source; event identity via event_arr
add_patches((events > 0) & (events < FIRE_CODE), "ardswc", year_const=0,
            min_px=10, event_arr=events.astype(np.int32))
add_patches(events == FIRE_CODE, "fire2021", year_const=2021, min_px=5)
print("total patches:", len(rows), f"{time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(rows)
df.loc[df.src == "hansen", "year"] = df.loc[df.src == "hansen", "year"] + 2000
# ardswc: year & agent from event code table
ev_year = {int(c): (int(v["date"][:4]) if v["date"] else 0) for c, v in event_codes.items()}
sel = df.src == "ardswc"
df.loc[sel, "year"] = df.loc[sel, "event_code"].map(ev_year)
df.to_parquet(f"{OUT}/patches_raw.parquet")

# ---- intact mask + surrounding canopy height for controls ----
write_grid(f"{OUT}/intact30.tif", intact.astype(np.uint8), "uint8", 255)
print("intact forest px:", int(intact.sum()), "->", round(intact.sum() * 0.09 / 100, 1), "kha")
print("STEP3 COMPLETE", f"{time.time()-t0:.0f}s")
