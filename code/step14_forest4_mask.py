"""Step 14 (chunked): replace the forest domain with the 4th National Forest
Resource Inventory stocked-forest-land map (林地蓄積圖 1061, TWD97 TM2,
348,617 polygons). Domain = all TypeName except 待成林地. Streams the
shapefile in 40k-feature chunks (7 GB RAM container) and rasterizes into one
30 m buffer, then rebuilds intact30 / intact2_30 with the survey domain in
place of WorldCover==10 (all other terms identical to step03/step03c).
Old masks preserved as *_wc.tif.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import time, warnings, os, shutil
warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
import rasterio.features as rfeat
from pyogrio import read_dataframe, read_info
import sys
sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import read_grid, write_grid, TRANSFORM, H, W, CRS

D = f"{_TROOT}/data"
U = "/mnt/user-data/uploads/文章發想與實踐"
t0 = time.time()
SHP = f"{D}/forest4/forest4.shp"

info = read_info(SHP)
NF = info["features"]
print("features:", NF, flush=True)
buf = np.zeros((H, W), np.uint8)
CHUNK = 40_000
kept = 0
for off in range(0, NF, CHUNK):
    g = read_dataframe(SHP, columns=["TypeName"], force_2d=True,
                       on_invalid="fix", skip_features=off,
                       max_features=CHUNK)
    g = g[~g.TypeName.fillna("").str.contains("待成林")]
    kept += len(g)
    if len(g):
        rfeat.rasterize(((geom, 1) for geom in g.geometry
                         if geom is not None and not geom.is_empty),
                        out=buf, transform=TRANSFORM, default_value=1)
    del g
    print(f"  chunk {off//CHUNK}: kept so far {kept:,} {time.time()-t0:.0f}s",
          flush=True)

f4 = buf == 1
write_grid(f"{D}/forest4_30.tif", f4.astype(np.uint8), "uint8", 255)
print("forest4 px:", int(f4.sum()), "=", round(f4.sum()*0.09/1000, 1), "kha",
      f"{time.time()-t0:.0f}s", flush=True)

tc2000 = read_grid(f"{D}/treecover2000_30.tif")
forest2000 = tc2000 >= 60
del tc2000
lossyear = read_grid(f"{D}/hansen_lossyear30.tif")
events30 = read_grid(f"{D}/events30.tif")
event_any = read_grid(f"{D}/event_any30.tif") == 1
wc = read_grid(f"{D}/worldcover2021.tif")

fire = gpd.read_file(f"{U}/Dataset/01_SOURCE/Hazard_Events/Huisun_fire_and_Taiwania_range/2021火燒範圍/2021火燒範圍/2021火燒範圍.shp",
                     encoding="utf-8").to_crs(CRS)
fire = fire[fire.Name == "火燒杜鵑嶺"]
fb = np.zeros((H, W), np.uint8)
rfeat.rasterize(((geom, 1) for geom in fire.geometry), out=fb,
                transform=TRANSFORM, default_value=1)
fire_m = fb == 1

for f in ["intact30.tif", "intact2_30.tif"]:
    if not os.path.exists(f"{D}/{f.replace('.tif', '_wc.tif')}"):
        shutil.copy(f"{D}/{f}", f"{D}/{f.replace('.tif', '_wc.tif')}")

old1 = read_grid(f"{D}/intact30_wc.tif") == 1
old2 = read_grid(f"{D}/intact2_30_wc.tif") == 1

intact_new = forest2000 & (lossyear == 0) & (events30 == 0) & f4
intact2_new = forest2000 & (lossyear == 0) & (~event_any) & (~fire_m) & f4
write_grid(f"{D}/intact30.tif", intact_new.astype(np.uint8), "uint8", 255)
write_grid(f"{D}/intact2_30.tif", intact2_new.astype(np.uint8), "uint8", 255)

for name, old, new in [("intact30", old1, intact_new),
                       ("intact2_30", old2, intact2_new)]:
    inter = (old & new).sum()
    print(f"{name}: old {old.sum():,} -> new {new.sum():,} px; "
          f"overlap {inter/max(new.sum(),1)*100:.1f}% of new, "
          f"{inter/max(old.sum(),1)*100:.1f}% of old", flush=True)
wc10 = wc == 10
print(f"domain only: wc10 {int(wc10.sum()):,} vs f4 {int(f4.sum()):,}; "
      f"jaccard {(wc10 & f4).sum() / (wc10 | f4).sum()*100:.1f}%", flush=True)
print("STEP14 COMPLETE", f"{time.time()-t0:.0f}s")
