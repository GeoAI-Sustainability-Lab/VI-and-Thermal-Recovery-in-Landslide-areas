"""Step 9: data provenance manifest — every source, URL, access time, counts,
checksums. Written for auditability (no unverifiable claims)."""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json, hashlib, datetime, subprocess
import numpy as np
import pandas as pd

D = f"{_TROOT}/data"
O = f"{_TROOT}/outputs"

def sha256(path, cap_mb=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk); n += len(chunk)
            if cap_mb and n > cap_mb << 20:
                return f"partial-{cap_mb}MB:" + h.hexdigest()
    return h.hexdigest()

man = {"generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
       "python": sys.version.split()[0]}
try:
    import rasterio, geopandas, shap, sklearn, pystac_client
    man["packages"] = {m.__name__: m.__version__ for m in
                       [rasterio, geopandas, shap, sklearn, np, pd, pystac_client]}
except Exception:
    pass

man["sources"] = [
    dict(name="Landsat Collection 2 Level-2 (LST ST_B10/ST_B6, SR red/nir, QA_PIXEL)",
         provider="USGS via Microsoft Planetary Computer STAC",
         collection="landsat-c2-l2",
         url="https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2",
         license="Public domain (USGS)"),
    dict(name="Sentinel-2 L2A true-colour (case chips)",
         provider="ESA Copernicus via Planetary Computer", collection="sentinel-2-l2a",
         url="https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a",
         license="Copernicus Sentinel data terms"),
    dict(name="Sentinel-3 SLSTR Level-2 LST (day/night swaths)",
         provider="ESA Copernicus via Planetary Computer",
         collection="sentinel-3-slstr-lst-l2-netcdf",
         url="https://planetarycomputer.microsoft.com/dataset/sentinel-3-slstr-lst-l2-netcdf",
         license="Copernicus Sentinel data terms"),
    dict(name="Copernicus DEM GLO-30", provider="ESA/Airbus via Planetary Computer",
         collection="cop-dem-glo-30",
         url="https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30",
         license="Copernicus DEM licence"),
    dict(name="ESA WorldCover 2021 v200", provider="ESA via Planetary Computer",
         collection="esa-worldcover",
         url="https://planetarycomputer.microsoft.com/dataset/esa-worldcover",
         license="CC-BY 4.0"),
    dict(name="ETH Global Canopy Height 10 m (2020), tiles N21E120 & N24E120",
         provider="Lang et al. 2023, ETH Zurich",
         url="https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt",
         reference="Lang, Jetz, Schindler, Wegner (2023) Nat. Ecol. Evol., doi:10.1038/s41559-023-02206-6",
         license="CC-BY 4.0"),
    dict(name="Hansen Global Forest Change 2024 v1.12 (lossyear, treecover2000), tile 30N_120E",
         provider="Hansen/UMD/Google/USGS/NASA",
         url="https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/",
         reference="Hansen et al. (2013) Science 342:850-853", license="CC-BY 4.0"),
    dict(name="ARDSWC 民國113年 事件型崩塌目錄 (Event_Inventory_2024_ARDSWC_V7)",
         provider="農業部農村發展及水土保持署 (user's local Dataset)",
         path="Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2024_113",
         note="8,315 polygons; per-event attribution incl. 凱米/康芮/山陀兒/0403花蓮地震"),
    dict(name="ARDSWC 民國114年 事件型崩塌目錄 (Event_Inventory_2025_ARDSWC_V2)",
         provider="農業部農村發展及水土保持署 (user's local Dataset)",
         path="Dataset/01_SOURCE/Hazard_Events/Landslide_Taiwan/2025_114",
         note="2,674 polygons; incl. 0728豪雨/樺加沙/丹娜絲/0121嘉義地震"),
    dict(name="惠蓀 2021 火燒範圍 (火燒杜鵑嶺 polygon, 41.2 ha)",
         provider="user's local Dataset (KMZ-derived shapefile)",
         path="Dataset/01_SOURCE/Hazard_Events/Huisun_fire_and_Taiwania_range",
         note="fire date 2021-05 (media-verified 2021-05-15 reports); "
              "50號1/50號2 polygons EXCLUDED (possible Taiwania planting compartments)"),
    dict(name="臺灣三類林型 20 m 分類 (taiwan_foresttype_20m.tif)",
         provider="user's local Dataset (GEE classification)",
         path="Dataset/01_SOURCE/Earth_Observation/Taiwan_forest_local_rasters",
         note="class semantics UNCONFIRMED (0/1/2/3) — used as auxiliary only; "
              "class 2 has conifer-like elevation profile (median 2,024 m)"),
]

# item usage from step01 logs
usage = {}
for key in ["s2023", "s2024", "s2025", "s2026"]:
    p = f"{D}/acc/{key}_items.json"
    if os.path.exists(p):
        meta = json.load(open(p))
        usage[key] = dict(items_total=len(meta),
                          items_used=sum(1 for m in meta if m.get("used")))
man["landsat_item_usage"] = usage

files = {}
for root, sub in [(f"{D}/raw", True), (D, False), (O, False)]:
    if not os.path.isdir(root):
        continue
    for fn in sorted(os.listdir(root)):
        p = os.path.join(root, fn)
        if os.path.isfile(p) and (sub or fn.endswith((".tif", ".parquet", ".json", ".npz"))):
            files[os.path.relpath(p, f"{_TROOT}")] = dict(
                bytes=os.path.getsize(p), sha256=sha256(p, cap_mb=64))
man["files"] = files

json.dump(man, open(f"{O}/DATA_PROVENANCE.json", "w"), ensure_ascii=False, indent=1)
print("provenance written:", len(files), "files")
