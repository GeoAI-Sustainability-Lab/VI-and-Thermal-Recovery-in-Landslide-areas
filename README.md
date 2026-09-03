# VI and Thermal Recovery in Landslide Areas

Data and code for a satellite-scale test of whether land surface temperature (LST) recovers
as fast as greenness after forest disturbance, in the montane forests of Taiwan, 2013–2026.

---

## 1. What the analysis establishes

Every estimate is defined as **a disturbed patch minus terrain-matched intact forest within
the same summer composite**, so interannual climate and long-term warming cancel in the
difference. What is measured is the functional gap relative to normal forest in the same year.

| Quantity | Value | Computed by | Result file |
| --- | --- | --- | --- |
| Summer daytime LST per +10 m of intact canopy | −0.48 ± 0.03 °C, saturating above 30 m | `step15_gradient_check.py` | `gradient_check.json` → `narrow_band.slope` |
| Elevation trend of that sensitivity, canopy-height support controlled | p = 0.29 | `step15_gradient_check.py` | `gradient_check.json` → `narrow_band.trend_p` |
| Net thermal shock, difference-in-differences | +0.82 to +2.31 °C | `step08n_did_uniform.py` | `results2.json` → `did_uniform` |
| Pre-event coefficients, seven summers pooled | −0.04 °C | `step20_eventstudy.py` | `eventstudy.json` → `lst_lead_pooled` |
| Thermal recovery constant τ_LST | 16.84 yr | `step16_recovery_clocks.py` | `recovery_clocks.json` → `thermal.tau` |
| Greenness recovery constant τ_NDVI | 14.66 yr | `step16_recovery_clocks.py` | `recovery_clocks.json` → `greenness.tau` |
| τ ratio | 1.147 (95% CI 1.089–1.205) | `step16_recovery_clocks.py` | `recovery_clocks.json` → `ratio` |
| Pooled thermal decline reproduced within patches | 74% | `step21_within_patch.py` | `within_patch.json` → `thermal.frac` |
| τ ratio by scar size, < 2 ha / ≥ 10 ha | 1.06 (CI spans 1) / 1.46 | `step17_strata_curves.py` | `strata_curves.json` |
| τ_LST above 2,000 m | 23.6 yr | `step17_strata_curves.py` | `strata_curves.json` → `gt2000` |
| Morakot cohort | 2,398 patches | `step18_morakot.py` | `morakot.json` |
| Per-trigger patch counts and largest single events | — | `step27_table2.py` | `table2_agents.json` |
| Robustness across epoch coverage eras | τ stable | `step22_epoch_era.py` | `epoch_era.json` |
| Disturbed patches analysed | 23,526 | `step05c_deltas_multi.py` | `results2.json` → `n_patches` |
| Landsat scenes / summer epochs | 792 / 14 (2013–2026) | acquisition | `data/acc/*_items.json` |

---

## 2. Repository layout

```
code/       the pipeline, step00 … step27, plus the figure scripts
data/       model-ready input tables extracted from the source datasets
outputs/    every statistic as JSON, plus the analysis tables and the provenance manifest
verify_numbers.py   recompute the statistics from the shipped tables and diff them
```

No imagery and no rendered figures are stored. Figure scripts are kept because they document
how each panel was produced; running one writes into a `figures/` directory that this
repository does not track.

---

## 3. Quick check, three minutes

```bash
python3 -m pip install -r requirements.txt
python3 verify_numbers.py --quick     # ~3 min
python3 verify_numbers.py             # ~10 min, adds the two slow steps
```

The script copies the released `outputs/*.json` aside, re-runs the analysis steps that need
only the tables in this repository, and compares each headline quantity with the released
value. Expect every line to read `OK`. Because the bootstraps are seeded, the confidence
intervals reproduce exactly, not merely closely.

本腳本先把釋出的結果檔複製到暫存區，再以本倉庫的表格重跑統計腳本，逐項比對。重抽皆設定
亂數種子，故連信賴區間都應逐位元相同。

---

## 4. Reproduction, tier A — from the tables shipped here

No downloads. These steps read only `data/` and `outputs/` and rewrite the corresponding
result file.

| Step | Reads | Writes | Runtime |
| --- | --- | --- | --- |
| `step15_gradient_check.py` | `buffering_sample.parquet` | `gradient_check.json`, `baseline_slope_by_height.json` | ~6 s |
| `step16_recovery_clocks.py` | `chrono2_long.parquet` | `recovery_clocks.json` | ~40 s |
| `step17_strata_curves.py` | `chrono2_long.parquet` | `strata_curves.json` | ~5 min |
| `step18_morakot.py` | `chrono2_long.parquet`, `patches_deltas2.parquet` | `morakot.json` | ~1 s |
| `step20_eventstudy.py` | `patches_deltas2.parquet` | `eventstudy.json` | ~1 s |
| `step21_within_patch.py` | `chrono2_long.parquet` | `within_patch.json` | ~26 s |
| `step22_epoch_era.py` | `chrono2_long.parquet` | `epoch_era.json` | ~4 min |
| `step27_table2.py` | `patches_deltas2.parquet` | `table2_agents.json` | ~1 s |
| `step08n_did_uniform.py` | `patches_deltas2.parquet` | `results2.json` (DiD keys) | ~5 s |
| `step08o_area_official.py` | `chrono2_long.parquet` | `results2.json` (area strata) | ~1 min |
| `step13g_s1_analysis.py` | `s1_pilot_rows.parquet` | `s1_pilot.json` | ~5 s |
| `step26_morakot_early.py` | `l57_deltas.parquet` | `morakot_early.json` | ~5 s |

Run any of them directly:

```bash
export THERMAL_ROOT="$PWD"        # optional; defaults to the parent of code/
python3 code/step16_recovery_clocks.py
```

Paths resolve from `THERMAL_ROOT`, defaulting to the repository root, so a plain
`git clone` runs with no editing.

`step24_l57_transfer.py` also reads the DEM raster, so it belongs to tier B; its result is
already in `outputs/l57_transfer.json`.

Two further quantities are recomputable from shipped tables with a few lines of pandas
rather than a dedicated step: the clear-sky mean against per-pixel median bias, from
`data/medval_pixels.parquet` grouped by `subarea`; and the day–night amplitude by canopy
class, from `data/s3_daynight_cells.parquet`.

---

## 5. Reproduction, tier B — from the original archives

To rebuild the tables themselves, the acquisition and compositing stages must be re-run.
They stream rasters from the Microsoft Planetary Computer STAC API and write about 18 GB
into `data/`. Order matters; each step writes what the next one reads.

| Step | Purpose | Needs |
| --- | --- | --- |
| `step00_grid_masks.py` | common 30 m grid (EPSG:3826), DEM, slope, aspect, land cover, forest domain | Copernicus DEM GLO-30, ESA WorldCover |
| `step00b_dem_fix.py` | repair the DEM mosaic with a running maximum | output of step00 |
| `step01_landsat_composites.py` | clear-sky summer LST and NDVI composites, 2020–2026 epochs | Landsat C2 L2 |
| `step19_hist_composites.py` | the same for 2013–2019, extending the stack backwards | Landsat C2 L2 |
| `step01b_median_validation.py` | mean-against-median compositing bias on three subtiles | Landsat C2 L2 |
| `step02_canopy.py` | ETH 10 m canopy height resampled to the 30 m grid | ETH canopy height model |
| `step14_forest4_mask.py` | replace the forest domain with the national inventory stocked-forest map | 4th National Forest Resource Inventory |
| `step03_events.py`, `step03c_events_multi.py`, `step03d_fix2004.py` | unified 2004–2025 disturbance patch database, 43,780 catalogue records plus Hansen annual loss | landslide catalogue (see §8), Hansen GFC-2024 |
| `step04_buffering_sample.py` | intact-forest sample pixels → `buffering_sample.parquet` | steps 00–02, 14 |
| `step05_patch_deltas.py`, `step05c_deltas_multi.py` | patch minus terrain-matched control at every epoch → `patches_deltas2.parquet` | steps 01, 03, 19 |
| `step08e_results2.py` | unified statistics and the long analysis table → `chrono2_long.parquet`, `results2.json` | step05c |
| `step06*`, `step07*` | case series and case chips | Landsat, Sentinel-2, Sentinel-3 |
| `step11_chm2_sample.py` | second canopy-height model for cross-checking → `chm2_sample.parquet` | Meta/WRI CHMv2 |
| `step13*` | Sentinel-1 structure pilot → `s1_pilot_rows.parquet` | Sentinel-1 RTC |
| `step23_l57_composites.py`, `step24_l57_transfer.py` | Landsat 5/7 bridge → `l57_deltas.parquet` | Landsat C2 L2 |
| `step25_harshness.py` | summer-harshness elasticity | intact-forest raster from step14 |
| `step09_provenance.py` | regenerate the provenance manifest with fresh checksums | all of the above |

`data/acc/*_items.json` lists, for each of the 14 summer epochs, every Landsat scene
identifier that entered the composite together with its clear-pixel count and whether it was
used. That list pins the composites scene for scene, so tier B can be reproduced exactly
rather than approximately.

---

## 6. Method in brief

Enough to interpret the shipped columns.

- **Grid.** Everything is resampled to a 30 m grid on TWD97 / TM2, EPSG:3826.
- **Summer composite.** For each year, clear-sky mean of Landsat 8/9 over 1 June – 30
  September, after removing cloud, cirrus, cloud shadow, snow and fill with the `QA_PIXEL`
  flags. Fourteen epochs, 2013–2026. Epoch codes in the column names are `c13` … `c26`;
  `c2425` is the pooled 2024–25 composite.
- **Patch.** A connected disturbance area from the catalogue or from Hansen annual loss,
  with its event date. Patches re-disturbed later are censored: `redist_frac` is the fraction
  of the patch disturbed again in a later year, and the recovery sample keeps `< 5%`.
- **Control.** Intact forest in a 120–900 m annulus around the patch, matched on elevation
  and slope, taken from the same summer composite. `n_ctrl` is the number of control pixels,
  `ctrl_elev` and `ctrl_slope` their means.
- **Anomaly.** `dlst` and `dndvi` are patch mean minus control mean. Zero means the patch is
  indistinguishable from intact forest that summer.
- **Age.** `age` is years since the event date at day precision, from the catalogue date and
  the composite mid-date.
- **Recovery clock.** Exponential decay `A · exp(−age/τ)` fitted on the patch × epoch table;
  confidence intervals from a block bootstrap resampling **patches**, not rows, so repeated
  observations of one patch do not inflate precision.
- **Difference-in-differences.** Post-event minus pre-event anomaly, which removes the
  standing warm bias of the sites where landslides occur.
- **Event study.** Coefficients on seven pre-event summers and the years after, relative to
  the summer before the event, which tests parallel trends instead of assuming them.

---

## 7. Data dictionary

### `outputs/chrono2_long.parquet` — 254,686 rows, the recovery-fit table

One row per patch per epoch.

`pid` patch identifier · `src` `event` or `hansen` · `agent` `typhoon_rain`, `rainfall`,
`earthquake` or `hansen` · `era` catalogue era · `dq` date quality · `event` event name ·
`year` composite year · `age` years since disturbance · `dlst` ΔLST, °C · `dndvi` ΔNDVI ·
`elev` m · `slope` degrees · `area_ha` patch area · `clean` passes the quality filter ·
`is_pre` observation precedes the event · `epoch` epoch code · `n_px` patch pixels ·
`redist_frac` later re-disturbance fraction

### `data/patches_deltas2.parquet` — 23,527 rows, one per patch

Patch attributes as above, plus `event_code`, `t_event` (event timestamp), `pre_tag`
(pre-event epoch), `forest2000_frac` (Hansen tree-cover-2000 fraction), `n_ctrl`,
`ctrl_elev`, `ctrl_slope`, and the paired anomalies at every epoch as `dlst_c13` … `dlst_c26`
and `dndvi_c13` … `dndvi_c26`, plus `dlst_c2425` / `dndvi_c2425`.

### `data/buffering_sample.parquet` — 400,000 intact-forest pixels

`lst` summer LST °C · `ndvi` · `chm` canopy height m · `elev` m · `slope` degrees ·
`northness`, `eastness` aspect components · `cos_i` terrain illumination · `nclear` clear
observations · `ftype` forest type code

### `data/chm2_sample.parquet` — 30,000 pixels

The same columns plus `tile`, `chm2` (Meta/WRI CHMv2 height) and `chm2_nvalid`, for the
canopy-height cross-check in `chm2_compare.json`.

### `data/l57_deltas.parquet` — 23,447 rows

Landsat 5/7 bridge. Patch attributes plus `dlst_*` / `dndvi_*` and their coverage fractions
`covlst_*` / `covndvi_*` for each per-sensor epoch, tagged `l5` or `l7`.

### `data/case_trajectories.parquet` — 2,318 rows

Per-scene case series, 2000–2026: `date`, `platform`, patch and control means
(`lst_p`, `lst_c`, `ndvi_p`, `ndvi_c`) with pixel counts, `case`, `t_event`, `elev`, `area_ha`.

### `data/s1_pilot_rows.parquet` — 1,533 rows

Sentinel-1 structure pilot: `pid`, `year_obs`, `age`, `dvh_db`, `dvv_db` (patch minus control
γ⁰ backscatter, dB, VH and VV).

### `data/s3_daynight_cells.parquet` — 8,133 cells

Sentinel-3 SLSTR summer 2025 on a 0.02° grid whose origin is 119.9 °E, 25.4 °N, 110 columns
by 180 rows: `day`, `night` LST °C, `chm`, `dem`, `treefrac`, and the clear-observation
counts `n_day`, `n_night`. Only cells with a day or night value are kept.

### `data/medval_pixels.parquet` — 147,485 pixels

Compositing check on three 256 × 256 subareas (`subarea` = `west_lowland`, `central_mid`,
`east_steep`): `median` per-pixel median LST, `mean` clear-sky mean LST, `n` clear
observations, `prod` product-count.

### `outputs/shap_buffering.parquet`, `outputs/shap_recovery_rate.parquet`

Model inputs and their SHAP values, one column pair per feature.

### `outputs/*.json`

`gradient_check.json` canopy-height sensitivity by elevation band and estimator ·
`recovery_clocks.json` pooled clocks, bootstrap ratio, milestone lags ·
`strata_curves.json` stratified fits by trigger, elevation and area ·
`eventstudy.json` leads and lags · `within_patch.json` patch fixed-effect decomposition ·
`morakot.json`, `morakot_early.json` the largest single event · `epoch_era.json` coverage-era
robustness · `table2_agents.json` per-trigger counts and largest events ·
`results.json`, `results2.json` the aggregate statistics · `s1_pilot.json` structure clock ·
`l57_transfer.json` cross-sensor transfer · `median_validation.json`, `chm2_compare.json`,
`harshness_check.json`, `s3_results.json` validity checks ·
`DATA_PROVENANCE.json` sources, URLs, retrieval times and file checksums

---

## 8. Source datasets

None of the raster archives are stored here. Each is public and re-downloadable; the pipeline
streams them from the Microsoft Planetary Computer STAC API at
`https://planetarycomputer.microsoft.com/api/stac/v1`.

| Dataset | STAC collection or source | Use | Licence |
| --- | --- | --- | --- |
| Landsat Collection 2 Level-2 | `landsat-c2-l2` | LST (`ST_B10`/`ST_B6`), NDVI (SR red, nir), `QA_PIXEL` | USGS, public domain |
| Sentinel-2 L2A | `sentinel-2-l2a` | case before/after panels; product identifiers in `data/s2_case_scenes.csv` | ESA Copernicus terms |
| Sentinel-3 SLSTR LST L2 | `sentinel-3-slstr-lst-l2-netcdf` | day–night amplitude, summer 2025 | ESA Copernicus terms |
| Sentinel-1 RTC | `sentinel-1-rtc` | γ⁰ structure pilot, descending relative orbit 105, 2023–2025 summers | ESA Copernicus terms |
| Copernicus DEM GLO-30 | `cop-dem-glo-30` | elevation, slope, aspect, illumination | ESA |
| ESA WorldCover 2021 | `esa-worldcover` | initial land-cover screen | CC BY 4.0 |
| ETH global canopy height 10 m, 2020 | Lang et al. 2023 | canopy structure | CC BY 4.0 |
| Meta / WRI CHMv2 1 m | public S3 COGs | canopy-height cross-check | CC BY 4.0 |
| Hansen Global Forest Change 2024 v1.12 | `lossyear`, `treecover2000` | annual forest loss, forest domain | CC BY 4.0 |
| 4th National Forest Resource Inventory stocked-forest map | Forestry and Nature Conservation Agency, Taiwan | forest domain | Taiwan Open Government Data Licence |
| Event-based landslide catalogue, 2004–2025 | Agency of Rural Development and Soil and Water Conservation, Taiwan | disturbance patches, trigger, date | request from the agency; see §9 |

`outputs/DATA_PROVENANCE.json` records, for every raster the analysis actually used, the
source, the access URL, the retrieval time and a SHA-256 checksum, so a rebuild can be
checked file by file. `data/acc/*_items.json` gives the Landsat scene list per epoch, and
`data/s2_case_scenes.csv` gives the twelve Sentinel-2 product identifiers with acquisition
dates, cloud fractions and chip corner coordinates.

---

## 9. What is not included, and why

The event-based landslide catalogue of the Agency of Rural Development and Soil and Water
Conservation, Ministry of Agriculture, Taiwan. Redistribution rights do not belong to this
study, so the catalogue must be requested from that agency. For the same reason every file
carrying patch pixel membership or grid indices has been withheld, and the released tables
keep the patch attributes while dropping every coordinate column. The patch identifier `pid`
is internal and carries no location.


Consequently tier B needs that catalogue, while tier A — every statistic reported — runs from
the tables shipped here.

---

## 10. Environment

Python 3.11. `pip install -r requirements.txt`. Tier A needs only numpy, pandas, pyarrow,
scipy, scikit-learn and shap; the geospatial stack and the STAC clients are needed for tier B.

All bootstraps set an explicit random seed, so repeated runs give identical intervals. The
recovery fits use `scipy.optimize.curve_fit` with fixed initial values, and the gradient
boosting uses `sklearn.ensemble.HistGradientBoostingRegressor` with a fixed `random_state`.
Results were produced on Linux with Python 3.11; minor last-digit differences are possible on
a different BLAS build, and `verify_numbers.py` reports the exact values so any drift is
visible rather than hidden.

---

## 11. Licence

Code is MIT, see `LICENSE`. Data and derived results are CC BY 4.0, see `LICENSE-DATA`.
Third-party inputs keep the licences listed in §8.

## 12. Citation

See `CITATION.cff`.
