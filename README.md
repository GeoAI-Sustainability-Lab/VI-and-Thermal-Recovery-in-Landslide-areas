# VI and Thermal Recovery in Landslide Areas

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22281543.svg)](https://doi.org/10.5281/zenodo.22281543)

Version 1.3.0. Cite as `doi:10.5281/zenodo.22281543` (resolves to the latest version); see `CITATION.cff`.

Data and code for a satellite-scale test of whether land surface temperature (LST) recovers
as fast as greenness after forest disturbance, in the montane forests of Taiwan, 2013–2026.

---

## 1.3.0 additions and correction (2026-09-23)

- The recovery long table no longer includes the pooled two-summer composite `c2425`
  (`step08e_results2.py`); see CHANGELOG. Every recovery result file is regenerated.
- New table-side steps, all run by `reproduce.py`:

| Step | Reads | Writes | Reported in |
| --- | --- | --- | --- |
| `step31_class_dissipation.py` | `outputs/chrono2_long.parquet` | `outputs/class_dissipation.json` | Fig. 10, Section 3.4.2 |
| `step32_severity_match.py` | `outputs/chrono2_long.parquet` | `outputs/severity_match.json` | Section 3.4.2, Fig. S4b |
| `step33_patch_elevation.py` | `outputs/chrono2_long.parquet`, `data/buffering_sample.parquet` | `outputs/patch_elevation.json` | Section 2.4 |
| `step34_functional_form.py` | `outputs/chrono2_long.parquet` | `outputs/functional_form.json` | Supplementary S4, Table S4 |
| `step35_form_robustness.py` | `outputs/chrono2_long.parquet`, `outputs/class_dissipation.json` | `outputs/form_robustness.json` | Section 3.4.2, Fig. S5b |
| `step36_modelfree_lag.py` | `outputs/chrono2_long.parquet` | `outputs/modelfree_lag.json` | Section 3.3.1, Fig. S5a |
| `step37_narrowband_elev.py` | `data/buffering_sample.parquet`, `outputs/gradient_check.json` | `outputs/narrowband_elev.json`, key `narrow_band_elev` | Sections 2.6 and 3.1, Table S1 |
| `step38_net_anomaly_recovery.py` | `outputs/chrono2_long.parquet` | `outputs/net_anomaly_recovery.json` | Section 3.4.3, Fig. S6 |
| `step39_within_patch_period.py` | `outputs/chrono2_long.parquet`, `outputs/recovery_clocks.json` | `outputs/within_patch_period.json` | Supplementary S4 |

- `step20_eventstudy.py` now also writes `lst_pretrend` and `ndvi_pretrend` (Wald test of the
  pre-event coefficients on their bootstrap covariance, pre-trend slopes, the largest year-to-year
  step and the first-year estimate under the relative-magnitude bound; Section 3.2.1, Table S5).
- `step35_form_robustness.py` evaluates the exponential over the same span as the observed bins
  (`dissipated_pct_exp_span`) and adds the `without_2026` sensitivity (Fig. S5b, Table 3 note).
- `step38_net_anomaly_recovery.py` adds `cohort_summary` (Fig. S6 caption).

## 1. Quick start

```bash
git clone https://github.com/GeoAI-Sustainability-Lab/VI-and-Thermal-Recovery-in-Landslide-areas.git
cd VI-and-Thermal-Recovery-in-Landslide-areas
python3 -m pip install numpy pandas pyarrow scipy scikit-learn shap
python3 reproduce.py            # about 5 minutes; --quick skips the two bootstrap-heavy steps
```

The script does not modify the repository. It links `data/` and `code/`, copies `outputs/`
into `reproduced/`, deletes every result file that the chain regenerates, runs the chain,
and then compares every regenerated file with the released one, key by key and row by row.
Expect every file to read `identical` and every headline quantity `OK`; the full comparison
is written to `reproduced/report.txt`. Bootstraps and model fits are seeded, so the
confidence intervals reproduce exactly, not merely closely.

---

## 2. Workflow

Every estimate is defined as **a disturbed patch minus terrain-matched intact forest within
the same summer composite**, so interannual climate and long-term warming cancel in the
difference. The pipeline has two tiers.

**Tier A — from the tables shipped here.** Everything the paper reports. `reproduce.py`
runs these steps; each can also be run alone with `python3 code/<step>.py`, paths
resolving from `THERMAL_ROOT` (default: the repository root).

| Stage | Paper section | Step | Reads | Writes | Reported quantities |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.4 patches, controls, DiD | `step08e_results2.py` | `patches_deltas2.parquet` | `chrono2_long.parquet`, `results2.json` | patch counts, the long recovery table, cohort DiD |
| 1 | 2.4 | `step08g_did_agents.py` | `patches_deltas2.parquet` | `results2.json` | DiD by trigger agent, pre-event (placebo) contrasts, quoted ranges |
| 1 | 2.4 | `step08m_area_hysteresis.py` | `chrono2_long.parquet` | `results2.json` | ΔLST–ΔNDVI hysteresis, area strata |
| 1 | 2.4, Table 3 | `step08n_did_uniform.py` | `patches_deltas2.parquet` | `results2.json` | uniform-standard DiD by agent, elevation and area |
| 1 | 2.5 | `step08o_area_official.py` | `chrono2_long.parquet` | `results2.json` | recovery fits for the < 2 / 2–10 / ≥ 10 ha classes |
| 1 | 2.4 | `step08p_rebuild_stats.py` | both | `results2.json` | event groups, τ ratios by agent, re-disturbance counts |
| 1 | 2.4 event study | `step20_eventstudy.py` | `patches_deltas2.parquet` | `eventstudy.json` | seven pre-event and the post-event coefficients |
| 1 | 2.4, Table 2 | `step27_table2.py` | `patches_deltas2.parquet` | `table2_agents.json` | patch counts, areas and largest event per agent |
| 2 | 2.5 recovery clocks | `step16_recovery_clocks.py` | `chrono2_long.parquet` | `recovery_clocks.json` | τ_LST, τ_NDVI, patch-bootstrap ratio interval, milestone lags |
| 2 | 2.5 fixed effects | `step21_within_patch.py` | `chrono2_long.parquet`, `recovery_clocks.json` | `within_patch.json` | fraction of the pooled decline reproduced within patches |
| 2 | 2.5 strata | `step17_strata_curves.py` | `chrono2_long.parquet` | `strata_curves.json` | clocks by agent, elevation band, area class, inventory era |
| 2 | 2.5 robustness | `step22_epoch_era.py` | `chrono2_long.parquet` | `epoch_era.json` | clocks by epoch-coverage era |
| 2 | Supplementary | `step25b_harshness_stats.py` | `epoch_harshness.json`, `chrono2_long.parquet` | `harshness_check.json` | summer-harshness elasticity β |
| 2 | Supplementary S2 | `step13g_s1_analysis.py` | `s1_pilot_rows.parquet`, `patches_deltas2.parquet` | `s1_pilot.json` | Sentinel-1 structural clock, within-patch slopes |
| 2 | 3.3.3 Morakot | `step18_morakot.py` | `chrono2_long.parquet`, `patches_deltas2.parquet`, `case_trajectories.parquet`, `s1_pilot.json` | `morakot.json` | the largest single event |
| 2 | Supplementary S3 | `step24b_l57_transfer_stats.py` | `l57_deltas.parquet` | `l57_transfer.json` | Landsat 5/7 against 8 transfer, by patch size |
| 2 | Supplementary S3 | `step26_morakot_early.py` | `l57_deltas.parquet`, `l57_transfer.json` | `morakot_early.json` | Morakot ages 1–3 through the bridge |
| 3 | 2.6 buffering baseline | `step29_buffering_model.py` | `buffering_sample.parquet` | `results.json`, `shap_buffering.parquet`, `baseline_slope_by_height.json` | band slopes, gradient-boosting R², mean \|SHAP\|, counterfactual, local slopes of the binned curve |
| 3 | 2.6, Fig. S1 | `step15_gradient_check.py` | `buffering_sample.parquet` | `gradient_check.json` | canopy sensitivity per 10 m with support controlled, elevation trend |
| 3 | 2.1, 2.4, 2.6 | `step28_descriptive_meta.py` | `event_codes2.json`, `buffering_sample.parquet`, `chrono2_long.parquet`, `shap_buffering.parquet` | `grid_meta.json` | catalogue breakdown, sample populations, curve anchors, mixed-source fit |
| 4 | Supplementary S1 | `step30_quality_checks.py` | `medval_pixels.parquet`, `s3_daynight_cells.parquet`, `chm2_sample.parquet` | `median_validation.json`, `s3_results.json`, `chm2_compare.json` | compositing bias, day–night amplitude, canopy-height cross-check |

Two result files are not regenerated by tier A and are shipped as produced:
`abs_series.json` and `abs_series2.json`, the population-level absolute LST and NDVI series
of patches and their controls per cohort and epoch, which need the composites and the
patch pixel membership (tier B, `step12_abs_series.py`, `step12b_abs_series2.py`).
`grid_meta.json` carries two values that only the rasters can give, the grid envelope and
the forest-inventory polygon count; `step28` keeps them from the released file and says so.

**Tier B — rebuilding the tables from the source archives.** These steps stream the
rasters from the Microsoft Planetary Computer STAC API and write about 18 GB into `data/`.
Order matters; each step writes what the next one reads.

| Step | Paper section | Purpose | Needs |
| --- | --- | --- | --- |
| `step00_grid_masks.py`, `step00b_dem_fix.py` | 2.1 | common 30 m grid (EPSG:3826), DEM, slope, aspect, illumination, land cover | Copernicus DEM GLO-30, ESA WorldCover |
| `step02_canopy.py` | 2.2 | ETH 10 m canopy height resampled to the grid | ETH canopy height model |
| `step14_forest4_mask.py` | 2.1 | forest domain from the national inventory stocked-forest map | 4th National Forest Resource Inventory |
| `step01_landsat_composites.py`, `step19_hist_composites.py` | 2.3 | clear-sky summer LST and NDVI composites, 2020–2026 and 2013–2019 | Landsat C2 L2 |
| `step01b_median_validation.py` | S1 | mean-against-median compositing check on three subareas; its per-pixel arrays are shipped as `medval_pixels.parquet` | Landsat C2 L2 |
| `step03_events.py`, `step03c_events_multi.py`, `step03d_fix2004.py` | 2.4 | unified 2004–2025 disturbance patch database, catalogue plus Hansen annual loss → `event_codes2.json` | landslide catalogue (§7), Hansen GFC-2024 |
| `step04_buffering_sample.py` | 2.6 | intact-forest sample pixels → `buffering_sample.parquet` | steps 00–02, 14 |
| `step05_patch_deltas.py`, `step05c_deltas_multi.py` | 2.4 | patch minus terrain-matched control at every epoch → `patches_deltas2.parquet` | steps 01, 03, 19 |
| `step06_trajectories.py`, `step06b_replacements.py`, `step06c_extra_cases.py` | S1 | per-scene case series 2000–2026 → `case_trajectories.parquet` | Landsat C2 L2 |
| `step07b_s3_slstr.py` | S1 | Sentinel-3 SLSTR day and night composites; the valid cells are shipped as `s3_daynight_cells.parquet` | Sentinel-3 SLSTR |
| `step11_chm2_sample.py` | S1 | second canopy-height model at the sample pixels → `chm2_sample.parquet` | Meta/WRI CHMv2 |
| `step12_abs_series.py`, `step12b_abs_series2.py` | 3.2 | absolute LST and NDVI series of patches and controls → `abs_series*.json` | composites, patch membership |
| `step13_s1_pilot.py` … `step13i_merge.py` | S2 | Sentinel-1 RTC γ⁰ extraction → `s1_pilot_rows.parquet` | Sentinel-1 RTC |
| `step23_l57_composites.py`, `step24_l57_transfer.py` | S3 | Landsat 5/7 per-sensor composites and the paired deltas → `l57_deltas.parquet` | Landsat C2 L2 |
| `step25_harshness.py` | S | per-epoch intact-forest LST anomaly → `epoch_harshness.json` | composites, intact-forest raster |
| `step09_provenance.py` | — | regenerate `DATA_PROVENANCE.json` with fresh checksums | all of the above |

`data/acc/*_items.json` lists, for each of the 14 summer epochs, every Landsat scene
identifier that entered the composite with its clear-pixel count and whether it was used
(792 scenes). That list pins the composites scene for scene, so tier B can be reproduced
exactly rather than approximately.

---

## 3. What the analysis establishes

| Quantity | Value | Step | Result file |
| --- | --- | --- | --- |
| Summer daytime LST per +10 m of intact canopy, canopy-height support controlled | −0.48 ± 0.03 °C (1.96 × SE), saturating above 30 m | `step15` | `gradient_check.json` → `narrow_band.slope`, `.se` |
| Elevation trend of that sensitivity | p = 0.29 | `step15` | `gradient_check.json` → `narrow_band.trend_p` |
| Local slope of the pooled curve at 15 / 25 / 35 m canopy | −1.86 / −0.52 / −0.35 °C per 10 m | `step29` | `baseline_slope_by_height.json` |
| Gradient-boosting model, held-out R²; mean \|SHAP\| of elevation and canopy height | 0.80; 2.97 and 0.30 °C | `step29` | `results.json` → `gbm_r2_test`, `shap_mean_abs` |
| Disturbed patches analysed | 15,679 landslide + 7,847 annual-loss = 23,526 | `step08e` | `results2.json` → `n_event`, `n_hansen` |
| Patches entering the recovery fits | 12,963 | `step08e` | `results2.json` → `n_clean` |
| Net thermal shock, DiD, cohorts with ≥ 50 patches | +0.82 to +2.31 °C | `step08g` | `results2.json` → `text_ranges`, `did_cohorts` |
| Pre-event coefficients, six summers pooled | −0.04 °C (95% CI −0.10 to +0.02); jointly non-zero (Wald χ² 47.7, 6 d.f.), slope through the reference summer +0.001 °C/yr (−0.018 to +0.017) | `step20` | `eventstudy.json` → `lst_lead_pooled`, `lst_pretrend` |
| First-year shock under the relative-magnitude bound | at least +1.02 °C (95% CI 0.84–1.19) | `step20` | `eventstudy.json` → `lst_pretrend.jump1_bound_m1` |
| Dissipated fraction, first to tenth observed year | 46.5% observed (42.6–50.4) against 41.5% from the exponential over the same span | `step35` | `form_robustness.json` → `classes.landslide_all` |
| Thermal recovery constant τ_LST | 16.84 yr | `step16` | `recovery_clocks.json` → `thermal.tau` |
| Greenness recovery constant τ_NDVI | 14.66 yr | `step16` | `recovery_clocks.json` → `greenness.tau` |
| τ_LST / τ_NDVI | 1.15, patch-bootstrap 95% CI 1.09–1.20 | `step16` | quotient of the two `tau`; `ratio.ci` |
| Pooled decline reproduced within patches | 74% (greenness 82%) | `step21` | `within_patch.json` → `thermal.frac`, `greenness.frac` |
| τ ratio by scar size, < 2 ha / ≥ 10 ha | 1.05 (CI 0.98–1.15) / 1.45 (CI 1.26–1.75) | `step17` | `strata_curves.json` → `lt2`, `ge10` |
| τ_LST above 2,000 m | 23.6 yr | `step17` | `strata_curves.json` → `gt2000.thermal.tau` |
| Morakot cohort | 2,398 patches | `step18` | `morakot.json` → `n_patches` |
| Summer-harshness elasticity β | +0.017 °C per °C (95% CI −0.003 to +0.038) | `step25b` | `harshness_check.json` |
| Landsat 7 against 8 patch deltas, 2013 / 2014 | r = 0.75 / 0.78 | `step24b` | `l57_transfer.json` |
| Sentinel-1 VH clock, Morakot cohort excluded | 23.7 ± 5.5 yr | `step13g` | `s1_pilot.json` → `fit_vh_xmor` |
| Catalogue records by dating quality | 15,475 canonical / 1,224 month-day / 16,652 midpoint / 10,429 mid-year, of 43,780 | `step28` | `grid_meta.json` → `catalogue` |
| Intact-forest sample | 400,000 drawn / 399,265 in the 250 m-slice analysis | `step28` | `grid_meta.json` → `intact_sample` |
| Compositing bias, clear-sky mean minus per-pixel median | −0.67 / −2.44 / −0.46 °C on the three subareas | `step30` | `median_validation.json` |
| Landsat scenes / summer epochs | 792 / 14 (2013–2026) | acquisition | `data/acc/*_items.json` |

The τ ratio is reported as the quotient of the two fitted τ; the patch-level bootstrap
supplies its interval. `ratio.mean` in the JSON files is the bootstrap mean and is not
quoted.

---

## 4. Method in brief

Enough to interpret the shipped columns.

- **Grid.** Everything is resampled to a 30 m grid on TWD97 / TM2, EPSG:3826
  (7,064 × 12,923 pixels).
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
- **Recovery clock.** Exponential decay `A · exp(−age/τ)` fitted on the binned patch × epoch
  table; confidence intervals from a bootstrap resampling **patches**, not rows, so repeated
  observations of one patch do not inflate precision.
- **Difference-in-differences.** Post-event minus pre-event anomaly, which removes the
  standing warm bias of the sites where landslides occur.
- **Event study.** Coefficients on seven pre-event summers and the years after, relative to
  the summer before the event, which tests parallel trends instead of assuming them.

---

## 5. Data dictionary

### `data/patches_deltas2.parquet` — 23,527 rows, one per patch

`pid` internal patch identifier · `src` `event` or `hansen` · `year` · `event_code`,
`event` catalogue event · `agent` `typhoon_rain`, `rainfall`, `earthquake` or `hansen` ·
`era` catalogue era · `dq` dating quality · `t_event` event timestamp · `n_px` patch pixels ·
`area_ha` · `redist_frac` later re-disturbance fraction · `forest2000_frac` Hansen
tree-cover-2000 fraction · `elev`, `slope`, `northness` patch terrain · `n_ctrl`, `ctrl_elev`,
`ctrl_slope` control annulus · `pre_tag` pre-event epoch · paired anomalies at every epoch as
`dlst_c13` … `dlst_c26` and `dndvi_c13` … `dndvi_c26`, plus `dlst_c2425` / `dndvi_c2425`.
The 23,527th row is one fire-record patch that enters no statistic.

### `outputs/chrono2_long.parquet` — 254,686 rows, the recovery-fit table

One row per patch per epoch, written by `step08e` from the table above: `pid`, `src`,
`agent`, `era`, `dq`, `event`, `year` composite year, `age`, `dlst`, `dndvi`, `elev`,
`slope`, `area_ha`, `clean` passes the quality filter, `is_pre` observation precedes the
event, `epoch`, `n_px`, `redist_frac`.

### `data/buffering_sample.parquet` — 400,000 intact-forest pixels

`lst` summer LST °C · `ndvi` · `chm` canopy height m · `elev` m · `slope` degrees ·
`northness`, `eastness` aspect components · `cos_i` terrain illumination · `nclear` clear
observations · `ftype` forest type code

### `data/chm2_sample.parquet` — 30,000 pixels

The same columns plus `tile`, `chm2` (Meta/WRI CHMv2 height) and `chm2_nvalid`, for the
canopy-height cross-check.

### `data/event_codes2.json` — 208 harmonised catalogue event codes

Per event code: `event` name as recorded, `year`, `agent`, `t_event` harmonised decimal
date, `dq` dating quality (`canonical`, `mmdd`, `midpoint`, `yearmid`), `era` (`annual_swcb`
2004–2017, `event_ardswc` 2018 on) and `n` polygons. No geometry.

### `data/l57_deltas.parquet` — 23,447 rows

Landsat 5/7 bridge. Patch attributes plus `dlst_*` / `dndvi_*` and their coverage fractions
`covlst_*` / `covndvi_*` for each per-sensor epoch, tagged `l5` or `l7`.

### `data/case_trajectories.parquet` — 2,318 rows

Per-scene case series, 2000–2026, for eight named cases: `date`, `platform`, patch and
control means (`lst_p`, `lst_c`, `ndvi_p`, `ndvi_c`) with pixel counts, `case`, `t_event`,
`elev`, `area_ha`.

### `data/s1_pilot_rows.parquet` — 1,533 rows

Sentinel-1 structure pilot: `pid`, `year_obs`, `age`, `dvh_db`, `dvv_db` (patch minus control
γ⁰ backscatter, dB, VH and VV).

### `data/s3_daynight_cells.parquet` — 8,133 cells

Sentinel-3 SLSTR summer 2025 on a 0.02° grid: `day`, `night` LST °C, `chm`, `dem`,
`treefrac`, and the clear-observation counts `n_day`, `n_night`. Only cells with a day or
night value are kept; no cell indices.

### `data/medval_pixels.parquet` — 147,485 pixels

Compositing check on three 256 × 256 subareas (`subarea` = `west_lowland`, `central_mid`,
`east_steep`): `median` per-pixel median LST, `mean` clear-sky mean LST, `n` clear
observations, `prod` production-composite value.

### `data/epoch_harshness.json`

The per-epoch intact-forest summer LST anomaly (°C) behind the harshness check, with its
definition.

### `outputs/shap_buffering.parquet` — 6,000 held-out pixels

Model inputs and their SHAP values, one column pair per feature, written by `step29`.

### `outputs/*.json`

`results2.json` the aggregate statistics of the unified 2004–2025 patch database ·
`results.json` the buffering baseline (band slopes, model R², SHAP, counterfactual) ·
`baseline_slope_by_height.json` local slopes of the pooled canopy curve ·
`gradient_check.json` canopy-height sensitivity by elevation band and estimator ·
`recovery_clocks.json` pooled clocks, bootstrap ratio, milestone lags ·
`within_patch.json` patch fixed-effect decomposition ·
`strata_curves.json` stratified fits · `epoch_era.json` coverage-era robustness ·
`eventstudy.json` leads and lags · `table2_agents.json` per-trigger counts and largest
events · `morakot.json`, `morakot_early.json` the largest single event ·
`harshness_check.json` summer-harshness elasticity · `l57_transfer.json` cross-sensor
transfer · `s1_pilot.json` structure clock · `grid_meta.json` the descriptive counts quoted
in the text · `median_validation.json`, `chm2_compare.json`, `s3_results.json` validity
checks · `abs_series.json`, `abs_series2.json` absolute population series (tier B) ·
`DATA_PROVENANCE.json` sources, URLs, retrieval times and file checksums

---

## 6. Source datasets

None of the raster archives are stored here. Each is public and re-downloadable; the pipeline
streams them from the Microsoft Planetary Computer STAC API at
`https://planetarycomputer.microsoft.com/api/stac/v1`.

| Dataset | STAC collection or source | Use | Licence |
| --- | --- | --- | --- |
| Landsat Collection 2 Level-2 | `landsat-c2-l2` | LST (`ST_B10`/`ST_B6`), NDVI (SR red, nir), `QA_PIXEL` | USGS, public domain |
| Sentinel-3 SLSTR LST L2 | `sentinel-3-slstr-lst-l2-netcdf` | day–night amplitude, summer 2025 | ESA Copernicus terms |
| Sentinel-1 RTC | `sentinel-1-rtc` | γ⁰ structure pilot, descending relative orbit 105, 2023–2025 summers | ESA Copernicus terms |
| Copernicus DEM GLO-30 | `cop-dem-glo-30` | elevation, slope, aspect, illumination | ESA |
| ESA WorldCover 2021 | `esa-worldcover` | initial land-cover screen | CC BY 4.0 |
| ETH global canopy height 10 m, 2020 | Lang et al. 2023 | canopy structure | CC BY 4.0 |
| Meta / WRI CHMv2 1 m | public S3 COGs | canopy-height cross-check | CC BY 4.0 |
| Hansen Global Forest Change 2024 v1.12 | `lossyear`, `treecover2000` | annual forest loss, forest domain | CC BY 4.0 |
| 4th National Forest Resource Inventory stocked-forest map | Forestry and Nature Conservation Agency, Taiwan | forest domain | Taiwan Open Government Data Licence |
| Event-based landslide catalogue, 2004–2025 | Agency of Rural Development and Soil and Water Conservation, Taiwan | disturbance patches, trigger, date | request from the agency; see §7 |

`outputs/DATA_PROVENANCE.json` records, for every raster the analysis actually used, the
source, the access URL, the retrieval time and a SHA-256 checksum, so a rebuild can be
checked file by file.

---

## 7. What is not included, and why

The event-based landslide catalogue of the Agency of Rural Development and Soil and Water
Conservation, Ministry of Agriculture, Taiwan. Redistribution rights do not belong to this
study, so the catalogue must be requested from that agency. For the same reason every file
carrying patch pixel membership or grid indices has been withheld, and the released tables
keep the patch attributes while dropping every coordinate column. The patch identifier `pid`
is internal and carries no location.

Consequently tier B needs that catalogue, while tier A — every statistic reported — runs from
the tables shipped here.

---

## 8. Environment

Python 3.11. Tier A needs only numpy, pandas, pyarrow, scipy, scikit-learn and shap; the
geospatial stack and the STAC clients in `requirements.txt` are needed for tier B only.

All bootstraps set an explicit random seed. The recovery fits use
`scipy.optimize.curve_fit` with fixed initial values, and the gradient boosting uses
`sklearn.ensemble.HistGradientBoostingRegressor` with a fixed `random_state`. The released
values were produced on Linux with Python 3.11, numpy 2.4, pandas 3.0, scipy 1.17,
scikit-learn 1.8 and shap 0.51; a different BLAS or library version may move the last digit
of a bootstrap interval or of the gradient-boosting R², and `reproduce.py` prints both values
so any drift is visible rather than hidden.

---

## 9. Licence and citation

Code is MIT, see `LICENSE`. Data and derived results are CC BY 4.0, see `LICENSE-DATA`.
Third-party inputs keep the licences listed in §6. Citation details are in `CITATION.cff`.
