# Changelog

## 1.3.0 — 2026-09-23

Correction and extension release, prepared for the Ecological Indicators submission. DOI for all
versions: 10.5281/zenodo.22281543.

- **Corrected age axis.** `step08e_results2.py` no longer lists the pooled two-summer composite
  `c2425` among the epochs of the recovery long table. Its two summers already enter through `c24`
  and `c25`, so 1.2.0 counted 2024-2025 twice for every patch observed in those years (3,674 of
  42,245 landslide rows). `outputs/chrono2_long.parquet` and every file downstream of it are
  regenerated: `results2.json` (fits2, year1_by_agent), `recovery_clocks.json`,
  `strata_curves.json`, `morakot.json`, `within_patch.json`, `epoch_era.json`. The pooled
  constants move from 16.84 / 14.66 yr to 16.78 / 14.36 yr
  (ratio 1.149 to 1.168); no direction changes. The pooled
  composite remains in `data/patches_deltas2.parquet` for cross-sectional use and feeds the
  intact-forest buffering baseline only. `step08e` now merges its keys into an existing
  `results2.json` instead of overwriting it, so the step can be rerun after the later steps.
- **Canopy sensitivity with the within-slice elevation term.** `step37_narrowband_elev.py` recomputes
  the 250 m-slice contrast with elevation in the within-slice residualisation and writes
  `outputs/narrowband_elev.json` plus the key `narrow_band_elev` in `gradient_check.json`. The
  paper's main estimate is now -0.39 +/- 0.03 °C per 10 m; the value
  without the term, -0.48 +/- 0.03, is reported as a sensitivity.
- **New table-side steps and result files** (all reachable from `data/` and `outputs/`):
  `step31_class_dissipation.py` (`class_dissipation.json`, dissipated fraction of the initial anomaly
  by disturbance class), `step32_severity_match.py` (`severity_match.json`), `step33_patch_elevation.py`
  (`patch_elevation.json`), `step34_functional_form.py` (`functional_form.json`, seven candidate forms
  with QAICc), `step35_form_robustness.py` (`form_robustness.json`, tau ratio under an asymptote and the
  observed-bin ten-year fraction), `step36_modelfree_lag.py` (`modelfree_lag.json`, the thermal versus
  greenness lag without a fitted curve), `step38_net_anomaly_recovery.py` (`net_anomaly_recovery.json`,
  the anomaly relative to each patch's pre-event summer), `step39_within_patch_period.py`
  (`within_patch_period.json`, the within-patch coefficient with calendar-epoch fixed effects).
- `step18_morakot.py` carries the 26-year case series forward from the released `morakot.json` when
  `data/case_trajectories.parquet` is absent, so the cohort statistics can be regenerated on their own.
- **Pre-trend evidence of the event study.** `step20_eventstudy.py` keeps the bootstrap replicates
  of every lead coefficient (one weight vector serves every year of a resample, so the replicates
  are jointly distributed) and writes `lst_pretrend` and `ndvi_pretrend` into `eventstudy.json`: a
  Wald test of the six pre-event coefficients on their bootstrap covariance, the slope of a linear
  pre-trend through the reference summer and with a free intercept, the first-year estimate after
  either trend is removed, the largest year-to-year step among the pre-event coefficients and the
  first-year estimate under the relative-magnitude restriction of Rambachan and Roth (2023). The
  coefficients themselves are unchanged.
- **Common time span for the fitted and observed dissipated fraction.** `step35_form_robustness.py`
  evaluates the exponential over the same span as the observed bins, from the first observed bin to
  the bin nearest ten years (`dissipated_pct_exp_span`, `span_yr`), records for every class whether
  that value lies inside the observed-bin interval, and adds `without_2026`, the pooled landslide
  clocks refitted with every observation of the truncated 2026 summer removed.
- `step38_net_anomaly_recovery.py` adds `cohort_summary`, the patch count, row count and oldest
  observed age of every cohort with a pre-event summer.
- `reproduce.py` runs the new steps in stages 2 and 3 (30 steps in all) and adds their headline
  quantities to the closing table. `--quick` skips the four bootstrap-heavy steps.

## 1.2.0 — 2026-09-04

Data-and-reproduction release. DOI for all versions: 10.5281/zenodo.22281543.

- **Tables and outputs are in the repository.** The 1.1.0 commit shipped the JSON results and
  `event_codes2.json` but not the parquet tables, so the statement that every statistic can be
  recomputed did not hold for that archive either. From this version `data/` and `outputs/`
  are complete (about 32 MB).
- **No figures, no figure scripts.** Every `make_*.py` and every figure-only step
  (`step08b/c/f/h/i/k`, `step07`, `step07d`, `step07e`), `data/s2_case_scenes.csv` and the
  matplotlib dependency are gone. The repository is the data and the analysis chain only.
- **`reproduce.py`** replaces `verify_numbers.py`. It rebuilds every tier-A result from the
  tables in the order of the paper's Methods section (four stages, 22 steps), in a work copy
  that leaves the repository untouched, and compares every regenerated file with the released
  copy key by key and row by row, not just a list of headline values. Every file reproduces
  identically on the release environment.
- Quantities that used to be computed only inside figure scripts or raster steps now have
  table-side steps, so the whole reported set is reachable from `data/`:
  `step29_buffering_model.py` (band slopes, gradient boosting, SHAP, counterfactual, and the
  local slopes of the binned canopy curve that `make_f2_buffering.py` used to write),
  `step30_quality_checks.py` (compositing bias, day–night amplitude, canopy-height
  cross-check), `step24b_l57_transfer_stats.py` (Landsat 5/7 transfer statistics),
  `step25b_harshness_stats.py` (harshness elasticity, from the new
  `data/epoch_harshness.json`), and `step13g_s1_analysis.py` now also produces the pooled
  Sentinel-1 fits.
- `outputs/results.json` now holds the buffering baseline only; the keys of the superseded
  single-epoch database (`recovery_fits`, `did`, `placebo_dlst`, `hansen_cohorts_c2425`,
  `rate_model`, …), together with `step08a_results.py`, `chronosequence_long.parquet`,
  `recovery_rates.parquet` and `shap_recovery_rate.parquet`, are removed; none of them is
  quoted in the paper.
- `outputs/s3_results.json` is regenerated from the shipped cell table; the values move in the
  seventh significant digit relative to the raster-side computation (float32 storage) and are
  identical at the precision reported.
- `data/acc/s2020_items.json`, a superseded partial run for 2020, is removed; `c20_items.json`
  is the production scene list. Fourteen epoch files, 792 scenes.
- README rewritten around the workflow: every step is listed with the paper section it
  implements, what it reads, what it writes and which reported quantities it yields.

## 1.1.0 — 2026-09-03

Pre-submission verification release.

- New `code/step28_descriptive_meta.py` and `outputs/grid_meta.json`: every descriptive count
  quoted in the text is computed from the tables (catalogue dating-quality and trigger
  breakdown, the four intact-forest sample populations, the binned canopy-curve anchors, the
  residual lapse rate inside a 250 m slice, the SHAP display subset, and a mixed-source
  recovery fit). Previously typed values that were wrong: canonical / bracket-midpoint dating
  counts 15,471 / 16,656 (correct 15,475 / 16,652); earthquake events "seven" (ten distinct
  dates); the intact-forest pixel count quoted as 397,978 or 399,990 for the 250 m-slice
  analysis (400,000 drawn, 399,990 in the six 500 m bands, 399,265 after the slice
  thresholds, 394,865 plotted); Landsat 7-to-8 transfer r "0.76–0.78" (0.75–0.78).
- New `data/event_codes2.json`: the harmonised catalogue summary (event name, year, trigger,
  harmonised date, dating quality, era, polygon count). No geometry; the catalogue itself is
  still withheld.
- τ ratio reported as the quotient of the two fitted τ, verifiable from the tables; the
  patch-level bootstrap supplies only the interval and the P(ratio > 1) star. The bootstrap
  mean of the ratio (still stored as `ratio.mean` in the JSON files) was previously printed
  instead, which for the heavy-rainfall stratum read 1.44 against a quotient of 1.36.
- `verify_numbers.py` also runs step 28 and checks the sample populations.

## 1.0.0 — 2026-08-31

First public release.

- Full pipeline, `step00` through `step27`, plus the figure scripts.
- Model-ready tables extracted from the source datasets, every coordinate column removed.
- `verify_numbers.py` recomputes the headline statistics from those tables and compares
  them with the released values; all 19 quantities match bit for bit.
- Unified 2004–2025 disturbance database: 43,780 catalogue records reduced to 23,526
  analysed patches with cross-year re-disturbance censoring.
- Fourteen Landsat 8/9 summer composites, 2013–2026, 792 scenes; the scene list for every
  epoch is in `data/acc/*_items.json`.
- No imagery and no rendered figures are stored. Source archives are documented in
  `README.md` and `outputs/DATA_PROVENANCE.json` with URLs, retrieval times and checksums.
- The landslide catalogue and every file carrying patch pixel membership or grid indices
  are withheld; see the data section of `README.md`.
