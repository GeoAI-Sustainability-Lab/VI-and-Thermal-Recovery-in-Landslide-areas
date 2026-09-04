# Changelog

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
