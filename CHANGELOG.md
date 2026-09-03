# Changelog

## 1.1.0 — 2026-09-03

Pre-submission verification release. DOI for all versions: 10.5281/zenodo.22281543.

- **Data and outputs are now in the repository.** Version 1.0.0 as archived on Zenodo held the
  code and documentation only; `data/` and `outputs/` were missing, so the statement that every
  statistic can be recomputed from the shipped tables did not hold for that archive. It holds
  from this version.
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
  `make_f_strata.py` prints the quotient.
- Figure scripts synchronised with the manuscript: `make_workflow_fig.py` (larger labels with
  an overflow guard), `make_f2_buffering.py` (legend and colour bar left-aligned),
  `make_s1_fig.py` (legend no longer covers the data).
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
