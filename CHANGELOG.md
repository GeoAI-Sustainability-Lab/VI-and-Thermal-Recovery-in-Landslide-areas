# Changelog

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
