# -*- coding: utf-8 -*-
"""Rebuild every reported statistic from the tables in this repository and
compare the result, value by value, with the released outputs.

以本倉庫釋出之表格，依論文方法節的順序重跑全部統計步驟，並將重算結果與釋出之
outputs/ 逐鍵比對。倉庫本身不被改寫：重算寫入 reproduced/outputs/，比對報告寫入
reproduced/report.txt。

    python3 reproduce.py            # all stages, about 15 minutes
    python3 reproduce.py --quick    # skips the two bootstrap-heavy steps, about 5 minutes
    python3 reproduce.py --stage 2  # one stage only (1-4), see STAGES below
    python3 reproduce.py --compare  # compare an existing reproduced/ again without re-running

Nothing here needs imagery. The stages follow Section 2 of the paper:
  1  disturbance patches, matched controls, event study and DiD   (Section 2.4)
  2  recovery trajectory, stratified clocks, Morakot, Sentinel-1  (Section 2.5, Supplementary S2-S3)
  3  intact-forest buffering baseline and descriptive counts      (Section 2.6, 2.1)
  4  compositing, day-night and canopy-height checks              (Supplementary S1)
"""
import json, math, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(ROOT, "reproduced")
QUICK = "--quick" in sys.argv
COMPARE_ONLY = "--compare" in sys.argv      # reuse an existing reproduced/ and only compare
ONLY = None
if "--stage" in sys.argv:
    ONLY = int(sys.argv[sys.argv.index("--stage") + 1])

# (script, files it (re)writes, slow)
STAGES = {
    1: ("Section 2.4  patches, controls, event study, DiD", [
        ("step08e_results2.py", ["results2.json", "chrono2_long.parquet"], False),
        ("step08g_did_agents.py", ["results2.json"], False),
        ("step08m_area_hysteresis.py", ["results2.json"], False),
        ("step08n_did_uniform.py", ["results2.json"], False),
        ("step08o_area_official.py", ["results2.json"], False),
        ("step08p_rebuild_stats.py", ["results2.json"], False),
        ("step20_eventstudy.py", ["eventstudy.json"], False),   # 1.3.1: also saves the joint covariance for step40
        ("step27_table2.py", ["table2_agents.json"], False),
    ]),
    2: ("Section 2.5  recovery clocks, strata, Morakot, Sentinel-1", [
        ("step16_recovery_clocks.py", ["recovery_clocks.json"], False),
        ("step21_within_patch.py", ["within_patch.json"], False),
        ("step17_strata_curves.py", ["strata_curves.json"], True),
        ("step22_epoch_era.py", ["epoch_era.json"], True),
        ("step25b_harshness_stats.py", ["harshness_check.json"], False),
        ("step13g_s1_analysis.py", ["s1_pilot.json"], False),
        ("step18_morakot.py", ["morakot.json"], False),
        ("step24b_l57_transfer_stats.py", ["l57_transfer.json"], False),
        ("step26_morakot_early.py", ["morakot_early.json"], False),
        # added in 1.3.0: class-level dissipated fraction, severity domains, elevation of the
        # analysed patches, functional-form comparison, model-free lag, pre-event baseline and the
        # period-effect sensitivity of the within-patch coefficient (Sections 2.5, 3.4, Supplementary S4)
        ("step31_class_dissipation.py", ["class_dissipation.json"], False),
        ("step32_severity_match.py", ["severity_match.json"], False),
        ("step33_patch_elevation.py", ["patch_elevation.json"], False),
        ("step34_functional_form.py", ["functional_form.json"], False),
        ("step35_form_robustness.py", ["form_robustness.json"], True),
        ("step36_modelfree_lag.py", ["modelfree_lag.json"], True),
        ("step38_net_anomaly_recovery.py", ["net_anomaly_recovery.json"], True),
        ("step39_within_patch_period.py", ["within_patch_period.json"], False),
    ]),
    3: ("Section 2.6  buffering baseline and descriptive counts", [
        ("step29_buffering_model.py", ["results.json", "shap_buffering.parquet",
                                       "baseline_slope_by_height.json"], False),
        ("step15_gradient_check.py", ["gradient_check.json"], False),
        # added in 1.3.0: the within-slice elevation term of the 250 m canopy sensitivity (Table S1);
        # it also writes the key narrow_band_elev into gradient_check.json
        ("step37_narrowband_elev.py", ["narrowband_elev.json", "gradient_check.json"], False),
        ("step28_descriptive_meta.py", ["grid_meta.json"], False),
    ]),
    4: ("Supplementary S1  compositing, day-night, canopy-height checks", [
        ("step30_quality_checks.py", ["median_validation.json", "s3_results.json",
                                      "chm2_compare.json"], False),
    ]),
    # added in 1.3.1: honest confidence sets for the first post-event effect (Table S6). The step
    # needs the honestdid package (a Python implementation of HonestDiD, with torch and cvxpy);
    # when it is not installed the step is skipped and the released honest_did.json is kept.
    5: ("Section 3.2.1  honest confidence sets under the relative-magnitude restriction", [
        ("step40_honest_did.py", ["honest_did.json"], True),
    ]),
}

# steps that need an optional package: (module to import, message when absent)
OPTIONAL = {"step40_honest_did.py": ("honestdid", "honestdid not installed; pip install honestdid")}
# files compared with a looser tolerance because their values come from a numerical test
# inversion on a grid and simulated critical values (relative 1e-3 instead of 1e-9)
LOOSE_TOL = {"honest_did.json": 1e-3}

# Files a step reads back before rewriting (tier-B values or legacy keys are
# carried over from the released copy); every other regenerated file is deleted
# from the work copy first, so nothing released can leak into the comparison.
CARRIED = {"results.json", "grid_meta.json", "median_validation.json", "s3_results.json"}

# Headline quantities, printed as a table at the end.
HEADLINE = [
    ("gradient_check.json", "narrow_band_elev.with_elevation.slope", "canopy sensitivity, °C per 10 m (main estimate)"),
    ("gradient_check.json", "narrow_band_elev.with_elevation.se", "its standard error"),
    ("gradient_check.json", "narrow_band.slope", "canopy sensitivity without the elevation term"),
    ("gradient_check.json", "narrow_band_elev.with_elevation.trend_p", "elevation trend p"),
    ("baseline_slope_by_height.json", "15", "local slope at 15 m canopy, °C per 10 m"),
    ("results.json", "gbm_r2_test", "gradient-boosting test R²"),
    ("results.json", "shap_mean_abs.chm", "mean |SHAP| of canopy height, °C"),
    ("results2.json", "n_event", "landslide patches"),
    ("results2.json", "n_hansen", "annual-loss patches"),
    ("results2.json", "n_clean", "patches entering the recovery fits"),
    ("eventstudy.json", "lst_lead_pooled.mean", "pooled pre-event coefficient, °C"),
    ("eventstudy.json", "lst_pretrend.wald", "Wald statistic of the six pre-event coefficients"),
    ("eventstudy.json", "lst_pretrend.slope_ref", "pre-trend slope through the reference summer, °C/yr"),
    ("honest_did.json", "lst.by_Mbar.3.robust_cs.0", "honest CS lower bound, first-year shock, Mbar = 2, °C"),
    ("honest_did.json", "lst.breakdown_Mbar", "breakdown value of Mbar"),
    ("recovery_clocks.json", "thermal.tau", "tau_LST, yr"),
    ("recovery_clocks.json", "greenness.tau", "tau_NDVI, yr"),
    ("recovery_clocks.json", "ratio.ci.0", "tau-ratio bootstrap CI, lower"),
    ("recovery_clocks.json", "ratio.ci.1", "tau-ratio bootstrap CI, upper"),
    ("recovery_clocks.json", "n_patches", "patches in the recovery fit"),
    ("within_patch.json", "thermal.frac", "decline reproduced within patches"),
    ("within_patch_period.json", "thermal.frac_patch_epoch_fe", "same with epoch fixed effects"),
    ("class_dissipation.json", "landslide_all.dissipated.10", "landslide anomaly dissipated by 10 yr, %"),
    ("class_dissipation.json", "ls_ge10.dissipated.10", "same, scars ≥ 10 ha"),
    ("class_dissipation.json", "al_ge2.dissipated.10", "same, annual loss ≥ 2 ha"),
    ("form_robustness.json", "classes.landslide_all.model_free.dissipated_pct", "observed-bin dissipation, first to tenth year, %"),
    ("form_robustness.json", "classes.landslide_all.dissipated_pct_exp_span", "exponential over the same span, %"),
    ("form_robustness.json", "without_2026.ratio", "tau ratio without the 2026 summer"),
    ("modelfree_lag.json", "remaining_at_10yr.difference_thermal_minus_greenness.point", "model-free lag at 10 yr"),
    ("functional_form.json", "landslide_dndvi.models.exponential.d_qaicc", "ΔQAICc of the exponential, greenness"),
    ("net_anomaly_recovery.json", "pre_share_of_first_year", "pre-event share of the first-year anomaly"),
    ("strata_curves.json", "lt2.thermal.tau", "tau_LST, scars < 2 ha"),
    ("strata_curves.json", "ge10.thermal.tau", "tau_LST, scars ≥ 10 ha"),
    ("strata_curves.json", "gt2000.thermal.tau", "tau_LST above 2,000 m"),
    ("epoch_era.json", "all.thermal.tau", "tau_LST, all epochs"),
    ("morakot.json", "n_patches", "Morakot cohort patches"),
    ("table2_agents.json", "residual.n", "unclassified-trigger patches"),
    ("s1_pilot.json", "fit_vh_xmor.tau", "Sentinel-1 VH clock excl. Morakot, yr"),
    ("harshness_check.json", "beta", "summer-harshness elasticity beta"),
    ("l57_transfer.json", "L7_vs_L8_2013_lst.all.r", "L7 vs L8 delta correlation, 2013"),
    ("morakot_early.json", "cal_l7_lst.2013.slope", "L7 → L8 calibration slope, LST 2013"),
    ("grid_meta.json", "catalogue.date_quality.canonical", "catalogue records, canonical dates"),
    ("grid_meta.json", "intact_sample.slice_250m", "intact pixels in the 250 m-slice analysis"),
    ("grid_meta.json", "mixed_fit.ratio", "tau ratio, sources pooled"),
    ("median_validation.json", "central_mid.mean_minus_median_bias", "mean − median bias, central subarea"),
    ("chm2_compare.json", "r", "ETH vs CHMv2 correlation"),
    ("s3_results.json", "stats.0.amp_mean", "day−night amplitude, short canopy 0–500 m"),
]


def dig(obj, path):
    for key in path.split("."):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def same(a, b, tol=1e-9):
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b or abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b


def diff_json(a, b, path="", out=None, tol=1e-9):
    """Recursive comparison; returns a list of (path, released, recomputed)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "absent" if k not in a else "present",
                            "absent" if k not in b else "present"))
            else:
                diff_json(a[k], b[k], f"{path}.{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"len {len(a)}", f"len {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff_json(x, y, f"{path}[{i}]", out, tol)
    elif not same(a, b, tol):
        out.append((path, a, b))
    return out


def diff_parquet(pa, pb):
    import pandas as pd
    import numpy as np
    a, b = pd.read_parquet(pa), pd.read_parquet(pb)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return [("shape/columns", f"{a.shape} {list(a.columns)[:6]}", f"{b.shape} {list(b.columns)[:6]}")]
    out = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            ok = np.isnan(xv) & np.isnan(yv)
            ok |= np.abs(xv - yv) <= 1e-9 * np.maximum(1.0, np.maximum(np.abs(xv), np.abs(yv)))
            if not ok.all():
                out.append((c, f"{int((~ok).sum())} rows differ", f"max |Δ| {np.nanmax(np.abs(xv - yv)):.3g}"))
        elif not x.astype(str).equals(y.astype(str)):
            out.append((c, f"{int((x.astype(str) != y.astype(str)).sum())} rows differ", ""))
    return out


def main():
    stages = [ONLY] if ONLY else sorted(STAGES)
    # --- work copy: data/ and code/ shared, outputs/ copied, regenerated files removed
    if COMPARE_ONLY and not os.path.isdir(WORK):
        print("nothing to compare: run without --compare first")
        return 2
    if os.path.isdir(WORK) and not COMPARE_ONLY:
        shutil.rmtree(WORK)
    os.makedirs(WORK, exist_ok=True)
    for name in ("data", "code"):
        if COMPARE_ONLY:
            break
        try:
            os.symlink(os.path.join(ROOT, name), os.path.join(WORK, name))
        except OSError:
            shutil.copytree(os.path.join(ROOT, name), os.path.join(WORK, name))
    if not COMPARE_ONLY:
        shutil.copytree(os.path.join(ROOT, "outputs"), os.path.join(WORK, "outputs"))
    import importlib.util
    skipped_optional = {s for s, (mod, _) in OPTIONAL.items() if importlib.util.find_spec(mod) is None}
    planned = [(s, f, slow) for st in stages for (s, f, slow) in STAGES[st][1]
               if not (QUICK and slow) and s not in skipped_optional]
    regenerated = []
    for _, files, _ in planned:
        for f in files:
            if f not in regenerated:
                regenerated.append(f)
    for f in regenerated:
        if f not in CARRIED and not COMPARE_ONLY:
            p = os.path.join(WORK, "outputs", f)
            if os.path.exists(p):
                os.remove(p)

    env = dict(os.environ, THERMAL_ROOT=WORK, PYTHONHASHSEED="0")
    log = open(os.path.join(WORK, "run.log"), "a" if COMPARE_ONLY else "w", encoding="utf-8")
    print(f"work copy: {WORK}\n")
    for st in ([] if COMPARE_ONLY else stages):
        title, steps = STAGES[st]
        print(f"stage {st}  {title}")
        for script, files, slow in steps:
            if script in skipped_optional:
                print(f"  {script:30s} skipped ({OPTIONAL[script][1]})")
                continue
            if QUICK and slow:
                print(f"  {script:30s} skipped (--quick)")
                continue
            t0 = time.time()
            print(f"  {script:30s} ", end="", flush=True)
            r = subprocess.run([sys.executable, os.path.join(ROOT, "code", script)],
                               env=env, capture_output=True, text=True)
            log.write(f"===== {script}\n{r.stdout}\n{r.stderr}\n")
            if r.returncode:
                print("FAILED")
                print(r.stderr.strip().splitlines()[-1] if r.stderr else "")
                log.close()
                return 2
            print(f"{time.time() - t0:6.1f} s  ->  {', '.join(files)}")
        print()
    log.close()

    # --- comparison
    lines = []
    n_bad = 0
    print(f"{'file':32s} {'result':s}")
    for f in regenerated:
        pa, pb = os.path.join(ROOT, "outputs", f), os.path.join(WORK, "outputs", f)
        if f.endswith(".json"):
            d = diff_json(json.load(open(pa, encoding="utf-8")), json.load(open(pb, encoding="utf-8")),
                          tol=LOOSE_TOL.get(f, 1e-9))
        else:
            d = diff_parquet(pa, pb)
        if d:
            n_bad += 1
            print(f"{f:32s} {len(d)} difference(s)")
            for p, x, y in d[:20]:
                lines.append(f"{f}{p}: released {x}  recomputed {y}")
        else:
            print(f"{f:32s} identical")
    print()
    print(f"{'quantity':44s} {'released':>12s} {'recomputed':>12s}")
    for fname, path, label in HEADLINE:
        if fname not in regenerated:
            continue
        try:
            a = dig(json.load(open(os.path.join(ROOT, "outputs", fname), encoding="utf-8")), path)
            b = dig(json.load(open(os.path.join(WORK, "outputs", fname), encoding="utf-8")), path)
        except (KeyError, IndexError, TypeError):
            print(f"{label:44s} {'?':>12s} {'?':>12s}  !! key missing")
            continue
        fa = f"{a:.6g}" if isinstance(a, (int, float)) else str(a)
        fb = f"{b:.6g}" if isinstance(b, (int, float)) else str(b)
        print(f"{label:44s} {fa:>12s} {fb:>12s}  {'OK' if same(a, b) else '!! MISMATCH'}")

    with open(os.path.join(WORK, "report.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"reproduce.py  {time.strftime('%Y-%m-%d %H:%M')}  quick={QUICK}  stages={stages}\n")
        fh.write(f"files compared: {len(regenerated)}, with differences: {n_bad}\n")
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"\n{len(regenerated) - n_bad} of {len(regenerated)} regenerated files identical to the released "
          f"copies; report in reproduced/report.txt")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
