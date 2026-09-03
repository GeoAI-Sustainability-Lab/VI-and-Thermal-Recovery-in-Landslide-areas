# -*- coding: utf-8 -*-
"""Recompute the study's headline statistics from the tables shipped in this
repository and compare them with the released values in outputs/.

以本倉庫釋出之表格重跑統計腳本，並與 outputs/ 中的既有結果逐項比對。
每一項都應顯示 OK；任何 MISMATCH 都代表資料或程式不一致。

    python3 verify_numbers.py            # 全部檢核，約 10 分鐘
    python3 verify_numbers.py --quick    # 略過兩個最慢的步驟，約 3 分鐘
"""
import json, os, shutil, subprocess, sys, tempfile, time

ROOT = os.environ.get("THERMAL_ROOT") or os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
QUICK = "--quick" in sys.argv

# 步驟與其產出的結果檔。這些步驟只需要本倉庫的表格，不需要任何柵格影像。
STEPS = [
    ("step15_gradient_check.py", "gradient_check.json", False),
    ("step16_recovery_clocks.py", "recovery_clocks.json", False),
    ("step20_eventstudy.py", "eventstudy.json", False),
    ("step21_within_patch.py", "within_patch.json", False),
    ("step18_morakot.py", "morakot.json", False),
    ("step27_table2.py", "table2_agents.json", False),
    ("step28_descriptive_meta.py", "grid_meta.json", False),
    ("step17_strata_curves.py", "strata_curves.json", True),
    ("step22_epoch_era.py", "epoch_era.json", True),
]

# 主要統計量，以「檔案, 鍵路徑, 說明」列出。
CHECKS = [
    ("gradient_check.json", "narrow_band.slope", "canopy sensitivity, °C per 10 m"),
    ("gradient_check.json", "narrow_band.se", "its standard error"),
    ("gradient_check.json", "narrow_band.trend_p", "elevation trend p"),
    ("recovery_clocks.json", "thermal.tau", "tau_LST, yr"),
    ("recovery_clocks.json", "greenness.tau", "tau_NDVI, yr"),
    ("recovery_clocks.json", "ratio.mean", "tau ratio"),
    ("recovery_clocks.json", "ratio.ci.0", "ratio CI lower"),
    ("recovery_clocks.json", "ratio.ci.1", "ratio CI upper"),
    ("recovery_clocks.json", "n_patches", "patches in the recovery fit"),
    ("eventstudy.json", "lst_lead_pooled.mean", "pooled pre-event coefficient"),
    ("within_patch.json", "thermal.frac", "decline reproduced within patches"),
    ("within_patch.json", "thermal.beta", "within-patch beta"),
    ("morakot.json", "n_patches", "Morakot cohort patches"),
    ("table2_agents.json", "residual.n", "unclassified trigger patches"),
    ("strata_curves.json", "lt2.ratio.mean", "tau ratio, scars < 2 ha"),
    ("strata_curves.json", "ge10.ratio.mean", "tau ratio, scars >= 10 ha"),
    ("strata_curves.json", "gt2000.thermal.tau", "tau_LST above 2,000 m"),
    ("strata_curves.json", "hansen.greenness.tau", "tau_NDVI, annual loss"),
    ("epoch_era.json", "all.thermal.tau", "tau_LST, all epochs"),
    ("grid_meta.json", "catalogue.date_quality.canonical", "catalogue records with canonical dates"),
    ("grid_meta.json", "intact_sample.slice_250m", "intact-forest pixels in the 250 m-slice analysis"),
    ("grid_meta.json", "curve.anom_at_15", "canopy curve at 15 m, °C"),
    ("grid_meta.json", "mixed_fit.ratio", "tau ratio when sources are pooled"),
]


def dig(obj, path):
    cur = obj
    for key in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(key)]
        else:
            cur = cur[key]
    return cur


def main():
    ref = tempfile.mkdtemp(prefix="verify_ref_")
    steps = [s for s in STEPS if not (QUICK and s[2])]
    needed = {s[1] for s in steps}
    for name in needed:
        shutil.copy(os.path.join(OUT, name), os.path.join(ref, name))

    env = dict(os.environ, THERMAL_ROOT=ROOT)
    print(f"recomputing {len(steps)} steps from the tables in this repository\n")
    for script, produces, slow in steps:
        t0 = time.time()
        print(f"  {script:28s} ", end="", flush=True)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "code", script)],
                           env=env, capture_output=True, text=True)
        if r.returncode:
            print("FAILED")
            print(r.stderr.strip().splitlines()[-1] if r.stderr else "")
            return 2
        print(f"{time.time() - t0:6.1f} s")

    print(f"\n{'quantity':38s} {'released':>13s} {'recomputed':>13s}")
    bad = 0
    for fname, path, label in CHECKS:
        if fname not in needed:
            continue
        a = dig(json.load(open(os.path.join(ref, fname))), path)
        b = dig(json.load(open(os.path.join(OUT, fname))), path)
        same = a == b or (isinstance(a, (int, float)) and isinstance(b, (int, float))
                          and abs(a - b) <= 1e-9 * max(1.0, abs(a)))
        bad += 0 if same else 1
        fa = f"{a:.6g}" if isinstance(a, (int, float)) else str(a)
        fb = f"{b:.6g}" if isinstance(b, (int, float)) else str(b)
        print(f"{label:38s} {fa:>13s} {fb:>13s}  {'OK' if same else '!! MISMATCH'}")

    shutil.rmtree(ref, ignore_errors=True)
    print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} match" if not QUICK else
          f"\nquick mode: {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
