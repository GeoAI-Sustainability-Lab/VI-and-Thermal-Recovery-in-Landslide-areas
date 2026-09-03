"""Render the analysis equations as stacked-fraction (分數式) mathtext images. Computer Modern fontset to match
the serif body. Output figures/eq/eq{1..6}.png at 600 dpi, fontsize 10 ->
natural size = 10 pt text when placed at px/600 inch.
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"

OUT = f"{_TROOT}/figures/eq"
os.makedirs(OUT, exist_ok=True)

EQS = {
 1: r"$\mathrm{LST} \;=\; 0.00341802\,\cdot\,\mathrm{DN}_{\mathrm{ST}} \;+\; 149.0 \;-\; 273.15$",
 2: r"$\mathrm{NDVI} \;=\; \dfrac{\rho_{\mathrm{NIR}} \,-\, \rho_{\mathrm{Red}}}{\rho_{\mathrm{NIR}} \,+\, \rho_{\mathrm{Red}}}$",
 3: r"$\Delta X \;=\; \dfrac{1}{n_{p}}\sum_{i \in \mathrm{patch}} X_{i} \;-\; \dfrac{1}{n_{c}}\sum_{j \in \mathrm{ctrl}} X_{j}$",
 4: r"$\mathrm{DiD} \;=\; \Delta X_{\mathrm{post}} \,-\, \Delta X_{\mathrm{pre}}$",
 5: r"$\Delta(t) \;=\; A\,e^{-t/\tau}, \qquad \left(\hat{A},\,\hat{\tau}\right) \;=\; \arg\min_{A,\,\tau}\;\sum_{b}\, \dfrac{\left[\,\overline{\Delta}_{b} - A\,e^{-t_{b}/\tau}\right]^{2}}{\sigma_{b}^{2}}$",
 6: r"$t_{50} \;=\; \tau\,\ln 2, \qquad t_{90} \;=\; \tau\,\ln 10$",
}

for n, tex in EQS.items():
    fig = plt.figure(figsize=(8, 1.6))
    fig.patch.set_alpha(0.0)
    t = fig.text(0.5, 0.5, tex, fontsize=10, ha="center", va="center")
    fig.savefig(f"{OUT}/eq{n}.png", dpi=600, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("eq", n, "ok")
print("EQS COMPLETE")
