"""Fig. 1 (v24) — method flow chart, drawn to formal flow-chart conventions.

Shape carries the meaning, so colour does not: parallelograms are data sources,
rectangles are processing steps, and every box has the same size. Boxes sit on
one grid (three columns, and the two-column rows on the column midpoints), rows
are evenly spaced, labels are noun phrases, and every connector runs vertically
or horizontally with lightly rounded right-angle corners, attached to box edges.
No results, no section numbers.
Output: figures/F2_workflow.{pdf,png}
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from matplotlib.path import Path

sys.path.insert(0, f"{_TROOT}/code")
from grid_utils import apply_mpl_standards

apply_mpl_standards()
O = f"{_TROOT}/outputs"
F = f"{_TROOT}/figures"
R2 = json.load(open(f"{O}/results2.json"))
N_PATCH = f"{R2['n_event'] + R2['n_hansen']:,}"

MM = 1 / 25.4
EC = "#333333"          # one outline colour for every box
FC_PROC = "#FFFFFF"     # process
FC_DATA = "#ECECEC"     # data source
TXT = "#111111"
FS = 8.3

fig = plt.figure(figsize=(183 * MM, 152 * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100); ax.set_ylim(20, 112)
ax.axis("off")

W, H = 26.0, 8.0                     # every box is the same size
HW, HH = W / 2, H / 2
SKEW = 2.2                           # parallelogram skew
COL = {"L": 20.0, "M": 50.0, "R": 80.0}
MID = {"L": 35.0, "R": 65.0}         # midpoints of the column pairs
ROW = [102.0, 88.0, 74.0, 60.0, 46.0, 32.0]   # evenly spaced


def rect(x, y, text):
    ax.add_patch(Rectangle((x - HW, y - HH), W, H, fc=FC_PROC, ec=EC, lw=1.0,
                           zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, color=TXT,
            zorder=4, linespacing=1.35)


def data(x, y, text):
    pts = [(x - HW + SKEW, y + HH), (x + HW + SKEW, y + HH),
           (x + HW - SKEW, y - HH), (x - HW - SKEW, y - HH)]
    ax.add_patch(Polygon(pts, closed=True, fc=FC_DATA, ec=EC, lw=1.0, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=FS, color=TXT,
            zorder=4, linespacing=1.35)


def ortho(points, r=0.0):
    """Orthogonal polyline with square corners."""
    pts = [np.asarray(p, float) for p in points]
    verts, codes = [pts[0]], [Path.MOVETO]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        d0, d1 = p1 - p0, p2 - p1
        l0, l1 = np.hypot(*d0), np.hypot(*d1)
        rr = min(r, l0 / 2, l1 / 2)
        if rr <= 0:
            verts.append(p1); codes.append(Path.LINETO)
            continue
        verts += [p1 - d0 / l0 * rr, p1, p1 + d1 / l1 * rr]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    verts.append(pts[-1]); codes.append(Path.LINETO)
    return Path(verts, codes)


def arrow(points, lw=0.9):
    ax.add_patch(FancyArrowPatch(path=ortho(points), arrowstyle="-|>",
                                 mutation_scale=7, lw=lw, color=EC, zorder=2,
                                 shrinkA=0, shrinkB=0, capstyle="butt"))


def line(points, lw=0.9):
    ax.add_patch(FancyArrowPatch(path=ortho(points), arrowstyle="-", lw=lw,
                                 color=EC, zorder=2, shrinkA=0, shrinkB=0,
                                 capstyle="butt"))


# ---------------- row 1: data sources ----------------
data(COL["L"], ROW[0], "DEM, canopy height,\nforest mask")
data(COL["M"], ROW[0], "Landsat C2 L2,\n792 summer scenes")
data(COL["R"], ROW[0], "Landslide catalogue,\nHansen forest loss")

# ---------------- row 2: products ----------------
rect(COL["L"], ROW[1], "Forest domain\nand terrain layers")
rect(COL["M"], ROW[1], "Summer composites,\n14 epochs at 30 m")
rect(COL["R"], ROW[1], f"Disturbance patches,\nn = {N_PATCH}")
for c in COL.values():
    arrow([(c, ROW[0] - HH), (c, ROW[1] + HH)])

# ---------------- row 3: the two comparison sets ----------------
rect(MID["L"], ROW[2], "Intact-forest\npixel sample")
rect(MID["R"], ROW[2], "Patch and matched\nintact control")
# two sources that feed the same step first join on a shared horizontal line,
# and a single arrow then leads down into that step
BUS1 = (ROW[1] + ROW[2]) / 2
GAPL, GAPR = 47.0, 53.0        # the middle box's two stubs, kept apart so the
                               # left and right merges read as separate lines
line([(COL["L"], ROW[1] - HH), (COL["L"], BUS1), (GAPL, BUS1), (GAPL, ROW[1] - HH)])
arrow([(MID["L"], BUS1), (MID["L"], ROW[2] + HH)])
line([(GAPR, ROW[1] - HH), (GAPR, BUS1), (COL["R"], BUS1), (COL["R"], ROW[1] - HH)])
arrow([(MID["R"], BUS1), (MID["R"], ROW[2] + HH)])

# ---------------- row 4: baseline and difference ----------------
rect(MID["L"], ROW[3], "Canopy–LST\nbuffering baseline")
rect(MID["R"], ROW[3], "Patch − control\ndifference")
for c in MID.values():
    arrow([(c, ROW[2] - HH), (c, ROW[3] + HH)])

# ---------------- row 5: the three analyses, fanned off the difference ------
rect(COL["L"], ROW[4], "Event study\nand DiD")
rect(COL["M"], ROW[4], "Recovery clocks")
rect(COL["R"], ROW[4], "Strata and\nMorakot case")
BUS2 = (ROW[3] + ROW[4]) / 2
line([(MID["R"], ROW[3] - HH), (MID["R"], BUS2)])
line([(COL["L"], BUS2), (COL["R"], BUS2)])
for c in COL.values():
    arrow([(c, BUS2), (c, ROW[4] + HH)])

# ---------------- row 6: robustness ----------------
rect(COL["M"], ROW[5], "Robustness tests")
BUS3 = (ROW[4] + ROW[5]) / 2
for c in (COL["L"], COL["R"]):
    line([(c, ROW[4] - HH), (c, BUS3)])
line([(COL["L"], BUS3), (COL["R"], BUS3)])
arrow([(COL["M"], ROW[4] - HH), (COL["M"], ROW[5] + HH)])

fig.savefig(f"{F}/F2_workflow.pdf", bbox_inches="tight")
fig.savefig(f"{F}/F2_workflow.png", bbox_inches="tight", dpi=500)
plt.close(fig)
print("saved F2_workflow v24")
