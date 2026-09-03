# -*- coding: utf-8 -*-
"""將系統 Noto CJK TC（OTF/CFF 外框，ReportLab 不支援）轉為 TrueType 供組稿使用。
產出：fonts/NotoSerifTC-{Regular,Bold}.ttf、fonts/NotoSansTC-{Regular,Bold}.ttf
依賴：pip install fonttools cu2qu
"""
import os as _os
# 專案根目錄：優先取環境變數 THERMAL_ROOT，否則取本檔所在 code/ 的上一層。
_TROOT = _os.environ.get("THERMAL_ROOT") or _os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))
from fontTools.ttLib import TTCollection, newTable
from cu2qu.pens import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
import os

def otf_to_ttf(font, out_path, max_err=1.0):
    glyphOrder = font.getGlyphOrder()
    glyphSet = font.getGlyphSet()
    glyf = newTable("glyf"); glyf.glyphOrder = glyphOrder; glyf.glyphs = {}
    for name in glyphOrder:
        pen = TTGlyphPen(glyphSet)
        qpen = Cu2QuPen(pen, max_err, reverse_direction=True)
        try:
            glyphSet[name].draw(qpen)
        except Exception:
            pen = TTGlyphPen(glyphSet)
        glyf.glyphs[name] = pen.glyph()
    font["glyf"] = glyf
    font["loca"] = newTable("loca")
    hmtx = font["hmtx"]
    for name, g in glyf.glyphs.items():
        g.recalcBounds(glyf)
        aw, _ = hmtx[name]
        hmtx[name] = (aw, getattr(g, "xMin", 0))
    mx = font["maxp"]; mx.tableVersion = 0x00010000
    for k in ["maxZones","maxTwilightPoints","maxStorage","maxFunctionDefs",
              "maxInstructionDefs","maxStackElements","maxSizeOfInstructions",
              "maxComponentElements","maxComponentDepth","maxPoints","maxContours",
              "maxCompositePoints","maxCompositeContours"]:
        setattr(mx, k, 0)
    mx.recalc(font)
    font["post"].formatType = 3.0          # >64k glyphs：不存字位名
    for t in ["CFF ", "VORG"]:
        if t in font: del font[t]
    font["head"].glyphDataFormat = 0
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(out_path)

if __name__ == "__main__":
    os.makedirs(f"{_TROOT}/fonts", exist_ok=True)
    src = "/usr/share/fonts/opentype/noto"
    jobs = [(f"{src}/NotoSerifCJK-Regular.ttc", "NotoSerifTC-Regular.ttf"),
            (f"{src}/NotoSerifCJK-Bold.ttc", "NotoSerifTC-Bold.ttf"),
            (f"{src}/NotoSansCJK-Regular.ttc", "NotoSansTC-Regular.ttf"),
            (f"{src}/NotoSansCJK-Bold.ttc", "NotoSansTC-Bold.ttf")]
    for s, o in jobs:
        f = TTCollection(s, lazy=True).fonts[3]      # index 3 = TC
        otf_to_ttf(f, f"{_TROOT}/fonts/{o}")
        print("done", o)
