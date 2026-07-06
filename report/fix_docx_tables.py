"""
fix_docx_tables.py — make pandoc 2.12 tables render with visible, equal-width
columns and full cell borders (needed because our result tables have empty cells,
and pandoc 2.12 writes an empty <w:tblGrid/> + tblW=0, which collapses empty
columns to zero width).

For every table it: sets a fixed layout, a full-width dxa table width, an equal
gridCol per column, a matching tcW on every cell, and single-line borders all
around so blank cells are clearly delineated.

Usage:  python fix_docx_tables.py report.docx
"""
import re, sys, zipfile, shutil, os

TOTAL = 9360  # twips = 6.5in text column (Letter, 1in margins)

BORDERS = (
    '<w:tblBorders>'
    '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
    '</w:tblBorders>'
)


def fix_table(tbl: str) -> str:
    # number of columns = number of <w:tc> in the first row
    m = re.search(r"<w:tr>.*?</w:tr>", tbl, flags=re.S)
    if not m:
        return tbl
    ncol = m.group(0).count("<w:tc>")
    if ncol == 0:
        return tbl
    colw = TOTAL // ncol

    # 1) table width: pct 0 -> dxa TOTAL
    tbl = re.sub(r'<w:tblW[^/]*/>', f'<w:tblW w:type="dxa" w:w="{TOTAL}"/>', tbl)

    # 2) fixed layout + borders, inserted right after <w:tblPr...> opening content.
    #    (place them just before <w:tblLook> which pandoc always writes)
    tbl = tbl.replace('<w:tblLook',
                      f'<w:tblLayout w:type="fixed"/>{BORDERS}<w:tblLook', 1)

    # 3) grid: replace empty grid with ncol equal columns
    grid = "<w:tblGrid>" + f'<w:gridCol w:w="{colw}"/>' * ncol + "</w:tblGrid>"
    tbl = re.sub(r'<w:tblGrid\s*/>', grid, tbl)
    tbl = re.sub(r'<w:tblGrid>.*?</w:tblGrid>', grid, tbl, flags=re.S)

    # 4) give every cell an explicit width (pandoc cells have no tcPr)
    tbl = tbl.replace('<w:tc><w:p',
                      f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{colw}"/></w:tcPr><w:p')
    return tbl


def main(path):
    tmp = path + ".tmpdir"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp)
    # Force Word/LibreOffice to refresh fields (incl. the Table of Contents) on open.
    settings = os.path.join(tmp, "word", "settings.xml")
    if os.path.exists(settings):
        sx = open(settings, encoding="utf-8").read()
        if "updateFields" not in sx:
            sx = sx.replace("<w:settings ", "<w:settings ", 1)
            sx = re.sub(r"(<w:settings[^>]*>)", r'\1<w:updateFields w:val="true"/>', sx, count=1)
            open(settings, "w", encoding="utf-8").write(sx)

    doc = os.path.join(tmp, "word", "document.xml")
    xml = open(doc, encoding="utf-8").read()

    out, i, n = [], 0, 0
    while True:
        a = xml.find("<w:tbl>", i)
        if a == -1:
            out.append(xml[i:]); break
        b = xml.find("</w:tbl>", a) + len("</w:tbl>")
        out.append(xml[i:a])
        out.append(fix_table(xml[a:b]))
        n += 1
        i = b
    open(doc, "w", encoding="utf-8").write("".join(out))

    # repackage
    newzip = path + ".new"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(newzip, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            if item == "word/document.xml":
                zout.write(doc, item)
            else:
                zout.writestr(item, zin.read(item))
    shutil.move(newzip, path)
    shutil.rmtree(tmp)
    print(f"fixed {n} tables in {path}")


if __name__ == "__main__":
    main(sys.argv[1])
