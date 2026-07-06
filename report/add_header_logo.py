"""
add_header_logo.py — put a small logo in the top-right corner of every page of a
pandoc-generated .docx by injecting a Word header part.

Usage: python add_header_logo.py report.docx assets/cnu_logo.png
"""
import os, re, sys, shutil, zipfile

W_IN = 1.55                      # logo width in inches
EMU = 914400
REL_HDR = "rId9100"              # header relationship id in document.xml.rels
REL_IMG = "rIdCNUlogo"           # image relationship id inside header1.xml.rels

HDR_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:drawing>
<wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>
<wp:docPr id="9101" name="cnu_logo"/>
<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
<pic:pic><pic:nvPicPr><pic:cNvPr id="9101" name="cnu_logo"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="{img}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
</pic:pic></a:graphicData></a:graphic></wp:inline>
</w:drawing></w:r></w:p>
</w:hdr>"""

HDR_RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="{REL_IMG}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/cnu_logo.png"/>'
            '</Relationships>')


def main(docx, logo):
    from PIL import Image
    w, h = Image.open(logo).size
    cx = int(W_IN * EMU)
    cy = int(cx * h / w)

    tmp = docx + ".hdrtmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    with zipfile.ZipFile(docx) as z:
        z.extractall(tmp)

    # 1) media
    shutil.copy(logo, os.path.join(tmp, "word", "media", "cnu_logo.png"))
    # 2) header part + its rels
    open(os.path.join(tmp, "word", "header1.xml"), "w", encoding="utf-8").write(
        HDR_XML.format(cx=cx, cy=cy, img=REL_IMG))
    open(os.path.join(tmp, "word", "_rels", "header1.xml.rels"), "w", encoding="utf-8").write(HDR_RELS)
    # 3) document.xml.rels: add header relationship
    drels_p = os.path.join(tmp, "word", "_rels", "document.xml.rels")
    drels = open(drels_p, encoding="utf-8").read()
    if REL_HDR not in drels:
        drels = drels.replace("</Relationships>",
            f'<Relationship Id="{REL_HDR}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/></Relationships>')
        open(drels_p, "w", encoding="utf-8").write(drels)
    # 4) content types: png default + header override
    ct_p = os.path.join(tmp, "[Content_Types].xml")
    ct = open(ct_p, encoding="utf-8").read()
    if 'Extension="png"' not in ct:
        ct = ct.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
    if "header1.xml" not in ct:
        ct = ct.replace("</Types>", '<Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/></Types>')
    open(ct_p, "w", encoding="utf-8").write(ct)
    # 5) reference the header from every sectPr (default = all pages)
    doc_p = os.path.join(tmp, "word", "document.xml")
    doc = open(doc_p, encoding="utf-8").read()
    ref = f'<w:headerReference w:type="default" r:id="{REL_HDR}"/>'
    # (a) self-closing empty sectPr  <w:sectPr/>  ->  <w:sectPr>REF</w:sectPr>
    doc = re.sub(r'<w:sectPr\s*/>', f'<w:sectPr>{ref}</w:sectPr>', doc)
    # (b) open sectPr with content  <w:sectPr ...>  ->  insert REF as first child
    #     (skip if a headerReference already follows, e.g. one we just wrote)
    doc = re.sub(r'<w:sectPr(\s[^>]*)?>(?!<w:headerReference)',
                 lambda m: m.group(0) + ref, doc)
    open(doc_p, "w", encoding="utf-8").write(doc)

    # repackage
    newzip = docx + ".new"
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(newzip, "w", zipfile.ZIP_DEFLATED) as zout:
        names = set(zin.namelist())
        for item in names:
            path = os.path.join(tmp, item)
            zout.write(path, item)
        # add newly created parts
        for extra in ["word/header1.xml", "word/_rels/header1.xml.rels", "word/media/cnu_logo.png"]:
            if extra not in names:
                zout.write(os.path.join(tmp, extra), extra)
    shutil.move(newzip, docx)
    shutil.rmtree(tmp)
    print(f"header logo added ({cx}x{cy} EMU) to {docx}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
