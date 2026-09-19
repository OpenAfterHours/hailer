"""A small, real XLSX workbook for reader tests, made with only the standard library."""

from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile


def write_workbook(path: Path) -> Path:
    """Two sheets with strings, numbers, a blank cell, dates and a cached formula result."""
    spreadsheet_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    relationship_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    sheets = {
        "Sales": [
            ["region", "revenue", "customer_id", "sold_on"],
            ["North", 120.5, "001", "2026-09-01"],
            ["South", 80, "002", "2026-09-02"],
            ["North", None, "003", "2026-09-03"],
        ],
        "Budget": [["region", "target"], ["North", 150], ["South", 100]],
    }
    with ZipFile(path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + '</Types>',
        )
        workbook.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{relationship_ns}/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>',
        )
        workbook.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{spreadsheet_ns}" xmlns:r="{relationship_ns}"><sheets>'
            + "".join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheets, 1))
            + '</sheets></workbook>',
        )
        workbook.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="{relationship_ns}/worksheet" Target="worksheets/sheet{i}.xml"/>'
                for i in range(1, len(sheets) + 1)
            )
            + '</Relationships>',
        )
        for i, (name, rows) in enumerate(sheets.items(), 1):
            xml_rows = []
            for row_number, row in enumerate(rows, 1):
                cells = []
                for col, value in enumerate(row):
                    ref = f"{chr(65 + col)}{row_number}"
                    if value is None:
                        continue
                    if name == "Sales" and col == 3 and row_number > 1:
                        cells.append(f'<c r="{ref}" t="d"><v>{value}T00:00:00</v></c>')
                    elif isinstance(value, str):
                        cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
                    else:
                        formula = '<f>100+50</f>' if name == "Budget" and ref == "B2" else ''
                        cells.append(f'<c r="{ref}">{formula}<v>{value}</v></c>')
                xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
            workbook.writestr(
                f"xl/worksheets/sheet{i}.xml",
                f'<worksheet xmlns="{spreadsheet_ns}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>',
            )
    return path
