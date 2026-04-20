import io
from typing import List

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from app.api.informe_patrimonial.portfolio.schemas import PortafolioItem

# ──────────────────────────────────────────────
# Styles — match the HTML original
# ──────────────────────────────────────────────

_FONT = Font(name="Arial", size=12)
_BOLD_FONT = Font(name="Arial", size=12, bold=True)

_HEADER_FILL = PatternFill("solid", fgColor="F0F0F0")  # light grey
_NO_FILL = PatternFill(fill_type=None)

_ALIGN_LEFT = Alignment(horizontal="left", vertical="center")
_ALIGN_LEFT_WRAP = Alignment(horizontal="left", vertical="center", wrap_text=True)

_BORDER = Border(
    left=Side(style="thin", color="000000"),
    right=Side(style="thin", color="000000"),
    top=Side(style="thin", color="000000"),
    bottom=Side(style="thin", color="000000"),
)

_COL_WIDTHS = {
    "A": 48, "B": 18,
    "C": 24, "D": 12, "E": 30,
    "F": 24, "G": 12, "H": 24,
    "I": 38, "J": 12, "K": 38,
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _set_cell(ws, row, col, value, bold=False, fill=None):
    cell = ws.cell(row=row, column=col, value=value)
    cell.font = _BOLD_FONT if bold else _FONT
    cell.alignment = _ALIGN_LEFT
    cell.border = _BORDER
    if fill:
        cell.fill = fill
    return cell


def _write_main_header(ws):
    """Row 1: nombre | monto | clases_activo (merged C-E) | foco_geografico (merged F-H) | tipo_activo (merged I-K)"""
    _set_cell(ws, 1, 1, "nombre", bold=True, fill=_HEADER_FILL)
    _set_cell(ws, 1, 2, "monto", bold=True, fill=_HEADER_FILL)

    for start_col, label in [(3, "clases_activo"), (6, "foco_geografico"), (9, "tipo_activo")]:
        ws.merge_cells(start_row=1, start_column=start_col,
                       end_row=1, end_column=start_col + 2)
        _set_cell(ws, 1, start_col, label, bold=True, fill=_HEADER_FILL)
        for c in range(start_col + 1, start_col + 3):
            ws.cell(row=1, column=c).border = _BORDER
            ws.cell(row=1, column=c).fill = _HEADER_FILL


def _write_sub_header_row(ws, row):
    """Write the repeating sub-header row (clase|porcentaje|slugs × 3) with grey fill."""
    sub_headers = [
        (3, "clase"), (4, "porcentaje"), (5, "slugs"),
        (6, "nombre"), (7, "porcentaje"), (8, "slugs"),
        (9, "nombre"), (10, "porcentaje"), (11, "slugs"),
    ]
    for col, label in sub_headers:
        _set_cell(ws, row, col, label, bold=True, fill=_HEADER_FILL)


def _write_item_block(ws, start_row, item: PortafolioItem) -> int:
    """
    Write one portfolio item block and return the number of rows used.

    Layout (matches the HTML nested-table rendering):
      Row 0 (sub-header):  nombre(merged) | monto(merged) | clase | porcentaje | slugs | nombre | porcentaje | slugs | nombre | porcentaje | slugs
      Row 1..N (data):     (merged)       | (merged)      | data cells...
    """
    n_clases = len(item.clases_activo)
    n_foco = len(item.foco_geografico)
    n_tipo = len(item.tipo_activo)
    data_rows = max(n_clases, n_foco, n_tipo, 1)
    total_rows = 1 + data_rows  # 1 sub-header + N data rows

    # ── Sub-header row ──
    _write_sub_header_row(ws, start_row)

    # ── nombre & monto (merged across all rows in the block) ──
    _set_cell(ws, start_row, 1, item.nombre)
    _set_cell(ws, start_row, 2, item.monto)
    ws.cell(row=start_row, column=1).alignment = _ALIGN_LEFT_WRAP

    if total_rows > 1:
        ws.merge_cells(start_row=start_row, start_column=1,
                       end_row=start_row + total_rows - 1, end_column=1)
        ws.merge_cells(start_row=start_row, start_column=2,
                       end_row=start_row + total_rows - 1, end_column=2)

    # ── Data rows ──
    for i in range(data_rows):
        r = start_row + 1 + i

        # clases_activo (cols C-E)
        if i < n_clases:
            ca = item.clases_activo[i]
            _set_cell(ws, r, 3, ca.clase)
            _set_cell(ws, r, 4, ca.porcentaje)
            _set_cell(ws, r, 5, ", ".join(ca.slugs))
        else:
            for c in (3, 4, 5):
                _set_cell(ws, r, c, None)

        # foco_geografico (cols F-H)
        if i < n_foco:
            fg = item.foco_geografico[i]
            _set_cell(ws, r, 6, fg.nombre)
            _set_cell(ws, r, 7, fg.porcentaje)
            _set_cell(ws, r, 8, ", ".join(fg.slugs))
        else:
            for c in (6, 7, 8):
                _set_cell(ws, r, c, None)

        # tipo_activo (cols I-K)
        if i < n_tipo:
            ta = item.tipo_activo[i]
            _set_cell(ws, r, 9, ta.nombre)
            _set_cell(ws, r, 10, ta.porcentaje)
            _set_cell(ws, r, 11, ", ".join(ta.slugs))
        else:
            for c in (9, 10, 11):
                _set_cell(ws, r, c, None)

    # Ensure borders on sub-header row for nombre/monto area
    for c in (1, 2):
        ws.cell(row=start_row, column=c).border = _BORDER

    return total_rows


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def generate_portfolio_excel(portafolio: List[PortafolioItem]) -> bytes:
    """
    Build a .xlsx workbook that replicates the layout of the
    HTML-based .xls reference file:

      Row 1: main headers (nombre, monto, clases_activo, foco_geografico, tipo_activo)
      Per item:
        - sub-header row (clase|porcentaje|slugs × 3) with grey background
        - 1–N data rows with actual values
        - nombre & monto merged vertically across the block
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Portfolio"

    _write_main_header(ws)

    current_row = 2
    for item in portafolio:
        rows_used = _write_item_block(ws, current_row, item)
        current_row += rows_used

    # Column widths
    for letter, width in _COL_WIDTHS.items():
        ws.column_dimensions[letter].width = width

    # Write to bytes
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()
