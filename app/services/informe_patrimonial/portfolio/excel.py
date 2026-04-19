from typing import List

from app.api.informe_patrimonial.portfolio.schemas import PortafolioItem, ClaseActivo, FocoGeografico, TipoActivo

CELL_STYLE = 'border:1px solid #000;'
HEADER_STYLE = f'{CELL_STYLE} background-color:#f0f0f0; font-weight:bold;'


def _th(text: str) -> str:
    return (
        f'<th align="left" valign="middle" style="{HEADER_STYLE}">'
        f'{text}</th>'
    )


def _td(text: str) -> str:
    return (
        f'<td align="left" valign="middle" style="{CELL_STYLE}">'
        f'{text}</td>'
    )


def _render_subtable(
        headers: List[str],
        rows: List[List[str]],
) -> str:
    """Build a nested HTML table to embed inside a parent cell."""
    thead = "<tr>" + "".join(_th(h) for h in headers) + "</tr>"
    tbody_rows = "".join(
        "<tr>" + "".join(_td(c) for c in row) + "</tr>"
        for row in rows
    )
    return (
        '<table border="1" style="border-collapse:collapse; width:100%;">'
        f"<thead>{thead}</thead>"
        f"<tbody>{tbody_rows}</tbody>"
        "</table>"
    )


def _build_clases_activo_subtable(items: List[ClaseActivo]) -> str:
    return _render_subtable(
        headers=["clase", "porcentaje", "slugs"],
        rows=[
            [i.clase, str(i.porcentaje), ", ".join(i.slugs)]
            for i in items
        ],
    )


def _build_foco_geografico_subtable(items: List[FocoGeografico]) -> str:
    return _render_subtable(
        headers=["nombre", "porcentaje", "slugs"],
        rows=[
            [i.nombre, str(i.porcentaje), ", ".join(i.slugs)]
            for i in items
        ],
    )


def _build_tipo_activo_subtable(items: List[TipoActivo]) -> str:
    return _render_subtable(
        headers=["nombre", "porcentaje", "slugs"],
        rows=[
            [i.nombre, str(i.porcentaje), ", ".join(i.slugs)]
            for i in items
        ],
    )


# ──────────────────────────────────────────────
# Main HTML builder
# ──────────────────────────────────────────────

def generate_portfolio_html(portafolio: List[PortafolioItem]) -> str:
    """
    Convert the portfolio list into an HTML string that Excel can
    open natively as a spreadsheet (.xls).
    """
    main_headers = [
        "nombre", "monto", "clases_activo",
        "foco_geografico", "tipo_activo",
    ]
    thead = "<tr>" + "".join(_th(h) for h in main_headers) + "</tr>"

    body_rows: list[str] = []
    for item in portafolio:
        clases_html = _build_clases_activo_subtable(item.clases_activo)
        foco_html = _build_foco_geografico_subtable(item.foco_geografico)
        tipo_html = _build_tipo_activo_subtable(item.tipo_activo)

        row = (
                "<tr>"
                + _td(item.nombre)
                + _td(str(item.monto))
                + _td(clases_html)
                + _td(foco_html)
                + _td(tipo_html)
                + "</tr>"
        )
        body_rows.append(row)

    table_style = (
        'border-collapse:collapse; width:100%; '
        'font-family: Arial, sans-serif; font-size: 12px;'
    )
    return (
        '<html xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns="http://www.w3.org/TR/REC-html40">\n'
        "         <head><meta charset=\"UTF-8\">"
        "<style>table { border-collapse:collapse; } "
        "th, td { border:1px solid #000; padding: 4px; "
        "font-family: Arial, sans-serif; font-size: 12px; }"
        "</style></head>\n"
        f'         <body><table border="1" style="{table_style}">\n'
        f"                          <thead>{thead}</thead>\n"
        f"                          <tbody>{''.join(body_rows)}</tbody>\n"
        "                        </table></body></html>"
    )
