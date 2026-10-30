"""Self-contained HTML reports.

Each phase writes ``outputs/<dataset>/<mode>/<NN_phase>/report.html``: a single
file with embedded PNG images, tables and key numbers. It can be opened
directly in a browser, also after copying it from the cluster with ``scp``.
"""

from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.prism.common.io import to_jsonable

_CSS = """
:root { --fg:#1d1d1f; --muted:#6e6e73; --line:#e3e3e8; --bg:#ffffff; --accent:#2f6fde; --card:#f7f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#ececf1; --muted:#a1a1aa; --line:#33333a; --bg:#151518; --accent:#7aa7ff; --card:#1e1e23; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px 64px; background:var(--bg); color:var(--fg);
       font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 32px 0 8px; border-bottom:1px solid var(--line); padding-bottom:4px; }
h3 { font-size: 16px; margin: 20px 0 6px; }
.meta { color: var(--muted); font-size: 13px; margin-bottom: 18px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap:10px; margin: 12px 0; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.kpi .v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.kpi .l { color: var(--muted); font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin: 8px 0 16px; display:block; overflow-x:auto; }
th, td { border-bottom: 1px solid var(--line); padding: 5px 8px; text-align: left; white-space: nowrap; font-variant-numeric: tabular-nums; }
th { background: var(--card); position: sticky; top: 0; }
img { max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background:#fff; }
figure { margin: 10px 0 18px; } figcaption { color: var(--muted); font-size: 13px; margin-top: 4px; }
pre { background: var(--card); border:1px solid var(--line); border-radius:8px; padding:10px; overflow-x:auto; font-size:12px; }
.note { background: var(--card); border-left: 3px solid var(--accent); padding: 8px 12px; border-radius: 4px; margin: 10px 0; }
.warn { border-left-color: #d9822b; }
.grid2 { display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
"""


def figure_to_png_bytes(fig: Any, dpi: int = 110) -> bytes:
    """Render a Matplotlib figure to PNG bytes and close it."""
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


class HtmlReport:
    """Incrementally build a single-file HTML report."""

    def __init__(self, title: str, subtitle: str = "") -> None:
        """Start a report with a title and an optional subtitle line."""
        self.title = title
        self.subtitle = subtitle
        self.parts: List[str] = []

    # ----------------------------------------------------------- primitives
    def heading(self, text: str, level: int = 2) -> None:
        """Add a section heading (``level`` 2 or 3)."""
        self.parts.append(f"<h{level}>{html.escape(text)}</h{level}>")

    def text(self, text: str) -> None:
        """Add a paragraph of plain text."""
        self.parts.append(f"<p>{html.escape(text)}</p>")

    def note(self, text: str, warn: bool = False) -> None:
        """Add a highlighted note (orange when ``warn`` is true)."""
        cls = "note warn" if warn else "note"
        self.parts.append(f'<div class="{cls}">{html.escape(text)}</div>')

    def kpis(self, values: Dict[str, Any]) -> None:
        """Add a row of key numbers (label -> value)."""
        cells = []
        for label, value in values.items():
            if isinstance(value, float):
                value = f"{value:.4g}"
            cells.append(f'<div class="kpi"><div class="v">{html.escape(str(value))}</div>'
                         f'<div class="l">{html.escape(str(label))}</div></div>')
        self.parts.append(f'<div class="kpis">{"".join(cells)}</div>')

    def table(self, rows: Sequence[Dict[str, Any]], max_rows: int = 200, floatfmt: str = "{:.4f}") -> None:
        """Add an HTML table from a list of dictionaries (union of keys as columns)."""
        if not rows:
            self.parts.append("<p><em>(empty)</em></p>")
            return
        columns: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in columns:
                    columns.append(key)
        head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
        body = []
        for row in list(rows)[:max_rows]:
            cells = []
            for c in columns:
                v = row.get(c, "")
                if isinstance(v, float):
                    v = floatfmt.format(v)
                cells.append(f"<td>{html.escape(str(v))}</td>")
            body.append(f"<tr>{''.join(cells)}</tr>")
        more = f"<p><em>{len(rows) - max_rows} more rows not shown</em></p>" if len(rows) > max_rows else ""
        self.parts.append(f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>{more}")

    def json_block(self, obj: Any, max_chars: int = 6000) -> None:
        """Add a pretty-printed JSON block (truncated to ``max_chars``)."""
        text = json.dumps(to_jsonable(obj), indent=2)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        self.parts.append(f"<pre>{html.escape(text)}</pre>")

    def image_bytes(self, png: bytes, caption: str = "") -> None:
        """Embed PNG bytes as a base64 image."""
        encoded = base64.b64encode(png).decode("ascii")
        cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
        self.parts.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">{cap}</figure>')

    def figure(self, fig: Any, caption: str = "", dpi: int = 110) -> None:
        """Embed a Matplotlib figure."""
        self.image_bytes(figure_to_png_bytes(fig, dpi=dpi), caption)

    def image_file(self, path: Path, caption: str = "") -> None:
        """Embed an existing PNG file."""
        self.image_bytes(Path(path).read_bytes(), caption or Path(path).name)

    def raw_html(self, snippet: str) -> None:
        """Append a raw HTML snippet (trusted content only)."""
        self.parts.append(snippet)

    # ----------------------------------------------------------------- save
    def save(self, path: Path) -> Path:
        """Write the report to ``path`` and return it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        sub = f"{html.escape(self.subtitle)} · " if self.subtitle else ""
        doc = (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{html.escape(self.title)}</title><style>{_CSS}</style></head><body><main>"
            f"<h1>{html.escape(self.title)}</h1><div class='meta'>{sub}generated {stamp}</div>"
            + "\n".join(self.parts)
            + "</main></body></html>"
        )
        path.write_text(doc, encoding="utf-8")
        print(f"  report: {path}")
        return path


def finalize_report(report: HtmlReport, out_dir: Path, mode: str, next_hint: Optional[str] = None) -> Path:
    """Add the standard footer (how to continue) and save ``report.html`` in ``out_dir``."""
    if mode == "preview":
        hint = next_hint or "If this looks right, re-run the same command with --mode full."
        report.note(f"Preview mode. {hint}")
    return report.save(out_dir / "report.html")
