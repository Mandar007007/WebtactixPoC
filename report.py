"""TPE (Test Protocol Execution) Word report generator.

Generates a .docx file with a table:
  Step # | Action | Expected Result | Actual Result | Screenshot

Each executed step becomes one row. Screenshots are embedded inline.
"""
from __future__ import annotations

import io
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor
from PIL import Image


@dataclass
class StepRecord:
    """One executed step in the run."""

    step_number: int
    action: str
    expected_result: str
    actual_result: str
    screenshot_b64: str = ""
    url: str = ""
    status: str = "pass"


@dataclass
class RunSummary:
    """Metadata about the completed run."""

    goal: str
    start_url: str
    total_steps: int = 0
    status: str = "complete"
    started_at: str = ""
    finished_at: str = ""
    duration_s: float = 0.0


def _resize_screenshot(b64: str, max_width: int = 500) -> bytes | None:
    """Decode a base64 screenshot and resize for the Word doc."""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        ratio = max_width / img.width
        new_h = int(img.height * ratio)
        img = img.resize((max_width, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _set_cell_shading(cell, color_hex: str) -> None:
    """Set the background color of a table cell."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color_hex)
    shading.set(qn("w:val"), "clear")
    tc_pr.append(shading)


def generate_tpe_report(
    summary: RunSummary,
    steps: list[StepRecord],
    output_path: Path,
) -> Path:
    """Generate a TPE Word document.

    Args:
        summary: Run metadata.
        steps: Executed step records, in order.
        output_path: Where to write the .docx file.

    Returns:
        The path the file was written to.
    """
    doc = Document()

    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    new_width, new_height = section.page_height, section.page_width
    section.page_width = new_width
    section.page_height = new_height
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)

    # Title
    title = doc.add_heading("Test Protocol Execution Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Summary section
    doc.add_heading("Run Summary", level=1)
    summary_table = doc.add_table(rows=6, cols=2)
    summary_table.style = "Light Grid Accent 1"
    summary_data = [
        ("Goal", summary.goal),
        ("Start URL", summary.start_url),
        ("Total Steps", str(summary.total_steps)),
        ("Status", summary.status.upper()),
        ("Started", summary.started_at or "N/A"),
        ("Duration", f"{summary.duration_s:.1f}s" if summary.duration_s else "N/A"),
    ]
    for i, (label, value) in enumerate(summary_data):
        row = summary_table.rows[i]
        row.cells[0].text = label
        row.cells[1].text = value
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.space_before = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(10)
        row.cells[0].paragraphs[0].runs[0].font.bold = True

    doc.add_paragraph()

    # Steps table
    doc.add_heading("Execution Steps", level=1)

    if not steps:
        doc.add_paragraph("No steps were executed.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        return output_path

    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # Header row
    headers = ["Step #", "Action", "Expected Result", "Actual Result", "Screenshot"]
    header_row = table.rows[0]
    for i, header_text in enumerate(headers):
        cell = header_row.cells[i]
        cell.text = header_text
        _set_cell_shading(cell, "1F4E79")
        for paragraph in cell.paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Column widths (landscape: ~9.5 inches usable)
    col_widths = [Inches(0.6), Inches(2.0), Inches(2.2), Inches(2.2), Inches(2.5)]

    # Data rows
    for step in steps:
        row = table.add_row()
        cells = row.cells

        # Step number
        cells[0].text = str(step.step_number)
        cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Action
        action_text = step.action
        if step.url:
            action_text += f"\n[URL: {step.url}]"
        cells[1].text = action_text

        # Expected result
        cells[2].text = step.expected_result

        # Actual result
        cells[3].text = step.actual_result

        # Status coloring
        if step.status == "fail":
            _set_cell_shading(cells[3], "FFE0E0")
        elif step.status == "pass":
            _set_cell_shading(cells[3], "E0FFE0")

        # Screenshot
        screenshot_bytes = _resize_screenshot(step.screenshot_b64)
        if screenshot_bytes:
            p = cells[4].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run()
            run.add_picture(io.BytesIO(screenshot_bytes), width=Inches(2.3))
        else:
            cells[4].text = "(no screenshot)"
            cells[4].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        # Format all cells
        for cell in cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.space_before = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(9)

    # Set column widths
    for i, width in enumerate(col_widths):
        for row in table.rows:
            row.cells[i].width = width

    # Footer
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = footer.add_run(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    return output_path
