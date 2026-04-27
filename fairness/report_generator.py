# report_generator.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Person 3 — Fairness Engineer
# Day 4: Generate a clean PDF audit report from fairness metric scores
#
# OUTPUT: bias_audit_report.pdf
# This is what an organisation receives after running the tool

import json
import os
from datetime import datetime

# We use reportlab for PDF generation
# Install it first: pip install reportlab
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import (
        HexColor, black, white, red, green
    )
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
except ImportError:
    print("Installing reportlab...")
    os.system("pip install reportlab")
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.colors import HexColor, black, white, red, green
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────

COLOR_PASS      = HexColor("#22c55e")   # green
COLOR_FAIL      = HexColor("#ef4444")   # red
COLOR_HEADER    = HexColor("#1e293b")   # dark slate
COLOR_SUBHEADER = HexColor("#334155")   # medium slate
COLOR_ACCENT    = HexColor("#6366f1")   # indigo
COLOR_LIGHT     = HexColor("#f8fafc")   # very light gray
COLOR_BORDER    = HexColor("#e2e8f0")   # light border
COLOR_WARNING   = HexColor("#f59e0b")   # amber


# ─────────────────────────────────────────────
# LOAD AUDIT RESULTS
# ─────────────────────────────────────────────

def load_audit_results(path="audit_results.json"):
    """Loads the audit results JSON produced by audit.py"""
    if not os.path.exists(path):
        print(f"[ERROR] audit_results.json not found at: {path}")
        print("       Run audit.py first to generate results")
        return None

    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# HELPER — verdict color
# ─────────────────────────────────────────────

def verdict_color(passed):
    return COLOR_PASS if passed else COLOR_FAIL


def verdict_text(passed):
    return "PASS ✓" if passed else "FAIL ✗"


def gap_to_percent(gap):
    if gap is None:
        return "N/A"
    return f"{gap * 100:.1f}%"


# ─────────────────────────────────────────────
# BUILD REPORT SECTIONS
# ─────────────────────────────────────────────

def build_cover_section(styles, results):
    """Title page content"""
    elements = []

    summary = results.get("summary", {})
    biased  = summary.get("biased_model", {})
    verdict = biased.get("verdict", "UNKNOWN")

    # Title
    elements.append(Spacer(1, 2*cm))
    elements.append(Paragraph(
        "AI Bias Audit Report",
        ParagraphStyle("Title",
            fontSize=28, fontName="Helvetica-Bold",
            textColor=COLOR_HEADER, alignment=TA_CENTER,
            spaceAfter=8)
    ))

    elements.append(Paragraph(
        "Fairness & Discrimination Analysis",
        ParagraphStyle("Subtitle",
            fontSize=14, fontName="Helvetica",
            textColor=COLOR_SUBHEADER, alignment=TA_CENTER,
            spaceAfter=4)
    ))

    # Date
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        ParagraphStyle("Date",
            fontSize=10, fontName="Helvetica",
            textColor=HexColor("#94a3b8"), alignment=TA_CENTER,
            spaceAfter=30)
    ))

    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=COLOR_BORDER, spaceAfter=20
    ))

    # Overall verdict box
    verdict_bg = COLOR_FAIL if verdict == "BIASED" else COLOR_PASS
    verdict_table = Table(
        [[Paragraph(
            f"OVERALL VERDICT: {verdict}",
            ParagraphStyle("Verdict",
                fontSize=18, fontName="Helvetica-Bold",
                textColor=white, alignment=TA_CENTER)
        )]],
        colWidths=[16*cm]
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), verdict_bg),
        ("ROUNDEDCORNERS", [8]),
        ("TOPPADDING",    (0,0), (-1,-1), 16),
        ("BOTTOMPADDING", (0,0), (-1,-1), 16),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 20))

    # Score summary
    score = biased.get("score", "N/A")
    failed = biased.get("failed", 0)

    elements.append(Paragraph(
        f"<b>{score}</b> &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<b>{failed}</b> critical bias issues detected",
        ParagraphStyle("Score",
            fontSize=13, fontName="Helvetica",
            textColor=COLOR_SUBHEADER, alignment=TA_CENTER,
            spaceAfter=20)
    ))

    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=COLOR_BORDER, spaceAfter=20
    ))

    return elements


def build_what_we_measured(styles):
    """Explains the 3 metrics in plain English"""
    elements = []

    elements.append(Paragraph(
        "What We Measured",
        ParagraphStyle("H1",
            fontSize=16, fontName="Helvetica-Bold",
            textColor=COLOR_HEADER, spaceBefore=10, spaceAfter=8)
    ))

    metrics_info = [
        ("Demographic Parity",
         "Are approval rates equal across groups? This measures whether "
         "the model approves people at the same rate regardless of their "
         "demographic group. A large gap means one group is systematically "
         "favoured over another in raw approval rates."),
        ("Equal Opportunity",
         "Among truly qualified people, does the model catch them equally? "
         "This only looks at people who genuinely qualify, and measures "
         "whether the model correctly identifies them at the same rate "
         "across groups. A gap here means qualified people from one group "
         "are being wrongly rejected at a higher rate."),
        ("False Positive Rate Parity",
         "Are false approvals distributed equally? This measures whether "
         "unqualified people are incorrectly approved at the same rate "
         "across groups. A gap here means the model is applying different "
         "standards of scrutiny to different groups."),
    ]

    for title, description in metrics_info:
        # Metric title row
        title_table = Table(
            [[Paragraph(title,
                ParagraphStyle("MetricTitle",
                    fontSize=11, fontName="Helvetica-Bold",
                    textColor=COLOR_ACCENT)
            )]],
            colWidths=[16*cm]
        )
        title_table.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), HexColor("#eef2ff")),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        elements.append(title_table)

        elements.append(Paragraph(
            description,
            ParagraphStyle("MetricDesc",
                fontSize=10, fontName="Helvetica",
                textColor=COLOR_SUBHEADER,
                leftIndent=10, spaceBefore=4, spaceAfter=10)
        ))

    elements.append(Paragraph(
        "<b>Threshold:</b> Any gap above 10% (0.10) is flagged as a "
        "potential bias issue requiring investigation.",
        ParagraphStyle("Threshold",
            fontSize=10, fontName="Helvetica-Oblique",
            textColor=HexColor("#64748b"), spaceAfter=16)
    ))

    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=COLOR_BORDER, spaceAfter=16
    ))

    return elements


def build_results_section(results):
    """The main results table for biased model"""
    elements = []

    elements.append(Paragraph(
        "Audit Results — Original Model",
        ParagraphStyle("H1",
            fontSize=16, fontName="Helvetica-Bold",
            textColor=COLOR_HEADER, spaceBefore=10, spaceAfter=12)
    ))

    biased_audits = results.get("biased_model", {}).get("audits", {})

    if not biased_audits:
        elements.append(Paragraph(
            "No audit results available.",
            ParagraphStyle("Body", fontSize=10)
        ))
        return elements

    for group_col, audit in biased_audits.items():

        # Group header
        elements.append(Paragraph(
            f"Sensitive Attribute: {group_col.upper()}",
            ParagraphStyle("GroupHeader",
                fontSize=13, fontName="Helvetica-Bold",
                textColor=COLOR_ACCENT, spaceBefore=8, spaceAfter=8)
        ))

        # Build results table
        table_data = [[
            Paragraph("<b>Metric</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white)),
            Paragraph("<b>Gap</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white,
                    alignment=TA_CENTER)),
            Paragraph("<b>Threshold</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white,
                    alignment=TA_CENTER)),
            Paragraph("<b>Status</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white,
                    alignment=TA_CENTER)),
        ]]

        metric_keys = [
            ("demographic_parity", "Demographic Parity"),
            ("equal_opportunity",  "Equal Opportunity"),
            ("fpr_parity",         "FPR Parity"),
        ]

        row_colors = []
        for i, (key, label) in enumerate(metric_keys):
            m = audit.get(key, {})
            gap    = m.get("gap")
            passed = m.get("passed", False)
            status = verdict_text(passed)
            color  = COLOR_PASS if passed else COLOR_FAIL

            table_data.append([
                Paragraph(label,
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica")),
                Paragraph(gap_to_percent(gap),
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica-Bold",
                        alignment=TA_CENTER,
                        textColor=color)),
                Paragraph("10.0%",
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica",
                        alignment=TA_CENTER,
                        textColor=HexColor("#64748b"))),
                Paragraph(status,
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica-Bold",
                        alignment=TA_CENTER,
                        textColor=color)),
            ])
            row_colors.append(passed)

        t = Table(table_data,
                  colWidths=[7*cm, 3*cm, 3*cm, 3*cm])

        style = [
            # Header row
            ("BACKGROUND",    (0,0), (-1,0), COLOR_HEADER),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("GRID",          (0,0), (-1,-1), 0.5, COLOR_BORDER),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [COLOR_LIGHT, white]),
        ]
        t.setStyle(TableStyle(style))
        elements.append(t)
        elements.append(Spacer(1, 12))

        # Group breakdown table
        elements.append(Paragraph(
            "Approval rates by group:",
            ParagraphStyle("SubHead",
                fontSize=10, fontName="Helvetica-Bold",
                textColor=COLOR_SUBHEADER, spaceAfter=4)
        ))

        dp = audit.get("demographic_parity", {})
        rates = dp.get("rates", {})
        if rates:
            rate_data = [[
                Paragraph("<b>Group</b>",
                    ParagraphStyle("TH2", fontSize=9,
                        fontName="Helvetica-Bold", textColor=white)),
                Paragraph("<b>Approval Rate</b>",
                    ParagraphStyle("TH2", fontSize=9,
                        fontName="Helvetica-Bold", textColor=white,
                        alignment=TA_CENTER)),
            ]]
            for group, rate in sorted(rates.items()):
                rate_data.append([
                    Paragraph(str(group),
                        ParagraphStyle("TD2", fontSize=9,
                            fontName="Helvetica")),
                    Paragraph(f"{rate*100:.1f}%",
                        ParagraphStyle("TD2", fontSize=9,
                            fontName="Helvetica-Bold",
                            alignment=TA_CENTER)),
                ])

            rt = Table(rate_data, colWidths=[8*cm, 8*cm])
            rt.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,0), COLOR_SUBHEADER),
                ("TOPPADDING",    (0,0), (-1,-1), 6),
                ("BOTTOMPADDING", (0,0), (-1,-1), 6),
                ("LEFTPADDING",   (0,0), (-1,-1), 10),
                ("GRID",          (0,0), (-1,-1), 0.5, COLOR_BORDER),
                ("ROWBACKGROUNDS",(0,1), (-1,-1),
                 [COLOR_LIGHT, white]),
            ]))
            elements.append(rt)

        elements.append(Spacer(1, 16))

    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=COLOR_BORDER, spaceAfter=16
    ))

    return elements


def build_comparison_section(results):
    """Before vs after comparison if fixed model exists"""
    elements = []

    fixed = results.get("fixed_model")
    if not fixed:
        elements.append(Paragraph(
            "Bias Mitigation",
            ParagraphStyle("H1",
                fontSize=16, fontName="Helvetica-Bold",
                textColor=COLOR_HEADER, spaceBefore=10, spaceAfter=8)
        ))
        elements.append(Paragraph(
            "Fixed model results not yet available. "
            "Run audit.py again once Person 2 provides "
            "predictions_fixed.csv.",
            ParagraphStyle("Body",
                fontSize=10, fontName="Helvetica-Oblique",
                textColor=HexColor("#94a3b8"))
        ))
        return elements

    elements.append(Paragraph(
        "Before vs After Bias Mitigation",
        ParagraphStyle("H1",
            fontSize=16, fontName="Helvetica-Bold",
            textColor=COLOR_HEADER, spaceBefore=10, spaceAfter=8)
    ))

    comparison = results.get("comparison", {})
    fixed_summary = results.get("summary", {}).get("fixed_model", {})

    # Fixed model verdict
    fixed_verdict = fixed_summary.get("verdict", "UNKNOWN")
    fixed_score   = fixed_summary.get("score", "N/A")
    verdict_bg    = COLOR_PASS if fixed_verdict == "FAIR" else COLOR_FAIL

    verdict_table = Table(
        [[Paragraph(
            f"Fixed Model Verdict: {fixed_verdict} — {fixed_score}",
            ParagraphStyle("FV",
                fontSize=13, fontName="Helvetica-Bold",
                textColor=white, alignment=TA_CENTER)
        )]],
        colWidths=[16*cm]
    )
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), verdict_bg),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 12))

    # Comparison table
    for group_col, metrics in comparison.items():
        elements.append(Paragraph(
            f"Improvement by {group_col}:",
            ParagraphStyle("SubHead",
                fontSize=11, fontName="Helvetica-Bold",
                textColor=COLOR_SUBHEADER, spaceAfter=6)
        ))

        comp_data = [[
            Paragraph("<b>Metric</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white)),
            Paragraph("<b>Before</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white,
                    alignment=TA_CENTER)),
            Paragraph("<b>After</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white,
                    alignment=TA_CENTER)),
            Paragraph("<b>Change</b>",
                ParagraphStyle("TH", fontSize=10,
                    fontName="Helvetica-Bold", textColor=white,
                    alignment=TA_CENTER)),
        ]]

        labels = {
            "demographic_parity": "Demographic Parity",
            "equal_opportunity":  "Equal Opportunity",
            "fpr_parity":         "FPR Parity",
        }

        for key, label in labels.items():
            m = metrics.get(key, {})
            before = m.get("biased_gap")
            after  = m.get("fixed_gap")
            imp    = m.get("improvement")
            improved = m.get("improved")

            if imp is not None:
                imp_str   = f"{imp*100:+.1f}%"
                imp_color = COLOR_PASS if improved else COLOR_FAIL
            else:
                imp_str   = "pending"
                imp_color = COLOR_WARNING

            comp_data.append([
                Paragraph(label,
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica")),
                Paragraph(gap_to_percent(before),
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica", alignment=TA_CENTER,
                        textColor=COLOR_FAIL)),
                Paragraph(gap_to_percent(after) if after else "pending",
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica", alignment=TA_CENTER,
                        textColor=COLOR_PASS if after else COLOR_WARNING)),
                Paragraph(imp_str,
                    ParagraphStyle("TD", fontSize=10,
                        fontName="Helvetica-Bold",
                        alignment=TA_CENTER,
                        textColor=imp_color)),
            ])

        ct = Table(comp_data,
                   colWidths=[7*cm, 3*cm, 3*cm, 3*cm])
        ct.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), COLOR_HEADER),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("GRID",          (0,0), (-1,-1), 0.5, COLOR_BORDER),
            ("ROWBACKGROUNDS",(0,1), (-1,-1),
             [COLOR_LIGHT, white]),
        ]))
        elements.append(ct)
        elements.append(Spacer(1, 12))

    elements.append(HRFlowable(
        width="100%", thickness=1,
        color=COLOR_BORDER, spaceAfter=16
    ))

    return elements


def build_recommendations_section(results):
    """Actionable recommendations based on audit findings"""
    elements = []

    elements.append(Paragraph(
        "Recommendations",
        ParagraphStyle("H1",
            fontSize=16, fontName="Helvetica-Bold",
            textColor=COLOR_HEADER, spaceBefore=10, spaceAfter=8)
    ))

    biased_audits = results.get("biased_model", {}).get("audits", {})
    recs = []

    for group_col, audit in biased_audits.items():
        dp  = audit.get("demographic_parity", {})
        eo  = audit.get("equal_opportunity", {})
        fpr = audit.get("fpr_parity", {})

        if not dp.get("passed", True):
            recs.append((
                f"Fix approval rate disparity by {group_col}",
                f"The model shows a {gap_to_percent(dp.get('gap'))} gap in "
                f"approval rates across {group_col} groups. Apply reweighting "
                f"or resampling to balance the training data.",
                "HIGH"
            ))

        if not eo.get("passed", True):
            recs.append((
                f"Improve detection of qualified candidates by {group_col}",
                f"The model misses qualified people from certain {group_col} "
                f"groups at a {gap_to_percent(eo.get('gap'))} higher rate. "
                f"Consider fairness-aware training constraints or "
                f"threshold adjustment per group.",
                "HIGH"
            ))

        if not fpr.get("passed", True):
            recs.append((
                f"Address unequal false approval rates by {group_col}",
                f"Unqualified people from certain {group_col} groups are "
                f"approved at a {gap_to_percent(fpr.get('gap'))} higher rate. "
                f"Calibrate decision thresholds separately per group.",
                "MEDIUM"
            ))

    if not recs:
        recs.append((
            "Continue monitoring",
            "No critical bias issues detected. Continue regular auditing "
            "as data distributions shift over time.",
            "LOW"
        ))

    priority_colors = {
        "HIGH":   COLOR_FAIL,
        "MEDIUM": COLOR_WARNING,
        "LOW":    COLOR_PASS,
    }

    for title, desc, priority in recs:
        color = priority_colors.get(priority, COLOR_ACCENT)

        row = Table(
            [[
                Paragraph(priority,
                    ParagraphStyle("Priority",
                        fontSize=9, fontName="Helvetica-Bold",
                        textColor=white, alignment=TA_CENTER)),
                Paragraph(
                    f"<b>{title}</b><br/><font size=9>{desc}</font>",
                    ParagraphStyle("RecText",
                        fontSize=10, fontName="Helvetica",
                        textColor=COLOR_SUBHEADER,
                        leftIndent=6))
            ]],
            colWidths=[2*cm, 14*cm]
        )
        row.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (0,0), color),
            ("BACKGROUND",    (1,0), (1,0), COLOR_LIGHT),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("BOX",           (0,0), (-1,-1), 0.5, COLOR_BORDER),
        ]))
        elements.append(row)
        elements.append(Spacer(1, 6))

    return elements


# ─────────────────────────────────────────────
# MAIN — generate the PDF
# ─────────────────────────────────────────────

def generate_report(
    audit_path="audit_results.json",
    output_path="bias_audit_report.pdf"
):
    """
    Main function — loads audit results and generates PDF report.
    Call this from Person 4's dashboard to generate downloadable reports.
    """
    print("\n[INFO] Generating bias audit report...")

    results = load_audit_results(audit_path)
    if not results:
        return None

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    elements = []

    # Build all sections
    elements += build_cover_section(styles, results)
    elements.append(Spacer(1, 0.5*cm))
    elements += build_what_we_measured(styles)
    elements += build_results_section(results)
    elements += build_comparison_section(results)
    elements += build_recommendations_section(results)

    # Footer note
    elements.append(Spacer(1, 1*cm))
    elements.append(Paragraph(
        "This report was generated automatically by the AI Bias Detection Tool. "
        "Findings should be reviewed by qualified professionals before making "
        "decisions that affect individuals.",
        ParagraphStyle("Footer",
            fontSize=8, fontName="Helvetica-Oblique",
            textColor=HexColor("#94a3b8"), alignment=TA_CENTER)
    ))

    doc.build(elements)
    print(f"[OK] Report saved to: {output_path}")
    return output_path


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    path = generate_report(
        audit_path="audit_results.json",
        output_path="bias_audit_report.pdf"
    )
    if path:
        print(f"\n[DONE] Report ready: {path}")
        print("       Share this file with your team or attach to the dashboard")
