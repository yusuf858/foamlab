"""
FoamLab — PDF Report Generator
"""
import io
import logging

logger = logging.getLogger(__name__)


def generate_pdf_report(result: dict) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.units import cm

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)

        ACCENT = colors.HexColor("#1a4a6b")
        RED    = colors.HexColor("#c0392b")
        ORANGE = colors.HexColor("#e67e22")
        GREEN  = colors.HexColor("#27ae60")

        styles = getSampleStyleSheet()
        title_style  = ParagraphStyle("title",  parent=styles["Title"],
                                       textColor=ACCENT, fontSize=18)
        h2_style     = ParagraphStyle("h2",     parent=styles["Heading2"],
                                       textColor=ACCENT)
        normal_style = styles["Normal"]

        sev = result.get("severity", "")
        sev_color = RED if sev == "Critical" else ORANGE if sev == "Moderate" else GREEN

        story = [
            Paragraph("FoamLab — Pollution Detection Report", title_style),
            HRFlowable(width="100%", thickness=2, color=ACCENT),
            Spacer(1, 0.3*cm),
            Paragraph(f"<b>Severity:</b> <font color='#{sev_color.hexval()[2:]}'>{sev}</font>", normal_style),
            Paragraph(f"<b>Dominant Pollution Type:</b> {result.get('dominant_label','—')}", normal_style),
            Paragraph(f"<b>Pollution Area:</b> {result.get('pollution_area_pct','0')}% of image", normal_style),
            Paragraph(f"<b>Total Regions Detected:</b> {result.get('total_regions', 0)}", normal_style),
            Paragraph(f"<b>Processing Time:</b> {result.get('process_time', 0)} s", normal_style),
            Spacer(1, 0.5*cm),
            Paragraph("Pipeline Parameters", h2_style),
        ]

        p = result.get("params", {})
        param_data = [
            ["Parameter", "Value"],
            ["Filter",    p.get("filter","—")],
            ["Enhancement", p.get("enhance","—")],
            ["Segmentation", p.get("segment","—")],
            ["SE Shape",  p.get("se_shape","—")],
            ["SE Size",   str(p.get("se_size","—"))],
        ]
        pt = Table(param_data, colWidths=[6*cm, 8*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), ACCENT),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f5fa")]),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("PADDING",  (0,0), (-1,-1), 5),
        ]))
        story += [pt, Spacer(1, 0.5*cm)]

        regions = result.get("regions", [])
        if regions:
            story.append(Paragraph("Region Analysis", h2_style))
            rdata = [["Label","Area(px²)","Circularity","Compactness","Aspect Ratio","% Image"]]
            for r in regions[:15]:
                rdata.append([
                    r.get("label","—"),
                    str(int(r.get("area",0))),
                    str(r.get("circularity",0)),
                    str(r.get("compactness",0)),
                    str(r.get("aspect_ratio",0)),
                    f"{r.get('pct_image',0)}%",
                ])
            rt = Table(rdata, colWidths=[4*cm,2.5*cm,2.5*cm,2.5*cm,2.5*cm,2*cm])
            rt.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), ACCENT),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f5fa")]),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("PADDING",  (0,0), (-1,-1), 4),
            ]))
            story += [rt, Spacer(1, 0.5*cm)]

        lc = result.get("label_counts", {})
        if lc:
            story.append(Paragraph("Label Distribution", h2_style))
            ldata = [["Label","Count"]] + [[k, str(v)] for k, v in lc.items()]
            lt = Table(ldata, colWidths=[9*cm, 3*cm])
            lt.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), ACCENT),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f5fa")]),
                ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("FONTSIZE", (0,0), (-1,-1), 9),
                ("PADDING",  (0,0), (-1,-1), 5),
            ]))
            story.append(lt)

        doc.build(story)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return b""
