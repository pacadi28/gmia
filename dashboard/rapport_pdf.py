from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.platypus import KeepTogether
from datetime import datetime
import pandas as pd
import io

# Couleurs Senelec
BLEU = colors.HexColor("#1e3a5f")
ROUGE = colors.HexColor("#e63946")
ORANGE = colors.HexColor("#ff8800")
VERT = colors.HexColor("#2dc653")
GRIS = colors.HexColor("#f5f5f5")
BLANC = colors.white

def generer_rapport(df_brut, df_final, titre="Rapport GMIA — Senelec DRS", periode=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    style_titre = ParagraphStyle(
        "titre",
        parent=styles["Title"],
        fontSize=20,
        textColor=BLEU,
        spaceAfter=6,
        fontName="Helvetica-Bold"
    )
    style_sous_titre = ParagraphStyle(
        "sous_titre",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.grey,
        spaceAfter=20
    )
    style_section = ParagraphStyle(
        "section",
        parent=styles["Heading1"],
        fontSize=14,
        textColor=BLEU,
        spaceBefore=16,
        spaceAfter=8,
        fontName="Helvetica-Bold"
    )
    style_normal = ParagraphStyle(
        "normal",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6
    )
    style_alerte = ParagraphStyle(
        "alerte",
        parent=styles["Normal"],
        fontSize=10,
        textColor=ROUGE,
        spaceAfter=4
    )
    style_ok = ParagraphStyle(
        "ok",
        parent=styles["Normal"],
        fontSize=10,
        textColor=VERT,
        spaceAfter=4
    )

    elements = []
    date_rapport = datetime.now().strftime("%d/%m/%Y à %H:%M")

    # ---- EN-TÊTE ----
    elements.append(Paragraph("⚡ GMIA — PowerInsight", style_titre))
    elements.append(Paragraph("Gestion de Maintenance Intelligente par IA", style_sous_titre))
    elements.append(Paragraph(f"Senelec — Délégation Régionale Sud", style_sous_titre))
    elements.append(Paragraph(f"Rapport généré le {date_rapport}", style_sous_titre))
    if periode:
        elements.append(Paragraph(f"Période analysée : {periode}", style_sous_titre))
    elements.append(HRFlowable(width="100%", thickness=2, color=BLEU, spaceAfter=20))

    # ---- KPIs GLOBAUX ----
    elements.append(Paragraph("1. Indicateurs clés de performance", style_section))

    total_incidents = len(df_brut)
    end_total = df_brut["end"].sum()
    nb_feeders = df_brut["feeder"].nunique()
    duree_moy = df_brut["duree_heures"].mean()
    feeder_critique = df_brut.groupby("feeder").size().idxmax()
    incidents_imputables = df_brut["imputable_drs"].sum() if "imputable_drs" in df_brut.columns else 0
    incidents_planifies = len(df_brut[df_brut["type_incident"] == "Planifié"]) if "type_incident" in df_brut.columns else 0
    incidents_reels = total_incidents - incidents_planifies

    kpi_data = [
        ["Indicateur", "Valeur", "Observation"],
        ["Total incidents", f"{total_incidents:,}", "Toutes natures confondues"],
        ["Incidents réels (hors planifiés)", f"{incidents_reels:,}", "Pannes imputables DRS"],
        ["Interventions planifiées", f"{incidents_planifies:,}", "Travaux et maintenances"],
        ["END total (kWh)", f"{end_total:,.0f}", "Énergie Non Distribuée"],
        ["Durée moyenne interruption (h)", f"{duree_moy:.2f}", "Par incident"],
        ["Nombre de feeders", f"{nb_feeders}", "Sous surveillance"],
        ["Feeder le plus critique", feeder_critique.strip()[:25], "Par nombre d'incidents"],
    ]

    table_kpi = Table(kpi_data, colWidths=[7*cm, 4*cm, 6*cm])
    table_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLEU),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLANC),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLANC, GRIS]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWHEIGHT", (0, 0), (-1, -1), 22),
    ]))
    elements.append(table_kpi)
    elements.append(Spacer(1, 20))

    # ---- ANALYSE PAR ANNÉE ----
    elements.append(Paragraph("2. Évolution annuelle", style_section))

    annee_stats = df_brut.groupby("annee").agg(
        nb_incidents=("end", "count"),
        end_total=("end", "sum"),
        puissance=("puissance_coupee", "sum"),
        duree_moy=("duree_heures", "mean")
    ).reset_index()

    annee_data = [["Année", "Nb incidents", "END (kWh)", "Puissance (kW)", "Durée moy (h)"]]
    for _, row in annee_stats.iterrows():
        annee_data.append([
            str(int(row["annee"])),
            f"{int(row['nb_incidents']):,}",
            f"{row['end_total']:,.0f}",
            f"{row['puissance']:,.0f}",
            f"{row['duree_moy']:.2f}"
        ])

    table_annee = Table(annee_data, colWidths=[3*cm, 3.5*cm, 3.5*cm, 3.5*cm, 3.5*cm])
    table_annee.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLEU),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLANC),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLANC, GRIS]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWHEIGHT", (0, 0), (-1, -1), 22),
    ]))
    elements.append(table_annee)
    elements.append(Spacer(1, 20))

    # ---- PARETO — TOP FEEDERS ----
    elements.append(Paragraph("3. Analyse Pareto — Feeders les plus impactés", style_section))

    pareto = df_brut.groupby("feeder").agg(
        nb_incidents=("end", "count"),
        end_total=("end", "sum"),
        duree_moy=("duree_heures", "mean")
    ).reset_index().sort_values("end_total", ascending=False)
    pareto["feeder"] = pareto["feeder"].str.strip()
    pareto["cumul_pct"] = pareto["end_total"].cumsum() / pareto["end_total"].sum() * 100

    elements.append(Paragraph(
        f"Les feeders suivants concentrent 80% de l'END total :",
        style_normal
    ))

    feeders_80 = pareto[pareto["cumul_pct"] <= 80]["feeder"].tolist()
    elements.append(Paragraph(
        f"🔴 {', '.join(feeders_80)}",
        style_alerte
    ))
    elements.append(Spacer(1, 8))

    pareto_data = [["Feeder", "Nb incidents", "END (kWh)", "Cumul (%)", "Durée moy (h)"]]
    for _, row in pareto.head(10).iterrows():
        pareto_data.append([
            row["feeder"][:20],
            f"{int(row['nb_incidents']):,}",
            f"{row['end_total']:,.0f}",
            f"{row['cumul_pct']:.1f}%",
            f"{row['duree_moy']:.2f}"
        ])

    table_pareto = Table(pareto_data, colWidths=[5*cm, 3*cm, 3.5*cm, 3*cm, 3*cm])
    table_pareto.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLEU),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLANC),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLANC, GRIS]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWHEIGHT", (0, 0), (-1, -1), 22),
        ("BACKGROUND", (0, 1), (-1, 3), colors.HexColor("#fff0f0")),
    ]))
    elements.append(table_pareto)
    elements.append(Spacer(1, 20))

    # ---- NATURE DES INCIDENTS ----
    elements.append(Paragraph("4. Nature des interruptions", style_section))

    nature_stats = df_brut.groupby("nature").agg(
        nb=("end", "count"),
        end_total=("end", "sum"),
        pct=("end", lambda x: len(x) / len(df_brut) * 100)
    ).reset_index().sort_values("nb", ascending=False)

    nature_data = [["Nature", "Nb incidents", "% total", "END (kWh)"]]
    for _, row in nature_stats.iterrows():
        nature_data.append([
            str(row["nature"]),
            f"{int(row['nb']):,}",
            f"{row['pct']:.1f}%",
            f"{row['end_total']:,.0f}"
        ])

    table_nature = Table(nature_data, colWidths=[5*cm, 4*cm, 3*cm, 5*cm])
    table_nature.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLEU),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLANC),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLANC, GRIS]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWHEIGHT", (0, 0), (-1, -1), 22),
    ]))
    elements.append(table_nature)
    elements.append(Spacer(1, 20))

    # ---- CAUSES PRINCIPALES ----
    elements.append(Paragraph("5. Causes principales", style_section))

    causes_stats = df_brut.groupby("cause").size().reset_index(name="nb")
    causes_stats["pct"] = causes_stats["nb"] / len(df_brut) * 100
    causes_stats = causes_stats.sort_values("nb", ascending=False).head(10)

    causes_data = [["Cause", "Nb incidents", "% total"]]
    for _, row in causes_stats.iterrows():
        causes_data.append([
            str(row["cause"])[:35],
            f"{int(row['nb']):,}",
            f"{row['pct']:.1f}%"
        ])

    table_causes = Table(causes_data, colWidths=[9*cm, 4*cm, 4*cm])
    table_causes.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLEU),
        ("TEXTCOLOR", (0, 0), (-1, 0), BLANC),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BLANC, GRIS]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWHEIGHT", (0, 0), (-1, -1), 22),
    ]))
    elements.append(table_causes)
    elements.append(Spacer(1, 20))

    # ---- RECOMMANDATIONS ----
    elements.append(Paragraph("6. Recommandations opérationnelles", style_section))

    top3_feeders = pareto.head(3)["feeder"].tolist()
    recommandations = [
        f"🔴 Priorité 1 — Maintenance urgente sur {top3_feeders[0]} : feeder le plus impactant en END",
        f"🔴 Priorité 2 — Surveillance renforcée sur {top3_feeders[1]} : risque élevé persistant",
        f"🟠 Priorité 3 — Plan d'action sur {top3_feeders[2]} : contribution significative à l'END",
        "🌧️ Préparer le réseau pour l'hivernage : élagage préventif, renforcement isolement",
        "📊 Réduire les incidents 'Non recherchés' (22%) par amélioration du système de diagnostic",
        "⏱️ Réduire le MTTR (temps moyen de rétablissement) par optimisation des procédures",
        "🔧 Planifier des inspections préventives sur les feeders avec forte durée moyenne",
    ]

    for rec in recommandations:
        elements.append(Paragraph(rec, style_normal))

    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        f"Document généré automatiquement par GMIA PowerInsight — {date_rapport}",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey, alignment=1)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer