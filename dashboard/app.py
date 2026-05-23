import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import requests
import ollama
import os
from datetime import date, timedelta
from datetime import datetime
from rapport_pdf import generer_rapport
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="GMIA — Senelec PowerInsight",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)



BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(BASE, "data") + "/"


# CSS
st.markdown("""
<style>
    .main-title { font-size: 28px; font-weight: 700; color: #1e3a5f; }
    .sub-title { font-size: 14px; color: #666; margin-bottom: 20px; }
    .kpi-box { background: #1e3a5f; color: white; padding: 15px; border-radius: 10px; text-align: center; }
    .kpi-val { font-size: 28px; font-weight: 700; }
    .kpi-label { font-size: 12px; opacity: 0.8; }
    .ai-box { background: #f0f7ff; border-left: 4px solid #1e3a5f; padding: 15px; border-radius: 5px; margin: 10px 0; }
    .alert-red { background: #fff0f0; border-left: 4px solid #e63946; padding: 10px; border-radius: 5px; }
    .alert-orange { background: #fff8f0; border-left: 4px solid #ff8800; padding: 10px; border-radius: 5px; }
    .alert-green { background: #f0fff4; border-left: 4px solid #00cc44; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DES DONNÉES ET MODÈLE
# ============================================================
@st.cache_data
def charger_donnees():
    df = pd.read_csv(PATH + "dataset_final_v2.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def charger_modele():
    model = joblib.load(PATH + "modele_gmia_v5.pkl")
    le_feeder = joblib.load(PATH + "encodeur_feeder_v5.pkl")
    le_zone = joblib.load(PATH + "encodeur_zone_v5.pkl")
    features = joblib.load(PATH + "features_v5.pkl")
    return model, le_feeder, le_zone, features

df = charger_donnees()
model, le_feeder, le_zone, features = charger_modele()

# Données enrichies brutes
@st.cache_data
def charger_brut():
    d = pd.read_csv(PATH + "fis_enrichi_2022_2026.csv")
    d["date"] = pd.to_datetime(d["date"])
    return d

df_brut = charger_brut()

# ============================================================
# FONCTION IA MISTRAL
# ============================================================
def analyser_avec_ia(contexte, question):
    try:
        prompt = f"""Tu es un expert en maintenance du réseau électrique de la Senelec (Sénégal).
Tu analyses les données de la Délégation Régionale Sud (DRS).

Contexte des données :
{contexte}

Question : {question}

Réponds en français, de manière précise et opérationnelle.
Donne : 1) Analyse des causes 2) Risques identifiés 3) Prédictions 4) Recommandations concrètes.
Sois direct et concis."""

        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"IA non disponible : {e}"

# ============================================================
# FONCTION PRÉDICTION
# ============================================================


def predire_risque(feeder, zone, date_pred, temp, prec, vent):
    d = pd.Timestamp(date_pred)
    mois = d.month
    jour_semaine = d.dayofweek
    semaine = int(d.isocalendar().week)
    trimestre = d.quarter
    hivernage = 1 if mois in [7, 8, 9, 10] else 0
    weekend = 1 if jour_semaine >= 5 else 0
    fin_mois = 1 if d.day >= 26 else 0
    annee = d.year

    jours_feries = pd.to_datetime(["2026-01-01", "2026-04-04", "2026-05-01"])
    jour_ferie = 1 if d in jours_feries else 0

    try:
        feeder_encode = le_feeder.transform([feeder])[0]
    except:
        feeder_encode = 0
    try:
        zone_encode = le_zone.transform([zone])[0]
    except:
        zone_encode = 0

    # Dataset principal filtré par feeder
    df_f = df[df["feeder"].str.strip() == feeder.strip()].sort_values("date")

    # Incidents réels sur fenêtres calendaires
    date_7j = d - pd.Timedelta(days=7)
    date_30j = d - pd.Timedelta(days=30)
    date_3mois = d - pd.Timedelta(days=90)

    col_inc = "incident_reel" if "incident_reel" in df_f.columns else "incident"

    inc_7j = float(df_f[(df_f["date"] >= date_7j) & (df_f["date"] < d)][col_inc].sum())
    inc_30j = float(df_f[(df_f["date"] >= date_30j) & (df_f["date"] < d)][col_inc].sum())
    inc_3mois = float(df_f[(df_f["date"] >= date_3mois) & (df_f["date"] < d)][col_inc].sum())

    # Ajuster selon météo normale
    if prec < 2 and temp < 36 and vent < 20:
        inc_7j *= 0.3
        inc_30j *= 0.3
        inc_3mois *= 0.3

    # Jours depuis dernier incident réel
    derniers = df_f[df_f[col_inc] == 1]
    if len(derniers) > 0:
        jours_depuis = max(1, (d - pd.Timestamp(derniers["date"].max())).days)
    else:
        jours_depuis = 90.0

    # Jours depuis dernière maintenance planifiée
    df_brut_pred = charger_brut()
    df_maint = df_brut_pred[
        (df_brut_pred["feeder"].str.strip() == feeder.strip()) &
        (df_brut_pred["type_incident"] == "Planifié") &
        (df_brut_pred["date"] < d)
    ]
    jours_maint = int((d - pd.Timestamp(df_maint["date"].max())).days) if len(df_maint) > 0 else 999

    # Taux historiques par type depuis fis_enrichi
    df_f_brut = df_brut_pred[df_brut_pred["feeder"].str.strip() == feeder.strip()]
    taux_total = len(df_f_brut)
    if taux_total > 0:
        taux_meteo = len(df_f_brut[df_f_brut["type_incident"] == "Panne météo"]) / taux_total
        taux_tech = len(df_f_brut[df_f_brut["type_incident"] == "Panne technique"]) / taux_total
        taux_grave = len(df_f_brut[df_f_brut["type_incident"] == "Panne grave"]) / taux_total
        taux_plan = len(df_f_brut[df_f_brut["type_incident"] == "Planifié"]) / taux_total
    else:
        taux_meteo = taux_tech = taux_grave = taux_plan = 0.0

    # Fragilité réelle
    fragilite_reelle_map = {
        "BIGNONA (D3)": 0.304, "LIVRAISON": 0.237,
        "CAP SKIRRING (D2)": 0.215, "TANAF": 0.189,
        "BOUNA": 0.139, "ZIGUINCHOR (D1)": 0.132,
        "GOUDOMP (D4)": 0.106, "DIATTACOUNDA": 0.032,
        "SEDHIOU": 0.030, "KEDOUGOU CENTRE": 0.016,
        "MEDINA YORO FOULAH": 0.010, "GOUYE MBINDE": 0.009,
        "AFIGNAM": 0.007, "BANDA FASSI": 0.004,
        "DIAOBE": 0.004, "GOUYE MINDE": 0.003,
        "MANDA DOUANE": 0.003, "SANDIARA": 0.002,
        "THIENABA": 0.0, "VELINGARA VILLE": 0.0,
    }
    fragilite = fragilite_reelle_map.get(feeder.strip(), 0.1)

    # Taux historique semaine
    df_semaine = df_f[df_f["semaine"] == semaine]
    taux_hist = float(df_semaine["taux_semaine_hist"].mean()) if len(df_semaine) > 0 else 0.2

    # Variables météo dérivées
    vague_chaleur = 1 if temp > 38 else 0
    jour_pluie = 1 if prec > 5 else 0
    vent_fort = 1 if vent > 30 else 0
    ecart_temp = temp - 32.0

    X = pd.DataFrame([{
        "feeder_encode": feeder_encode, "zone_encode": zone_encode,
        "mois": mois, "jour_semaine": jour_semaine, "semaine": semaine,
        "trimestre": trimestre, "hivernage": hivernage, "weekend": weekend,
        "fin_mois": fin_mois, "jour_ferie": jour_ferie, "annee_x": annee,
        "temperature_max": temp, "precipitation": prec, "vent_max": vent,
        "vague_chaleur": vague_chaleur, "jour_pluie": jour_pluie,
        "vent_fort": vent_fort, "pluie_3jours": prec, "ecart_temp": ecart_temp,
        "incidents_7j": inc_7j, "incidents_30j": inc_30j, "incidents_3mois": inc_3mois,
        "jours_depuis_incident": jours_depuis, "fragilite_feeder": fragilite,
        "taux_semaine_hist": taux_hist,
        "taux_meteo_hist": taux_meteo, "taux_tech_hist": taux_tech,
        "taux_grave_hist": taux_grave, "taux_planifie_hist": taux_plan,
        "jours_depuis_maintenance": jours_maint
    }])

    proba = model.predict_proba(X)[0][1]

    if proba < 0.3:
        niveau, couleur = "FAIBLE", "green"
    elif proba < 0.6:
        niveau, couleur = "MODÉRÉ", "orange"
    else:
        niveau, couleur = "ÉLEVÉ", "red"

    return proba, niveau, couleur


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## ⚡ GMIA PowerInsight")
st.sidebar.markdown("**Senelec — Délégation Sud**")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("Navigation", [
    "🏠 Tableau de bord",
    "📊 Suivi & KPIs",
    "🔥 Pareto & Analyse",
    "🗺️ Carte des risques",
    "🔮 Prédiction IA",
    "💬 Assistant IA",
    "📅 Comparaison annuelle",
    "⚡ Feeders critiques",
    "📈 Tendances",
])

st.sidebar.markdown("### Filtres globaux")
annees_dispo = sorted(df_brut["annee"].unique())
annees_sel = st.sidebar.multiselect("Années", annees_dispo, default=annees_dispo, key="filtre_annees")

feeders_dispo = sorted(df_brut["feeder"].str.strip().unique())
feeders_glob = st.sidebar.multiselect("Feeders", feeders_dispo, default=feeders_dispo, key="filtre_feeders")

# Appliquer filtres
df_filtre = df[
    df["annee_x"].isin(annees_sel) & 
    df["feeder"].str.strip().isin(feeders_glob)
]
df_brut_filtre = df_brut[
    df_brut["annee"].isin(annees_sel) & 
    df_brut["feeder"].str.strip().isin(feeders_glob)
]

zones_sel = df_filtre["zone"].unique().tolist()

# ============================================================
# PAGE 1 — TABLEAU DE BORD
# ============================================================
if page == "🏠 Tableau de bord":
    st.markdown('<p class="main-title">⚡ GMIA — Tableau de bord Senelec DRS</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Gestion de Maintenance Intelligente par IA — Délégation Régionale Sud</p>', unsafe_allow_html=True)

    # KPIs
    total_inc = int(df_filtre["incident"].sum())
    end_total = df_brut_filtre["end"].sum()
    nb_feeders = df_filtre["feeder"].nunique()
    taux_inc = df_filtre["incident"].mean() * 100
    feeder_critique = df_filtre.groupby("feeder")["incident"].sum().idxmax()
    inc_30j = df_filtre[df_filtre["date"] >= df_filtre["date"].max() - timedelta(days=30)]["incident"].sum()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("📋 Total incidents", f"{total_inc:,}")
    with col2:
        st.metric("⚡ END total (kWh)", f"{end_total:,.0f}")
    with col3:
        st.metric("🔌 Feeders", f"{nb_feeders}")
    with col4:
        st.metric("📊 Taux incidents", f"{taux_inc:.1f}%")
    with col5:
        st.metric("⚠️ Feeder critique", feeder_critique.strip()[:12])
    with col6:
        st.metric("📅 Incidents 30j", f"{int(inc_30j)}")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("📈 Évolution END par année")
        end_annee = df_brut_filtre.groupby("annee")["end"].sum().reset_index()
        fig1 = px.bar(end_annee, x="annee", y="end",
                      color_discrete_sequence=["#1e3a5f"],
                      labels={"end": "END (kWh)", "annee": "Année"},
                      text="end")
        fig1.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        if st.button("🤖 Analyser évolution END", key="btn_end"):
            contexte = f"END par année : {end_annee.to_string()}"
            with st.spinner("Mistral analyse..."):
                analyse = analyser_avec_ia(contexte, "Analyse l'évolution de l'END par année. Quelles sont les tendances, causes probables et prévisions pour l'année prochaine ?")
            st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

    with col_g2:
        st.subheader("🔥 Nature des incidents")
        nature_inc = df_brut_filtre["nature"].value_counts().reset_index()
        nature_inc.columns = ["nature", "count"]
        fig2 = px.pie(nature_inc, values="count", names="nature",
                      color_discrete_sequence=px.colors.sequential.Blues_r,
                      hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

        if st.button("🤖 Analyser nature incidents", key="btn_nature"):
            contexte = f"Distribution des natures d'incidents : {nature_inc.to_string()}"
            with st.spinner("Mistral analyse..."):
                analyse = analyser_avec_ia(contexte, "Analyse la distribution des types d'incidents. Quels types sont les plus critiques et pourquoi ? Quelles actions préventives recommandes-tu ?")
            st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

    col_g3, col_g4 = st.columns(2)

    with col_g3:
        st.subheader("📊 Incidents par mois (toutes années)")
        inc_mois = df_filtre.groupby(["mois", "annee_x"])["incident"].sum().reset_index()
        inc_mois["annee"] = inc_mois["annee_x"].astype(str)
        fig3 = px.line(inc_mois, x="mois", y="incident", color="annee",
                       markers=True,
                       labels={"incident": "Nb incidents", "mois": "Mois"},
                       color_discrete_sequence=px.colors.qualitative.Set1)
        fig3.update_xaxes(tickvals=list(range(1,13)),
                          ticktext=["Jan","Fév","Mar","Avr","Mai","Jun",
                                    "Jul","Aoû","Sep","Oct","Nov","Déc"])
        st.plotly_chart(fig3, use_container_width=True)

        if st.button("🤖 Analyser saisonnalité", key="btn_saison"):
            contexte = f"Incidents par mois et année : {inc_mois.to_string()}"
            with st.spinner("Mistral analyse..."):
                analyse = analyser_avec_ia(contexte, "Analyse la saisonnalité des incidents. Quels mois sont les plus critiques ? Quel est l'impact de l'hivernage ? Que prévoir pour les prochains mois ?")
            st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)


    with col_g4:
        st.subheader("⚡ END par feeder")
        end_feeder = df_brut_filtre.groupby("feeder")["end"].sum().reset_index()
        end_feeder["feeder"] = end_feeder["feeder"].str.strip()
        end_feeder = end_feeder.sort_values("end", ascending=True)
 
        fig4 = px.bar(end_feeder,
              x="end", y="feeder", orientation="h",
              color="end",
              color_continuous_scale="Blues",
              labels={"end": "END (kWh)", "feeder": "Feeder"})
        fig4.update_layout(
            height=600,
            margin=dict(l=200, r=20, t=20, b=20),
            yaxis=dict(tickfont=dict(size=11))
        )
        st.plotly_chart(fig4, use_container_width=True)

        if st.button("🤖 Analyser feeders", key="btn_feeder_end"):
            contexte = f"END par feeder : {end_feeder.to_string()}"
            with st.spinner("Mistral analyse..."):
                analyse = analyser_avec_ia(
                    contexte,
                    "Analyse la répartition de l'END par feeder. Quels feeders sont les plus impactés ? Quelles actions prioritaires recommandes-tu ?"
                )
            st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)



    st.markdown("---")
    col_pdf1, col_pdf2, col_pdf3 = st.columns([1, 2, 1])
    with col_pdf2:
        if st.button("📄 Générer rapport PDF", type="primary", use_container_width=True):
            with st.spinner("Génération du rapport PDF..."):
                periode = f"{df_brut_filtre['date'].min().date()} → {df_brut_filtre['date'].max().date()}"
                buffer = generer_rapport(df_brut_filtre, df_filtre, periode=periode)
                st.download_button(
                    label="⬇️ Télécharger le rapport PDF",
                    data=buffer,
                    file_name=f"rapport_gmia_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )







# ============================================================
# PAGE 2 — SUIVI & KPIs
# ============================================================
elif page == "📊 Suivi & KPIs":
    st.title("📊 Suivi des performances — KPIs détaillés")

    # Tableau comparatif multi-années
    st.subheader("Tableau comparatif par feeder et par année")



    pivot = df_brut_filtre.groupby(["feeder", "annee"]).agg(
        nb_incidents=("feeder", "count"),
        end_total=("end", "sum"),
        puissance=("puissance_coupee", "sum"),
        duree_moy=("duree_heures", "mean")
    ).reset_index()

    pivot["feeder"] = pivot["feeder"].str.strip()
    pivot["duree_moy"] = pivot["duree_moy"].round(2)
    pivot["end_total"] = pivot["end_total"].round(0)
    pivot["puissance"] = pivot["puissance"].round(0)

    # Filtres
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        feeders_sel = st.multiselect("Filtrer par feeder",
                                      sorted(pivot["feeder"].unique()),
                                      default=sorted(pivot["feeder"].unique())[:5])
    with col_f2:
        metrique = st.selectbox("Métrique", ["nb_incidents", "end_total", "puissance", "duree_moy"])

    pivot_filtre = pivot[pivot["feeder"].isin(feeders_sel)]

    # Afficher tableau
    st.dataframe(pivot_filtre, use_container_width=True, height=300)

    # Bouton analyse IA sur le tableau
    if st.button("🤖 Analyser ce tableau avec l'IA"):
        contexte = pivot_filtre.to_string()
        with st.spinner("Mistral analyse le tableau..."):
            analyse = analyser_avec_ia(
                contexte,
                f"Analyse ce tableau de performances par feeder. Quels feeders se dégradent ? Lesquels s'améliorent ? Si cette tendance continue, que va-t-il se passer ? Quelles actions urgentes recommandes-tu ?"
            )
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Graphique métrique sélectionnée
    metriques_labels = {
        "nb_incidents": "Nombre d'incidents",
        "end_total": "END total (kWh)",
        "puissance": "Puissance coupée (kW)",
        "duree_moy": "Durée moyenne (h)"
    }

    fig_suivi = px.bar(
        pivot_filtre, x="annee", y=metrique, color="feeder",
        barmode="group",
        labels={"annee": "Année", metrique: metriques_labels.get(metrique, metrique)},
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"{metriques_labels.get(metrique, metrique)} par feeder"
    )
    fig_suivi.update_layout(height=450)

    st.plotly_chart(fig_suivi, use_container_width=True)



    if st.button("🤖 Analyser graphique", key="btn_graph_suivi"):
        contexte = pivot_filtre[["feeder", "annee", metrique]].to_string()
        with st.spinner("Mistral analyse..."):
            analyse = analyser_avec_ia(contexte, f"Analyse l'évolution de {metrique} par feeder sur les années. Identifie les tendances inquiétantes et les améliorations. Que prévoir pour la suite ?")
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Nature des interruptions par zone
    st.subheader("Nature des interruptions par zone")
    nature_zone = df_brut_filtre.groupby(["zone", "nature"]).size().reset_index(name="count")
    fig_nature = px.bar(nature_zone, x="zone", y="count", color="nature",
                        barmode="stack",
                        color_discrete_sequence=px.colors.qualitative.Pastel,
                        labels={"count": "Nb incidents", "zone": "Zone"})
    st.plotly_chart(fig_nature, use_container_width=True)

    if st.button("🤖 Analyser nature par zone", key="btn_nature_zone"):
        contexte = nature_zone.to_string()
        with st.spinner("Mistral analyse..."):
            analyse = analyser_avec_ia(contexte, "Analyse la nature des interruptions par zone. Quelles zones ont le plus de défauts permanents ? Quelles zones sont sensibles aux fugitifs ? Que recommandes-tu ?")
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 3 — PARETO & ANALYSE
# ============================================================
elif page == "🔥 Pareto & Analyse":
    st.title("🔥 Diagramme de Pareto — Règle 80/20")
    st.markdown("20% des feeders causent 80% des pannes")

    annee_pareto = st.selectbox("Année", sorted(df_brut_filtre["annee"].unique(), reverse=True))
    df_pareto = df_brut_filtre[df_brut_filtre["annee"] == annee_pareto]

    # Calculer Pareto par END
    pareto = df_pareto.groupby("feeder")["end"].sum().reset_index()
    pareto["feeder"] = pareto["feeder"].str.strip()
    pareto = pareto.sort_values("end", ascending=False)
    pareto["cumul_pct"] = pareto["end"].cumsum() / pareto["end"].sum() * 100
    pareto["feeder_80"] = pareto["cumul_pct"] <= 80

    # Graphique Pareto
    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(
        go.Bar(x=pareto["feeder"], y=pareto["end"],
               name="END (kWh)",
               marker_color=["#e63946" if f else "#1e3a5f" for f in pareto["feeder_80"]]),
        secondary_y=False
    )
    fig_pareto.add_trace(
        go.Scatter(x=pareto["feeder"], y=pareto["cumul_pct"],
                   name="Cumul %", line=dict(color="orange", width=2),
                   mode="lines+markers"),
        secondary_y=True
    )
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="red",
                          annotation_text="80%", secondary_y=True)
    fig_pareto.update_layout(title=f"Pareto END {annee_pareto}")
    fig_pareto.update_yaxes(title_text="END (kWh)", secondary_y=False)
    fig_pareto.update_yaxes(title_text="Cumul (%)", secondary_y=True)
    st.plotly_chart(fig_pareto, use_container_width=True)

    feeders_80 = pareto[pareto["feeder_80"]]["feeder"].tolist()
    st.warning(f"⚠️ Ces {len(feeders_80)} feeders représentent 80% de l'END en {annee_pareto} : **{', '.join(feeders_80)}**")

    if st.button("🤖 Analyser Pareto avec l'IA"):
        contexte = pareto.to_string()
        with st.spinner("Mistral analyse..."):
            analyse = analyser_avec_ia(
                contexte,
                f"Analyse ce diagramme Pareto pour {annee_pareto}. Quels feeders concentrent les problèmes ? Pourquoi ces feeders sont-ils si critiques ? Que recommandes-tu pour réduire l'END sur ces feeders prioritaires ?"
            )
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Diagramme bulle
    st.subheader("🫧 Diagramme bulle — Position des feeders")
    annees_bulle = sorted(df_brut_filtre["annee"].unique())
    if len(annees_bulle) >= 2:
        a1, a2 = annees_bulle[-2], annees_bulle[-1]
        bulle1 = df_brut_filtre[df_brut_filtre["annee"] == a1].groupby("feeder")["end"].sum().reset_index()
        bulle2 = df_brut_filtre[df_brut_filtre["annee"] == a2].groupby("feeder")["end"].sum().reset_index()
        bulle = bulle1.merge(bulle2, on="feeder", suffixes=(f"_{a1}", f"_{a2}"))
        bulle["feeder"] = bulle["feeder"].str.strip()
        bulle["evolution"] = ((bulle[f"end_{a2}"] - bulle[f"end_{a1}"]) / (bulle[f"end_{a1}"] + 1) * 100).round(1)
        bulle["couleur"] = bulle["evolution"].apply(lambda x: "Dégradé" if x > 0 else "Amélioré")

        fig_bulle = px.scatter(bulle,
                               x=f"end_{a1}", y=f"end_{a2}",
                               size=bulle[f"end_{a2}"].abs() + 1,
                               color="couleur",
                               text="feeder",
                               color_discrete_map={"Dégradé": "#e63946", "Amélioré": "#2dc653"},
                               labels={f"end_{a1}": f"END {a1}", f"end_{a2}": f"END {a2}"},
                               title=f"Position feeders {a1} vs {a2}")
        fig_bulle.add_shape(type="line", x0=0, y0=0,
                             x1=bulle[f"end_{a1}"].max(),
                             y1=bulle[f"end_{a1}"].max(),
                             line=dict(dash="dash", color="gray"))
        st.plotly_chart(fig_bulle, use_container_width=True)

        if st.button("🤖 Analyser diagramme bulle"):
            contexte = bulle.to_string()
            with st.spinner("Mistral analyse..."):
                analyse = analyser_avec_ia(contexte, f"Analyse ce diagramme bulle comparant {a1} vs {a2}. Quels feeders se sont dégradés ? Lesquels se sont améliorés ? Quelles actions urgentes pour les feeders dégradés ?")
            st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 4 — CARTE DES RISQUES
# ============================================================
elif page == "🗺️ Carte des risques":
    st.title("🗺️ Carte des risques — Délégation Sud")

    try:
        import folium
        from streamlit_folium import st_folium

        coords_feeders = {
            "BIGNONA (D3)":         (12.8697, -16.2270, "ZIGUINCHOR"),
            "LIVRAISON":            (12.5500, -14.9500, "KOLDA"),
            "TANAF":                (12.6833, -15.5333, "TANAF"),
            "CAP SKIRRING (D2)":    (12.3833, -16.7333, "ZIGUINCHOR"),
            "GOUDOMP (D4)":         (12.5667, -15.8667, "ZIGUINCHOR"),
            "BOUNA":                (12.9000, -14.7500, "KOLDA"),
            "ZIGUINCHOR (D1)":      (12.5833, -16.2719, "ZIGUINCHOR"),
            "DIATTACOUNDA":         (12.9167, -14.5833, "TANAF"),
            "SEDHIOU":              (12.7000, -15.5500, "TANAF"),
            "MEDINA YORO FOULAH":   (13.1833, -14.5000, "VELINGARA"),
            "AFIGNAM":              (12.6000, -16.0000, "ZIGUINCHOR"),
            "MANDA DOUANE":         (12.8000, -15.2000, "TANAF"),
            "DIAOBE":               (13.0000, -14.4000, "VELINGARA"),
            "VELINGARA VILLE":      (13.1500, -14.1167, "VELINGARA"),
            "KEDOUGOU CENTRE":      (12.5605, -12.1747, "KEDOUGOU"),
        }

        fragilite_map = df_filtre.groupby("feeder")["incident"].mean().reset_index()
        fragilite_map.columns = ["feeder", "risque"]

        carte = folium.Map(location=[12.7, -15.5], zoom_start=7, tiles="CartoDB positron")

        for _, row in fragilite_map.iterrows():
            feeder = row["feeder"].strip()
            risque = row["risque"]
            if feeder in coords_feeders:
                lat, lon, zone = coords_feeders[feeder]
                couleur = "red" if risque >= 0.4 else ("orange" if risque >= 0.2 else "green")
                niveau = "ÉLEVÉ" if risque >= 0.4 else ("MODÉRÉ" if risque >= 0.2 else "FAIBLE")
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=15 + risque * 30,
                    color=couleur, fill=True, fill_color=couleur, fill_opacity=0.7,
                    popup=folium.Popup(f"<b>{feeder}</b><br>Zone: {zone}<br>Risque: {niveau}<br>Score: {risque:.1%}", max_width=200),
                    tooltip=f"{feeder} — {risque:.1%}"
                ).add_to(carte)

        st_folium(carte, width=1200, height=600)
        st.markdown("🔴 Risque élevé (>40%)  🟠 Risque modéré (20-40%)  🟢 Risque faible (<20%)")

    except Exception as e:
        st.error(f"Erreur carte : {e}")

# ============================================================
# PAGE 5 — PRÉDICTION IA
# ============================================================
elif page == "🔮 Prédiction IA":
    st.title("🔮 Prédiction de pannes par IA")

    col1, col2 = st.columns(2)
    col1, col2 = st.columns(2)
    with col1:
        # Sélectionner zone d'abord
        zone_pred = st.selectbox("Zone", sorted(df["zone"].unique()), key="zone_pred")
        
        # Feeders filtrés selon la zone
        feeders_zone = sorted(df[df["zone"] == zone_pred]["feeder"].str.strip().unique())
        feeder_pred = st.selectbox("Feeder", feeders_zone, key="feeder_pred")
        
        date_pred = st.date_input("Date", value=date.today() + timedelta(days=1))
    with col2:
        temp = st.slider("Température max (°C)", 25.0, 45.0, 35.0)
        prec = st.slider("Précipitations (mm)", 0.0, 100.0, 0.0)
        vent = st.slider("Vent max (km/h)", 0.0, 80.0, 15.0)

    if st.button("🔮 Lancer la prédiction", type="primary"):
        proba, niveau, couleur = predire_risque(feeder_pred, zone_pred, date_pred, temp, prec, vent)

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Score de risque", f"{proba*100:.1f}%")
        with col_r2:
            st.metric("Niveau", niveau)
        with col_r3:
            st.metric("Statut", "⚠️ Intervention" if proba >= 0.5 else "✅ Stable")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={"text": f"Risque — {feeder_pred}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "#00cc44"},
                    {"range": [30, 60], "color": "#ff8800"},
                    {"range": [60, 100], "color": "#ff4444"}
                ]
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if st.button("🤖 Analyser cette prédiction avec l'IA"):
            df_hist = df_brut[df_brut["feeder"] == feeder_pred]
            contexte = f"""
Feeder : {feeder_pred} | Zone : {zone_pred} | Date : {date_pred}
Score de risque prédit : {proba*100:.1f}% ({niveau})
Température : {temp}°C | Précipitations : {prec}mm | Vent : {vent}km/h
Historique feeder : {len(df_hist)} incidents enregistrés
Causes principales : {df_hist['cause'].value_counts().head(3).to_string()}
END historique total : {df_hist['end'].sum():,.0f} kWh
"""
            with st.spinner("Mistral analyse la prédiction..."):
                analyse = analyser_avec_ia(contexte, "Analyse cette prédiction de risque. Quelles sont les causes probables ? Quels risques concrets pour le réseau ? Quelles actions préventives recommandes-tu avant cette date ?")
            st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 6 — ASSISTANT IA
# ============================================================
elif page == "💬 Assistant IA":
    st.title("💬 Assistant IA — Langage naturel")
    st.markdown("Posez n'importe quelle question sur le réseau en français")

    # Exemples rapides
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        if st.button("Feeders les plus risqués"):
            st.session_state["question_ia"] = "Quels sont les feeders les plus risqués et pourquoi ?"
    with col_e2:
        if st.button("Impact hivernage"):
            st.session_state["question_ia"] = "Quel est l'impact de l'hivernage sur le réseau ?"
    with col_e3:
        if st.button("Prévisions 2026"):
            st.session_state["question_ia"] = "Quelles sont les prévisions pour le reste de 2026 ?"
    with col_e4:
        if st.button("Recommandations maintenance"):
            st.session_state["question_ia"] = "Quelles sont les recommandations de maintenance prioritaires ?"

    question = st.text_area(
        "Votre question :",
        value=st.session_state.get("question_ia", ""),
        height=100,
        placeholder="Ex: Si BIGNONA D3 continue comme ça, que va-t-il se passer ?"
    )

    if st.button("🤖 Analyser", type="primary") and question:
        # Construire contexte global
        contexte = f"""
Dataset Senelec DRS 2022-2026 :
- Total incidents : {len(df_brut_filtre):,}
- Feeders : {df_brut_filtre['feeder'].nunique()}
- Zones : {list(df_brut_filtre['zone'].unique())}
- END total : {df_brut_filtre['end'].sum():,.0f} kWh
- Top 5 feeders par incidents : {df_brut_filtre.groupby('feeder').size().nlargest(5).to_string()}
- Distribution natures : {df_brut_filtre['nature'].value_counts().to_string()}
- Distribution causes : {df_brut_filtre['cause'].value_counts().head(5).to_string()}
- Incidents par année : {df_brut_filtre.groupby('annee').size().to_string()}
- END par année : {df_brut_filtre.groupby('annee')['end'].sum().to_string()}
"""
        with st.spinner("Mistral réfléchit..."):
            reponse = analyser_avec_ia(contexte, question)
        st.markdown(f'<div class="ai-box">{reponse}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 7 — COMPARAISON ANNUELLE
# ============================================================
elif page == "📅 Comparaison annuelle":
    st.title("📅 Comparaison annuelle")

    annees_comp = sorted(df_brut_filtre["annee"].unique())

    # KPIs par année
    st.subheader("KPIs par année")
    kpi_annee = df_brut_filtre.groupby("annee").agg(
        nb_incidents=("end", "count"),
        end_total=("end", "sum"),
        puissance=("puissance_coupee", "sum"),
        duree_moy=("duree_heures", "mean")
    ).reset_index()
    kpi_annee["duree_moy"] = kpi_annee["duree_moy"].round(2)
    st.dataframe(kpi_annee, use_container_width=True)

    if st.button("🤖 Analyser évolution annuelle"):
        contexte = kpi_annee.to_string()
        with st.spinner("Mistral analyse..."):
            analyse = analyser_avec_ia(contexte, "Analyse l'évolution des KPIs année par année. Quelles sont les tendances ? Le réseau s'améliore-t-il ? Que prévoir pour 2026 et après ?")
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Incidents par année et nature")
        nature_annee = df_brut_filtre.groupby(["annee", "nature"]).size().reset_index(name="count")
        fig_na = px.bar(nature_annee, x="annee", y="count", color="nature",
                        barmode="stack",
                        color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_na, use_container_width=True)

    with col2:
        st.subheader("END par année et zone")
        end_az = df_brut_filtre.groupby(["annee", "zone"])["end"].sum().reset_index()
        fig_az = px.bar(end_az, x="annee", y="end", color="zone",
                        barmode="group",
                        color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_az, use_container_width=True)

    # Patterns récurrents
    st.subheader("🔁 Patterns récurrents hors hivernage")
    df_hors_hiv = df_filtre[df_filtre["hivernage"] == 0]
    pattern = df_hors_hiv.groupby(["semaine", "annee"])["incident"].mean().reset_index()
    pattern["annee"] = pattern["annee"].astype(str)
    fig_pat = px.line(pattern, x="semaine", y="incident", color="annee",
                      markers=True,
                      title="Taux d'incidents par semaine hors hivernage",
                      labels={"incident": "Taux", "semaine": "Semaine"})
    st.plotly_chart(fig_pat, use_container_width=True)

    if st.button("🤖 Analyser patterns récurrents"):
        contexte = pattern.to_string()
        with st.spinner("Mistral analyse..."):
            analyse = analyser_avec_ia(contexte, "Identifie les patterns récurrents hors hivernage. Quelles semaines reviennent systématiquement avec des pics d'incidents ? Quelles en sont les causes probables ?")
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 8 — FEEDERS CRITIQUES
# ============================================================
elif page == "⚡ Feeders critiques":
    st.title("⚡ Feeders critiques — Analyse approfondie")

    fragilite_df = df_brut_filtre.groupby("feeder").agg(
        nb_incidents=("end", "count"),
        end_total=("end", "sum"),
        puissance_totale=("puissance_coupee", "sum"),
        duree_moy=("duree_heures", "mean"),
        taux_incident=("end", lambda x: (x > 0).mean())
    ).reset_index().sort_values("end_total", ascending=False)

    fragilite_df["feeder"] = fragilite_df["feeder"].str.strip()
    fragilite_df["duree_moy"] = fragilite_df["duree_moy"].round(2)
    fragilite_df["end_total"] = fragilite_df["end_total"].round(0)

    # Classement
    st.subheader("Classement des feeders")
    st.dataframe(fragilite_df, use_container_width=True, height=350)

    # Feeder sélectionné pour analyse
    feeder_analyse = st.selectbox("Analyser un feeder en détail",
                                   sorted(fragilite_df["feeder"].unique()))

    df_fa = df_brut_filtre[df_brut_filtre["feeder"].str.strip() == feeder_analyse]

    if len(df_fa) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Nb incidents", len(df_fa))
        with col2:
            st.metric("END total (kWh)", f"{df_fa['end'].sum():,.0f}")
        with col3:
            st.metric("Durée moy (h)", f"{df_fa['duree_heures'].mean():.2f}")
        with col4:
            st.metric("Puissance coupée (kW)", f"{df_fa['puissance_coupee'].sum():,.0f}")

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("Causes principales")
            causes = df_fa["cause"].value_counts().head(8).reset_index()
            fig_causes = px.bar(causes, x="count", y="cause", orientation="h",
                                color_discrete_sequence=["#1e3a5f"])
            st.plotly_chart(fig_causes, use_container_width=True)

        with col_g2:
            st.subheader("Distribution par nature")
            natures = df_fa["nature"].value_counts().reset_index()
            fig_nat = px.pie(natures, values="count", names="nature",
                             color_discrete_sequence=px.colors.sequential.Blues_r,
                             hole=0.4)
            st.plotly_chart(fig_nat, use_container_width=True)

        # Évolution END dans le temps
        st.subheader("Évolution END dans le temps")
        end_temps = df_fa.groupby("date")["end"].sum().reset_index()
        fig_temps = px.line(end_temps, x="date", y="end",
                            color_discrete_sequence=["#e63946"])
        st.plotly_chart(fig_temps, use_container_width=True)

        if st.button(f"🤖 Analyse IA complète de {feeder_analyse}"):
            contexte = f"""
Feeder : {feeder_analyse}
Nb incidents : {len(df_fa)}
END total : {df_fa['end'].sum():,.0f} kWh
Durée moyenne : {df_fa['duree_heures'].mean():.2f}h
Top causes : {df_fa['cause'].value_counts().head(5).to_string()}
Distribution natures : {df_fa['nature'].value_counts().to_string()}
Incidents par année : {df_fa.groupby('annee').size().to_string()}
END par année : {df_fa.groupby('annee')['end'].sum().to_string()}
Incidents en hivernage : {df_fa[df_fa['mois'].isin([7,8,9,10])].shape[0]}
Incidents hors hivernage : {df_fa[~df_fa['mois'].isin([7,8,9,10])].shape[0]}
"""
            with st.spinner(f"Mistral analyse {feeder_analyse}..."):
                analyse = analyser_avec_ia(
                    contexte,
                    f"Fais une analyse complète du feeder {feeder_analyse}. Identifie les causes structurelles, les patterns temporels, les risques pour l'hivernage prochain, et donne des recommandations de maintenance précises et prioritaires."
                )
            st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

# ============================================================
# PAGE 9 — TENDANCES
# ============================================================
elif page == "📈 Tendances":
    st.title("📈 Tendances et prévisions")

    # Heatmap incidents feeder x mois
    st.subheader("🗓️ Heatmap — Incidents par feeder et mois")
    heatmap_data = df_filtre.groupby(["feeder", "mois"])["incident"].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index="feeder", columns="mois", values="incident").fillna(0)
    heatmap_pivot.index = heatmap_pivot.index.str.strip()
    heatmap_pivot.columns = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

    fig_heat = px.imshow(heatmap_pivot,
                         color_continuous_scale="Reds",
                         title="Intensité des incidents par feeder et mois",
                         labels=dict(x="Mois", y="Feeder", color="Incidents"))
    st.plotly_chart(fig_heat, use_container_width=True)

    if st.button("🤖 Analyser heatmap"):
        top_combinaisons = heatmap_data.nlargest(10, "incident").to_string()
        with st.spinner("Mistral analyse..."):
            analyse = analyser_avec_ia(
                f"Top combinaisons feeder/mois : {top_combinaisons}",
                "Analyse cette heatmap d'incidents. Quelles combinaisons feeder/mois sont les plus critiques ? Quels patterns saisonniers identifies-tu ? Que prévoir pour les prochains mois ?"
            )
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Tendance par feeder
    st.subheader("📊 Tendance par feeder")
    feeder_tendance = st.selectbox("Choisir feeder", sorted(df_filtre["feeder"].str.strip().unique()))
    df_tend = df_filtre[df_filtre["feeder"].str.strip() == feeder_tendance]
    tend_mois = df_tend.groupby(["annee", "mois"])["incident"].sum().reset_index()
    tend_mois["periode"] = tend_mois["annee"].astype(str) + "-" + tend_mois["mois"].astype(str).str.zfill(2)
    tend_mois = tend_mois.sort_values("periode")

    fig_tend = px.bar(tend_mois, x="periode", y="incident",
                      color_discrete_sequence=["#1e3a5f"],
                      title=f"Évolution mensuelle — {feeder_tendance}")
    fig_tend.update_xaxes(tickangle=45)
    st.plotly_chart(fig_tend, use_container_width=True)

    if st.button("🤖 Prédire tendance future"):
        contexte = f"Feeder : {feeder_tendance}\nHistorique mensuel :\n{tend_mois[['periode','incident']].to_string()}"
        with st.spinner("Mistral analyse et prédit..."):
            analyse = analyser_avec_ia(
                contexte,
                f"Analyse la tendance historique du feeder {feeder_tendance}. Si cette tendance continue, que va-t-il se passer dans les 3 prochains mois ? Quelles périodes seront critiques ? Quelles actions préventives recommandes-tu ?"
            )
        st.markdown(f'<div class="ai-box">{analyse}</div>', unsafe_allow_html=True)