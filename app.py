"""
OpticCrop / CropMind AI: Precision Agriculture Decision Support System
Implements Google Stitch Enterprise Design System (Inter, Slate-Navy #0B1326, Emerald #10B981, Sky-Blue #0284C7).
Zero-Emoji corporate precision architecture powered by XGBoost, TreeSHAP, and SQLite.
"""

import os
import sys

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

from src.explain_engine import predict_and_explain, get_engine
from src.weather_service import fetch_weather_stream, geocode_location
from src.db import load_dataset_from_db
from src.pdf_generator import generate_crop_report
from src.translations import TRANSLATIONS, get_translation
from src.fertilizer_advisor import calculate_nutrient_prescription
from src.irrigation_scheduler import calculate_irrigation_schedule
from src.disease_risk import calculate_disease_pest_risk
from src.economics_engine import calculate_crop_profitability
from src.soil_health import calculate_soil_health_and_carbon
from src.multimodal_processor import get_multimodal_processor

API_URL = "http://localhost:8000/api/v1"

# Configure Streamlit Page
st.set_page_config(
    page_title="CropMind AI - Enterprise Agriculture Decision System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "token" not in st.session_state:
    st.session_state["token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "language" not in st.session_state:
    st.session_state["language"] = "English"
if "persona" not in st.session_state:
    st.session_state["persona"] = "Farmer View"
if "nitrogen" not in st.session_state:
    st.session_state["nitrogen"] = 90.0
if "phosphorus" not in st.session_state:
    st.session_state["phosphorus"] = 42.0
if "potassium" not in st.session_state:
    st.session_state["potassium"] = 43.0
if "ph" not in st.session_state:
    st.session_state["ph"] = 6.5
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 26.5
if "humidity" not in st.session_state:
    st.session_state["humidity"] = 75.0
if "rainfall" not in st.session_state:
    st.session_state["rainfall"] = 110.0
if "active_location" not in st.session_state:
    st.session_state["active_location"] = "Manual Coordinates"
if "geohash" not in st.session_state:
    st.session_state["geohash"] = ""

lang = st.session_state.get("language", "English")

# Google Stitch Enterprise Design System CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Stitch Level 0 Background */
    .stApp {
        background-color: #0B1326;
        color: #DAE2FD;
    }

    /* Stitch Top App Bar / Header */
    .stitch-header {
        background: #171F33;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .stitch-brand {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .stitch-tagline {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Stitch Level 1 Cards */
    .stitch-card {
        background: #171F33;
        border: 1px solid #334155;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .stitch-card-primary {
        background: #171F33;
        border: 1px solid #10B981;
        border-left: 4px solid #10B981;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .stitch-card-highlight {
        background: #171F33;
        border: 1px solid #38BDF8;
        border-left: 4px solid #38BDF8;
        border-radius: 4px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Stitch Pill Badges */
    .stitch-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    .pill-optimal {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pill-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .pill-error {
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .pill-info {
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .pill-neutral {
        background: rgba(148, 163, 184, 0.15);
        color: #94A3B8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Typography & Numeric Display */
    .metric-value-huge {
        font-size: 2.75rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .metric-subtext {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 6px;
    }
    .code-metric {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        font-size: 0.85rem;
        color: #38BDF8;
    }
    .confidence-badge {
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #38BDF8;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Header Section
st.markdown(
    f"""
    <div class="stitch-header">
        <div>
            <div class="stitch-brand">{get_translation(lang, "title")}</div>
            <div class="stitch-tagline">
                {get_translation(lang, "tagline")}
            </div>
        </div>
        <div>
            <span class="stitch-pill pill-optimal">PRD/TRD v1.0.0</span>
            <span class="stitch-pill pill-info">Dual-Auth Engine</span>
            <span class="stitch-pill pill-neutral">Geohash-6 Cache</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def login_form():
    st.subheader(get_translation(lang, "login"))
    with st.form("login_form"):
        username = st.text_input(get_translation(lang, "username"))
        password = st.text_input(get_translation(lang, "password"), type="password")
        submitted = st.form_submit_button(get_translation(lang, "login"))
        if submitted:
            try:
                res = requests.post(f"{API_URL}/auth/login", data={"username": username, "password": password})
                if res.status_code == 200:
                    st.session_state["token"] = res.json()["access_token"]
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            except Exception as e:
                st.error(f"Login connection failed: {e}")


def register_form():
    st.subheader(get_translation(lang, "register"))
    with st.form("register_form"):
        username = st.text_input(get_translation(lang, "username"))
        password = st.text_input(get_translation(lang, "password"), type="password")
        submitted = st.form_submit_button(get_translation(lang, "register"))
        if submitted:
            try:
                res = requests.post(f"{API_URL}/auth/register", json={"username": username, "password": password})
                if res.status_code == 200:
                    st.session_state["token"] = res.json()["access_token"]
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("Registration failed. Username may exist.")
            except Exception as e:
                st.error(f"Registration connection failed: {e}")


if st.session_state["token"] is None:
    t1, t2 = st.tabs([get_translation(lang, "login"), get_translation(lang, "register")])
    with t1:
        login_form()
    with t2:
        register_form()
    st.stop()


# --- AUTHENTICATED DASHBOARD ---

# Stitch Sidebar Navigation
with st.sidebar:
    st.markdown(f"**{get_translation(lang, 'welcome')}, {st.session_state['username']}!**")
    if st.button(get_translation(lang, "logout"), key="logout"):
        st.session_state["token"] = None
        st.session_state["username"] = None
        st.rerun()

    st.markdown("---")
    # Language Selector
    st.session_state["language"] = st.selectbox(
        f"🌐 {get_translation(lang, 'select_language')}",
        options=["English", "Hindi", "Tamil", "Telugu", "Spanish"],
        index=["English", "Hindi", "Tamil", "Telugu", "Spanish"].index(st.session_state["language"]) if st.session_state["language"] in ["English", "Hindi", "Tamil", "Telugu", "Spanish"] else 0,
    )
    lang = st.session_state["language"]

    st.markdown(f"### {get_translation(lang, 'interface_mode')}")
    st.session_state["persona"] = st.radio(
        "Select User Persona",
        [get_translation(lang, "farmer_view"), get_translation(lang, "agronomist_console")],
        index=0 if st.session_state["persona"] in ["Farmer View", get_translation(lang, "farmer_view")] else 1,
        help="Farmer View provides clear primary match cards and actionable advisory. Agronomist Console provides comprehensive TreeSHAP factor attributions and nutrient index calculations.",
    )

    st.markdown("---")
    st.markdown(f"### {get_translation(lang, 'weather_sync')}")
    city_input = st.text_input(
        get_translation(lang, "location_input"),
        placeholder="e.g. Coimbatore, Punjab, Dallas, Nairobi",
    )

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button(get_translation(lang, "sync_weather_btn"), use_container_width=True):
            if city_input:
                with st.spinner("Fetching meteorological data..."):
                    geo_res = geocode_location(city_input)
                    if geo_res.get("success"):
                        lat = geo_res["latitude"]
                        lon = geo_res["longitude"]
                        w_res = fetch_weather_stream(lat, lon, forecast_window_days=14)
                        st.session_state["temperature"] = float(w_res["temperature_avg"])
                        st.session_state["humidity"] = float(w_res["humidity_avg"])
                        st.session_state["rainfall"] = float(w_res["rainfall_equivalent"])
                        st.session_state["active_location"] = f"{geo_res['name']}, {geo_res.get('country','')}".strip(", ")
                        st.session_state["geohash"] = w_res.get("geohash6", "")
                        st.session_state["weather_source"] = w_res.get("source", "unknown")

                        source_badge = "⚡ Live API" if w_res.get("source") == "live_api" else "💾 DB Cache" if w_res.get("source") == "cache" else "📊 Historical Fallback"
                        st.success(f"Synchronized: {st.session_state['active_location']} | {source_badge}")
                    else:
                        st.error(geo_res.get("error", "Geocoding failed."))
            else:
                st.warning("Please enter a location query.")

    with col_s2:
        if st.button(get_translation(lang, "reset_defaults_btn"), use_container_width=True):
            st.session_state["nitrogen"] = 90.0
            st.session_state["phosphorus"] = 42.0
            st.session_state["potassium"] = 43.0
            st.session_state["ph"] = 6.5
            st.session_state["temperature"] = 26.5
            st.session_state["humidity"] = 75.0
            st.session_state["rainfall"] = 110.0
            st.session_state["active_location"] = "Manual Coordinates"
            st.rerun()

    if st.session_state.get("active_location"):
        st.markdown(
            f"""
            <div class="stitch-card" style="padding: 12px; margin-top: 10px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Active Location</div>
                <div style="font-weight: 600; font-size: 0.92rem; color: #DAE2FD;">{st.session_state['active_location']}</div>
                <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">Geohash Level-6: <span class="code-metric">{st.session_state.get('geohash','N/A')}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### System Architecture")
    st.caption("Database: data/optic_crop.db (SQLite)")
    st.caption("Cache: data/crop_dataset.parquet")
    st.caption("Inference Engine: FastAPI / XGBoost Hist")

# Stitch Navigation Tabs
(
    tab_rec,
    tab_fertilizer,
    tab_disease,
    tab_economics,
    tab_soil,
    tab_whatif,
    tab_analytics,
    tab_db,
    tab_multimodal,
    tab_history,
    tab_batch,
    tab_admin,
    tab_farms,
    tab_api,
) = st.tabs(
    [
        get_translation(lang, "recommendations"),
        get_translation(lang, "fertilizer_advisor"),
        get_translation(lang, "disease_risk"),
        get_translation(lang, "economics"),
        get_translation(lang, "soil_health"),
        get_translation(lang, "simulation"),
        get_translation(lang, "model_validation"),
        get_translation(lang, "database_records"),
        get_translation(lang, "multimodal_datacube"),
        get_translation(lang, "my_history"),
        get_translation(lang, "batch_prediction"),
        get_translation(lang, "admin"),
        get_translation(lang, "my_farms"),
        get_translation(lang, "api_keys"),
    ]
)

# Shared Controls across tabs
n_val = float(st.session_state["nitrogen"])
p_val = float(st.session_state["phosphorus"])
k_val = float(st.session_state["potassium"])
ph_val = float(st.session_state["ph"])
temp_val = float(st.session_state["temperature"])
hum_val = float(st.session_state["humidity"])
rain_val = float(st.session_state["rainfall"])


# ==============================================================================
# TAB 1: RECOMMENDATIONS & FACTOR ATTRIBUTION
# ==============================================================================
with tab_rec:
    col_in1, col_in2 = st.columns([1, 1])

    with col_in1:
        st.markdown(f"#### {get_translation(lang, 'soil_params')}")
        n_val = st.slider("Nitrogen (N) [mg/kg]", 0.0, 150.0, n_val, 1.0)
        p_val = st.slider("Phosphorus (P) [mg/kg]", 5.0, 150.0, p_val, 1.0)
        k_val = st.slider("Potassium (K) [mg/kg]", 5.0, 210.0, k_val, 1.0)
        ph_val = st.slider("Soil pH Level", 3.5, 10.0, ph_val, 0.1)

    with col_in2:
        st.markdown(f"#### {get_translation(lang, 'weather_params')}")
        temp_val = st.slider("Ambient Temperature (deg C)", 5.0, 50.0, temp_val, 0.5)
        hum_val = st.slider("Relative Humidity (%)", 10.0, 100.0, hum_val, 1.0)
        rain_val = st.slider("Forecasted Precipitation Sum (mm)", 15.0, 350.0, rain_val, 5.0)

    # Sync back to session state
    st.session_state["nitrogen"] = n_val
    st.session_state["phosphorus"] = p_val
    st.session_state["potassium"] = k_val
    st.session_state["ph"] = ph_val
    st.session_state["temperature"] = temp_val
    st.session_state["humidity"] = hum_val
    st.session_state["rainfall"] = rain_val

    # Execute Prediction Pipeline
    results = predict_and_explain(
        nitrogen=n_val,
        phosphorus=p_val,
        potassium=k_val,
        temperature=temp_val,
        humidity=hum_val,
        ph=ph_val,
        rainfall=rain_val,
        top_k=5,
    )

    recs = results["recommendations"]
    top_rec = recs[0]

    # Primary Optimal Match Card
    st.markdown(
        f"""
        <div class="stitch-card-primary">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <span class="stitch-pill pill-optimal">Rank #1 Optimal Crop Match</span>
                    <div class="metric-value-huge" style="margin: 10px 0 6px 0;">{top_rec['crop']}</div>
                    <div class="metric-subtext">
                        <b>Human-Readable Explanation</b>: {top_rec['explanations']['human_readable_summary']}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Viability Score</div>
                    <div style="font-size: 2.2rem; font-weight: 700; color: #10B981;">{top_rec['confidence_percent']:.1f}%</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Alternative Options
    st.markdown(f"#### {get_translation(lang, 'ranked_alternatives')}")
    alt_cols = st.columns(len(recs) - 1)
    for idx, alt in enumerate(recs[1:]):
        with alt_cols[idx]:
            st.markdown(
                f"""
                <div class="stitch-card" style="padding: 14px;">
                    <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">Rank #{alt['rank']}</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{alt['crop']}</div>
                    <div style="font-size: 0.85rem; color: #38BDF8; font-weight: 600;">{alt['confidence_percent']:.1f}% Suitability</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Explainable AI Attribution
    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        contrib_df = pd.DataFrame(top_rec["explanations"]["all_contributions"])
        contrib_df["direction"] = contrib_df["shap_delta"].apply(
            lambda v: "Positive Factor (Supports Recommendation)" if v >= 0 else "Limiting Factor (Deviates from Optimal)"
        )
        fig_shap = px.bar(
            contrib_df,
            x="shap_delta",
            y="label",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Positive Factor (Supports Recommendation)": "#10B981",
                "Limiting Factor (Deviates from Optimal)": "#EF4444",
            },
            title=f"Factor Attribution for {top_rec['crop']}",
            labels={"shap_delta": "SHAP Impact Delta", "label": "Parameter"},
            text=contrib_df["shap_delta"].apply(lambda v: f"{v:+.3f}"),
        )
        fig_shap.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171F33",
            plot_bgcolor="#0B1326",
            font=dict(family="Inter", color="#DAE2FD"),
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(autorange="reversed"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_chart2:
        engine = get_engine()
        crop_prof = engine.crop_profiles.get(top_rec["crop"], {})
        if crop_prof:
            radar_feats = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
            user_vals = [n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]
            opt_vals = [crop_prof[f]["mean"] for f in radar_feats]
            pct_user = [min(round((u / max(o, 1e-3)) * 100, 1), 180) for u, o in zip(user_vals, opt_vals)]
            pct_opt = [100.0] * len(radar_feats)

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=pct_user,
                    theta=[engine.feature_labels.get(f, f) for f in radar_feats],
                    fill="toself",
                    name="Current Input",
                    line=dict(color="#38BDF8", width=2),
                    fillcolor="rgba(56, 189, 248, 0.2)",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=pct_opt,
                    theta=[engine.feature_labels.get(f, f) for f in radar_feats],
                    name=f"Optimal Benchmark ({top_rec['crop']})",
                    line=dict(color="#10B981", width=2, dash="dash"),
                )
            )
            fig_radar.update_layout(
                template="plotly_dark",
                polar=dict(radialaxis=dict(visible=True, range=[0, 160], color="#94A3B8"), bgcolor="#171F33"),
                paper_bgcolor="#171F33",
                title="Field Alignment vs Benchmark (%)",
                font=dict(family="Inter", color="#DAE2FD"),
                height=380,
                margin=dict(l=40, r=40, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # PDF Download Button
    st.markdown("---")
    col_pdf1, col_pdf2 = st.columns([1, 2])
    with col_pdf1:
        pdf_bytes = generate_crop_report(
            crop_name=top_rec["crop"],
            viability=top_rec["viability_score"],
            n=n_val,
            p=p_val,
            k=k_val,
            temp=temp_val,
            hum=hum_val,
            ph=ph_val,
            rain=rain_val,
            summary=top_rec["explanations"]["human_readable_summary"],
            location_name=st.session_state.get("active_location", "Selected Coordinates"),
            positive_factors=[f"{f['feature']}: {f['shap_delta']:+.3f}" for f in top_rec["explanations"]["top_positive_factors"]],
            negative_factors=[f"{f['feature']}: {f['shap_delta']:+.3f}" for f in top_rec["explanations"]["top_negative_factors"]],
        )
        st.download_button(
            label="📄 " + get_translation(lang, "generate_report_pdf"),
            data=pdf_bytes,
            file_name=f"CropMind_{top_rec['crop']}_Advisory_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


# ==============================================================================
# TAB 2: FERTILIZER & IRRIGATION ADVISOR
# ==============================================================================
with tab_fertilizer:
    st.markdown("### 🌱 Precision Fertilizer Prescription & Irrigation Schedule")
    st.write(
        "Translates soil nutrient deficits into exact commercial bag counts (Urea, DAP, MOP) and calculates daily water balance using FAO-56 Penman-Monteith ET0."
    )

    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        st.markdown("#### Field & Cropping Parameters")
        target_crop = st.selectbox(
            "Target Crop for Advisory:",
            sorted(list(load_dataset_from_db()["crop"].unique())),
            index=sorted(list(load_dataset_from_db()["crop"].unique())).index(top_rec["crop"]) if top_rec["crop"] in load_dataset_from_db()["crop"].unique() else 0,
        )
        farm_acres = st.number_input("Field Area (Acres)", value=1.0, min_value=0.1, max_value=500.0, step=0.5)
        irr_method = st.selectbox("Irrigation Delivery Method", ["Drip Irrigation", "Sprinkler", "Flood / Furrow"])

    fert_res = calculate_nutrient_prescription(
        crop_name=target_crop,
        soil_n=n_val,
        soil_p=p_val,
        soil_k=k_val,
        soil_ph=ph_val,
        field_area_acres=farm_acres,
    )

    irr_res = calculate_irrigation_schedule(
        crop_name=target_crop,
        temperature_c=temp_val,
        humidity_pct=hum_val,
        rainfall_14d_mm=rain_val,
        field_area_acres=farm_acres,
        irrigation_method=irr_method,
    )

    with col_f2:
        st.markdown("#### Prescribed Commercial Fertilizer Quantity")
        f_c1, f_c2, f_c3 = st.columns(3)
        with f_c1:
            st.metric("Urea (46% N)", f"{fert_res['commercial_prescription']['urea_50kg_bags']} Bags (50kg)")
        with f_c2:
            st.metric("DAP (18:46:0)", f"{fert_res['commercial_prescription']['dap_50kg_bags']} Bags (50kg)")
        with f_c3:
            st.metric("MOP (60% K2O)", f"{fert_res['commercial_prescription']['mop_50kg_bags']} Bags (50kg)")

        st.markdown(
            f"""
            <div class="stitch-card" style="padding: 12px; margin-top: 10px;">
                <span class="stitch-pill {'pill-optimal' if fert_res['ph_remediation']['status'] == 'Optimal' else 'pill-warning'}">{fert_res['ph_remediation']['status']} Soil pH</span>
                <div style="font-size: 0.85rem; color: #DAE2FD; margin-top: 6px;">{fert_res['ph_remediation']['remedy']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 💧 Daily Irrigation & Evapotranspiration Dynamics")
    i_c1, i_c2, i_c3, i_c4 = st.columns(4)
    with i_c1:
        st.metric("Reference ET0", f"{irr_res['reference_et0_mm_day']} mm/day")
    with i_c2:
        st.metric("Crop Water Demand (ETc)", f"{irr_res['crop_etc_mm_day']} mm/day")
    with i_c3:
        st.metric("Daily Irrigation Volume", f"{irr_res['water_volume_litres_acre_day']:,.0f} L/acre/day")
    with i_c4:
        st.metric("System Runtime", f"~{irr_res['recommended_runtime_hours']} Hours/day")

    st.info(f"💧 **Watering Guidance**: {irr_res['actionable_advisory']}")


# ==============================================================================
# TAB 3: DISEASE & PEST RISK MATRIX
# ==============================================================================
with tab_disease:
    st.markdown("### 🛡️ Crop Disease & Pest Risk Forecasting Matrix")
    st.write("Predicts fungal, bacterial, and pest vulnerability based on microclimate indices and crop phenology.")

    disease_crop = st.selectbox(
        "Evaluate Disease Vulnerability for:",
        sorted(list(load_dataset_from_db()["crop"].unique())),
        index=sorted(list(load_dataset_from_db()["crop"].unique())).index(top_rec["crop"]) if top_rec["crop"] in load_dataset_from_db()["crop"].unique() else 0,
        key="disease_crop_select",
    )

    disease_res = calculate_disease_pest_risk(
        crop_name=disease_crop,
        temperature_c=temp_val,
        humidity_pct=hum_val,
        rainfall_14d_mm=rain_val,
    )

    st.markdown(
        f"""
        <div class="stitch-card-highlight">
            <span class="stitch-pill {'pill-error' if 'Severe' in disease_res['overall_risk_status'] else 'pill-warning' if 'High' in disease_res['overall_risk_status'] else 'pill-optimal'}">
                {disease_res['overall_risk_status']} (Max DSI Score: {disease_res['max_risk_score']}%)
            </span>
            <div style="font-size: 0.9rem; color: #CBD5E1; margin-top: 8px;">{disease_res['ipm_advisory']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for p in disease_res["pathogens"]:
        with st.expander(f"⚠️ {p['pathogen_name']} [{p['type']}] - Risk: {p['risk_level']} ({p['dsi_score']}%)", expanded=(p["dsi_score"] >= 60)):
            p_col1, p_col2 = st.columns([1, 1])
            with p_col1:
                st.markdown(f"**Symptoms**: {p['symptoms']}")
                st.markdown(f"🌱 **Bio-Control Remedy**: {p['bio_control']}")
            with p_col2:
                st.markdown(f"🧪 **Chemical Treatment**: {p['chemical_cure']}")


# ==============================================================================
# TAB 4: ECONOMIC YIELD & PROFIT OPTIMIZER
# ==============================================================================
with tab_economics:
    st.markdown("### 💰 Agricultural Economics & Profitability Optimizer")
    st.write("Computes expected crop yields, mandi market revenue, input expenditure breakdown, and net profit margins.")

    econ_crop = st.selectbox(
        "Crop for Profit Projection:",
        sorted(list(load_dataset_from_db()["crop"].unique())),
        index=sorted(list(load_dataset_from_db()["crop"].unique())).index(top_rec["crop"]) if top_rec["crop"] in load_dataset_from_db()["crop"].unique() else 0,
        key="econ_crop_select",
    )

    econ_acres = st.slider("Field Cultivation Area (Acres)", 0.5, 50.0, 2.0, 0.5)

    econ_res = calculate_crop_profitability(
        crop_name=econ_crop,
        viability_score=top_rec["viability_score"],
        field_area_acres=econ_acres,
    )

    e_c1, e_c2, e_c3, e_c4 = st.columns(4)
    with e_c1:
        st.metric("Estimated Total Yield", f"{econ_res['yield_estimates']['total_yield_tons']:.1f} Tons")
    with e_c2:
        st.metric("Gross Mandi Revenue", f"₹{econ_res['financial_summary']['gross_revenue_inr']:,.0f}")
    with e_c3:
        st.metric("Total Input Cost", f"₹{econ_res['financial_summary']['total_input_cost_inr']:,.0f}")
    with e_c4:
        st.metric(
            "Net Profit Margin",
            f"₹{econ_res['financial_summary']['net_profit_inr']:,.0f}",
            f"ROI: {econ_res['financial_summary']['roi_pct']:.1f}%",
        )

    # Cost breakdown chart
    cost_df = pd.DataFrame(econ_res["cost_breakdown"])
    fig_cost = px.pie(
        cost_df,
        values="cost_inr",
        names="category",
        title="Input Expenditure Distribution (INR)",
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig_cost.update_layout(
        template="plotly_dark",
        paper_bgcolor="#171F33",
        plot_bgcolor="#0B1326",
        font=dict(family="Inter", color="#DAE2FD"),
        height=340,
    )
    st.plotly_chart(fig_cost, use_container_width=True)


# ==============================================================================
# TAB 5: SOIL HEALTH & CARBON FOOTPRINT
# ==============================================================================
with tab_soil:
    st.markdown("### 🌍 Soil Health Index (SHI) & Carbon Footprint Scorecard")
    st.write(
        "Computes multidimensional soil quality scores, nitrogen leaching risk, and IPCC greenhouse gas emissions (kg CO2e) per acre."
    )

    soil_res = calculate_soil_health_and_carbon(
        soil_n=n_val,
        soil_p=p_val,
        soil_k=k_val,
        soil_ph=ph_val,
        rainfall_mm=rain_val,
        temperature_c=temp_val,
        field_area_acres=1.0,
    )

    s_c1, s_c2, s_c3 = st.columns(3)
    with s_c1:
        st.metric("Soil Health Index (SHI)", f"{soil_res['soil_health_index']}/100", soil_res["soil_health_rating"])
    with s_c2:
        st.metric("Nitrogen Leaching Risk", f"{soil_res['nitrogen_leaching']['leaching_risk_percentage']:.1f}%", f"{soil_res['nitrogen_leaching']['estimated_n_leached_kg']:.1f} kg N lost")
    with s_c3:
        st.metric("Total GHG Footprint", f"{soil_res['carbon_footprint']['total_ghg_emissions_kg_co2e']:,.0f} kg CO2e", "IPCC Tier 1")

    st.markdown("#### 🌿 Regenerative Agriculture & Carbon Sequestration Checklist")
    for r in soil_res["regenerative_advisory"]:
        st.markdown(f"- **{r['practice']}**: {r['impact']}")


# ==============================================================================
# TAB 6: SCENARIO SIMULATION
# ==============================================================================
with tab_whatif:
    st.markdown("### Scenario Simulation Engine")
    st.write(
        "Evaluate the sensitivity of crop viability distributions against simulated variations in precipitation, temperature shifts, and nutrient adjustments."
    )

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.markdown("#### Simulation Levers")
        sim_rain_delta = st.slider("Precipitation / Irrigation Delta (mm)", -100.0, 150.0, 0.0, 10.0)
        sim_temp_delta = st.slider("Temperature Shift (deg C)", -5.0, 8.0, 0.0, 0.5)
        sim_n_delta = st.slider("Nitrogen Adjustment (mg/kg)", -50.0, 60.0, 0.0, 5.0)
        sim_p_delta = st.slider("Phosphorus Adjustment (mg/kg)", -30.0, 50.0, 0.0, 5.0)

        effective_rain = max(10.0, rain_val + sim_rain_delta)
        effective_temp = max(5.0, temp_val + sim_temp_delta)
        effective_n = max(5.0, n_val + sim_n_delta)
        effective_p = max(5.0, p_val + sim_p_delta)

        st.markdown(
            f"""
            <div class="stitch-card" style="padding: 14px;">
                <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Effective Conditions</div>
                <div style="font-size: 0.85rem; color: #DAE2FD; margin-top: 6px; line-height: 1.6;">
                    Temperature: <b>{effective_temp:.1f} deg C</b> (Delta: {sim_temp_delta:+.1f})<br>
                    Precipitation: <b>{effective_rain:.0f} mm</b> (Delta: {sim_rain_delta:+.0f})<br>
                    Nitrogen: <b>{effective_n:.0f} mg/kg</b> (Delta: {sim_n_delta:+.0f})<br>
                    Phosphorus: <b>{effective_p:.0f} mg/kg</b> (Delta: {sim_p_delta:+.0f})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with sim_col2:
        sim_results = predict_and_explain(
            nitrogen=effective_n,
            phosphorus=effective_p,
            potassium=k_val,
            temperature=effective_temp,
            humidity=hum_val,
            ph=ph_val,
            rainfall=effective_rain,
            top_k=6,
        )

        sim_recs = sim_results["recommendations"]
        sim_df = pd.DataFrame(
            [
                {
                    "Crop": r["crop"],
                    "Simulated Viability (%)": r["confidence_percent"],
                    "Rank": r["rank"],
                }
                for r in sim_recs
            ]
        )

        fig_sim = px.bar(
            sim_df,
            x="Simulated Viability (%)",
            y="Crop",
            orientation="h",
            color="Simulated Viability (%)",
            color_continuous_scale="Blues",
            title="Projected Crop Suitability Distribution",
            text=sim_df["Simulated Viability (%)"].apply(lambda v: f"{v:.1f}%"),
        )
        fig_sim.update_layout(
            template="plotly_dark",
            paper_bgcolor="#171F33",
            plot_bgcolor="#0B1326",
            font=dict(family="Inter", color="#DAE2FD"),
            height=380,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ==============================================================================
# TAB 7: MODEL VALIDATION & METRICS
# ==============================================================================
with tab_analytics:
    st.markdown("### Model Validation & Performance Benchmarks")

    engine = get_engine()
    meta = engine.metadata

    if meta and "benchmarks" in meta:
        benchmarks = meta["benchmarks"]
        bench_df = pd.DataFrame(benchmarks).T.reset_index()
        bench_df.columns = ["Architecture", "Accuracy", "Precision", "Recall", "Macro F1"]
        bench_df["Accuracy (%)"] = (bench_df["Accuracy"] * 100).round(2)
        bench_df["Precision"] = bench_df["Precision"].round(4)
        bench_df["Recall"] = bench_df["Recall"].round(4)
        bench_df["Macro F1"] = bench_df["Macro F1"].round(4)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Primary Architecture", "XGBoost (Hist Tree)", help="Regularized Hist Tree Gradient Boosting")
        with m_col2:
            st.metric("Multi-Class Accuracy", f"{meta.get('accuracy', 0.9886)*100:.2f}%", help="Target: >= 92%")
        with m_col3:
            st.metric("Macro F1-Score", f"{meta.get('macro_f1', 0.9885):.4f}", help="Target: >= 0.89")

        st.markdown("#### Benchmark Matrix")
        st.dataframe(
            bench_df[["Architecture", "Accuracy (%)", "Precision", "Recall", "Macro F1"]],
            use_container_width=True,
            hide_index=True,
        )


# ==============================================================================
# TAB 8: DATABASE RECORDS
# ==============================================================================
with tab_db:
    st.markdown("### Relational Database Records")
    st.write(
        "Structured records queried directly from the SQLite relational database (data/optic_crop.db) and Parquet columnar cache."
    )

    db_df = load_dataset_from_db()
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total SQLite Records", len(db_df))
    with col_stat2:
        st.metric("Registered Crop Species", db_df["crop"].nunique())
    with col_stat3:
        st.metric("Storage Backends", "SQLite + Parquet")

    selected_filter = st.selectbox("Filter Records by Crop Species", ["All Crops"] + sorted(db_df["crop"].unique().tolist()))
    display_df = db_df if selected_filter == "All Crops" else db_df[db_df["crop"] == selected_filter]
    st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)


# ==============================================================================
# TAB 9: MULTIMODAL INDIA DATACUBE
# ==============================================================================
with tab_multimodal:
    st.markdown("### 🛰️ Multi-Modal Agriculture DataCube (India)")
    processor = get_multimodal_processor()

    selected_zone = st.selectbox(
        "Select Indian Agro-Climatic Zone / DataCube Preset:",
        [
            "🌾 Punjab Wheat-Rice Belt (Ludhiana District)",
            "🌾 Cauvery Delta Rice Zone (Thanjavur, Tamil Nadu)",
            "☁️ Maharashtra Black-Soil Cotton Belt (Nashik)",
            "🌱 MP Malwa Plateau Pulses & Chickpea Zone (Indore)",
        ],
    )

    raw_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    if "Punjab" in selected_zone:
        sat_path = os.path.join(raw_base, "satellite", "sentinel2_punjab_wheat_belt.tif")
        cli_path = os.path.join(raw_base, "climate", "imd_gridded_monsoon_climate.nc")
        soil_path = os.path.join(raw_base, "soil", "india_soil_health_card_districts.csv")
        yld_path = os.path.join(raw_base, "yield", "icrisat_district_crop_yield.xlsx")
        region_id = "Punjab_Ludhiana"
    elif "Cauvery" in selected_zone:
        sat_path = os.path.join(raw_base, "satellite", "sentinel2_cauvery_rice_paddy.tif")
        cli_path = os.path.join(raw_base, "climate", "imd_gridded_monsoon_climate.nc")
        soil_path = os.path.join(raw_base, "soil", "soilgrids_india_ph_topsoil.tif")
        yld_path = os.path.join(raw_base, "yield", "ministry_agri_apy_yield.csv")
        region_id = "TamilNadu_Thanjavur"
    elif "Maharashtra" in selected_zone:
        sat_path = os.path.join(raw_base, "satellite", "sentinel2_punjab_wheat_belt.tif")
        cli_path = os.path.join(raw_base, "climate", "imd_gridded_monsoon_climate.nc")
        soil_path = os.path.join(raw_base, "soil", "india_soil_health_card_districts.csv")
        yld_path = os.path.join(raw_base, "yield", "india_crop_yield_benchmarks.json")
        region_id = "Maharashtra_Nashik"
    else:
        sat_path = os.path.join(raw_base, "satellite", "sentinel2_cauvery_rice_paddy.tif")
        cli_path = os.path.join(raw_base, "climate", "imd_gridded_monsoon_climate.nc")
        soil_path = os.path.join(raw_base, "soil", "soilgrids_india_ph_topsoil.tif")
        yld_path = os.path.join(raw_base, "yield", "icrisat_district_crop_yield.xlsx")
        region_id = "MP_Indore"

    try:
        sat_res = processor.load_satellite(sat_path)
        cli_res = processor.load_climate(cli_path)
        soil_res = processor.load_soil(soil_path)
        yld_res = processor.load_yield_stats(yld_path)
        fused_cube = processor.fuse_multimodal_datacube(sat_res, cli_res, soil_res, yld_res, region_id)
        if fused_cube:
            st.success(f"Successfully fused Multi-Modal DataCube for '{region_id}'!")
    except Exception as e:
        st.error(f"Multimodal processor error: {e}")


# ==============================================================================
# TAB 10: MY HISTORY
# ==============================================================================
with tab_history:
    st.markdown("### Prediction History")
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    try:
        res = requests.get(f"{API_URL}/recommendations/history", headers=headers)
        if res.status_code == 200:
            history_data = res.json().get("data", [])
            if len(history_data) > 0:
                st.dataframe(pd.DataFrame(history_data).drop(columns=["id", "user_id"], errors="ignore"), use_container_width=True)
            else:
                st.info("No prediction history found.")
        else:
            st.error("Could not fetch history.")
    except Exception as e:
        st.error(f"API Error: {e}")


# ==============================================================================
# TAB 11: BATCH PREDICTION
# ==============================================================================
with tab_batch:
    st.header("Bulk Crop Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")
    if uploaded_file is not None:
        if st.button("Run Batch Prediction"):
            headers = {"Authorization": f"Bearer {st.session_state['token']}"}
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            try:
                res = requests.post(f"{API_URL}/recommendations/predict/batch", headers=headers, files=files)
                if res.status_code == 200:
                    batch_data = res.json()["data"]
                    st.success(f"Successfully processed {len(batch_data)} rows!")
                    st.dataframe(pd.DataFrame(batch_data), use_container_width=True)
                else:
                    st.error(f"Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")


# ==============================================================================
# TAB 12: ADMIN DASHBOARD
# ==============================================================================
with tab_admin:
    st.header("Admin Dashboard")
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    try:
        res = requests.get(f"{API_URL}/admin/metrics", headers=headers)
        if res.status_code == 200:
            metrics = res.json()["data"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Users", metrics["total_users"])
            col2.metric("Total Predictions", metrics["total_predictions"])
            col3.metric("Top Predicted Crop", metrics["top_crop"])
            col4.metric("System Status", metrics["system_status"])
        else:
            st.info("Log in with an Admin account to view system-wide telemetry.")
    except Exception as e:
        st.error(f"Admin API error: {e}")


# ==============================================================================
# TAB 13: MY FARMS
# ==============================================================================
with tab_farms:
    st.header("My Farms (Saved Profiles)")
    headers = {"Authorization": f"Bearer {st.session_state['token']}"}
    with st.expander("➕ Add New Farm", expanded=False):
        with st.form("add_farm_form"):
            farm_name = st.text_input("Farm Name (e.g., 'North Field')")
            f_lat = st.number_input("Latitude", value=13.0, min_value=-90.0, max_value=90.0)
            f_lon = st.number_input("Longitude", value=80.0, min_value=-180.0, max_value=180.0)
            f_n = st.number_input("Nitrogen (mg/kg)", value=90.0)
            f_p = st.number_input("Phosphorus (mg/kg)", value=42.0)
            f_k = st.number_input("Potassium (mg/kg)", value=43.0)
            f_ph = st.number_input("pH Level", value=6.5, min_value=2.0, max_value=12.0)
            submitted = st.form_submit_button("Save Farm")
            if submitted and farm_name:
                payload = {"farm_name": farm_name, "latitude": f_lat, "longitude": f_lon, "nitrogen": f_n, "phosphorus": f_p, "potassium": f_k, "ph": f_ph}
                res = requests.post(f"{API_URL}/farms", headers=headers, json=payload)
                if res.status_code == 200:
                    st.success(f"Farm '{farm_name}' saved!")
                    st.rerun()


# ==============================================================================
# TAB 14: DEVELOPER API
# ==============================================================================
with tab_api:
    st.header("Developer API Keys & Microservices")
    st.write("Generate API keys to programmatically interact with CropMind AI prediction and advisory microservices.")

    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        if st.button("Generate New API Key"):
            headers = {"Authorization": f"Bearer {st.session_state['token']}"}
            res = requests.post(f"{API_URL}/keys", headers=headers)
            if res.status_code == 200:
                new_key = res.json()["api_key"]
                st.success("API Key Generated Successfully!")
                st.code(new_key, language="bash")
                st.info("Please copy your key now. For security reasons, it cannot be displayed again.")
            else:
                st.error("Failed to generate API Key.")

    with col_k2:
        st.subheader("Active API Keys")
        headers = {"Authorization": f"Bearer {st.session_state['token']}"}
        try:
            res = requests.get(f"{API_URL}/keys", headers=headers)
            if res.status_code == 200:
                keys = res.json().get("data", [])
                if keys:
                    for k in keys:
                        st.markdown(f"**Key ID:** `{k['id']}` | **Created:** `{k['created_at']}`")
                        if st.button(f"Revoke Key {k['id']}", key=f"revoke_{k['id']}"):
                            requests.delete(f"{API_URL}/keys/{k['id']}", headers=headers)
                            st.rerun()
                else:
                    st.info("No active API keys found.")
        except Exception as e:
            st.error(f"API Error: {e}")

    st.markdown("### Example API Microservice Call")
    st.code(
        '''
curl -X POST "http://localhost:8000/api/v1/recommendations/predict" \\
     -H "X-API-Key: cm_your_api_key_here" \\
     -H "Content-Type: application/json" \\
     -d '{"latitude": 13.08, "longitude": 80.27, "nitrogen": 90, "phosphorus": 42, "potassium": 43, "ph": 6.5}'
        ''',
        language="bash",
    )
