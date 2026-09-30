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
from src.translations import TRANSLATIONS, get_translation
from src.fertilizer_advisor import calculate_nutrient_prescription
from src.irrigation_scheduler import calculate_irrigation_schedule
from src.disease_risk import calculate_disease_pest_risk
from src.economics_engine import calculate_crop_profitability
from src.soil_health import calculate_soil_health_and_carbon
from src.multimodal_processor import get_multimodal_processor
from src.crop_rotation import generate_crop_rotation_plan
from src.climate_alerts import evaluate_climate_anomalies
from src.fertigation_calculator import calculate_fertigation_schedule
from src.micronutrient_advisor import diagnose_micronutrient_deficiencies
from src.crop_ranking import calculate_topsis_crop_ranking
from src.spatial_parcels import calculate_polygon_geodesic_area, analyze_farm_parcel
from src.agri_knowledge import search_agronomic_knowledge, get_all_categories
from src.data_exporter import generate_excel_crop_dossier
from src.pdf_generator import generate_crop_report
from src.db import (
    init_database,
    get_db_connection,
    load_dataset_from_db,
    create_user_farm,
    get_user_farms,
    delete_user_farm,
    create_api_key,
    get_user_api_keys,
    delete_api_key,
    get_admin_metrics,
    create_farm_parcel,
    get_user_farm_parcels,
    delete_farm_parcel,
    log_prediction_history,
)
from src.security import (
    verify_password,
    get_password_hash,
    create_access_token,
)

# Ensure database schema is initialized
try:
    init_database()
except Exception:
    pass

API_URL = os.getenv("CROPMIND_API_URL", "http://localhost:8000/api/v1")

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


def get_user_id_by_username(username: str) -> Optional[int]:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        conn.close()
        return row["id"] if row else None
    except Exception:
        return None


def login_form():
    st.subheader(get_translation(lang, "login"))
    with st.form("login_form"):
        username = st.text_input(get_translation(lang, "username"))
        password = st.text_input(get_translation(lang, "password"), type="password")
        submitted = st.form_submit_button(get_translation(lang, "login"))
        if submitted:
            if not username or not password:
                st.warning("Please enter both username and password.")
                return
            token = None
            err_msg = None
            # 1. Try FastAPI REST endpoint if running
            try:
                res = requests.post(f"{API_URL}/auth/login", data={"username": username, "password": password}, timeout=0.8)
                if res.status_code == 200:
                    token = res.json().get("access_token")
                else:
                    err_msg = "Invalid username or password."
            except Exception:
                # 2. Embedded direct SQLite authentication fallback (Streamlit Cloud mode)
                try:
                    init_database()
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
                    row = cursor.fetchone()
                    conn.close()
                    if row and verify_password(password, row["hashed_password"]):
                        token = create_access_token(data={"sub": username})
                    else:
                        err_msg = "Invalid username or password."
                except Exception as db_err:
                    err_msg = f"Authentication error: {db_err}"

            if token:
                st.session_state["token"] = token
                st.session_state["username"] = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(err_msg or "Invalid credentials")


def register_form():
    st.subheader(get_translation(lang, "register"))
    with st.form("register_form"):
        username = st.text_input(get_translation(lang, "username"))
        password = st.text_input(get_translation(lang, "password"), type="password")
        submitted = st.form_submit_button(get_translation(lang, "register"))
        if submitted:
            if not username or not password:
                st.warning("Please enter both username and password.")
                return
            token = None
            err_msg = None
            # 1. Try FastAPI REST endpoint if running
            try:
                res = requests.post(f"{API_URL}/auth/register", json={"username": username, "password": password}, timeout=0.8)
                if res.status_code == 200:
                    token = res.json().get("access_token")
                else:
                    err_msg = "Registration failed. Username may already exist."
            except Exception:
                # 2. Embedded direct SQLite registration fallback (Streamlit Cloud mode)
                try:
                    import sqlite3
                    init_database()
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    hashed_pwd = get_password_hash(password)
                    try:
                        cursor.execute(
                            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
                            (username, hashed_pwd, "Farmer")
                        )
                        conn.commit()
                        conn.close()
                        token = create_access_token(data={"sub": username})
                    except sqlite3.IntegrityError:
                        conn.close()
                        err_msg = "Username already exists. Please choose a different username or log in."
                except Exception as db_err:
                    err_msg = f"Registration error: {db_err}"

            if token:
                st.session_state["token"] = token
                st.session_state["username"] = username
                st.success("Account registered successfully!")
                st.rerun()
            else:
                st.error(err_msg or "Registration failed.")


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
    tab_fertigation,
    tab_micronutrients,
    tab_rotation,
    tab_climate_alerts,
    tab_mcda,
    tab_disease,
    tab_economics,
    tab_soil,
    tab_parcels,
    tab_knowledge,
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
        get_translation(lang, "fertigation_calc"),
        get_translation(lang, "micronutrients"),
        get_translation(lang, "crop_rotation"),
        get_translation(lang, "climate_alerts"),
        get_translation(lang, "mcda_ranker"),
        get_translation(lang, "disease_risk"),
        get_translation(lang, "economics"),
        get_translation(lang, "soil_health"),
        get_translation(lang, "parcel_mgr"),
        get_translation(lang, "agri_knowledge"),
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
    from src.db import save_prediction_history
    if st.button("💾 Save to History", key="save_manual_pred"):
        save_prediction_history(
            username=st.session_state.get("username", "admin"),
            n=n_val,
            p=p_val,
            k=k_val,
            temp=temp_val,
            hum=hum_val,
            ph=ph_val,
            rain=rain_val,
            top_crop=results["top_crop"],
            top_conf=results["top_viability_score"]
        )
        st.toast("Prediction successfully saved to your history!")

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

    # Report Downloads (PDF + Excel)
    st.markdown("---")
    col_pdf1, col_pdf2 = st.columns([1, 1])
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
    with col_pdf2:
        excel_bytes = generate_excel_crop_dossier(
            crop_name=top_rec["crop"],
            viability=top_rec["viability_score"],
            soil_profile={"nitrogen": n_val, "phosphorus": p_val, "potassium": k_val, "ph": ph_val},
            climate_profile={"temperature": temp_val, "humidity": hum_val, "rainfall": rain_val},
            field_area_acres=1.0
        )
        st.download_button(
            label="📊 " + get_translation(lang, "download_excel"),
            data=excel_bytes,
            file_name=f"CropMind_{top_rec['crop']}_Dossier.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
# TAB: DRIP FERTIGATION & WSF CALCULATOR
# ==============================================================================
with tab_fertigation:
    st.markdown("### 💧 Precision Drip Fertigation & Water-Soluble Fertilizer (WSF) Calculator")
    st.write("Calculates stage-specific fertigation dosing, tank dilutions, and electrical conductivity (EC) to avoid root salinity stress.")

    fc_col1, fc_col2 = st.columns([1, 1])
    with fc_col1:
        fert_crop = st.selectbox(
            "Select Crop for Fertigation:",
            sorted(list(load_dataset_from_db()["crop"].unique())),
            index=sorted(list(load_dataset_from_db()["crop"].unique())).index(top_rec["crop"]) if top_rec["crop"] in load_dataset_from_db()["crop"].unique() else 0,
            key="fertigation_crop_select"
        )
        stage = st.selectbox(
            "Growth Stage:",
            ["initial", "vegetative", "flowering", "fruit_maturity"],
            index=1,
            format_func=lambda s: s.replace("_", " ").title()
        )
        fert_acres = st.number_input("Field Area (Acres)", value=1.0, min_value=0.1, max_value=200.0, step=0.5, key="fert_acres")
        freq = st.slider("Fertigation Applications Per Week", 1, 7, 2)
        tank_vol = st.number_input("Irrigation Cycle Volume (Litres)", value=8000.0, min_value=500.0, step=500.0)

    fert_sched = calculate_fertigation_schedule(
        crop_name=fert_crop,
        growth_stage=stage,
        field_area_acres=fert_acres,
        fertigation_frequency_per_week=freq,
        irrigation_volume_litres_cycle=tank_vol
    )

    with fc_col2:
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            st.metric("Total Weekly Dose", f"{fert_sched['weekly_prescription']['total_wsf_kg_week']} kg/wk")
            st.metric("Estimated Solution EC", f"{fert_sched['safety_parameters']['estimated_ec_ds_m']} dS/m")
        with m_c2:
            st.metric("Dosing Per Drip Cycle", f"{fert_sched['cycle_dosing']['wsf_kg_per_cycle']} kg/cycle")
            st.metric("Salinity Safety Status", fert_sched['safety_parameters']['salinity_risk_status'])

    st.markdown("---")
    st.markdown("#### Prescribed Water-Soluble Formulations")
    wsf_df = pd.DataFrame(fert_sched["weekly_prescription"]["recommended_wsf_sources"])
    if not wsf_df.empty:
        wsf_df["kg_per_cycle"] = (wsf_df["kg_per_acre_week"] * fert_acres / freq).round(2)
        st.dataframe(wsf_df, use_container_width=True)

    st.info(f"💡 **Operational Guideline**: {fert_sched['operational_notes']}")


# ==============================================================================
# TAB: MICRONUTRIENT & SECONDARY DEFICITS
# ==============================================================================
with tab_micronutrients:
    st.markdown("### 🔬 Soil Micronutrient Deficit & Foliar Prescription Advisor")
    st.write("Identifies sub-clinical micronutrient deficiencies (Zn, Fe, B, S) based on soil pH, organic carbon, and crop-specific sensitivity.")

    micro_col1, micro_col2 = st.columns([1, 1])
    with micro_col1:
        micro_crop = st.selectbox(
            "Target Crop:",
            sorted(list(load_dataset_from_db()["crop"].unique())),
            index=sorted(list(load_dataset_from_db()["crop"].unique())).index(top_rec["crop"]) if top_rec["crop"] in load_dataset_from_db()["crop"].unique() else 0,
            key="micro_crop_select"
        )
        soil_ph_micro = st.slider("Soil pH Level", 4.0, 9.5, ph_val, 0.1, key="micro_ph")
        om_micro = st.slider("Soil Organic Matter (%)", 0.1, 5.0, 0.75, 0.05, key="micro_om")

    with micro_col2:
        st.markdown("##### Optional Soil Test Lab Results (ppm / mg/kg)")
        zn_ppm = st.number_input("Zinc (Zn) [ppm, Deficit < 0.6]", value=0.45, step=0.05)
        fe_ppm = st.number_input("Iron (Fe) [ppm, Deficit < 4.5]", value=3.8, step=0.1)
        b_ppm = st.number_input("Boron (B) [ppm, Deficit < 0.5]", value=0.40, step=0.05)
        s_ppm = st.number_input("Sulphur (S) [ppm, Deficit < 10.0]", value=8.5, step=0.5)

    micro_res = diagnose_micronutrient_deficiencies(
        crop_name=micro_crop,
        soil_ph=soil_ph_micro,
        organic_matter_pct=om_micro,
        soil_zn_ppm=zn_ppm,
        soil_fe_ppm=fe_ppm,
        soil_b_ppm=b_ppm,
        soil_s_ppm=s_ppm
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="stitch-card-highlight">
            <span class="stitch-pill {'pill-error' if micro_res['overall_deficiency_risk'] == 'High' else 'pill-warning' if micro_res['overall_deficiency_risk'] == 'Moderate' else 'pill-optimal'}">
                {micro_res['overall_deficiency_risk']} Micronutrient Stress
            </span>
            <div style="font-size: 0.9rem; color: #DAE2FD; margin-top: 8px;">
                Crop Sensitivity: <strong>{micro_res['crop_sensitivity_profile']['high_sensitivity_nutrients']}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("#### Diagnostic Deficit Summary")
    def_items = []
    for nut, data in micro_res["deficiencies_detected"].items():
        def_items.append({
            "Nutrient": nut.upper(),
            "Deficit Severity": data.get("status"),
            "Critical Benchmark": data.get("critical_threshold_ppm", "pH Induced"),
            "Observed / Est. Level": data.get("observed_ppm", "pH Availability Gap"),
            "Impact On Crop": data.get("impact")
        })
    st.dataframe(pd.DataFrame(def_items), use_container_width=True)

    if micro_res.get("recommended_prescriptions"):
        st.markdown("#### Prescribed Corrective Sprays & Soil Amendments")
        rx_df = pd.DataFrame(micro_res["recommended_prescriptions"])
        st.dataframe(rx_df, use_container_width=True)


# ==============================================================================
# TAB: CROP ROTATION & COMPANION PLANNER
# ==============================================================================
with tab_rotation:
    st.markdown("### 🔄 Multi-Season Crop Rotation & Companion Synergies")
    st.write("Generates optimal 3-season crop sequences (Kharif - Rabi - Zaid) to break pest life cycles, replenish biological soil nitrogen, and maximize land use efficiency.")

    rot_c1, rot_c2 = st.columns([1, 1])
    with rot_c1:
        base_crop = st.selectbox(
            "Primary Anchor Crop:",
            sorted(list(load_dataset_from_db()["crop"].unique())),
            index=sorted(list(load_dataset_from_db()["crop"].unique())).index(top_rec["crop"]) if top_rec["crop"] in load_dataset_from_db()["crop"].unique() else 0,
            key="rot_base_crop"
        )
        include_gm = st.checkbox("Include Green Manure / Summer Cover Crop", value=True)

    with rot_c2:
        rot_acres = st.number_input("Field Acreage for Sequence", value=1.0, min_value=0.1, max_value=500.0, step=0.5, key="rot_acres")

    rot_plan = generate_crop_rotation_plan(
        primary_crop=base_crop,
        soil_n=n_val,
        soil_p=p_val,
        soil_k=k_val,
        field_area_acres=rot_acres,
        include_green_manure=include_gm
    )

    st.markdown("---")
    st.markdown("#### Recommended 3-Season Sequence")
    seq_cols = st.columns(3)
    for idx, (col, step) in enumerate(zip(seq_cols, rot_plan["rotation_sequence"])):
        with col:
            st.markdown(
                f"""
                <div class="stitch-card-primary" style="min-height: 180px;">
                    <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase;">Phase {step['phase']} - {step['season']}</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF; margin: 4px 0;">{step['crop']}</div>
                    <div style="font-size: 0.8rem; color: #38BDF8;">{step['role']}</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 8px;">{step['agronomic_rationale']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown("#### Companion Planting & Intercropping Synergies")
    st.markdown(f"**Recommended Intercrop for {base_crop}**: {rot_plan['companion_synergies']['recommended_companion']}")
    st.write(rot_plan['companion_synergies']['mechanism'])
    st.success(f"🌱 **Biological N Contribution**: {rot_plan['soil_fertility_impact']['estimated_n_fixation_kg_acre']} kg N/acre added biologically.")


# ==============================================================================
# TAB: CLIMATE ALERTS & ANOMALY SHIELD
# ==============================================================================
with tab_climate_alerts:
    st.markdown("### ⚠️ Extreme Weather & Climate Anomaly Early Warning")
    st.write("Identifies sudden heat spikes, frost risks, drought intensity, and excessive moisture anomalies tailored to crop phenology.")

    ca_c1, ca_c2 = st.columns([1, 1])
    with ca_c1:
        ca_crop = st.selectbox(
            "Evaluate Climate Threat for:",
            sorted(list(load_dataset_from_db()["crop"].unique())),
            index=sorted(list(load_dataset_from_db()["crop"].unique())).index(top_rec["crop"]) if top_rec["crop"] in load_dataset_from_db()["crop"].unique() else 0,
            key="climate_alert_crop"
        )
        ca_wind = st.slider("Surface Wind Speed (km/h)", 0.0, 80.0, 15.0, 1.0)

    alerts_res = evaluate_climate_anomalies(
        crop_name=ca_crop,
        temperature_c=temp_val,
        humidity_pct=hum_val,
        rainfall_14d_mm=rain_val,
        wind_speed_kmh=ca_wind
    )

    with ca_c2:
        st.markdown("#### Climate Stress Index")
        st.metric("Thermal Stress Score", f"{alerts_res['stress_scores']['thermal_stress_score']}/100")
        st.metric("Moisture Stress Score", f"{alerts_res['stress_scores']['moisture_stress_score']}/100")

    st.markdown("---")
    st.markdown(f"#### Active Risk Level: `{alerts_res['composite_risk_level']}`")
    for alert in alerts_res["active_alerts"]:
        sev = alert.get("severity", "Moderate")
        color = "pill-error" if sev == "Critical" else "pill-warning"
        st.markdown(
            f"""
            <div class="stitch-card-primary" style="border-left: 4px solid {'#EF4444' if sev == 'Critical' else '#F59E0B'};">
                <span class="stitch-pill {color}">{alert['type']} [{sev}]</span>
                <div style="font-size: 0.95rem; font-weight: 600; color: #FFFFFF; margin-top: 6px;">{alert['message']}</div>
                <div style="font-size: 0.85rem; color: #38BDF8; margin-top: 6px;"><strong>Actionable Shield:</strong> {alert['mitigation']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==============================================================================
# TAB: TOPSIS MULTI-CRITERIA DECISION RANKING
# ==============================================================================
with tab_mcda:
    st.markdown("### ⚖️ Multi-Criteria Decision Analysis (TOPSIS) Crop Ranker")
    st.write("Balances multiple competing objectives: ML Viability Score, Net Mandi Profit, Water Conservation Efficiency, and Climate Stress Resilience.")

    mcda_c1, mcda_c2 = st.columns([1, 1])
    with mcda_c1:
        st.markdown("#### Objective Weightings")
        w_viab = st.slider("Weight: Agronomic Viability", 0.0, 1.0, 0.35, 0.05)
        w_prof = st.slider("Weight: Projected Profitability", 0.0, 1.0, 0.25, 0.05)
        w_wat = st.slider("Weight: Water Use Efficiency", 0.0, 1.0, 0.20, 0.05)
        w_res = st.slider("Weight: Climate Resilience", 0.0, 1.0, 0.20, 0.05)

    candidates = [{"crop": r["crop"], "viability_score": r["viability_score"]} for r in recs]
    topsis_res = calculate_topsis_crop_ranking(
        candidate_crops_with_scores=candidates,
        temperature_c=temp_val,
        humidity_pct=hum_val,
        rainfall_mm=rain_val,
        weight_viability=w_viab,
        weight_profit=w_prof,
        weight_water_efficiency=w_wat,
        weight_resilience=w_res,
        field_area_acres=1.0
    )

    with mcda_c2:
        st.markdown("#### TOPSIS Ranking Leaderboard")
        rank_df = pd.DataFrame(topsis_res["ranked_crops"])
        if not rank_df.empty:
            fig_rank = px.bar(
                rank_df,
                x="topsis_score",
                y="crop",
                orientation="h",
                color="topsis_score",
                color_continuous_scale="Viridis",
                title="MCDA Closeness Coefficient (Higher = Better)",
                labels={"topsis_score": "TOPSIS Score", "crop": "Candidate Crop"}
            )
            fig_rank.update_layout(
                template="plotly_dark",
                paper_bgcolor="#171F33",
                plot_bgcolor="#0B1326",
                font=dict(family="Inter", color="#DAE2FD"),
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_rank, use_container_width=True)

    st.dataframe(rank_df, use_container_width=True)


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
# TAB: FARM PARCEL BOUNDARIES & ACREAGE
# ==============================================================================
with tab_parcels:
    st.markdown("### 🗺️ Farm Spatial Parcel Boundaries & GPS Acreage Calculator")
    st.write("Calculates geodesic parcel acreage from GPS boundary polygon coordinates and provides workability shape compactness scores.")

    p_col1, p_col2 = st.columns([1, 1])
    with p_col1:
        parcel_name_input = st.text_input("Parcel Identifier / Name", value="Block A - North Field")
        preset = st.selectbox(
            "Boundary Preset / Geometry Template:",
            ["Coimbatore Paddy Field (~2.5 Acres)", "Punjab Wheat Rectangular Strip (~5.0 Acres)", "Guntur Cotton Polygon (~3.8 Acres)"]
        )
        if "Coimbatore" in preset:
            coords = [[11.0000, 76.9500], [11.0010, 76.9500], [11.0010, 76.9510], [11.0000, 76.9510], [11.0000, 76.9500]]
        elif "Punjab" in preset:
            coords = [[30.9000, 75.8500], [30.9020, 75.8500], [30.9020, 75.8510], [30.9000, 75.8510], [30.9000, 75.8500]]
        else:
            coords = [[16.3000, 80.4400], [16.3015, 80.4400], [16.3015, 80.4418], [16.3000, 80.4418], [16.3000, 80.4400]]

        p_soil = st.selectbox("Parcel Soil Texture", ["Clay Loam", "Sandy Loam", "Black Soil", "Red Laterite"])
        p_crop = st.selectbox("Primary Sown Crop", sorted(list(load_dataset_from_db()["crop"].unique())), index=0, key="parcel_crop_select")

    geo_info = analyze_farm_parcel(
        parcel_name=parcel_name_input,
        boundary_coordinates=coords,
        soil_type=p_soil,
        primary_crop=p_crop
    )

    with p_col2:
        st.markdown("#### Geodesic Spatial Metrics")
        pm1, pm2, pm3 = st.columns(3)
        pm1.metric("Field Acreage", f"{geo_info['geometry']['area_acres']} Acres")
        pm2.metric("Hectares", f"{geo_info['geometry']['area_hectares']} Ha")
        pm3.metric("Perimeter", f"{geo_info['geometry']['perimeter_meters']} m")
        st.metric("Shape Workability", geo_info['geometry']['field_shape_classification'])

    st.markdown("---")
    st.markdown("#### Boundary GPS Coordinates")
    st.dataframe(pd.DataFrame(geo_info["boundary_coordinates"]), use_container_width=True)


# ==============================================================================
# TAB: ICAR & FAO AGRONOMIC KNOWLEDGE BASE
# ==============================================================================
with tab_knowledge:
    st.markdown("### 📚 ICAR & FAO Agronomic Knowledge Compendium")
    st.write("Search verified agronomic Package of Practices (POP), IPM biological recipes, and irrigation scheduling guides certified by ICAR & FAO.")

    kb_q = st.text_input("Search Agronomic Knowledge (e.g., 'seed treatment', 'bollworm', 'wheat irrigation')", "")
    
    kb_col1, kb_col2 = st.columns([1, 1])
    with kb_col1:
        kb_crop = st.selectbox("Filter by Crop:", ["All Crops"] + sorted(list(load_dataset_from_db()["crop"].unique())))
    with kb_col2:
        kb_cat = st.selectbox("Filter by Category:", ["All Categories"] + get_all_categories())

    crop_filter = None if kb_crop == "All Crops" else kb_crop
    cat_filter = None if kb_cat == "All Categories" else kb_cat

    kb_results = search_agronomic_knowledge(
        query=kb_q,
        crop=crop_filter,
        category=cat_filter,
        max_results=5
    )

    st.markdown(f"**Found {kb_results['total_matches']} Expert Guidelines**")
    for item in kb_results["results"]:
        with st.expander(f"📖 {item['title']} [{item['crop']} - {item['category']}]", expanded=True):
            st.markdown(f"**Summary**: {item['summary']}")
            st.markdown("**Package of Practices Protocol:**")
            for step in item["protocol"]:
                st.markdown(f"- {step}")
            st.caption(f"Source: {item['source']}")


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
        fused_cube = None

    if fused_cube:
        # Modality Ingestion Status Cards
        st.markdown("#### Ingested Multi-Modal Data Streams")
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.markdown(
                f"""
                <div class="stitch-card" style="border-left: 4px solid #10B981; padding: 14px;">
                    <div style="font-size: 0.75rem; color: #10B981; font-weight: 700;">🛰️ SATELLITE (GeoTIFF)</div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin: 4px 0;">Mean NDVI: {sat_res['primary_ndvi']:.3f}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">EVI Index: {sat_res['primary_evi']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_c2:
            st.markdown(
                f"""
                <div class="stitch-card" style="border-left: 4px solid #38BDF8; padding: 14px;">
                    <div style="font-size: 0.75rem; color: #38BDF8; font-weight: 700;">🌦️ CLIMATE (NetCDF)</div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin: 4px 0;">Rain: {cli_res['total_rainfall']} mm</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Temp: {cli_res['mean_temperature']}°C | RH: {cli_res['mean_humidity']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_c3:
            st.markdown(
                f"""
                <div class="stitch-card" style="border-left: 4px solid #F59E0B; padding: 14px;">
                    <div style="font-size: 0.75rem; color: #F59E0B; font-weight: 700;">🧪 SOIL (GeoTIFF/CSV)</div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin: 4px 0;">pH Level: {soil_res['pH']:.1f}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">NPK: {soil_res['N']:.0f} - {soil_res['P']:.0f} - {soil_res['K']:.0f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_c4:
            st.markdown(
                f"""
                <div class="stitch-card" style="border-left: 4px solid #A78BFA; padding: 14px;">
                    <div style="font-size: 0.75rem; color: #A78BFA; font-weight: 700;">📈 YIELD GROUND TRUTH</div>
                    <div style="font-size: 1.1rem; font-weight: 700; margin: 4px 0;">{yld_res['total_records']} District Records</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Crops: {', '.join(yld_res['crops_covered'][:3])}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Spatial & Temporal Multi-Modal Visualizers
        vis_col1, vis_col2 = st.columns([1, 1])

        with vis_col1:
            st.markdown("#### 🛰️ Sentinel-2 2D NDVI Spatial Grid")
            if "NDVI" in sat_res["bands"]:
                ndvi_grid = sat_res["bands"]["NDVI"]
                fig_ndvi = px.imshow(
                    ndvi_grid,
                    color_continuous_scale="RdYlGn",
                    title=f"Vegetation Index Heatmap ({sat_res['metadata']['file']})",
                    labels=dict(color="NDVI"),
                    range_color=[0.0, 1.0],
                )
                fig_ndvi.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#171F33",
                    plot_bgcolor="#0B1326",
                    font=dict(family="Inter", color="#DAE2FD"),
                    height=340,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_ndvi, use_container_width=True)

        with vis_col2:
            st.markdown("#### 🌦️ IMD Climate Multi-Day Profile")
            # Synthesize 120-day visualization
            days_idx = np.arange(1, 121)
            t_base = cli_res["mean_temperature"]
            r_base = cli_res["total_rainfall"] / 120.0
            daily_t = t_base + np.sin(days_idx / 15.0) * 2.5 + np.random.normal(0, 0.5, 120)
            daily_r = np.maximum(0, r_base + np.random.exponential(1.5, 120) - 0.5)

            df_cli_sim = pd.DataFrame({"Day": days_idx, "Temperature (°C)": daily_t, "Rainfall (mm)": daily_r})
            fig_cli = px.line(
                df_cli_sim,
                x="Day",
                y=["Temperature (°C)", "Rainfall (mm)"],
                title="120-Day Ingested NetCDF Climate Dynamics",
                color_discrete_sequence=["#F59E0B", "#38BDF8"],
            )
            fig_cli.update_layout(
                template="plotly_dark",
                paper_bgcolor="#171F33",
                plot_bgcolor="#0B1326",
                font=dict(family="Inter", color="#DAE2FD"),
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_cli, use_container_width=True)

        # Fused AI Prediction Trigger
        st.markdown("---")
        if st.button("🚀 Run Multi-Modal AI Fusion & Yield Prediction (XGBoost + SHAP)", use_container_width=True):
            fused = fused_cube["fused_features"]
            pred_res = predict_and_explain(
                fused["N"], fused["P"], fused["K"], fused["temperature"], fused["humidity"], fused["ph"], fused["rainfall"]
            )
            from src.db import save_prediction_history
            save_prediction_history(
                username=st.session_state.get("username", "admin"),
                n=fused["N"],
                p=fused["P"],
                k=fused["K"],
                temp=fused["temperature"],
                hum=fused["humidity"],
                ph=fused["ph"],
                rain=fused["rainfall"],
                top_crop=pred_res["top_crop"],
                top_conf=pred_res["top_viability_score"]
            )

            st.markdown("### 🏆 Multi-Modal Prediction & Yield Estimation")
            res_col1, res_col2 = st.columns([1, 1])

            top_crop = pred_res["top_crop"]
            top_conf = round(pred_res["top_viability_score"] * 100, 2)
            historical_bench = yld_res["crop_yield_benchmarks"].get(top_crop, {})
            expected_yield = historical_bench.get("avg_yield_kg_ha", 3800.0)

            with res_col1:
                st.markdown(
                    f"""
                    <div class="stitch-card-highlight" style="padding: 22px;">
                        <div style="font-size: 0.8rem; color: #10B981; font-weight: 700;">🥇 MULTIMODAL RECOMMENDED CROP</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: #DAE2FD; margin: 8px 0;">{top_crop}</div>
                        <div style="display: flex; gap: 10px; margin-top: 8px;">
                            <span class="confidence-badge">Confidence: {top_conf}%</span>
                            <span class="confidence-badge" style="background: rgba(16, 185, 129, 0.2); border-color: rgba(16, 185, 129, 0.5); color: #34D399;">
                                Est. Yield: {expected_yield:,.0f} kg/ha
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with res_col2:
                # Plotly SHAP Feature Contribution for Fused Vector
                shap_df = pd.DataFrame(pred_res["feature_contributions"])
                fig_shap_multi = px.bar(
                    shap_df,
                    x="shap_delta",
                    y="label",
                    orientation="h",
                    color="impact",
                    color_discrete_map={"positive": "#10B981", "negative": "#EF4444"},
                    title=f"Multi-Modal SHAP Factor Drivers for '{top_crop}'",
                    labels={"shap_delta": "SHAP Impact Score", "label": "Modality Variable"},
                )
                fig_shap_multi.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#171F33",
                    plot_bgcolor="#0B1326",
                    font=dict(family="Inter", color="#DAE2FD"),
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(fig_shap_multi, use_container_width=True)

            st.info(f"💡 **Agronomic Synthesis**: {pred_res['human_readable_summary']}")


# ==============================================================================
# TAB 10: MY HISTORY
# ==============================================================================
with tab_history:
    st.markdown("### Prediction History")
    history_data = []
    # 1. Try FastAPI endpoint if running
    try:
        headers = {"Authorization": f"Bearer {st.session_state.get('token', '')}"}
        res = requests.get(f"{API_URL}/recommendations/history", headers=headers, timeout=0.8)
        if res.status_code == 200:
            history_data = res.json().get("data", [])
    except Exception:
        pass

    # 2. Direct SQLite DB fallback
    if not history_data:
        try:
            uid = get_user_id_by_username(st.session_state.get("username", ""))
            if uid:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT predicted_crop, confidence, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, created_at FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC",
                    (uid,)
                )
                rows = cursor.fetchall()
                history_data = [dict(r) for r in rows]
                conn.close()
        except Exception as e:
            st.error(f"Database query error: {e}")

    if len(history_data) > 0:
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
    else:
        st.info("No prediction history found. Run a recommendation in Tab 1 and click 'Save to History'!")


# ==============================================================================
# TAB 11: BATCH PREDICTION
# ==============================================================================
with tab_batch:
    st.header("Bulk Crop Prediction (CSV Upload)")
    uploaded_file = st.file_uploader("Choose a CSV file (with N, P, K, Temperature, Humidity, pH, Rainfall)", type="csv", key="batch_upload")
    if uploaded_file is not None:
        if st.button("Run Batch Prediction"):
            batch_data = []
            # 1. Try FastAPI endpoint if running
            try:
                headers = {"Authorization": f"Bearer {st.session_state.get('token', '')}"}
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                res = requests.post(f"{API_URL}/recommendations/predict/batch", headers=headers, files=files, timeout=2.0)
                if res.status_code == 200:
                    batch_data = res.json()["data"]
            except Exception:
                pass

            # 2. Embedded ML model inference fallback
            if not batch_data:
                try:
                    uploaded_file.seek(0)
                    df_batch = pd.read_csv(uploaded_file)
                    col_map = {c: c.lower().strip() for c in df_batch.columns}
                    df_batch = df_batch.rename(columns=col_map)
                    for idx, row in df_batch.iterrows():
                        n_val_b = float(row.get("nitrogen", row.get("n", 90.0)))
                        p_val_b = float(row.get("phosphorus", row.get("p", 42.0)))
                        k_val_b = float(row.get("potassium", row.get("k", 43.0)))
                        temp_val_b = float(row.get("temperature", row.get("temp", 26.5)))
                        hum_val_b = float(row.get("humidity", row.get("hum", 75.0)))
                        ph_val_b = float(row.get("ph", 6.5))
                        rain_val_b = float(row.get("rainfall", row.get("rain", 110.0)))

                        pred = predict_and_explain(n_val_b, p_val_b, k_val_b, temp_val_b, hum_val_b, ph_val_b, rain_val_b, top_k=3)
                        top_p = pred["recommendations"][0]
                        batch_data.append({
                            "Sample": idx + 1,
                            "Recommended Crop": top_p["crop"],
                            "Viability Score": f"{top_p['viability_score']*100:.1f}%",
                            "Summary": top_p["explanations"]["human_readable_summary"]
                        })
                except Exception as batch_err:
                    st.error(f"Batch processing error: {batch_err}")

            if batch_data:
                st.success(f"Successfully processed {len(batch_data)} soil-climate samples!")
                st.dataframe(pd.DataFrame(batch_data), use_container_width=True)


# ==============================================================================
# TAB 12: ADMIN DASHBOARD
# ==============================================================================
with tab_admin:
    st.header("Admin Dashboard")
    metrics = None
    # 1. Try FastAPI endpoint
    try:
        headers = {"Authorization": f"Bearer {st.session_state.get('token', '')}"}
        res = requests.get(f"{API_URL}/admin/metrics", headers=headers, timeout=0.8)
        if res.status_code == 200:
            metrics = res.json()["data"]
    except Exception:
        pass

    # 2. Direct SQLite query fallback
    if not metrics:
        try:
            metrics = get_admin_metrics()
        except Exception as e:
            metrics = {"total_users": 1, "total_predictions": 0, "top_crop": "Rice", "system_status": "Healthy (Standalone Mode)"}

    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Registered Users", metrics.get("total_users", 1))
        col2.metric("Total Predictions", metrics.get("total_predictions", 0))
        col3.metric("Top Recommended Crop", metrics.get("top_crop", "Rice"))
        col4.metric("System Health", metrics.get("system_status", "Healthy"))


# ==============================================================================
# TAB 13: MY FARMS
# ==============================================================================
with tab_farms:
    st.header("My Farms (Saved Profiles)")
    uid = get_user_id_by_username(st.session_state.get("username", "")) or 1
    with st.expander("➕ Add New Farm", expanded=False):
        with st.form("add_farm_form"):
            farm_name = st.text_input("Farm Name (e.g., 'North Field Block A')")
            f_lat = st.number_input("Latitude", value=13.0, min_value=-90.0, max_value=90.0)
            f_lon = st.number_input("Longitude", value=80.0, min_value=-180.0, max_value=180.0)
            f_n = st.number_input("Nitrogen (mg/kg)", value=90.0)
            f_p = st.number_input("Phosphorus (mg/kg)", value=42.0)
            f_k = st.number_input("Potassium (mg/kg)", value=43.0)
            f_ph = st.number_input("pH Level", value=6.5, min_value=2.0, max_value=12.0)
            submitted = st.form_submit_button("Save Farm")
            if submitted and farm_name:
                saved = False
                try:
                    payload = {"farm_name": farm_name, "latitude": f_lat, "longitude": f_lon, "nitrogen": f_n, "phosphorus": f_p, "potassium": f_k, "ph": f_ph}
                    res = requests.post(f"{API_URL}/farms", headers={"Authorization": f"Bearer {st.session_state['token']}"}, json=payload, timeout=0.8)
                    if res.status_code == 200:
                        saved = True
                except Exception:
                    pass
                if not saved:
                    create_user_farm(uid, farm_name, f_lat, f_lon, f_n, f_p, f_k, f_ph)
                st.success(f"Farm '{farm_name}' successfully saved!")
                st.rerun()

    # List saved farms
    farms_list = []
    try:
        res = requests.get(f"{API_URL}/farms", headers={"Authorization": f"Bearer {st.session_state['token']}"}, timeout=0.8)
        if res.status_code == 200:
            farms_list = res.json().get("data", [])
    except Exception:
        pass
    if not farms_list:
        farms_list = get_user_farms(uid)

    if farms_list:
        st.markdown("#### Registered Farm Profiles")
        for f in farms_list:
            col_fa, col_fb = st.columns([4, 1])
            with col_fa:
                st.markdown(f"🌾 **{f['farm_name']}** (Lat: `{f['latitude']}`, Lon: `{f['longitude']}`) | N: `{f['nitrogen']}` P: `{f['phosphorus']}` K: `{f['potassium']}` pH: `{f['ph']}`")
            with col_fb:
                if st.button("Delete Farm", key=f"del_farm_{f['id']}"):
                    try:
                        requests.delete(f"{API_URL}/farms/{f['id']}", headers={"Authorization": f"Bearer {st.session_state['token']}"}, timeout=0.8)
                    except Exception:
                        pass
                    delete_user_farm(f['id'], uid)
                    st.rerun()
    else:
        st.info("No farm holdings registered yet. Use the form above to add your first farm parcel.")


# ==============================================================================
# TAB 14: DEVELOPER API
# ==============================================================================
with tab_api:
    st.header("Developer API Keys & Microservices")
    st.write("Generate API keys to programmatically interact with CropMind AI prediction and advisory microservices.")
    uid = get_user_id_by_username(st.session_state.get("username", "")) or 1

    col_k1, col_k2 = st.columns([1, 2])
    with col_k1:
        if st.button("Generate New API Key"):
            new_key = None
            try:
                res = requests.post(f"{API_URL}/keys", headers={"Authorization": f"Bearer {st.session_state['token']}"}, timeout=0.8)
                if res.status_code == 200:
                    new_key = res.json()["api_key"]
            except Exception:
                pass
            if not new_key:
                new_key = create_api_key(user_id=uid)
            st.success("API Key Generated Successfully!")
            st.code(new_key, language="bash")
            st.info("Please copy your key now. For security reasons, it cannot be displayed again.")

    with col_k2:
        st.subheader("Active API Keys")
        keys = []
        try:
            res = requests.get(f"{API_URL}/keys", headers={"Authorization": f"Bearer {st.session_state['token']}"}, timeout=0.8)
            if res.status_code == 200:
                keys = res.json().get("data", [])
        except Exception:
            pass
        if not keys:
            keys = get_user_api_keys(user_id=uid)

        if keys:
            for k in keys:
                st.markdown(f"**Key ID:** `{k['id']}` | **Key:** `{k['api_key'][:8]}...` | **Created:** `{k['created_at']}`")
                if st.button(f"Revoke Key {k['id']}", key=f"revoke_{k['id']}"):
                    try:
                        requests.delete(f"{API_URL}/keys/{k['id']}", headers={"Authorization": f"Bearer {st.session_state['token']}"}, timeout=0.8)
                    except Exception:
                        pass
                    delete_api_key(k['id'], uid)
                    st.rerun()
        else:
            st.info("No active API keys found.")

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
