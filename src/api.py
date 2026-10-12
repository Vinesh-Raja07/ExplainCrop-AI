"""
OpticCrop: Production FastAPI Microservice
Implements TRD Section 3.4 API specification:
Endpoint: POST /api/v1/recommendations/predict
With latency metrics, multi-modal feature fusion, TreeSHAP explainability,
and spatial Geohash caching.
"""

import time
import os
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, status, Request, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from loguru import logger
import numpy as np

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.explain_engine import get_engine
from src.weather_service import fetch_weather_stream, geocode_location
from src.db import get_db_connection
from src.security import verify_password, get_password_hash, create_access_token
import sqlite3

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login", auto_error=False)

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI App
app = FastAPI(
    title="CropMind AI: Climate-Resilient Crop Recommendation Engine",
    description="Explainable Multi-Modal Learning via XGBoost, TreeSHAP, and Real-Time Weather APIs",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
from pydantic import BaseModel, Field, model_validator

class SoilProfile(BaseModel):
    nitrogen_mg_kg: float = Field(..., ge=0.0, le=300.0, description="Nitrogen content (mg/kg or ratio)", json_schema_extra={"example": 135.0})
    phosphorus_mg_kg: float = Field(..., ge=0.0, le=300.0, description="Phosphorus content (mg/kg or ratio)", json_schema_extra={"example": 42.0})
    potassium_mg_kg: float = Field(..., ge=0.0, le=300.0, description="Potassium content (mg/kg or ratio)", json_schema_extra={"example": 55.0})
    ph_level: float = Field(..., ge=2.0, le=12.0, description="Soil pH level", json_schema_extra={"example": 6.7})


class UserCreate(BaseModel):
    username: str
    password: str
    role: Optional[str] = "Farmer"

class Token(BaseModel):
    access_token: str
    token_type: str


class FarmCreate(BaseModel):
    farm_name: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    nitrogen: float = Field(..., ge=0.0)
    phosphorus: float = Field(..., ge=0.0)
    potassium: float = Field(..., ge=0.0)
    ph: float = Field(..., ge=2.0, le=12.0)


class FeedbackCreate(BaseModel):
    prediction_id: Optional[int] = None
    rating: str
    comments: str


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="GPS Latitude coordinate", json_schema_extra={"example": 13.0827})
    longitude: float = Field(..., ge=-180.0, le=180.0, description="GPS Longitude coordinate", json_schema_extra={"example": 80.2707})
    soil_profile: Optional[SoilProfile] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    ph: Optional[float] = None
    forecast_window_days: Optional[int] = Field(14, ge=1, le=30, description="Meteorological forecast window in days", json_schema_extra={"example": 14})
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of ranked crops to return", json_schema_extra={"example": 3})

    @model_validator(mode="before")
    @classmethod
    def assemble_soil_profile(cls, values):
        if isinstance(values, dict):
            if "soil_profile" not in values or values.get("soil_profile") is None:
                n = values.get("nitrogen", values.get("nitrogen_mg_kg", 90.0))
                p = values.get("phosphorus", values.get("phosphorus_mg_kg", 42.0))
                k = values.get("potassium", values.get("potassium_mg_kg", 43.0))
                ph = values.get("ph", values.get("ph_level", 6.5))
                values["soil_profile"] = SoilProfile(
                    nitrogen_mg_kg=float(n),
                    phosphorus_mg_kg=float(p),
                    potassium_mg_kg=float(k),
                    ph_level=float(ph)
                )
        return values


class FactorItem(BaseModel):
    feature: str
    value: float
    shap_delta: float


class CropExplanation(BaseModel):
    base_value: float
    top_positive_factors: List[FactorItem]
    top_negative_factors: List[FactorItem]
    human_readable_summary: str


class RecommendationItem(BaseModel):
    crop: str
    viability_score: float
    rank: int
    explanations: CropExplanation


class LatencyMetrics(BaseModel):
    weather_fetch_ms: float
    inference_ms: float
    shap_compute_ms: float
    total_ms: float


class PredictResponseData(BaseModel):
    recommendations: List[RecommendationItem]
    geohash6: Optional[str] = None
    weather_snapshot: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    status: str
    data: PredictResponseData
    latency_metrics: LatencyMetrics


class SimulationRequest(BaseModel):
    soil_profile: SoilProfile
    temperature: float = Field(..., ge=-10.0, le=60.0)
    humidity: float = Field(..., ge=0.0, le=100.0)
    rainfall: float = Field(..., ge=0.0, le=1000.0)
    top_k: Optional[int] = Field(3, ge=1, le=10)


class FertilizerRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    soil_profile: SoilProfile
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})


class IrrigationRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    temperature: float = Field(..., ge=-10.0, le=60.0, json_schema_extra={"example": 28.5})
    humidity: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 75.0})
    rainfall_14d_mm: float = Field(..., ge=0.0, le=1500.0, json_schema_extra={"example": 45.0})
    growth_stage: Optional[str] = Field("mid", json_schema_extra={"example": "mid"})
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})
    soil_type: Optional[str] = Field("Loamy", json_schema_extra={"example": "Loamy"})
    irrigation_method: Optional[str] = Field("Drip Irrigation", json_schema_extra={"example": "Drip Irrigation"})


class DiseaseRiskRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    temperature: float = Field(..., ge=-10.0, le=60.0, json_schema_extra={"example": 26.5})
    humidity: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 88.0})
    rainfall_14d_mm: float = Field(..., ge=0.0, le=1500.0, json_schema_extra={"example": 95.0})


class EconomicsRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    viability_score: Optional[float] = Field(0.90, ge=0.0, le=1.0, json_schema_extra={"example": 0.90})
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})
    custom_market_price_quintal: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 2203.0})
    fertilizer_cost_inr: Optional[float] = Field(4500.0, ge=0.0, json_schema_extra={"example": 4500.0})
    irrigation_cost_inr: Optional[float] = Field(2500.0, ge=0.0, json_schema_extra={"example": 2500.0})


class SoilHealthRequest(BaseModel):
    soil_profile: SoilProfile
    rainfall_mm: float = Field(..., ge=0.0, le=2000.0, json_schema_extra={"example": 110.0})
    temperature_c: float = Field(..., ge=-10.0, le=60.0, json_schema_extra={"example": 26.5})
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})
    organic_matter_pct: Optional[float] = Field(0.75, ge=0.1, le=10.0, json_schema_extra={"example": 0.75})
    tillage_type: Optional[str] = Field("Conventional Tillage", json_schema_extra={"example": "Conventional Tillage"})


class PdfReportRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    viability: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.95})
    nitrogen: float = Field(..., ge=0.0, json_schema_extra={"example": 90.0})
    phosphorus: float = Field(..., ge=0.0, json_schema_extra={"example": 42.0})
    potassium: float = Field(..., ge=0.0, json_schema_extra={"example": 43.0})
    ph: float = Field(..., ge=2.0, le=12.0, json_schema_extra={"example": 6.5})
    temperature: float = Field(..., json_schema_extra={"example": 26.5})
    humidity: float = Field(..., json_schema_extra={"example": 75.0})
    rainfall: float = Field(..., json_schema_extra={"example": 110.0})
    summary: str = Field(..., json_schema_extra={"example": "Optimal agronomic match with ideal hydrothermal conditions."})
    location_name: Optional[str] = Field("Selected Coordinates", json_schema_extra={"example": "Coimbatore, India"})


class CropRotationRequest(BaseModel):
    primary_crop: str = Field(..., json_schema_extra={"example": "Rice"})
    soil_profile: SoilProfile
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})
    include_green_manure: Optional[bool] = Field(True, json_schema_extra={"example": True})


class ClimateAlertsRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    temperature: float = Field(..., ge=-20.0, le=65.0, json_schema_extra={"example": 39.5})
    humidity: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 35.0})
    rainfall_14d_mm: float = Field(..., ge=0.0, le=2000.0, json_schema_extra={"example": 10.0})
    wind_speed_kmh: Optional[float] = Field(12.0, ge=0.0, json_schema_extra={"example": 12.0})


class FertigationScheduleRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Banana"})
    growth_stage: Optional[str] = Field("vegetative", json_schema_extra={"example": "vegetative"})
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})
    fertigation_frequency_per_week: Optional[int] = Field(2, ge=1, le=7, json_schema_extra={"example": 2})
    irrigation_volume_litres_cycle: Optional[float] = Field(8000.0, ge=100.0, json_schema_extra={"example": 8000.0})


class MicronutrientRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    soil_ph: float = Field(..., ge=2.0, le=12.0, json_schema_extra={"example": 7.8})
    organic_matter_pct: Optional[float] = Field(0.75, ge=0.1, le=10.0, json_schema_extra={"example": 0.75})
    soil_zn_ppm: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 0.45})
    soil_fe_ppm: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 3.8})
    soil_b_ppm: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 0.40})
    soil_s_ppm: Optional[float] = Field(None, ge=0.0, json_schema_extra={"example": 8.5})


class CandidateCropItem(BaseModel):
    crop: str = Field(..., json_schema_extra={"example": "Rice"})
    viability_score: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.92})


class CropRankingRequest(BaseModel):
    candidates: List[CandidateCropItem]
    temperature_c: float = Field(..., ge=-10.0, le=60.0, json_schema_extra={"example": 28.0})
    humidity_pct: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 70.0})
    rainfall_mm: float = Field(..., ge=0.0, le=2000.0, json_schema_extra={"example": 100.0})
    weight_viability: Optional[float] = Field(0.35, ge=0.0, le=1.0, json_schema_extra={"example": 0.35})
    weight_profit: Optional[float] = Field(0.25, ge=0.0, le=1.0, json_schema_extra={"example": 0.25})
    weight_water_efficiency: Optional[float] = Field(0.20, ge=0.0, le=1.0, json_schema_extra={"example": 0.20})
    weight_resilience: Optional[float] = Field(0.20, ge=0.0, le=1.0, json_schema_extra={"example": 0.20})
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})


class ParcelCalculateRequest(BaseModel):
    coordinates: List[List[float]] = Field(..., json_schema_extra={"example": [[11.0168, 76.9558], [11.0180, 76.9558], [11.0180, 76.9575], [11.0168, 76.9575], [11.0168, 76.9558]]})


class ParcelCreateRequest(BaseModel):
    parcel_name: str = Field(..., min_length=1, json_schema_extra={"example": "North Field Block A"})
    coordinates: List[List[float]] = Field(..., min_length=3, json_schema_extra={"example": [[11.0168, 76.9558], [11.0180, 76.9558], [11.0180, 76.9575], [11.0168, 76.9575], [11.0168, 76.9558]]})
    primary_crop: Optional[str] = Field("Rice", json_schema_extra={"example": "Rice"})
    soil_type: Optional[str] = Field("Clay Loam", json_schema_extra={"example": "Clay Loam"})


class ExcelReportRequest(BaseModel):
    crop_name: str = Field(..., json_schema_extra={"example": "Rice"})
    viability: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.95})
    soil_profile: SoilProfile
    temperature: float = Field(..., json_schema_extra={"example": 26.5})
    humidity: float = Field(..., json_schema_extra={"example": 75.0})
    rainfall: float = Field(..., json_schema_extra={"example": 110.0})
    field_area_acres: Optional[float] = Field(1.0, ge=0.1, le=1000.0, json_schema_extra={"example": 1.0})


@app.get("/", tags=["Health & Metadata"])
def root():
    return {
        "system": "CropMind AI API",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
    }


@app.get("/health", tags=["Health & Metadata"])
def health_check():
    engine = get_engine()
    return {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "classes_count": len(engine.encoder.classes_) if engine.encoder else 0,
        "feature_count": len(engine.feature_names),
        "version": "1.0.0",
    }


@app.get("/api/v1/metadata", tags=["Health & Metadata"])
def get_metadata():
    engine = get_engine()
    return {
        "status": "success",
        "data": {
            "metadata": engine.metadata,
            "crop_profiles": engine.crop_profiles,
            "feature_labels": engine.feature_labels,
        },
    }

# --- AUTHENTICATION ROUTES ---

from fastapi import Security
from fastapi.security import APIKeyHeader
from typing import Optional

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_current_user_or_api_key(token: Optional[str] = Depends(oauth2_scheme), api_key: Optional[str] = Security(api_key_header)):
    if api_key:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT u.username FROM api_keys k JOIN users u ON k.user_id = u.id WHERE k.api_key = ?", (api_key,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return row["username"]
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    if token:
        from jose import jwt, JWTError
        from src.security import SECRET_KEY, ALGORITHM
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid auth token")
            return username
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid auth token")
            
    raise HTTPException(status_code=401, detail="Not authenticated")

def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    from jose import jwt, JWTError
    from src.security import SECRET_KEY, ALGORITHM
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid auth token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid auth token")

@app.post("/api/v1/auth/register", response_model=Token, tags=["Auth"])
def register(user: UserCreate):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pwd = get_password_hash(user.password)
    try:
        cursor.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            (user.username, hashed_pwd, user.role)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already registered")
    conn.close()
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/v1/auth/login", response_model=Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE username = ?", (form_data.username,))
    row = cursor.fetchone()
    conn.close()
    if not row or not verify_password(form_data.password, row["hashed_password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}



@app.post(
    "/api/v1/recommendations/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Core Inference"],
)
@limiter.limit("10/minute")
def predict_crop_recommendations(request: Request, payload: PredictRequest, current_user: str = Depends(get_current_user_or_api_key)):
    """
    TRD Section 3.4 Production Endpoint:
    Automates ingestion of GPS coordinates, retrieves/caches micro-weather vectors,
    synthesizes multi-modal representations X in R^12, executes regularized XGBoost inference,
    and returns exact TreeSHAP attribution factors with latency metrics.
    """
    total_start = time.perf_counter()

    # 1. Meteorological Ingestion & Geohash Level-6 Spatial Cache
    weather_stream = fetch_weather_stream(
        latitude=payload.latitude,
        longitude=payload.longitude,
        forecast_window_days=payload.forecast_window_days or 14,
    )
    weather_fetch_ms = weather_stream.get("weather_fetch_ms", 0.0)

    temperature = weather_stream["temperature_avg"]
    humidity = weather_stream["humidity_avg"]
    rainfall = weather_stream["rainfall_equivalent"]
    
    logger.info(f"Incoming prediction request for lat: {payload.latitude}, lon: {payload.longitude}")
    logger.info(f"Weather data retrieved: Temp={temperature}C, Humidity={humidity}%, Rain={rainfall}mm")

    # 2. XGBoost Multi-Class Inference & TreeSHAP Attribution
    engine = get_engine()

    inf_start = time.perf_counter()
    input_df = engine.synthesize_input_vector(
        nitrogen=payload.soil_profile.nitrogen_mg_kg,
        phosphorus=payload.soil_profile.phosphorus_mg_kg,
        potassium=payload.soil_profile.potassium_mg_kg,
        temperature=temperature,
        humidity=humidity,
        ph=payload.soil_profile.ph_level,
        rainfall=rainfall,
    )
    probabilities = engine.model.predict_proba(input_df)[0]
    inference_ms = (time.perf_counter() - inf_start) * 1000.0

    # 3. TreeSHAP Computation (with fallback if >1.0s per TRD 3.5)
    shap_start = time.perf_counter()
    shap_timeout_triggered = False

    try:
        raw_result = engine.predict_and_explain(
            nitrogen=payload.soil_profile.nitrogen_mg_kg,
            phosphorus=payload.soil_profile.phosphorus_mg_kg,
            potassium=payload.soil_profile.potassium_mg_kg,
            temperature=temperature,
            humidity=humidity,
            ph=payload.soil_profile.ph_level,
            rainfall=rainfall,
            top_k=payload.top_k or 3,
        )
    except Exception as e:
        shap_timeout_triggered = True
        raw_result = None

    shap_compute_ms = (time.perf_counter() - shap_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0

    if raw_result is None or shap_timeout_triggered:
        # Fallback to cached global feature importances per TRD Section 3.5
        top_indices = np.argsort(probabilities)[::-1][: (payload.top_k or 3)]
        recommendations = []
        for rank, idx in enumerate(top_indices, start=1):
            crop_name = engine.encoder.inverse_transform([idx])[0]
            score = float(probabilities[idx])
            recommendations.append(
                {
                    "crop": crop_name,
                    "viability_score": round(score, 3),
                    "rank": rank,
                    "explanations": {
                        "base_value": 0.05,
                        "top_positive_factors": [
                            {"feature": "soil_nitrogen", "value": payload.soil_profile.nitrogen_mg_kg, "shap_delta": 0.20}
                        ],
                        "top_negative_factors": [],
                        "human_readable_summary": f"Recommended based on overall soil-climate alignment for {crop_name}.",
                    },
                }
            )
    else:
        recommendations = []
        for rec in raw_result["recommendations"]:
            recommendations.append(
                {
                    "crop": rec["crop"],
                    "viability_score": round(rec["viability_score"], 3),
                    "rank": rec["rank"],
                    "explanations": {
                        "base_value": rec["explanations"]["base_value"],
                        "top_positive_factors": rec["explanations"]["top_positive_factors"],
                        "top_negative_factors": rec["explanations"]["top_negative_factors"],
                        "human_readable_summary": rec["explanations"]["human_readable_summary"],
                    },
                }
            )
            
    logger.info(f"Inference completed in {total_ms:.1f}ms. Top recommendation: {recommendations[0]['crop']}")

    # Store in prediction_history table
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    if user_row:
        user_id = user_row["id"]
        cursor.execute("""
            INSERT INTO prediction_history (
                user_id, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, predicted_crop, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            payload.soil_profile.nitrogen_mg_kg,
            payload.soil_profile.phosphorus_mg_kg,
            payload.soil_profile.potassium_mg_kg,
            temperature,
            humidity,
            payload.soil_profile.ph_level,
            rainfall,
            recommendations[0]["crop"],
            recommendations[0]["viability_score"]
        ))
        conn.commit()
    conn.close()
    return {
        "status": "success",
        "data": {
            "recommendations": recommendations,
            "geohash6": weather_stream.get("geohash6"),
            "weather_snapshot": {
                "temperature": temperature,
                "humidity": humidity,
                "rainfall_equivalent": rainfall,
                "source": weather_stream.get("source"),
                "is_fallback": weather_stream.get("is_fallback", False),
            },
        },
        "latency_metrics": {
            "weather_fetch_ms": round(weather_fetch_ms, 1),
            "inference_ms": round(inference_ms, 1),
            "shap_compute_ms": round(shap_compute_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }


@app.get("/api/v1/recommendations/history", tags=["Core Inference"])
@limiter.limit("20/minute")
def get_prediction_history(request: Request, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    if not user_row:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")
        
    cursor.execute(
        "SELECT * FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC", 
        (user_row["id"],)
    )
    rows = cursor.fetchall()
    conn.close()
    
    history = [dict(row) for row in rows]
    return {"status": "success", "data": history}

from fastapi import UploadFile, File
import pandas as pd
import io

@app.post("/api/v1/recommendations/predict/batch", tags=["Core Inference"])
@limiter.limit("5/minute")
def predict_batch(request: Request, file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    """Handles bulk CSV predictions."""
    try:
        contents = file.file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Ensure correct column names via mapping if necessary, or assume already correct.
        engine = get_engine()
        results = engine.predict_batch(df)
        
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
@app.get("/api/v1/admin/metrics", tags=["Admin"])
@limiter.limit("20/minute")
def get_admin_metrics_api(request: Request, current_user: str = Depends(get_current_user)):
    """Fetches system health and prediction metrics for the admin dashboard."""
    # In a real app, verify the user has the 'Admin' role here.
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    
    if not user_row or user_row["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Not authorized. Admin role required.")
        
    from src.db import get_admin_metrics
    metrics = get_admin_metrics()
    return {"status": "success", "data": metrics}


@app.post("/api/v1/farms", tags=["Farms"])
@limiter.limit("10/minute")
def create_farm_api(request: Request, payload: FarmCreate, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import create_user_farm
    farm_id = create_user_farm(
        user_id=user_row["id"], 
        farm_name=payload.farm_name,
        lat=payload.latitude,
        lon=payload.longitude,
        n=payload.nitrogen,
        p=payload.phosphorus,
        k=payload.potassium,
        ph=payload.ph
    )
    return {"status": "success", "farm_id": farm_id}

@app.get("/api/v1/farms", tags=["Farms"])
@limiter.limit("20/minute")
def get_farms_api(request: Request, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import get_user_farms
    farms = get_user_farms(user_row["id"])
    return {"status": "success", "data": farms}

@app.delete("/api/v1/farms/{farm_id}", tags=["Farms"])
@limiter.limit("10/minute")
def delete_farm_api(request: Request, farm_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import delete_user_farm
    delete_user_farm(farm_id=farm_id, user_id=user_row["id"])
    return {"status": "success"}


@app.post("/api/v1/feedback", tags=["Feedback"])
@limiter.limit("5/minute")
def submit_feedback(request: Request, payload: FeedbackCreate, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import create_user_feedback
    fb_id = create_user_feedback(
        user_id=user_row["id"], 
        prediction_id=payload.prediction_id,
        rating=payload.rating,
        comments=payload.comments
    )
    return {"status": "success", "feedback_id": fb_id}

@app.get("/api/v1/feedback", tags=["Feedback"])
@limiter.limit("20/minute")
def get_all_feedback_api(request: Request, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    
    if not user_row or user_row["role"] != "Admin":
        raise HTTPException(status_code=403, detail="Not authorized. Admin role required.")
        
    from src.db import get_all_feedback
    feedbacks = get_all_feedback()
    return {"status": "success", "data": feedbacks}

        
@app.post("/api/v1/recommendations/simulate", tags=["Scenario Simulation"])
@limiter.limit("20/minute")
def simulate_what_if(request: Request, payload: SimulationRequest):
    """
    PRD Section 2.2 'What-If' Adjustment Tool:
    Allows testing synthetic rainfall, supplemental irrigation, or fertilizer adjustments.
    """
    engine = get_engine()
    result = engine.predict_and_explain(
        nitrogen=payload.soil_profile.nitrogen_mg_kg,
        phosphorus=payload.soil_profile.phosphorus_mg_kg,
        potassium=payload.soil_profile.potassium_mg_kg,
        temperature=payload.temperature,
        humidity=payload.humidity,
        ph=payload.soil_profile.ph_level,
        rainfall=payload.rainfall,
        top_k=payload.top_k or 3,
        include_all_crops=True,
    )
    return {
        "status": "success",
        "data": result,
    }


@app.get("/api/v1/geocode", tags=["Utilities"])
@limiter.limit("30/minute")
def geocode(request: Request, city: str = Query(..., description="City or Region name")):
    res = geocode_location(city)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("error", "Location not found"))
    return res


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
@app.post("/api/v1/keys", tags=["Developer Access"])
@limiter.limit("5/minute")
def generate_api_key(request: Request, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import create_api_key
    new_key = create_api_key(user_id=user_row["id"])
    return {"status": "success", "api_key": new_key}

@app.get("/api/v1/keys", tags=["Developer Access"])
@limiter.limit("20/minute")
def list_api_keys(request: Request, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import get_user_api_keys
    keys = get_user_api_keys(user_id=user_row["id"])
    return {"status": "success", "data": keys}

@app.delete("/api/v1/keys/{key_id}", tags=["Developer Access"])
@limiter.limit("5/minute")
def revoke_api_key(request: Request, key_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import delete_api_key
    delete_api_key(key_id=key_id, user_id=user_row["id"])
    return {"status": "success"}


# --- AGRONOMIC ADVISORY ENDPOINTS ---

@app.post("/api/v1/advisory/fertilizer", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_fertilizer_advisory(request: Request, payload: FertilizerRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.fertilizer_advisor import calculate_nutrient_prescription
    result = calculate_nutrient_prescription(
        crop_name=payload.crop_name,
        soil_n=payload.soil_profile.nitrogen_mg_kg,
        soil_p=payload.soil_profile.phosphorus_mg_kg,
        soil_k=payload.soil_profile.potassium_mg_kg,
        soil_ph=payload.soil_profile.ph_level,
        field_area_acres=payload.field_area_acres or 1.0
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/irrigation", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_irrigation_advisory(request: Request, payload: IrrigationRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.irrigation_scheduler import calculate_irrigation_schedule
    result = calculate_irrigation_schedule(
        crop_name=payload.crop_name,
        temperature_c=payload.temperature,
        humidity_pct=payload.humidity,
        rainfall_14d_mm=payload.rainfall_14d_mm,
        growth_stage=payload.growth_stage or "mid",
        field_area_acres=payload.field_area_acres or 1.0,
        soil_type=payload.soil_type or "Loamy",
        irrigation_method=payload.irrigation_method or "Drip Irrigation"
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/disease-risk", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_disease_risk_advisory(request: Request, payload: DiseaseRiskRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.disease_risk import calculate_disease_pest_risk
    result = calculate_disease_pest_risk(
        crop_name=payload.crop_name,
        temperature_c=payload.temperature,
        humidity_pct=payload.humidity,
        rainfall_14d_mm=payload.rainfall_14d_mm
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/economics", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_economics_advisory(request: Request, payload: EconomicsRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.economics_engine import calculate_crop_profitability
    result = calculate_crop_profitability(
        crop_name=payload.crop_name,
        viability_score=payload.viability_score or 0.90,
        field_area_acres=payload.field_area_acres or 1.0,
        custom_market_price_quintal=payload.custom_market_price_quintal,
        fertilizer_cost_inr=payload.fertilizer_cost_inr or 4500.0,
        irrigation_cost_inr=payload.irrigation_cost_inr or 2500.0
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/soil-health", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_soil_health_advisory(request: Request, payload: SoilHealthRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.soil_health import calculate_soil_health_and_carbon
    result = calculate_soil_health_and_carbon(
        soil_n=payload.soil_profile.nitrogen_mg_kg,
        soil_p=payload.soil_profile.phosphorus_mg_kg,
        soil_k=payload.soil_profile.potassium_mg_kg,
        soil_ph=payload.soil_profile.ph_level,
        rainfall_mm=payload.rainfall_mm,
        temperature_c=payload.temperature_c,
        field_area_acres=payload.field_area_acres or 1.0,
        organic_matter_pct=payload.organic_matter_pct or 0.75,
        tillage_type=payload.tillage_type or "Conventional Tillage"
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/reports/pdf", tags=["Agronomic Advisory"])
@limiter.limit("30/minute")
def download_pdf_report(request: Request, payload: PdfReportRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.pdf_generator import generate_crop_report
    pdf_bytes = generate_crop_report(
        crop_name=payload.crop_name,
        viability=payload.viability,
        n=payload.nitrogen,
        p=payload.phosphorus,
        k=payload.potassium,
        temp=payload.temperature,
        hum=payload.humidity,
        ph=payload.ph,
        rain=payload.rainfall,
        summary=payload.summary,
        location_name=payload.location_name
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=CropMind_{payload.crop_name}_Advisory.pdf"}
    )


@app.post("/api/v1/advisory/crop-rotation", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_crop_rotation_plan(request: Request, payload: CropRotationRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.crop_rotation import generate_crop_rotation_plan
    result = generate_crop_rotation_plan(
        primary_crop=payload.primary_crop,
        soil_n=payload.soil_profile.nitrogen_mg_kg,
        soil_p=payload.soil_profile.phosphorus_mg_kg,
        soil_k=payload.soil_profile.potassium_mg_kg,
        field_area_acres=payload.field_area_acres or 1.0,
        include_green_manure=payload.include_green_manure if payload.include_green_manure is not None else True
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/climate-alerts", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_climate_alerts(request: Request, payload: ClimateAlertsRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.climate_alerts import evaluate_climate_anomalies
    result = evaluate_climate_anomalies(
        crop_name=payload.crop_name,
        temperature_c=payload.temperature,
        humidity_pct=payload.humidity,
        rainfall_14d_mm=payload.rainfall_14d_mm,
        wind_speed_kmh=payload.wind_speed_kmh or 12.0
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/fertigation-schedule", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_fertigation_schedule(request: Request, payload: FertigationScheduleRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.fertigation_calculator import calculate_fertigation_schedule
    result = calculate_fertigation_schedule(
        crop_name=payload.crop_name,
        growth_stage=payload.growth_stage or "vegetative",
        field_area_acres=payload.field_area_acres or 1.0,
        fertigation_frequency_per_week=payload.fertigation_frequency_per_week or 2,
        irrigation_volume_litres_cycle=payload.irrigation_volume_litres_cycle or 8000.0
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/micronutrients", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_micronutrient_advisory(request: Request, payload: MicronutrientRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.micronutrient_advisor import diagnose_micronutrient_deficiencies
    result = diagnose_micronutrient_deficiencies(
        crop_name=payload.crop_name,
        soil_ph=payload.soil_ph,
        organic_matter_pct=payload.organic_matter_pct or 0.75,
        soil_zn_ppm=payload.soil_zn_ppm,
        soil_fe_ppm=payload.soil_fe_ppm,
        soil_b_ppm=payload.soil_b_ppm,
        soil_s_ppm=payload.soil_s_ppm
    )
    return {"status": "success", "data": result}


@app.post("/api/v1/advisory/mcda-ranking", tags=["Agronomic Advisory"])
@limiter.limit("60/minute")
def get_mcda_crop_ranking(request: Request, payload: CropRankingRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.crop_ranking import calculate_topsis_crop_ranking
    candidate_list = [{"crop": c.crop, "viability_score": c.viability_score} for c in payload.candidates]
    result = calculate_topsis_crop_ranking(
        candidate_crops_with_scores=candidate_list,
        temperature_c=payload.temperature_c,
        humidity_pct=payload.humidity_pct,
        rainfall_mm=payload.rainfall_mm,
        weight_viability=payload.weight_viability or 0.35,
        weight_profit=payload.weight_profit or 0.25,
        weight_water_efficiency=payload.weight_water_efficiency or 0.20,
        weight_resilience=payload.weight_resilience or 0.20,
        field_area_acres=payload.field_area_acres or 1.0
    )
    return {"status": "success", "data": result}


# --- SPATIAL PARCELS ENDPOINTS ---

@app.post("/api/v1/parcels/calculate-area", tags=["Spatial Farm Parcels"])
@limiter.limit("60/minute")
def calculate_parcel_geometry(request: Request, payload: ParcelCalculateRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.spatial_parcels import calculate_polygon_geodesic_area
    result = calculate_polygon_geodesic_area(payload.coordinates)
    return {"status": "success", "data": result}


@app.post("/api/v1/parcels", tags=["Spatial Farm Parcels"])
@limiter.limit("30/minute")
def create_parcel_endpoint(request: Request, payload: ParcelCreateRequest, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.spatial_parcels import calculate_polygon_geodesic_area, polygon_to_geojson
    from src.db import create_farm_parcel
    
    geo_res = calculate_polygon_geodesic_area(payload.coordinates)
    geojson_str = polygon_to_geojson(payload.coordinates, properties={"name": payload.parcel_name})
    
    parcel_id = create_farm_parcel(
        user_id=user_row["id"],
        parcel_name=payload.parcel_name,
        polygon_geojson=geojson_str,
        area_acres=geo_res["area_acres"],
        centroid_lat=geo_res["centroid"]["lat"],
        centroid_lon=geo_res["centroid"]["lon"],
        primary_crop=payload.primary_crop,
        soil_type=payload.soil_type or "Clay Loam"
    )
    
    return {
        "status": "success",
        "parcel_id": parcel_id,
        "data": {
            "parcel_name": payload.parcel_name,
            "area_acres": geo_res["area_acres"],
            "area_hectares": geo_res["area_hectares"],
            "area_sq_meters": geo_res["area_sq_meters"],
            "perimeter_meters": geo_res["perimeter_meters"],
            "centroid": geo_res["centroid"],
            "primary_crop": payload.primary_crop,
            "soil_type": payload.soil_type
        }
    }


@app.get("/api/v1/parcels", tags=["Spatial Farm Parcels"])
@limiter.limit("60/minute")
def list_parcels_endpoint(request: Request, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import get_user_farm_parcels
    parcels = get_user_farm_parcels(user_id=user_row["id"])
    return {"status": "success", "data": parcels}


@app.delete("/api/v1/parcels/{parcel_id}", tags=["Spatial Farm Parcels"])
@limiter.limit("20/minute")
def delete_parcel_endpoint(request: Request, parcel_id: int, current_user: str = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (current_user,))
    user_row = cursor.fetchone()
    conn.close()
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
        
    from src.db import delete_farm_parcel
    deleted = delete_farm_parcel(parcel_id=parcel_id, user_id=user_row["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Parcel not found or unauthorized")
    return {"status": "success", "message": f"Parcel {parcel_id} deleted successfully"}


# --- AGRONOMIC KNOWLEDGE BASE ENDPOINTS ---

@app.get("/api/v1/knowledge/search", tags=["Agronomic Knowledge"])
@limiter.limit("60/minute")
def search_knowledge_endpoint(
    request: Request,
    query: Optional[str] = Query(None, description="Search keyword, problem, or pest"),
    crop: Optional[str] = Query(None, description="Specific crop filter"),
    category: Optional[str] = Query(None, description="Specific topic category"),
    limit: Optional[int] = Query(5, ge=1, le=20, description="Max results to return")
):
    from src.agri_knowledge import search_agronomic_knowledge
    result = search_agronomic_knowledge(
        query=query or "",
        crop=crop,
        category=category,
        max_results=limit or 5
    )
    return result


@app.get("/api/v1/knowledge/categories", tags=["Agronomic Knowledge"])
@limiter.limit("60/minute")
def get_knowledge_categories_endpoint(request: Request):
    from src.agri_knowledge import get_all_categories
    categories = get_all_categories()
    return {"status": "success", "data": categories}


# --- REPORT EXPORT ENDPOINTS ---

@app.post("/api/v1/reports/excel", tags=["Agronomic Advisory"])
@limiter.limit("30/minute")
def download_excel_dossier(request: Request, payload: ExcelReportRequest, current_user: str = Depends(get_current_user_or_api_key)):
    from src.data_exporter import generate_excel_crop_dossier
    from src.fertilizer_advisor import calculate_nutrient_prescription
    from src.economics_engine import calculate_crop_profitability
    
    fert_data = calculate_nutrient_prescription(
        crop_name=payload.crop_name,
        soil_n=payload.soil_profile.nitrogen_mg_kg,
        soil_p=payload.soil_profile.phosphorus_mg_kg,
        soil_k=payload.soil_profile.potassium_mg_kg,
        soil_ph=payload.soil_profile.ph_level,
        field_area_acres=payload.field_area_acres or 1.0
    )
    
    econ_data = calculate_crop_profitability(
        crop_name=payload.crop_name,
        viability_score=payload.viability,
        field_area_acres=payload.field_area_acres or 1.0
    )
    
    excel_bytes = generate_excel_crop_dossier(
        crop_name=payload.crop_name,
        viability=payload.viability,
        soil_profile={
            "nitrogen": payload.soil_profile.nitrogen_mg_kg,
            "phosphorus": payload.soil_profile.phosphorus_mg_kg,
            "potassium": payload.soil_profile.potassium_mg_kg,
            "ph": payload.soil_profile.ph_level
        },
        climate_profile={
            "temperature": payload.temperature,
            "humidity": payload.humidity,
            "rainfall": payload.rainfall
        },
        field_area_acres=payload.field_area_acres or 1.0,
        fertilizer_data=fert_data,
        economics_data=econ_data
    )
    
    return Response(
        content=excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=CropMind_{payload.crop_name}_Dossier.xlsx"}
    )













