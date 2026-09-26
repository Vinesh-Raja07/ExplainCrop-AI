"""
Integration tests for CropMind AI FastAPI Microservice.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api import app
from src.db import init_database, get_db_connection, create_api_key

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_test_auth():
    init_database()
    conn = get_db_connection()
    cursor = conn.cursor()
    # Create test user if not existing
    from src.security import get_password_hash
    cursor.execute("SELECT id FROM users WHERE username = 'testuser'")
    row = cursor.fetchone()
    if not row:
        cursor.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            ("testuser", get_password_hash("testpass123"), "Farmer")
        )
        user_id = cursor.lastrowid
    else:
        user_id = row["id"]
    conn.commit()
    conn.close()
    
    test_key = create_api_key(user_id)
    return {"api_key": test_key}


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metadata():
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    assert "data" in response.json()


def test_fertilizer_advisory_endpoint(setup_test_auth):
    headers = {"X-API-Key": setup_test_auth["api_key"]}
    payload = {
        "crop_name": "Rice",
        "soil_profile": {
            "nitrogen_mg_kg": 40.0,
            "phosphorus_mg_kg": 25.0,
            "potassium_mg_kg": 20.0,
            "ph_level": 6.2
        },
        "field_area_acres": 2.0
    }
    response = client.post("/api/v1/advisory/fertilizer", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["crop"] == "Rice"
    assert "commercial_prescription" in data


def test_irrigation_advisory_endpoint(setup_test_auth):
    headers = {"X-API-Key": setup_test_auth["api_key"]}
    payload = {
        "crop_name": "Maize",
        "temperature": 30.0,
        "humidity": 65.0,
        "rainfall_14d_mm": 20.0,
        "growth_stage": "mid",
        "field_area_acres": 1.5,
        "irrigation_method": "Drip Irrigation"
    }
    response = client.post("/api/v1/advisory/irrigation", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["crop"] == "Maize"
    assert data["water_volume_litres_acre_day"] > 0


def test_disease_risk_endpoint(setup_test_auth):
    headers = {"X-API-Key": setup_test_auth["api_key"]}
    payload = {
        "crop_name": "Rice",
        "temperature": 27.0,
        "humidity": 90.0,
        "rainfall_14d_mm": 80.0
    }
    response = client.post("/api/v1/advisory/disease-risk", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert len(data["pathogens"]) > 0


def test_economics_endpoint(setup_test_auth):
    headers = {"X-API-Key": setup_test_auth["api_key"]}
    payload = {
        "crop_name": "Rice",
        "viability_score": 0.95,
        "field_area_acres": 1.0
    }
    response = client.post("/api/v1/advisory/economics", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["financial_summary"]["gross_revenue_inr"] > 0


def test_soil_health_endpoint(setup_test_auth):
    headers = {"X-API-Key": setup_test_auth["api_key"]}
    payload = {
        "soil_profile": {
            "nitrogen_mg_kg": 90.0,
            "phosphorus_mg_kg": 45.0,
            "potassium_mg_kg": 50.0,
            "ph_level": 6.8
        },
        "rainfall_mm": 120.0,
        "temperature_c": 28.0,
        "field_area_acres": 1.0
    }
    response = client.post("/api/v1/advisory/soil-health", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["soil_health_index"] > 0


def test_pdf_report_endpoint(setup_test_auth):
    headers = {"X-API-Key": setup_test_auth["api_key"]}
    payload = {
        "crop_name": "Rice",
        "viability": 0.96,
        "nitrogen": 90.0,
        "phosphorus": 42.0,
        "potassium": 43.0,
        "ph": 6.5,
        "temperature": 26.5,
        "humidity": 75.0,
        "rainfall": 110.0,
        "summary": "Excellent hydrothermal balance for high-yield paddy cultivation."
    }
    response = client.post("/api/v1/reports/pdf", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 500
