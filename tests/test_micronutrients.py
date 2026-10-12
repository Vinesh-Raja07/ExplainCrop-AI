"""
Unit and Integration tests for CropMind AI Micronutrient Diagnostics & Foliar Prescription Module.
"""

import pytest
from src.micronutrient_advisor import diagnose_micronutrient_deficiencies
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_alkaline_soil_fe_zn_diagnosis():
    res = diagnose_micronutrient_deficiencies(
        crop_name="Rice",
        soil_ph=8.2,
        organic_matter_pct=0.9,
        soil_zn_ppm=0.40
    )
    assert res["crop"] == "Rice"
    fe_item = [e for e in res["deficiency_profiles"] if "Iron" in e["nutrient"]][0]
    zn_item = [e for e in res["deficiency_profiles"] if "Zinc" in e["nutrient"]][0]
    assert "High" in fe_item["risk_level"]
    assert "High" in zn_item["risk_level"]
    assert "EDDHA" in fe_item["soil_remedy"] or "Ferrous Sulfate" in fe_item["foliar_remedy"]
    assert "Zinc Sulfate" in zn_item["foliar_remedy"]


def test_low_om_boron_sulfur_diagnosis():
    res = diagnose_micronutrient_deficiencies(
        crop_name="Pomegranate",
        soil_ph=6.8,
        organic_matter_pct=0.35,
        soil_b_ppm=0.30
    )
    assert res["crop"] == "Pomegranate"
    b_item = [e for e in res["deficiency_profiles"] if "Boron" in e["nutrient"]][0]
    s_item = [e for e in res["deficiency_profiles"] if "Sulfur" in e["nutrient"]][0]
    assert "High" in b_item["risk_level"]
    assert "High" in s_item["risk_level"]
    assert "Solubor" in b_item["foliar_remedy"]


def test_micronutrient_api_endpoint():
    from src.db import get_db_connection, create_api_key
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = 'testuser'")
    row = cursor.fetchone()
    user_id = row["id"] if row else 1
    conn.close()
    
    test_key = create_api_key(user_id)
    headers = {"X-API-Key": test_key}
    
    payload = {
        "crop_name": "Cotton",
        "soil_ph": 8.0,
        "organic_matter_pct": 0.5,
        "soil_zn_ppm": 0.5,
        "soil_b_ppm": 0.35
    }
    
    response = client.post("/api/v1/advisory/micronutrients", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["crop"] == "Cotton"
    assert len(data["deficiency_profiles"]) >= 4
