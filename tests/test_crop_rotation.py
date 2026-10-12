"""
Unit and Integration tests for CropMind AI Crop Rotation & Companion Synergy Module.
"""

import pytest
from src.crop_rotation import generate_crop_rotation_plan, COMPANION_SYNERGY_DB
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_cereal_rotation_plan():
    res = generate_crop_rotation_plan(
        primary_crop="Rice",
        soil_n=80.0,
        soil_p=40.0,
        soil_k=35.0,
        field_area_acres=2.0,
        include_green_manure=True
    )
    assert res["primary_crop"] == "Rice"
    assert len(res["rotation_schedule"]) == 3
    assert res["ecological_benefits"]["biological_n_fixed_kg"] > 0
    assert len(res["companion_synergies"]) > 0


def test_perennial_rotation_plan():
    res = generate_crop_rotation_plan(
        primary_crop="Banana",
        soil_n=100.0,
        soil_p=50.0,
        soil_k=60.0,
        field_area_acres=1.5
    )
    assert res["primary_crop"] == "Banana"
    assert len(res["rotation_schedule"]) == 3
    assert "Year-Round" in res["rotation_schedule"][0]["stage"] or "Primary" in res["rotation_schedule"][0]["role"]


def test_crop_rotation_api_endpoint():
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
        "primary_crop": "Maize",
        "soil_profile": {
            "nitrogen_mg_kg": 75.0,
            "phosphorus_mg_kg": 40.0,
            "potassium_mg_kg": 30.0,
            "ph_level": 6.5
        },
        "field_area_acres": 1.0,
        "include_green_manure": True
    }
    
    response = client.post("/api/v1/advisory/crop-rotation", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["primary_crop"] == "Maize"
    assert "rotation_schedule" in data
