"""
Unit and Integration tests for CropMind AI Drip Fertigation & WSF Dosing Module.
"""

import pytest
from src.fertigation_calculator import calculate_fertigation_schedule
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_banana_fertigation_vegetative():
    res = calculate_fertigation_schedule(
        crop_name="Banana",
        growth_stage="vegetative",
        field_area_acres=2.0,
        fertigation_frequency_per_week=2,
        irrigation_volume_litres_cycle=12000.0
    )
    assert res["crop"] == "Banana"
    assert res["total_wsf_kg_per_cycle"] > 0
    assert 0.5 <= res["target_solution_ec_dsm"] <= 3.5
    assert len(res["tank_separation"]["tank_a_calcium_nitrate_compatible"]) > 0
    assert len(res["tank_separation"]["tank_b_phosphate_sulfate_compatible"]) > 0


def test_fertigation_api_endpoint():
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
        "growth_stage": "boll_development",
        "field_area_acres": 1.5,
        "fertigation_frequency_per_week": 2,
        "irrigation_volume_litres_cycle": 9000.0
    }
    
    response = client.post("/api/v1/advisory/fertigation-schedule", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["crop"] == "Cotton"
    assert "tank_separation" in data
    assert data["total_wsf_kg_per_cycle"] > 0
