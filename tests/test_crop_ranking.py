"""
Unit and Integration tests for CropMind AI Multi-Criteria Decision Analysis (TOPSIS) Crop Ranking Module.
"""

import pytest
from src.crop_ranking import calculate_topsis_crop_ranking
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_topsis_ranking_calculation():
    candidates = [
        {"crop": "Rice", "viability_score": 0.95},
        {"crop": "Maize", "viability_score": 0.88},
        {"crop": "Chickpea", "viability_score": 0.82},
        {"crop": "Cotton", "viability_score": 0.80}
    ]
    res = calculate_topsis_crop_ranking(
        candidate_crops_with_scores=candidates,
        temperature_c=28.0,
        humidity_pct=70.0,
        rainfall_mm=100.0,
        field_area_acres=2.0
    )
    assert len(res["ranked_crops"]) == 4
    assert res["ranked_crops"][0]["topsis_rank"] == 1
    for r in res["ranked_crops"]:
        assert 0.0 <= r["topsis_score"] <= 1.0


def test_water_conservation_priority_weighting():
    candidates = [
        {"crop": "Rice", "viability_score": 0.90}, # High water requirement ~1200mm
        {"crop": "Mothbeans", "viability_score": 0.80} # Extremely low water requirement ~300mm
    ]
    # Heavy weight on water efficiency (70%)
    res = calculate_topsis_crop_ranking(
        candidate_crops_with_scores=candidates,
        temperature_c=34.0,
        humidity_pct=40.0,
        rainfall_mm=20.0,
        weight_viability=0.10,
        weight_profit=0.10,
        weight_water_efficiency=0.70,
        weight_resilience=0.10
    )
    # Mothbeans should score high when water efficiency is prioritized
    assert res["ranked_crops"][0]["crop"] == "Mothbeans"


def test_mcda_api_endpoint():
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
        "candidates": [
            {"crop": "Rice", "viability_score": 0.92},
            {"crop": "Banana", "viability_score": 0.89},
            {"crop": "Cotton", "viability_score": 0.85}
        ],
        "temperature_c": 27.5,
        "humidity_pct": 72.0,
        "rainfall_mm": 110.0,
        "field_area_acres": 1.0
    }
    
    response = client.post("/api/v1/advisory/mcda-ranking", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert "ranked_crops" in data
    assert len(data["ranked_crops"]) == 3
