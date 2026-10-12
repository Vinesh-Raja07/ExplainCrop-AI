"""
Unit and Integration tests for CropMind AI Extreme Weather & Climate Anomaly Module.
"""

import pytest
from src.climate_alerts import evaluate_climate_anomalies, calculate_vpd
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_vpd_calculation():
    vpd = calculate_vpd(temperature_c=32.0, humidity_pct=40.0)
    assert 2.0 <= vpd <= 3.5


def test_heatwave_alert_trigger():
    res = evaluate_climate_anomalies(
        crop_name="Rice",
        temperature_c=41.0,
        humidity_pct=60.0,
        rainfall_14d_mm=30.0
    )
    assert res["total_active_alerts"] >= 1
    heat_alert = [a for a in res["alerts"] if "Heatwave" in a["anomaly_type"]][0]
    assert heat_alert["severity"] in ["Warning", "Critical"]
    assert "Kaolin" in heat_alert["emergency_mitigation"]


def test_frost_alert_trigger():
    res = evaluate_climate_anomalies(
        crop_name="Banana",
        temperature_c=4.0,
        humidity_pct=85.0,
        rainfall_14d_mm=5.0
    )
    assert res["total_active_alerts"] >= 1
    frost_alert = [a for a in res["alerts"] if "Frost" in a["anomaly_type"]][0]
    assert frost_alert["severity"] == "Critical"
    assert "smoke" in frost_alert["emergency_mitigation"].lower() or "irrigate" in frost_alert["emergency_mitigation"].lower()


def test_climate_alerts_api_endpoint():
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
        "temperature": 42.5,
        "humidity": 30.0,
        "rainfall_14d_mm": 5.0
    }
    
    response = client.post("/api/v1/advisory/climate-alerts", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["crop"] == "Cotton"
    assert data["total_active_alerts"] > 0
