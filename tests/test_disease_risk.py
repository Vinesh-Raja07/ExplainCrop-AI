"""
Unit tests for CropMind AI Disease & Pest Risk Forecasting Module.
"""

import pytest
from src.disease_risk import calculate_disease_pest_risk


def test_rice_disease_high_humidity_risk():
    # Warm temperature and high humidity triggers severe fungal risk in rice
    res = calculate_disease_pest_risk(
        crop_name="Rice",
        temperature_c=26.0,
        humidity_pct=92.0,
        rainfall_14d_mm=120.0
    )
    assert res["crop"] == "Rice"
    assert res["overall_risk_status"] in ["High Risk", "Severe Risk"]
    assert len(res["pathogens"]) >= 2
    blast = [p for p in res["pathogens"] if "Blast" in p["pathogen_name"]][0]
    assert blast["dsi_score"] >= 70.0
    assert blast["risk_level"] in ["High", "Severe"]


def test_cotton_pest_risk():
    res = calculate_disease_pest_risk(
        crop_name="Cotton",
        temperature_c=29.0,
        humidity_pct=68.0,
        rainfall_14d_mm=25.0
    )
    assert res["crop"] == "Cotton"
    bollworm = [p for p in res["pathogens"] if "Bollworm" in p["pathogen_name"]][0]
    assert bollworm["dsi_score"] > 40.0


def test_generic_fallback_pathogen_risk():
    res = calculate_disease_pest_risk(
        crop_name="Chickpea",
        temperature_c=24.0,
        humidity_pct=75.0,
        rainfall_14d_mm=15.0
    )
    assert res["crop"] == "Chickpea"
    assert len(res["pathogens"]) > 0
    assert "ipm_advisory" in res
