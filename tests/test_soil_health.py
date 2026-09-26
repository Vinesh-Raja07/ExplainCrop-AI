"""
Unit tests for CropMind AI Soil Health Index & Carbon Footprint Module.
"""

import pytest
from src.soil_health import calculate_soil_health_and_carbon


def test_optimal_soil_health_score():
    res = calculate_soil_health_and_carbon(
        soil_n=90.0,
        soil_p=45.0,
        soil_k=45.0,
        soil_ph=6.8,
        rainfall_mm=100.0,
        temperature_c=26.0,
        field_area_acres=1.0,
        organic_matter_pct=1.2,
        tillage_type="Minimum Tillage"
    )
    assert res["soil_health_index"] >= 75.0
    assert "Sustainable" in res["soil_health_rating"] or "High" in res["soil_health_rating"]
    assert res["carbon_footprint"]["total_ghg_emissions_kg_co2e"] > 0
    assert len(res["regenerative_advisory"]) == 4


def test_degraded_soil_low_organic_matter():
    res = calculate_soil_health_and_carbon(
        soil_n=10.0,
        soil_p=10.0,
        soil_k=10.0,
        soil_ph=4.2,
        rainfall_mm=350.0,
        temperature_c=42.0,
        field_area_acres=2.0,
        organic_matter_pct=0.3,
        tillage_type="Conventional Tillage"
    )
    assert res["soil_health_index"] < 60.0
    assert res["nitrogen_leaching"]["leaching_risk_percentage"] > 0
    assert res["carbon_footprint"]["total_ghg_emissions_kg_co2e"] > 0
