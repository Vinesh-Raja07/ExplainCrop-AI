"""
Unit tests for CropMind AI Fertilizer Prescription & Irrigation Scheduling Modules.
"""

import pytest
from src.fertilizer_advisor import calculate_nutrient_prescription, CROP_NUTRIENT_BENCHMARKS
from src.irrigation_scheduler import calculate_irrigation_schedule, estimate_et0


def test_fertilizer_prescription_deficit():
    # Low nitrogen, phosphorus, and potassium
    res = calculate_nutrient_prescription(
        crop_name="Rice",
        soil_n=30.0,
        soil_p=20.0,
        soil_k=15.0,
        soil_ph=6.5,
        field_area_acres=2.0
    )
    assert res["crop"] == "Rice"
    assert res["deficit_status"]["nitrogen_status"] == "Deficient"
    assert res["deficit_status"]["phosphorus_status"] == "Deficient"
    assert res["deficit_status"]["potassium_status"] == "Deficient"
    assert res["commercial_prescription"]["dap_50kg_bags"] > 0
    assert res["commercial_prescription"]["urea_50kg_bags"] >= 0
    assert res["commercial_prescription"]["mop_50kg_bags"] > 0


def test_fertilizer_ph_remediation_acidic():
    res = calculate_nutrient_prescription(
        crop_name="Maize",
        soil_n=90.0,
        soil_p=50.0,
        soil_k=30.0,
        soil_ph=4.5,
        field_area_acres=1.0
    )
    assert res["ph_remediation"]["status"] == "Acidic"
    assert "Lime" in res["ph_remediation"]["remedy"]


def test_et0_estimation():
    et0 = estimate_et0(temperature_c=30.0, humidity_pct=60.0, latitude=13.0)
    assert 2.0 <= et0 <= 8.5


def test_irrigation_schedule_calculation():
    res = calculate_irrigation_schedule(
        crop_name="Rice",
        temperature_c=32.0,
        humidity_pct=65.0,
        rainfall_14d_mm=10.0,
        growth_stage="mid",
        field_area_acres=1.0,
        irrigation_method="Drip Irrigation"
    )
    assert res["crop"] == "Rice"
    assert res["reference_et0_mm_day"] > 0
    assert res["crop_etc_mm_day"] > 0
    assert res["water_volume_litres_acre_day"] > 0
    assert res["recommended_runtime_hours"] > 0
