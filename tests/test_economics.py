"""
Unit tests for CropMind AI Economics & Profitability Engine.
"""

import pytest
from src.economics_engine import calculate_crop_profitability, CROP_ECONOMIC_DATA


def test_rice_profitability_calculation():
    res = calculate_crop_profitability(
        crop_name="Rice",
        viability_score=0.92,
        field_area_acres=2.0
    )
    assert res["crop"] == "Rice"
    assert res["yield_estimates"]["yield_per_acre_tons"] > 0
    assert res["yield_estimates"]["total_yield_tons"] > 0
    assert res["financial_summary"]["gross_revenue_inr"] > 0
    assert res["financial_summary"]["total_input_cost_inr"] > 0
    assert len(res["cost_breakdown"]) == 6


def test_custom_market_price():
    res = calculate_crop_profitability(
        crop_name="Banana",
        viability_score=0.95,
        field_area_acres=1.0,
        custom_market_price_quintal=3000.0
    )
    assert res["market_pricing"]["mandi_price_per_quintal_inr"] == 3000.0
    assert res["financial_summary"]["gross_revenue_inr"] > 0
    assert res["financial_summary"]["roi_pct"] > 0
