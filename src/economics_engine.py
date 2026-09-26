"""
CropMind AI: Precision Agricultural Economics & Yield Profitability Engine
Computes expected crop yields, mandi market revenue, input expenditure breakdown, and net profit margins per acre.
"""

from typing import Dict, Any, List, Optional
import math

# Baseline yield (Tons/acre), mandi price (INR/Quintal), and standard production costs (INR/acre)
CROP_ECONOMIC_DATA = {
    "Rice": {"base_yield_tons": 2.2, "price_per_quintal": 2203, "seed_cost": 1800, "machinery_cost": 4500, "labor_cost": 8500, "crop_cycle_days": 120},
    "Maize": {"base_yield_tons": 2.8, "price_per_quintal": 2090, "seed_cost": 2200, "machinery_cost": 3800, "labor_cost": 5500, "crop_cycle_days": 110},
    "Chickpea": {"base_yield_tons": 0.9, "price_per_quintal": 5440, "seed_cost": 2400, "machinery_cost": 3000, "labor_cost": 4500, "crop_cycle_days": 100},
    "Kidneybeans": {"base_yield_tons": 0.8, "price_per_quintal": 7200, "seed_cost": 3200, "machinery_cost": 3200, "labor_cost": 5000, "crop_cycle_days": 90},
    "Pigeonpeas": {"base_yield_tons": 0.85, "price_per_quintal": 7000, "seed_cost": 2000, "machinery_cost": 3200, "labor_cost": 4800, "crop_cycle_days": 150},
    "Mothbeans": {"base_yield_tons": 0.5, "price_per_quintal": 6500, "seed_cost": 1500, "machinery_cost": 2500, "labor_cost": 3500, "crop_cycle_days": 80},
    "Mungbean": {"base_yield_tons": 0.6, "price_per_quintal": 8558, "seed_cost": 1800, "machinery_cost": 2800, "labor_cost": 4000, "crop_cycle_days": 75},
    "Blackgram": {"base_yield_tons": 0.65, "price_per_quintal": 6950, "seed_cost": 1900, "machinery_cost": 2800, "labor_cost": 4200, "crop_cycle_days": 80},
    "Lentil": {"base_yield_tons": 0.7, "price_per_quintal": 6425, "seed_cost": 2100, "machinery_cost": 2900, "labor_cost": 4000, "crop_cycle_days": 100},
    "Pomegranate": {"base_yield_tons": 4.5, "price_per_quintal": 8500, "seed_cost": 8000, "machinery_cost": 6500, "labor_cost": 14000, "crop_cycle_days": 365},
    "Banana": {"base_yield_tons": 18.0, "price_per_quintal": 2100, "seed_cost": 12000, "machinery_cost": 8000, "labor_cost": 22000, "crop_cycle_days": 300},
    "Mango": {"base_yield_tons": 4.0, "price_per_quintal": 4500, "seed_cost": 6000, "machinery_cost": 5000, "labor_cost": 11000, "crop_cycle_days": 365},
    "Grapes": {"base_yield_tons": 8.0, "price_per_quintal": 6000, "seed_cost": 15000, "machinery_cost": 12000, "labor_cost": 30000, "crop_cycle_days": 210},
    "Watermelon": {"base_yield_tons": 12.0, "price_per_quintal": 1200, "seed_cost": 4500, "machinery_cost": 4000, "labor_cost": 9000, "crop_cycle_days": 85},
    "Muskmelon": {"base_yield_tons": 9.0, "price_per_quintal": 1600, "seed_cost": 4200, "machinery_cost": 3800, "labor_cost": 8500, "crop_cycle_days": 80},
    "Apple": {"base_yield_tons": 5.0, "price_per_quintal": 7500, "seed_cost": 10000, "machinery_cost": 8000, "labor_cost": 18000, "crop_cycle_days": 240},
    "Orange": {"base_yield_tons": 6.0, "price_per_quintal": 4200, "seed_cost": 8000, "machinery_cost": 6000, "labor_cost": 14000, "crop_cycle_days": 365},
    "Papaya": {"base_yield_tons": 25.0, "price_per_quintal": 1400, "seed_cost": 6000, "machinery_cost": 5000, "labor_cost": 15000, "crop_cycle_days": 300},
    "Coconut": {"base_yield_tons": 4.0, "price_per_quintal": 3500, "seed_cost": 5000, "machinery_cost": 4000, "labor_cost": 9000, "crop_cycle_days": 365},
    "Cotton": {"base_yield_tons": 1.1, "price_per_quintal": 7020, "seed_cost": 2800, "machinery_cost": 4500, "labor_cost": 9500, "crop_cycle_days": 160},
    "Jute": {"base_yield_tons": 1.4, "price_per_quintal": 5050, "seed_cost": 1600, "machinery_cost": 3500, "labor_cost": 7500, "crop_cycle_days": 120},
    "Coffee": {"base_yield_tons": 0.6, "price_per_quintal": 28000, "seed_cost": 9000, "machinery_cost": 7000, "labor_cost": 20000, "crop_cycle_days": 365},
}

DEFAULT_ECONOMICS = {"base_yield_tons": 1.5, "price_per_quintal": 3000, "seed_cost": 2500, "machinery_cost": 4000, "labor_cost": 7000, "crop_cycle_days": 120}


def calculate_crop_profitability(
    crop_name: str,
    viability_score: float = 0.90,
    field_area_acres: float = 1.0,
    custom_market_price_quintal: Optional[float] = None,
    fertilizer_cost_inr: float = 4500.0,
    irrigation_cost_inr: float = 2500.0
) -> Dict[str, Any]:
    """
    Computes expected yield output, total production costs, projected revenue, and net profit margin.
    """
    crop_key = crop_name.strip().capitalize()
    econ = CROP_ECONOMIC_DATA.get(crop_key, DEFAULT_ECONOMICS)
    
    # Adjusted yield based on viability score (TreeSHAP confidence factor)
    # Range multiplier 0.70 to 1.15
    yield_multiplier = max(0.60, min(1.20, 0.50 + (viability_score * 0.65)))
    expected_yield_tons_per_acre = round(econ["base_yield_tons"] * yield_multiplier, 2)
    total_yield_tons = round(expected_yield_tons_per_acre * field_area_acres, 2)
    total_yield_quintals = round(total_yield_tons * 10.0, 1) # 1 Ton = 10 Quintals
    
    # Mandi price per quintal
    mandi_price = custom_market_price_quintal if (custom_market_price_quintal and custom_market_price_quintal > 0) else econ["price_per_quintal"]
    price_per_ton = mandi_price * 10.0
    
    # Gross Revenue
    gross_revenue_inr = round(total_yield_quintals * mandi_price, 2)
    
    # Input Cost Breakdown per acre
    seed_cost = econ["seed_cost"] * field_area_acres
    machinery_cost = econ["machinery_cost"] * field_area_acres
    labor_cost = econ["labor_cost"] * field_area_acres
    fert_cost = fertilizer_cost_inr * field_area_acres
    irr_cost = irrigation_cost_inr * field_area_acres
    plant_protection_cost = 1800.0 * field_area_acres
    
    total_input_cost_inr = round(
        seed_cost + machinery_cost + labor_cost + fert_cost + irr_cost + plant_protection_cost,
        2
    )
    
    # Net Profit
    net_profit_inr = round(gross_revenue_inr - total_input_cost_inr, 2)
    profit_margin_pct = round((net_profit_inr / gross_revenue_inr) * 100.0, 1) if gross_revenue_inr > 0 else 0.0
    roi_pct = round((net_profit_inr / total_input_cost_inr) * 100.0, 1) if total_input_cost_inr > 0 else 0.0
    
    # Break-even yield (tons/acre)
    break_even_yield_tons = round(total_input_cost_inr / (price_per_ton * field_area_acres), 2) if price_per_ton > 0 else 0.0
    
    cost_breakdown = [
        {"category": "Seed / Saplings", "cost_inr": seed_cost, "pct": round((seed_cost / total_input_cost_inr) * 100, 1)},
        {"category": "Land Prep & Machinery", "cost_inr": machinery_cost, "pct": round((machinery_cost / total_input_cost_inr) * 100, 1)},
        {"category": "Fertilizers & Nutrients", "cost_inr": fert_cost, "pct": round((fert_cost / total_input_cost_inr) * 100, 1)},
        {"category": "Irrigation & Pumping", "cost_inr": irr_cost, "pct": round((irr_cost / total_input_cost_inr) * 100, 1)},
        {"category": "Plant Protection & Bio-control", "cost_inr": plant_protection_cost, "pct": round((plant_protection_cost / total_input_cost_inr) * 100, 1)},
        {"category": "Labor & Harvesting", "cost_inr": labor_cost, "pct": round((labor_cost / total_input_cost_inr) * 100, 1)},
    ]

    return {
        "crop": crop_key,
        "field_area_acres": field_area_acres,
        "viability_score": viability_score,
        "yield_estimates": {
            "yield_per_acre_tons": expected_yield_tons_per_acre,
            "total_yield_tons": total_yield_tons,
            "total_yield_quintals": total_yield_quintals,
            "break_even_yield_tons_acre": break_even_yield_tons,
        },
        "market_pricing": {
            "mandi_price_per_quintal_inr": mandi_price,
            "price_per_ton_inr": price_per_ton,
            "pricing_source": "Government MSP / Mandi Benchmark 2024-2025"
        },
        "financial_summary": {
            "gross_revenue_inr": gross_revenue_inr,
            "total_input_cost_inr": total_input_cost_inr,
            "net_profit_inr": net_profit_inr,
            "net_profit_per_acre_inr": round(net_profit_inr / field_area_acres, 2),
            "profit_margin_pct": profit_margin_pct,
            "roi_pct": roi_pct,
            "financial_viability": "Highly Profitable" if roi_pct >= 40 else ("Profitable" if roi_pct >= 15 else "Marginal / Risky")
        },
        "cost_breakdown": cost_breakdown,
        "crop_duration_days": econ["crop_cycle_days"]
    }
