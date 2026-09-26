"""
CropMind AI: Soil Health Index (SHI) & Carbon Footprint Evaluation Engine
Computes multidimensional soil quality scores, nitrogen leaching risk, and GHG emissions per acre.
"""

from typing import Dict, Any, List, Optional
import math


def calculate_soil_health_and_carbon(
    soil_n: float,
    soil_p: float,
    soil_k: float,
    soil_ph: float,
    rainfall_mm: float,
    temperature_c: float,
    field_area_acres: float = 1.0,
    organic_matter_pct: float = 0.75, # Typical Indian soils range 0.4% - 1.2%
    tillage_type: str = "Conventional Tillage"
) -> Dict[str, Any]:
    """
    Evaluates Soil Health Index (0-100), Carbon Footprint (kg CO2e), and Nitrogen Leaching risk.
    """
    # 1. pH Sub-Index (Optimal 6.5 - 7.2)
    ph_dev = abs(soil_ph - 6.8)
    ph_score = max(20.0, 100.0 - (ph_dev * 25.0))
    
    # 2. Nutrient Balance Sub-Index (Harmonic balance of N-P-K)
    # Target ideal ratio N:P:K ~ 4:2:1 (or approx 100:50:25 ppm)
    n_score = max(20.0, min(100.0, 100.0 - abs(soil_n - 90.0) * 0.7))
    p_score = max(20.0, min(100.0, 100.0 - abs(soil_p - 45.0) * 1.1))
    k_score = max(20.0, min(100.0, 100.0 - abs(soil_k - 45.0) * 1.1))
    nutrient_score = (n_score + p_score + k_score) / 3.0
    
    # 3. Organic Matter & Biological Proxy Sub-Index
    # Optimal organic matter > 1.5%
    om_score = min(100.0, (organic_matter_pct / 1.5) * 100.0)
    
    # 4. Climatic Resilience Sub-Index
    climate_score = 100.0
    if rainfall_mm > 250.0:
        climate_score -= (rainfall_mm - 250.0) * 0.25 # Erosion / Leaching risk
    if temperature_c > 38.0:
        climate_score -= (temperature_c - 38.0) * 3.0 # Organic carbon oxidation
    climate_score = max(25.0, min(100.0, climate_score))
    
    # Composite Soil Health Index (SHI, 0 - 100)
    shi = round((ph_score * 0.25) + (nutrient_score * 0.35) + (om_score * 0.25) + (climate_score * 0.15), 1)
    
    if shi >= 80:
        shi_rating = "Excellent / High Fertility"
        shi_color = "#10B981"
    elif shi >= 65:
        shi_rating = "Good / Sustainable"
        shi_color = "#0284C7"
    elif shi >= 50:
        shi_rating = "Moderate / Needs Organic Inputs"
        shi_color = "#F59E0B"
    else:
        shi_rating = "Degraded / Low Fertility"
        shi_color = "#E11D48"

    # 5. Nitrogen Leaching Risk Calculation
    # High rainfall + high nitrogen = high leaching into groundwater
    leach_factor = (soil_n / 150.0) * (rainfall_mm / 200.0)
    leaching_pct = min(45.0, max(5.0, round(leach_factor * 25.0, 1)))
    leached_n_kg_acre = round((soil_n * 0.907) * (leaching_pct / 100.0) * field_area_acres, 2)
    
    # 6. Carbon & GHG Footprint Modeling (IPCC Tier 1 Formulation)
    # Synthetic N fertilizer emission factor: ~1.25 kg N2O-N per 100 kg N = ~4.5 kg CO2e / kg N
    n_applied_kg = max(20.0, soil_n * 0.907) * field_area_acres
    fert_ghg_kg_co2e = n_applied_kg * 4.5
    
    # Diesel fuel / tillage emission
    tillage_ghg_map = {
        "Zero Tillage": 25.0,
        "Minimum Tillage": 45.0,
        "Conventional Tillage": 85.0
    }
    tillage_ghg_kg_co2e = tillage_ghg_map.get(tillage_type, 85.0) * field_area_acres
    
    # Irrigation pumping emissions (~40 kg CO2e / acre)
    irrigation_ghg_kg_co2e = 40.0 * field_area_acres
    
    total_ghg_emissions_kg_co2e = round(fert_ghg_kg_co2e + tillage_ghg_kg_co2e + irrigation_ghg_kg_co2e, 1)
    
    # Potential Carbon Sequestration through regenerative practices (Tonnes CO2 / acre / year)
    carbon_sequestration_potential_kg = round(field_area_acres * 450.0, 1) # ~0.45 t CO2/acre/yr with biochar + cover crop
    
    # Regenerative Recommendations
    regenerative_advisory = [
        {"practice": "Cover Cropping (e.g. Dhaincha / Sunnhemp)", "impact": "Supplies +20 kg natural N/acre and builds organic carbon."},
        {"practice": "Biochar Application @ 500 kg/acre", "impact": "Permanently sequesters carbon and boosts soil water retention by 28%."},
        {"practice": "Adopt Zero/Reduced Tillage", "impact": f"Reduces machinery diesel emissions by ~{tillage_ghg_kg_co2e * 0.6:.0f} kg CO2e."},
        {"practice": "Split Nitrogen Application with Neem Coating", "impact": f"Cuts nitrogen leaching by up to 35% and reduces N2O emissions."}
    ]

    return {
        "field_area_acres": field_area_acres,
        "soil_health_index": shi,
        "soil_health_rating": shi_rating,
        "rating_color": shi_color,
        "sub_scores": {
            "ph_balance_score": round(ph_score, 1),
            "nutrient_balance_score": round(nutrient_score, 1),
            "organic_matter_score": round(om_score, 1),
            "climatic_resilience_score": round(climate_score, 1)
        },
        "nitrogen_leaching": {
            "leaching_risk_percentage": leaching_pct,
            "estimated_n_leached_kg": leached_n_kg_acre,
            "leaching_risk_level": "High" if leaching_pct > 25 else ("Moderate" if leaching_pct > 12 else "Low")
        },
        "carbon_footprint": {
            "total_ghg_emissions_kg_co2e": total_ghg_emissions_kg_co2e,
            "emissions_per_acre_kg_co2e": round(total_ghg_emissions_kg_co2e / field_area_acres, 1),
            "breakdown": {
                "fertilizer_induced_n2o_co2e": round(fert_ghg_kg_co2e, 1),
                "tillage_machinery_diesel_co2e": round(tillage_ghg_kg_co2e, 1),
                "irrigation_pumping_co2e": round(irrigation_ghg_kg_co2e, 1)
            },
            "carbon_sequestration_potential_kg": carbon_sequestration_potential_kg
        },
        "regenerative_advisory": regenerative_advisory
    }
