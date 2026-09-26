"""
CropMind AI: Precision Fertilizer Prescription & Nutrient Management Engine
Calculates crop-specific nutrient deficit/excess and prescribes balanced commercial & organic fertigation schedules.
"""

from typing import Dict, Any, List, Optional
import math

# Reference optimal nutrient benchmarks (mg/kg / kg/ha baseline) for all 22 crop classes
CROP_NUTRIENT_BENCHMARKS = {
    "Rice": {"n": 80.0, "p": 48.0, "k": 40.0, "ph_min": 5.5, "ph_max": 7.5, "water_req_mm": 1200},
    "Maize": {"n": 78.0, "p": 48.0, "k": 20.0, "ph_min": 5.8, "ph_max": 7.2, "water_req_mm": 650},
    "Chickpea": {"n": 40.0, "p": 68.0, "k": 80.0, "ph_min": 6.0, "ph_max": 7.8, "water_req_mm": 350},
    "Kidneybeans": {"n": 20.0, "p": 67.0, "k": 20.0, "ph_min": 5.5, "ph_max": 6.8, "water_req_mm": 450},
    "Pigeonpeas": {"n": 20.0, "p": 68.0, "k": 20.0, "ph_min": 6.0, "ph_max": 7.5, "water_req_mm": 400},
    "Mothbeans": {"n": 21.0, "p": 48.0, "k": 20.0, "ph_min": 6.0, "ph_max": 8.0, "water_req_mm": 300},
    "Mungbean": {"n": 21.0, "p": 47.0, "k": 20.0, "ph_min": 6.2, "ph_max": 7.5, "water_req_mm": 350},
    "Blackgram": {"n": 40.0, "p": 67.0, "k": 19.0, "ph_min": 6.0, "ph_max": 7.5, "water_req_mm": 400},
    "Lentil": {"n": 19.0, "p": 68.0, "k": 19.0, "ph_min": 6.0, "ph_max": 7.5, "water_req_mm": 300},
    "Pomegranate": {"n": 19.0, "p": 19.0, "k": 40.0, "ph_min": 6.5, "ph_max": 7.5, "water_req_mm": 700},
    "Banana": {"n": 100.0, "p": 73.0, "k": 50.0, "ph_min": 6.0, "ph_max": 7.5, "water_req_mm": 1800},
    "Mango": {"n": 20.0, "p": 27.0, "k": 30.0, "ph_min": 5.5, "ph_max": 7.5, "water_req_mm": 900},
    "Grapes": {"n": 23.0, "p": 133.0, "k": 200.0, "ph_min": 6.0, "ph_max": 7.5, "water_req_mm": 600},
    "Watermelon": {"n": 99.0, "p": 17.0, "k": 50.0, "ph_min": 6.0, "ph_max": 7.0, "water_req_mm": 500},
    "Muskmelon": {"n": 100.0, "p": 18.0, "k": 50.0, "ph_min": 6.0, "ph_max": 7.2, "water_req_mm": 450},
    "Apple": {"n": 21.0, "p": 134.0, "k": 200.0, "ph_min": 5.5, "ph_max": 6.8, "water_req_mm": 800},
    "Orange": {"n": 20.0, "p": 17.0, "k": 10.0, "ph_min": 6.0, "ph_max": 7.5, "water_req_mm": 900},
    "Papaya": {"n": 50.0, "p": 59.0, "k": 50.0, "ph_min": 6.0, "ph_max": 7.0, "water_req_mm": 1400},
    "Coconut": {"n": 22.0, "p": 17.0, "k": 31.0, "ph_min": 5.2, "ph_max": 8.0, "water_req_mm": 1500},
    "Cotton": {"n": 118.0, "p": 46.0, "k": 19.0, "ph_min": 6.0, "ph_max": 8.0, "water_req_mm": 700},
    "Jute": {"n": 78.0, "p": 47.0, "k": 40.0, "ph_min": 6.0, "ph_max": 7.4, "water_req_mm": 600},
    "Coffee": {"n": 101.0, "p": 29.0, "k": 30.0, "ph_min": 6.0, "ph_max": 6.8, "water_req_mm": 1500},
}

DEFAULT_BENCHMARK = {"n": 50.0, "p": 50.0, "k": 50.0, "ph_min": 6.0, "ph_max": 7.5, "water_req_mm": 600}


def calculate_nutrient_prescription(
    crop_name: str,
    soil_n: float,
    soil_p: float,
    soil_k: float,
    soil_ph: float,
    field_area_acres: float = 1.0
) -> Dict[str, Any]:
    """
    Computes exact elemental nutrient deficit/excess and translates into commercial fertilizer bag quantities.
    """
    crop_key = crop_name.strip().capitalize()
    benchmark = CROP_NUTRIENT_BENCHMARKS.get(crop_key, DEFAULT_BENCHMARK)
    
    # Deficit in mg/kg (ppm) converted to kg/hectare equivalent factor ~2.24, and kg/acre factor ~0.907
    delta_n_ppm = benchmark["n"] - soil_n
    delta_p_ppm = benchmark["p"] - soil_p
    delta_k_ppm = benchmark["k"] - soil_k
    
    # kg per acre requirement (assuming top 15cm soil depth)
    kg_per_acre_n = max(0.0, delta_n_ppm * 0.907)
    kg_per_acre_p = max(0.0, delta_p_ppm * 0.907)
    kg_per_acre_k = max(0.0, delta_k_ppm * 0.907)
    
    total_req_n = kg_per_acre_n * field_area_acres
    total_req_p = kg_per_acre_p * field_area_acres
    total_req_k = kg_per_acre_k * field_area_acres
    
    # Commercial fertilizer calculations:
    # 1. Urea provides 46% N
    # 2. DAP provides 18% N and 46% P2O5
    # 3. MOP provides 60% K2O
    # 4. Single Super Phosphate (SSP) provides 16% P2O5
    dap_bags_50kg = (total_req_p / 0.46) / 50.0 if total_req_p > 0 else 0.0
    n_from_dap = dap_bags_50kg * 50.0 * 0.18
    remaining_n = max(0.0, total_req_n - n_from_dap)
    urea_bags_50kg = (remaining_n / 0.46) / 50.0 if remaining_n > 0 else 0.0
    mop_bags_50kg = (total_req_k / 0.60) / 50.0 if total_req_k > 0 else 0.0
    
    # Soil pH rectification advice
    ph_status = "Optimal"
    ph_remedy = "Soil pH is within the optimal physiological uptake range."
    if soil_ph < benchmark["ph_min"]:
        ph_status = "Acidic"
        lime_kg_per_acre = (benchmark["ph_min"] - soil_ph) * 350.0
        ph_remedy = f"Soil is acidic. Apply Agricultural Lime (CaCO3) @ {lime_kg_per_acre:.0f} kg/acre 3 weeks before sowing."
    elif soil_ph > benchmark["ph_max"]:
        ph_status = "Alkaline / Saline"
        gypsum_kg_per_acre = (soil_ph - benchmark["ph_max"]) * 400.0
        ph_remedy = f"Soil is alkaline. Apply Agricultural Gypsum (CaSO4) @ {gypsum_kg_per_acre:.0f} kg/acre with adequate leaching irrigation."

    # Stage split application
    split_schedule = [
        {
            "stage": "Basal Application (At Sowing / Transplanting)",
            "urea_pct": 30,
            "dap_pct": 100,
            "mop_pct": 50,
            "notes": "Incorporate thoroughly into seedbed at 5-10 cm depth."
        },
        {
            "stage": "Vegetative Growth (20-30 Days After Sowing)",
            "urea_pct": 40,
            "dap_pct": 0,
            "mop_pct": 0,
            "notes": "Top-dress alongside crop rows during soil moist condition."
        },
        {
            "stage": "Panicle / Flowering / Tuber Formation (45-60 Days)",
            "urea_pct": 30,
            "dap_pct": 0,
            "mop_pct": 50,
            "notes": "Foliar or band placement before final irrigation cycle."
        }
    ]

    # Organic bio-fertilizer alternatives
    organic_alternatives = [
        {"name": "Well-decomposed Farmyard Manure (FYM)", "dosage": f"{field_area_acres * 3.0:.1f} Tonnes", "purpose": "Improves organic carbon and soil water holding capacity."},
        {"name": "Vermicompost", "dosage": f"{field_area_acres * 1.5:.1f} Tonnes", "purpose": "Supplies micro-nutrients and enhances microbial biomass."},
        {"name": "Azotobacter / Rhizobium Bio-inoculant", "dosage": f"{field_area_acres * 2.0:.1f} kg", "purpose": "Fixes atmospheric Nitrogen non-symbiotically/symbiotically."},
        {"name": "Phosphate Solubilizing Bacteria (PSB)", "dosage": f"{field_area_acres * 2.0:.1f} kg", "purpose": "Mobilizes fixed legacy phosphorus in soil matrix."}
    ]

    return {
        "crop": crop_key,
        "field_area_acres": field_area_acres,
        "soil_profile": {"n": soil_n, "p": soil_p, "k": soil_k, "ph": soil_ph},
        "target_benchmark": benchmark,
        "deficit_status": {
            "nitrogen_status": "Deficient" if delta_n_ppm > 5 else ("Excess" if delta_n_ppm < -15 else "Adequate"),
            "phosphorus_status": "Deficient" if delta_p_ppm > 5 else ("Excess" if delta_p_ppm < -15 else "Adequate"),
            "potassium_status": "Deficient" if delta_k_ppm > 5 else ("Excess" if delta_k_ppm < -15 else "Adequate"),
            "n_deficit_kg_acre": round(kg_per_acre_n, 2),
            "p_deficit_kg_acre": round(kg_per_acre_p, 2),
            "k_deficit_kg_acre": round(kg_per_acre_k, 2),
        },
        "commercial_prescription": {
            "urea_50kg_bags": math.ceil(urea_bags_50kg * 10) / 10,
            "dap_50kg_bags": math.ceil(dap_bags_50kg * 10) / 10,
            "mop_50kg_bags": math.ceil(mop_bags_50kg * 10) / 10,
            "total_fertilizer_cost_inr": round(
                (math.ceil(urea_bags_50kg * 10) / 10 * 270) +
                (math.ceil(dap_bags_50kg * 10) / 10 * 1350) +
                (math.ceil(mop_bags_50kg * 10) / 10 * 1700),
                2
            )
        },
        "ph_remediation": {
            "status": ph_status,
            "remedy": ph_remedy
        },
        "split_schedule": split_schedule,
        "organic_alternatives": organic_alternatives
    }
