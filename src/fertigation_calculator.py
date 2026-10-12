"""
CropMind AI: Precision Drip Fertigation & Water-Soluble Fertilizer (WSF) Calculator
Calculates growth-stage specific fertigation dosing, 2-tank compatibility separation, and target EC/pH solution management.
"""

from typing import Dict, Any, List, Optional
import math

# WSF Stage-wise Dosing Matrix (kg/acre/week) for major commercial drip-irrigated crops
WSF_DOSING_DATABASE = {
    "Rice": {
        "vegetative": {"map_12_61_0": 4.0, "urea": 8.0, "potassium_nitrate_13_0_45": 3.0, "zinc_edta_g": 250},
        "panicle": {"npk_19_19_19": 6.0, "potassium_nitrate_13_0_45": 6.0, "mkp_0_52_34": 2.0, "zinc_edta_g": 150},
        "grain_filling": {"potassium_nitrate_13_0_45": 8.0, "mkp_0_52_34": 3.0, "zinc_edta_g": 0}
    },
    "Banana": {
        "establishment": {"map_12_61_0": 6.0, "urea": 10.0, "calcium_nitrate": 5.0, "zinc_edta_g": 300},
        "vegetative": {"urea": 15.0, "npk_19_19_19": 10.0, "calcium_nitrate": 8.0, "magnesium_sulfate": 4.0, "zinc_edta_g": 250},
        "shooting_flowering": {"mkp_0_52_34": 8.0, "potassium_nitrate_13_0_45": 14.0, "calcium_nitrate": 6.0, "boron_g": 200},
        "bunch_development": {"potassium_nitrate_13_0_45": 20.0, "mop_soluble": 10.0, "calcium_nitrate": 6.0, "zinc_edta_g": 150}
    },
    "Cotton": {
        "seedling": {"map_12_61_0": 3.0, "urea": 5.0, "zinc_edta_g": 200},
        "square_formation": {"npk_19_19_19": 6.0, "urea": 8.0, "magnesium_sulfate": 3.0, "boron_g": 150},
        "boll_development": {"potassium_nitrate_13_0_45": 10.0, "mkp_0_52_34": 4.0, "calcium_nitrate": 4.0, "zinc_edta_g": 100}
    },
    "Pomegranate": {
        "defoliation_bahar": {"map_12_61_0": 5.0, "ammonium_sulfate": 8.0, "zinc_edta_g": 200},
        "flowering": {"mkp_0_52_34": 6.0, "calcium_nitrate": 6.0, "boron_g": 250},
        "fruit_development": {"potassium_nitrate_13_0_45": 12.0, "magnesium_sulfate": 4.0, "calcium_nitrate": 6.0, "zinc_edta_g": 150}
    },
    "Grapes": {
        "bud_burst": {"map_12_61_0": 4.0, "urea": 6.0, "calcium_nitrate": 4.0, "zinc_edta_g": 200},
        "flowering_berry_set": {"mkp_0_52_34": 5.0, "potassium_nitrate_13_0_45": 8.0, "boron_g": 200},
        "veraison_ripening": {"potassium_nitrate_13_0_45": 14.0, "potassium_sulfate": 8.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 0}
    }
}

DEFAULT_WSF = {
    "vegetative": {"npk_19_19_19": 8.0, "urea": 6.0, "zinc_edta_g": 200},
    "flowering": {"mkp_0_52_34": 6.0, "potassium_nitrate_13_0_45": 8.0, "calcium_nitrate": 4.0, "boron_g": 150},
    "maturation": {"potassium_nitrate_13_0_45": 10.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 100}
}


def calculate_fertigation_schedule(
    crop_name: str,
    growth_stage: str = "vegetative",
    field_area_acres: float = 1.0,
    fertigation_frequency_per_week: int = 2,
    irrigation_volume_litres_cycle: float = 8000.0
) -> Dict[str, Any]:
    """
    Computes precise Water-Soluble Fertilizer (WSF) quantity per fertigation run,
    splits into Tank A and Tank B to avoid chemical precipitation, and estimates target EC.
    """
    crop_key = crop_name.strip().capitalize()
    crop_wsf_profile = WSF_DOSING_DATABASE.get(crop_key, DEFAULT_WSF)
    
    # Select closest matching growth stage
    available_stages = list(crop_wsf_profile.keys())
    matched_stage = growth_stage.lower()
    if matched_stage not in crop_wsf_profile:
        matched_stage = available_stages[0]
        
    weekly_recipe = crop_wsf_profile[matched_stage]
    freq = max(1, fertigation_frequency_per_week)
    
    # Scale recipe by acreage and fertigation cycle frequency
    cycle_dosing = {}
    total_wsf_kg_cycle = 0.0
    
    # Two-Tank System Formulation:
    # Tank A (Fertilizers containing Calcium and Iron): Calcium Nitrate, Iron-EDTA, Urea
    # Tank B (Fertilizers containing Phosphates & Sulfates): MAP, MKP, Magnesium Sulfate, Zinc-EDTA, Boron
    tank_a = []
    tank_b = []
    
    for fert_name, amount in weekly_recipe.items():
        if "_g" in fert_name: # Micronutrients in grams
            per_cycle_g = round((amount * field_area_acres) / freq, 1)
            cycle_dosing[fert_name] = per_cycle_g
            if "zinc" in fert_name or "boron" in fert_name:
                tank_b.append(f"{fert_name.replace('_g', '').replace('_', ' ').title()}: {per_cycle_g} grams")
            else:
                tank_a.append(f"{fert_name.replace('_g', '').replace('_', ' ').title()}: {per_cycle_g} grams")
        else: # Macronutrients in kg
            per_cycle_kg = round((amount * field_area_acres) / freq, 2)
            cycle_dosing[fert_name] = per_cycle_kg
            total_wsf_kg_cycle += per_cycle_kg
            
            if "calcium" in fert_name:
                tank_a.append(f"{fert_name.replace('_', ' ').title()}: {per_cycle_kg} kg")
            elif "urea" in fert_name or "potassium_nitrate" in fert_name:
                tank_a.append(f"{fert_name.replace('_', ' ').title()}: {per_cycle_kg} kg")
            else:
                tank_b.append(f"{fert_name.replace('_', ' ').title()}: {per_cycle_kg} kg")
                
    # Target EC Estimation (ppm concentration -> dS/m)
    # Standard agricultural water baseline EC (~0.4 dS/m) + salt contribution
    concentration_ppm = (total_wsf_kg_cycle * 1_000_000.0) / max(5000.0, irrigation_volume_litres_cycle)
    target_ec_dsm = round(min(3.2, max(0.6, 0.4 + (concentration_ppm / 1500.0))), 2)
    
    return {
        "crop": crop_key,
        "growth_stage": matched_stage.capitalize(),
        "field_area_acres": field_area_acres,
        "fertigation_frequency_per_week": freq,
        "irrigation_volume_litres_cycle": irrigation_volume_litres_cycle,
        "total_wsf_kg_per_cycle": round(total_wsf_kg_cycle, 2),
        "target_solution_ec_dsm": target_ec_dsm,
        "target_solution_ph_range": "5.8 - 6.5",
        "tank_separation": {
            "tank_a_calcium_nitrate_compatible": tank_a if tank_a else ["Urea / Potassium Nitrate solution as needed"],
            "tank_b_phosphate_sulfate_compatible": tank_b if tank_b else ["MAP / MKP / Micronutrient solution as needed"],
            "safety_warning": "NEVER mix Calcium Nitrate (Tank A) with Phosphates or Sulfates (Tank B) in concentrated form as insoluble Calcium Phosphate / Gypsum sludge will clog emitters."
        },
        "cycle_dosing_breakdown": cycle_dosing,
        "injection_protocol": (
            "1. Run clean irrigation water for first 15 minutes to pressurize system.\n"
            "2. Inject Tank A and Tank B solutions concurrently via Venturi / dosing pumps over 30-40 minutes.\n"
            "3. Flush lateral lines with clean water for final 15 minutes to clear residual salts."
        )
    }
