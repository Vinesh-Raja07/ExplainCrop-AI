"""
CropMind AI: Precision Drip Fertigation & Water-Soluble Fertilizer (WSF) Calculator
Calculates growth-stage specific fertigation dosing, 2-tank compatibility separation, and target EC/pH solution management.
"""

from typing import Dict, Any, List, Optional
import math

FERTILIZER_METADATA = {
    "npk_19_19_19": {
        "name": "NPK 19:19:19 (Water Soluble)",
        "grade": "19-19-19",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Balanced vegetative growth, shoot & root development",
        "is_micronutrient": False
    },
    "map_12_61_0": {
        "name": "Monoammonium Phosphate (MAP 12-61-0)",
        "grade": "12-61-0",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Root establishment, energy transfer & high phosphorus priming",
        "is_micronutrient": False
    },
    "mkp_0_52_34": {
        "name": "Monopotassium Phosphate (MKP 0-52-34)",
        "grade": "0-52-34",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Flower bud differentiation, bloom vigor & fruit setting",
        "is_micronutrient": False
    },
    "potassium_nitrate_13_0_45": {
        "name": "Potassium Nitrate (13-0-45)",
        "grade": "13-0-45",
        "tank": "Tank A (Nitrate / Calcium)",
        "role": "Fruit sizing, sugar accumulation, drought tolerance & grain filling",
        "is_micronutrient": False
    },
    "calcium_nitrate": {
        "name": "Calcium Nitrate",
        "grade": "15.5-0-0 + 18.8% Ca",
        "tank": "Tank A (Calcium / Nitrate)",
        "role": "Cell wall structural strength, root tip development & blossom end rot prevention",
        "is_micronutrient": False
    },
    "urea": {
        "name": "Technical Grade Urea",
        "grade": "46-0-0",
        "tank": "Tank A (Nitrate / Calcium)",
        "role": "Rapid nitrogen assimilation for vegetative canopy flush",
        "is_micronutrient": False
    },
    "ammonium_sulfate": {
        "name": "Ammonium Sulfate",
        "grade": "21-0-0 + 24% S",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Acidifying nitrogen and sulfur supply for chlorophyll synthesis",
        "is_micronutrient": False
    },
    "magnesium_sulfate": {
        "name": "Magnesium Sulfate (Epsom Salt)",
        "grade": "9.6% Mg + 12% S",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Central chlorophyll molecule activator & sulfur enzyme cofactor",
        "is_micronutrient": False
    },
    "potassium_sulfate": {
        "name": "Sulfate of Potash (SOP 0-0-50)",
        "grade": "0-0-50 + 17.5% S",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Chloride-free potassium and sulfur for high fruit quality & rind color",
        "is_micronutrient": False
    },
    "mop_soluble": {
        "name": "White Muriate of Potash (Soluble MOP)",
        "grade": "0-0-60",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Potassium osmoprotectant for bunch development & yield bulk",
        "is_micronutrient": False
    },
    "npk_13_40_13": {
        "name": "NPK 13:40:13 (Starter Booster)",
        "grade": "13-40-13",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Phosphorus booster for early root establishment and tillering",
        "is_micronutrient": False
    },
    "npk_12_32_16": {
        "name": "NPK 12:32:16",
        "grade": "12-32-16",
        "tank": "Tank B (Phosphate / Sulfate)",
        "role": "Pre-flowering bloom boost and energy transfer",
        "is_micronutrient": False
    },
    "zinc_edta_g": {
        "name": "Chelated Zinc (Zn-EDTA 12%)",
        "grade": "12% Zn Chelated",
        "tank": "Tank B (Micronutrients)",
        "role": "Auxin (IAA) hormone synthesis and internode expansion",
        "is_micronutrient": True
    },
    "boron_g": {
        "name": "Solubor (Disodium Octaborate 20% B)",
        "grade": "20% B",
        "tank": "Tank B (Micronutrients)",
        "role": "Pollen tube elongation, flower viability & fruit set integrity",
        "is_micronutrient": True
    },
    "iron_edta_g": {
        "name": "Chelated Iron (Fe-EDTA 12%)",
        "grade": "12% Fe Chelated",
        "tank": "Tank A (Calcium / Nitrate / Iron)",
        "role": "Electron transport, chlorophyll synthesis & prevention of chlorosis",
        "is_micronutrient": True
    }
}

# WSF Stage-wise Dosing Matrix (kg/acre/week, micronutrients in g/acre/week)
WSF_DOSING_DATABASE: Dict[str, Dict[str, Dict[str, float]]] = {
    "Rice": {
        "initial": {"map_12_61_0": 3.5, "urea": 5.0, "zinc_edta_g": 200},
        "vegetative": {"map_12_61_0": 4.0, "urea": 8.0, "potassium_nitrate_13_0_45": 3.0, "zinc_edta_g": 250},
        "flowering": {"npk_19_19_19": 6.0, "potassium_nitrate_13_0_45": 6.0, "mkp_0_52_34": 2.0, "zinc_edta_g": 150},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 8.0, "mkp_0_52_34": 3.0, "zinc_edta_g": 0}
    },
    "Banana": {
        "initial": {"map_12_61_0": 6.0, "urea": 10.0, "calcium_nitrate": 5.0, "zinc_edta_g": 300},
        "vegetative": {"urea": 15.0, "npk_19_19_19": 10.0, "calcium_nitrate": 8.0, "magnesium_sulfate": 4.0, "zinc_edta_g": 250},
        "flowering": {"mkp_0_52_34": 8.0, "potassium_nitrate_13_0_45": 14.0, "calcium_nitrate": 6.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 20.0, "mop_soluble": 10.0, "calcium_nitrate": 6.0, "zinc_edta_g": 150}
    },
    "Cotton": {
        "initial": {"map_12_61_0": 3.0, "urea": 5.0, "zinc_edta_g": 200},
        "vegetative": {"npk_19_19_19": 6.0, "urea": 8.0, "magnesium_sulfate": 3.0, "boron_g": 150},
        "flowering": {"mkp_0_52_34": 5.0, "potassium_nitrate_13_0_45": 8.0, "calcium_nitrate": 4.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 10.0, "mkp_0_52_34": 4.0, "calcium_nitrate": 4.0, "zinc_edta_g": 100}
    },
    "Pomegranate": {
        "initial": {"map_12_61_0": 5.0, "ammonium_sulfate": 8.0, "zinc_edta_g": 200},
        "vegetative": {"npk_19_19_19": 8.0, "urea": 6.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 200},
        "flowering": {"mkp_0_52_34": 6.0, "calcium_nitrate": 6.0, "boron_g": 250},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 12.0, "magnesium_sulfate": 4.0, "calcium_nitrate": 6.0, "zinc_edta_g": 150}
    },
    "Grapes": {
        "initial": {"map_12_61_0": 4.0, "urea": 6.0, "calcium_nitrate": 4.0, "zinc_edta_g": 200},
        "vegetative": {"npk_19_19_19": 8.0, "urea": 6.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 150},
        "flowering": {"mkp_0_52_34": 5.0, "potassium_nitrate_13_0_45": 8.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 14.0, "potassium_sulfate": 8.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 0}
    },
    "Apple": {
        "initial": {"map_12_61_0": 4.0, "urea": 5.0, "zinc_edta_g": 200},
        "vegetative": {"npk_19_19_19": 7.0, "calcium_nitrate": 5.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 150},
        "flowering": {"mkp_0_52_34": 5.0, "calcium_nitrate": 6.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 12.0, "potassium_sulfate": 6.0, "calcium_nitrate": 4.0}
    },
    "Maize": {
        "initial": {"map_12_61_0": 4.0, "urea": 6.0, "zinc_edta_g": 250},
        "vegetative": {"npk_19_19_19": 8.0, "urea": 10.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 200},
        "flowering": {"mkp_0_52_34": 5.0, "potassium_nitrate_13_0_45": 8.0, "urea": 4.0, "boron_g": 150},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 10.0, "mkp_0_52_34": 3.0}
    },
    "Mango": {
        "initial": {"map_12_61_0": 4.0, "urea": 6.0, "zinc_edta_g": 200},
        "vegetative": {"npk_19_19_19": 8.0, "urea": 6.0, "magnesium_sulfate": 3.0},
        "flowering": {"mkp_0_52_34": 6.0, "calcium_nitrate": 5.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 12.0, "potassium_sulfate": 6.0, "magnesium_sulfate": 3.0}
    },
    "Orange": {
        "initial": {"map_12_61_0": 4.0, "urea": 6.0, "zinc_edta_g": 250},
        "vegetative": {"npk_19_19_19": 8.0, "calcium_nitrate": 5.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 200},
        "flowering": {"mkp_0_52_34": 6.0, "calcium_nitrate": 6.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 14.0, "potassium_sulfate": 5.0, "magnesium_sulfate": 3.0}
    },
    "Papaya": {
        "initial": {"map_12_61_0": 5.0, "urea": 6.0, "zinc_edta_g": 200},
        "vegetative": {"npk_19_19_19": 9.0, "calcium_nitrate": 6.0, "magnesium_sulfate": 4.0, "zinc_edta_g": 200},
        "flowering": {"mkp_0_52_34": 7.0, "potassium_nitrate_13_0_45": 10.0, "calcium_nitrate": 5.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 16.0, "mkp_0_52_34": 5.0, "calcium_nitrate": 5.0}
    },
    "Watermelon": {
        "initial": {"map_12_61_0": 4.0, "urea": 5.0, "zinc_edta_g": 150},
        "vegetative": {"npk_19_19_19": 8.0, "calcium_nitrate": 6.0, "magnesium_sulfate": 3.0},
        "flowering": {"mkp_0_52_34": 6.0, "potassium_nitrate_13_0_45": 8.0, "calcium_nitrate": 5.0, "boron_g": 150},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 14.0, "potassium_sulfate": 6.0, "magnesium_sulfate": 3.0}
    },
    "Muskmelon": {
        "initial": {"map_12_61_0": 4.0, "urea": 5.0, "zinc_edta_g": 150},
        "vegetative": {"npk_19_19_19": 7.5, "calcium_nitrate": 5.5, "magnesium_sulfate": 3.0},
        "flowering": {"mkp_0_52_34": 5.5, "potassium_nitrate_13_0_45": 7.5, "calcium_nitrate": 5.0, "boron_g": 150},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 12.5, "potassium_sulfate": 5.5, "magnesium_sulfate": 3.0}
    },
    "Coconut": {
        "initial": {"map_12_61_0": 4.0, "urea": 6.0, "magnesium_sulfate": 3.0},
        "vegetative": {"npk_19_19_19": 8.0, "urea": 8.0, "magnesium_sulfate": 4.0, "zinc_edta_g": 200},
        "flowering": {"potassium_nitrate_13_0_45": 10.0, "mkp_0_52_34": 5.0, "boron_g": 200},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 14.0, "mop_soluble": 8.0, "magnesium_sulfate": 4.0}
    },
    "Coffee": {
        "initial": {"map_12_61_0": 3.5, "urea": 5.0, "zinc_edta_g": 150},
        "vegetative": {"npk_19_19_19": 7.0, "urea": 6.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 200},
        "flowering": {"mkp_0_52_34": 5.0, "potassium_nitrate_13_0_45": 7.0, "calcium_nitrate": 4.0, "boron_g": 150},
        "fruit_maturity": {"potassium_nitrate_13_0_45": 10.0, "potassium_sulfate": 5.0, "magnesium_sulfate": 3.0}
    }
}

# Generic Pulse template for Blackgram, Chickpea, Kidneybeans, Lentil, Mothbeans, Mungbean, Pigeonpeas
GENERIC_PULSE_WSF = {
    "initial": {"map_12_61_0": 3.0, "urea": 2.5, "zinc_edta_g": 150},
    "vegetative": {"npk_19_19_19": 5.0, "magnesium_sulfate": 2.5, "zinc_edta_g": 150},
    "flowering": {"mkp_0_52_34": 4.0, "potassium_nitrate_13_0_45": 4.0, "calcium_nitrate": 3.0, "boron_g": 120},
    "fruit_maturity": {"potassium_nitrate_13_0_45": 6.0, "mkp_0_52_34": 3.0}
}

# Populate pulse crops
for pulse in ["Blackgram", "Chickpea", "Kidneybeans", "Lentil", "Mothbeans", "Mungbean", "Pigeonpeas", "Jute"]:
    if pulse not in WSF_DOSING_DATABASE:
        WSF_DOSING_DATABASE[pulse] = GENERIC_PULSE_WSF

DEFAULT_WSF = {
    "initial": {"map_12_61_0": 4.0, "urea": 5.0, "zinc_edta_g": 200},
    "vegetative": {"npk_19_19_19": 8.0, "urea": 6.0, "zinc_edta_g": 200},
    "flowering": {"mkp_0_52_34": 6.0, "potassium_nitrate_13_0_45": 8.0, "calcium_nitrate": 4.0, "boron_g": 150},
    "fruit_maturity": {"potassium_nitrate_13_0_45": 10.0, "magnesium_sulfate": 3.0, "zinc_edta_g": 100}
}

# Stage alias dictionary for matching standard and specific agronomic terms
STAGE_ALIASES = {
    "initial": ["initial", "seedling", "establishment", "bud_burst", "defoliation_bahar", "dormancy", "starter", "transplanting"],
    "vegetative": ["vegetative", "tillering", "square_formation", "canopy", "growth", "foliage", "branching"],
    "flowering": ["flowering", "panicle", "shooting_flowering", "flowering_berry_set", "bloom", "blossom", "tasseling"],
    "fruit_maturity": ["fruit_maturity", "grain_filling", "bunch_development", "boll_development", "fruit_development", "veraison_ripening", "maturation", "harvest", "ripening", "nut_development"]
}


def normalize_growth_stage(raw_stage: str, crop_profile: Dict[str, Any]) -> str:
    """Matches any stage string (including domain aliases) to an available key in crop_profile."""
    s = raw_stage.strip().lower().replace(" ", "_")
    
    # 1. Exact match in profile
    if s in crop_profile:
        return s
        
    # 2. Check alias group
    for std_stage, aliases in STAGE_ALIASES.items():
        if s in aliases or any(alias in s for alias in aliases):
            if std_stage in crop_profile:
                return std_stage
            # Check if any alias exists as key in crop_profile
            for alias in aliases:
                if alias in crop_profile:
                    return alias
                    
    # 3. Fallback to first available stage
    return list(crop_profile.keys())[0]


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
    
    matched_stage_key = normalize_growth_stage(growth_stage, crop_wsf_profile)
    weekly_recipe = crop_wsf_profile[matched_stage_key]
    
    freq = max(1, int(fertigation_frequency_per_week))
    acres = max(0.1, float(field_area_acres))
    cycle_water = max(500.0, float(irrigation_volume_litres_cycle))
    
    # Scale recipe by acreage and fertigation cycle frequency
    cycle_dosing = {}
    tank_a = []
    tank_b = []
    recommended_wsf_sources = []
    
    total_wsf_kg_week = 0.0
    total_wsf_kg_acre_week = 0.0
    
    for fert_key, amount in weekly_recipe.items():
        meta = FERTILIZER_METADATA.get(fert_key, {
            "name": fert_key.replace("_", " ").title(),
            "grade": "Standard Grade",
            "tank": "Tank B (Phosphate / Sulfate)" if "map" in fert_key or "mkp" in fert_key or "sulfate" in fert_key else "Tank A (Nitrate / Calcium)",
            "role": "Nutritional supplement",
            "is_micronutrient": "_g" in fert_key
        })
        
        is_micro = meta["is_micronutrient"]
        
        if is_micro:
            per_cycle_g = round((amount * acres) / freq, 1)
            kg_acre_wk = round(amount / 1000.0, 3)
            cycle_dosing[fert_key] = per_cycle_g
            
            item_text = f"{meta['name']}: {per_cycle_g} g/cycle ({round(amount * acres, 1)} g/wk total)"
            if "Tank A" in meta["tank"]:
                tank_a.append(item_text)
            else:
                tank_b.append(item_text)
                
            recommended_wsf_sources.append({
                "Fertilizer Name": meta["name"],
                "Grade": meta["grade"],
                "kg_per_acre_week": kg_acre_wk,
                "Weekly Dose (g/acre)": f"{amount} g",
                "Tank Compatibility": meta["tank"],
                "Agronomic Purpose": meta["role"]
            })
        else:
            per_cycle_kg = round((amount * acres) / freq, 2)
            total_wsf_kg_week += amount * acres
            total_wsf_kg_acre_week += amount
            cycle_dosing[fert_key] = per_cycle_kg
            
            item_text = f"{meta['name']}: {per_cycle_kg} kg/cycle ({round(amount * acres, 2)} kg/wk total)"
            if "Tank A" in meta["tank"]:
                tank_a.append(item_text)
            else:
                tank_b.append(item_text)
                
            recommended_wsf_sources.append({
                "Fertilizer Name": meta["name"],
                "Grade": meta["grade"],
                "kg_per_acre_week": round(amount, 2),
                "Weekly Dose (kg/acre)": f"{amount} kg",
                "Tank Compatibility": meta["tank"],
                "Agronomic Purpose": meta["role"]
            })
            
    total_wsf_kg_cycle = round(total_wsf_kg_week / freq, 2)
    
    # Target EC Estimation (ppm concentration -> dS/m)
    # Baseline irrigation water EC (~0.4 dS/m) + dissolved salt contribution
    concentration_ppm = (total_wsf_kg_cycle * 1_000_000.0) / cycle_water
    target_ec_dsm = round(min(3.5, max(0.6, 0.4 + (concentration_ppm / 1500.0))), 2)
    
    # Determine Salinity Status
    if target_ec_dsm <= 1.8:
        salinity_risk = "Safe / Optimal (EC < 1.8 dS/m)"
    elif target_ec_dsm <= 2.5:
        salinity_risk = "Moderate Salinity - Monitor Drainage EC"
    else:
        salinity_risk = "High Salinity Risk (> 2.5 dS/m) - Increase Irrigation Volume"
        
    tank_a_items = tank_a if tank_a else ["Urea / Potassium Nitrate solution as needed"]
    tank_b_items = tank_b if tank_b else ["MAP / MKP / Micronutrient solution as needed"]
    
    return {
        "crop": crop_key,
        "growth_stage": matched_stage_key.replace("_", " ").title(),
        "field_area_acres": acres,
        "fertigation_frequency_per_week": freq,
        "irrigation_volume_litres_cycle": cycle_water,
        "total_wsf_kg_per_cycle": total_wsf_kg_cycle,
        "target_solution_ec_dsm": target_ec_dsm,
        "target_solution_ph_range": "5.8 - 6.5",
        "tank_separation": {
            "tank_a_calcium_nitrate_compatible": tank_a_items,
            "tank_b_phosphate_sulfate_compatible": tank_b_items,
            "safety_warning": "NEVER mix Calcium Nitrate (Tank A) with Phosphates or Sulfates (Tank B) in concentrated stock form. Insoluble Calcium Phosphate / Gypsum sludge will precipitate and clog dripper emitters."
        },
        "cycle_dosing_breakdown": cycle_dosing,
        "injection_protocol": (
            "1. Clean Water Prep: Run clean irrigation water for first 15 minutes to pressurize laterals.\n"
            "2. Dosing Injection: Inject Tank A and Tank B solutions concurrently via separate Venturi injectors over 30-45 minutes.\n"
            "3. System Flush: Flush lateral lines with clean water for final 15 minutes to clear residual nutrient salts."
        ),
        "weekly_prescription": {
            "total_wsf_kg_week": round(total_wsf_kg_week, 2),
            "total_wsf_kg_acre_week": round(total_wsf_kg_acre_week, 2),
            "recommended_wsf_sources": recommended_wsf_sources
        },
        "safety_parameters": {
            "estimated_ec_ds_m": target_ec_dsm,
            "salinity_risk_status": salinity_risk,
            "recommended_ph_range": "5.8 - 6.5"
        },
        "cycle_dosing": {
            "wsf_kg_per_cycle": total_wsf_kg_cycle,
            "applications_per_week": freq,
            "cycle_water_litres": cycle_water
        },
        "operational_notes": (
            "Inject fertilizer in the middle third of the irrigation cycle. "
            "Maintain system operating pressure at 1.2 - 2.0 bar and monitor dripper discharge uniformity. "
            "Always dissolve fertilizers in water (never pour water onto dry powder)."
        )
    }
