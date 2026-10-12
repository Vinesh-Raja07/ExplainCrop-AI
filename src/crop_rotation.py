"""
CropMind AI: Precision Crop Rotation & Companion Planting Synergy Engine
Plans multi-season agro-ecological rotations (Kharif -> Rabi -> Zaid) and identifies symbiotic intercropping companion pairs.
"""

from typing import Dict, Any, List, Optional

# Classification of crops by agronomic botanical family and functional type
CROP_FAMILY_CLASSIFICATION = {
    "Rice": {"family": "Poaceae (Gramineae)", "type": "Cereal / Heavy Feeder", "season": "Kharif", "n_drain": 65.0, "p_drain": 30.0, "k_drain": 25.0},
    "Maize": {"family": "Poaceae (Gramineae)", "type": "Cereal / Heavy Feeder", "season": "Kharif", "n_drain": 70.0, "p_drain": 35.0, "k_drain": 20.0},
    "Jute": {"family": "Malvaceae", "type": "Fiber / Moderate Feeder", "season": "Kharif", "n_drain": 45.0, "p_drain": 20.0, "k_drain": 30.0},
    "Cotton": {"family": "Malvaceae", "type": "Fiber / Heavy Feeder", "season": "Kharif", "n_drain": 80.0, "p_drain": 30.0, "k_drain": 20.0},
    "Chickpea": {"family": "Fabaceae (Legume)", "type": "Pulse / N-Fixer", "season": "Rabi", "n_fixation_kg": 28.0, "root_depth": "Deep Taproot"},
    "Kidneybeans": {"family": "Fabaceae (Legume)", "type": "Pulse / N-Fixer", "season": "Rabi", "n_fixation_kg": 22.0, "root_depth": "Moderate Taproot"},
    "Pigeonpeas": {"family": "Fabaceae (Legume)", "type": "Pulse / N-Fixer", "season": "Kharif/Rabi", "n_fixation_kg": 35.0, "root_depth": "Deep Taproot"},
    "Mothbeans": {"family": "Fabaceae (Legume)", "type": "Drought-Resilient Pulse", "season": "Kharif", "n_fixation_kg": 18.0, "root_depth": "Deep"},
    "Mungbean": {"family": "Fabaceae (Legume)", "type": "Short-Duration Pulse / N-Fixer", "season": "Zaid/Kharif", "n_fixation_kg": 24.0, "root_depth": "Shallow"},
    "Blackgram": {"family": "Fabaceae (Legume)", "type": "Pulse / Soil Restorer", "season": "Kharif/Rabi", "n_fixation_kg": 25.0, "root_depth": "Moderate"},
    "Lentil": {"family": "Fabaceae (Legume)", "type": "Pulse / N-Fixer", "season": "Rabi", "n_fixation_kg": 20.0, "root_depth": "Moderate"},
    "Watermelon": {"family": "Cucurbitaceae", "type": "Cucurbit / Light Feeder", "season": "Zaid", "n_drain": 30.0, "p_drain": 15.0, "k_drain": 40.0},
    "Muskmelon": {"family": "Cucurbitaceae", "type": "Cucurbit / Light Feeder", "season": "Zaid", "n_drain": 28.0, "p_drain": 14.0, "k_drain": 35.0},
    "Banana": {"family": "Musaceae", "type": "Perennial Fruit / Heavy Feeder", "season": "Perennial", "n_drain": 110.0, "p_drain": 45.0, "k_drain": 60.0},
    "Mango": {"family": "Anacardiaceae", "type": "Perennial Orchard", "season": "Perennial", "n_drain": 30.0, "p_drain": 15.0, "k_drain": 30.0},
    "Grapes": {"family": "Vitaceae", "type": "Perennial Vine", "season": "Perennial", "n_drain": 40.0, "p_drain": 25.0, "k_drain": 80.0},
    "Pomegranate": {"family": "Lythraceae", "type": "Perennial Orchard", "season": "Perennial", "n_drain": 35.0, "p_drain": 18.0, "k_drain": 35.0},
    "Apple": {"family": "Rosaceae", "type": "Temperate Perennial", "season": "Perennial", "n_drain": 35.0, "p_drain": 20.0, "k_drain": 70.0},
    "Orange": {"family": "Rutaceae", "type": "Citrus Perennial", "season": "Perennial", "n_drain": 30.0, "p_drain": 15.0, "k_drain": 20.0},
    "Papaya": {"family": "Caricaceae", "type": "Semi-Perennial", "season": "Perennial", "n_drain": 55.0, "p_drain": 30.0, "k_drain": 45.0},
    "Coconut": {"family": "Arecaceae", "type": "Palm Plantation", "season": "Perennial", "n_drain": 40.0, "p_drain": 20.0, "k_drain": 60.0},
    "Coffee": {"family": "Rubiaceae", "type": "Shade Plantation", "season": "Perennial", "n_drain": 60.0, "p_drain": 25.0, "k_drain": 40.0},
}

# Companion Planting and Intercropping Synergy Matrix
COMPANION_SYNERGY_DB = {
    "Rice": [
        {"companion": "Azolla pinnata (Water fern)", "role": "Bio-fertilizer", "benefit": "Fixes 30-40 kg atmospheric N/ha and suppresses 60% weed growth."},
        {"companion": "Fish / Duck Co-culture", "role": "Integrated Agro-Aqua", "benefit": "Controls insect larvae & snails while providing organic excreta fertilization."},
        {"companion": "Blackgram (Relay Cropping)", "role": "Residual Moisture Utilization", "benefit": "Broadcast into standing paddy 5-7 days prior to harvest."}
    ],
    "Maize": [
        {"companion": "Cowpea / Beans (Three Sisters)", "role": "Nitrogen Fixer & Ground Cover", "benefit": "Climbs corn stalks, fixes nitrogen, and smothers weeds."},
        {"companion": "Sunnhemp / Desmodium", "role": "Push-Pull Trap Crop", "benefit": "Repels stem borer moths ('Push') and attracts them to Napier border ('Pull')."},
        {"companion": "Pumpkins / Squash", "role": "Living Mulch", "benefit": "Broad leaves shade soil, conserve moisture, and prevent soil erosion."}
    ],
    "Cotton": [
        {"companion": "Pigeonpeas (Border 4:1 Ratio)", "role": "Trap & Refugia Crop", "benefit": "Attracts Helicoverpa armigera away from cotton bolls."},
        {"companion": "Castor / Marigold", "role": "Nematode & Spodoptera Trap", "benefit": "Traps tobacco caterpillars and suppresses root-knot nematodes."},
        {"companion": "Green Gram / Mungbean", "role": "Intercrop Cash & Soil Cover", "benefit": "Generates 60-day intermediate income and enriches organic nitrogen."}
    ],
    "Banana": [
        {"companion": "Papaya / Pineapple", "role": "Multistorey Intercropping", "benefit": "Utilizes canopy light tiers effectively during first 6 months."},
        {"companion": "Ginger / Turmeric", "role": "Shade-tolerant Cash Crop", "benefit": "Repels soil-borne pests with essential oils and provides high market value."},
        {"companion": "Mucuna / Velvet Bean", "role": "Cover Crop & Weed Suppression", "benefit": "Suppresses stubborn weeds and prevents nematode infestation."}
    ],
    "Coconut": [
        {"companion": "Cocoa / Pepper Vines", "role": "Two-Tier Canopy Integration", "benefit": "Maximizes vertical land productivity and enhances palm microclimate."},
        {"companion": "Banana / Nutmeg", "role": "Intermediate Tier", "benefit": "Improves organic residue return and soil moisture conservation."},
        {"companion": "Pineapple", "role": "Ground Layer", "benefit": "Utilizes interspaces effectively with shallow root systems."}
    ],
    "Chickpea": [
        {"companion": "Mustard / Coriander", "role": "Pest Repellent Intercrop", "benefit": "Mustard glucosinolates deter pod borer adults and enhance pollinator visitations."},
        {"companion": "Barley / Wheat", "role": "Mixed Cropping", "benefit": "Improves grain yield stability and reduces root rot incidence."}
    ]
}

GENERIC_COMPANIONS = [
    {"companion": "Marigold (Tagetes erecta)", "role": "Nematode Deterrent", "benefit": "Roots exude alpha-terthienyl which suppresses harmful root-knot nematodes."},
    {"companion": "Sesbania / Dhaincha", "role": "Green Manure", "benefit": "Adds 15-20 tonnes fresh organic biomass and 40 kg N/acre when incorporated."},
    {"companion": "Coriander / Fennel", "role": "Beneficial Insect Attractant", "benefit": "Attracts predatory wasps, hoverflies, and ladybird beetles."}
]


def generate_crop_rotation_plan(
    primary_crop: str,
    soil_n: float,
    soil_p: float,
    soil_k: float,
    field_area_acres: float = 1.0,
    include_green_manure: bool = True
) -> Dict[str, Any]:
    """
    Formulates a 3-Season (Kharif -> Rabi -> Zaid) rotational cropping plan.
    """
    crop_key = primary_crop.strip().capitalize()
    crop_info = CROP_FAMILY_CLASSIFICATION.get(crop_key, {
        "family": "General Agri", "type": "Standard Crop", "season": "Kharif", "n_drain": 50.0, "p_drain": 20.0, "k_drain": 25.0
    })
    
    # Identify rotation sequencing based on primary crop type
    if "Legume" in crop_info.get("family", "") or "Pulse" in crop_info.get("type", ""):
        # Pulse primary -> Rotate to Heavy Feeder cereal, then Green manure
        s1 = {"season": "Kharif (Monsoon)", "crop": "Maize" if crop_key != "Maize" else "Rice", "role": "High-Yield Cereal (Leverages fixed N)"}
        s2 = {"season": "Rabi (Winter)", "crop": crop_key, "role": "Legume Phase (Biological Nitrogen Fixation)"}
        s3 = {"season": "Zaid (Summer)", "crop": "Mungbean / Watermelon", "role": "Short-duration Cash & Soil Cover"}
        n_restoration = 25.0
    elif crop_info.get("season") == "Perennial":
        # Perennial Orchard / Plantation -> Recommend Multistorey Companion Matrix
        s1 = {"season": "Year-Round Main", "crop": crop_key, "role": "Primary Orchard Canopy"}
        s2 = {"season": "Inter-row Monsoonal", "crop": "Pigeonpea / Sunnhemp", "role": "Biomass & Nitrogen Enricher"}
        s3 = {"season": "Inter-row Winter/Summer", "crop": "Turmeric / Ginger / Legume Cover", "role": "High-Value Ground Layer"}
        n_restoration = 35.0
    else:
        # Heavy feeder (Rice, Maize, Cotton, Jute) -> Followed by Legume pulse -> Followed by Green Manure / Cucurbit
        s1 = {"season": "Kharif (Monsoon)", "crop": crop_key, "role": "Main Commercial Cereal / Fiber Crop"}
        s2 = {"season": "Rabi (Winter)", "crop": "Chickpea" if crop_key != "Chickpea" else "Lentil", "role": "Pulse Restorer (Breaks Cereal Pests & Fixes ~25kg N/acre)"}
        s3 = {"season": "Zaid (Summer)", "crop": "Dhaincha (Green Manure)" if include_green_manure else "Watermelon / Mungbean", "role": "Summer Biomass & Soil Protection"}
        n_restoration = 28.0
        
    companions = COMPANION_SYNERGY_DB.get(crop_key, GENERIC_COMPANIONS)
    
    # Soil nutrient trajectory
    est_n_saved_kg = n_restoration * field_area_acres
    urea_equiv_bags = round((est_n_saved_kg / 0.46) / 50.0, 1)
    
    rotation_schedule = [
        {
            "stage": "Phase 1: " + s1["season"],
            "crop": s1["crop"],
            "role": s1["role"],
            "nutrient_impact": "Nutrient utilization & high biomass output.",
            "disease_break_impact": "Establishes baseline crop with tailored pest scouting."
        },
        {
            "stage": "Phase 2: " + s2["season"],
            "crop": s2["crop"],
            "role": s2["role"],
            "nutrient_impact": f"Fixes ~{n_restoration:.0f} kg biological Nitrogen per acre into soil matrix.",
            "disease_break_impact": "Disrupts mono-cropping fungal and nematode pathogen lifecycles by 75%."
        },
        {
            "stage": "Phase 3: " + s3["season"],
            "crop": s3["crop"],
            "role": s3["role"],
            "nutrient_impact": "Prevents soil crusting and adds 2-3 tonnes organic carbon biomass.",
            "disease_break_impact": "Prevents weed seed banks from establishing during fallow window."
        }
    ]

    return {
        "primary_crop": crop_key,
        "botanical_family": crop_info.get("family", "Unknown"),
        "crop_type": crop_info.get("type", "Standard"),
        "field_area_acres": field_area_acres,
        "rotation_schedule": rotation_schedule,
        "companion_synergies": companions,
        "ecological_benefits": {
            "biological_n_fixed_kg": est_n_saved_kg,
            "urea_bags_saved_50kg": urea_equiv_bags,
            "soil_organic_matter_gain_pct": "+0.25% - 0.40% over 2 rotation cycles",
            "pesticide_reduction_potential_pct": "30% - 45% reduction in synthetic spray cycles",
            "water_use_efficiency_gain": "+22% enhanced infiltration through varied root depth profiles"
        },
        "agronomy_rule_of_thumb": (
            "Never cultivate crops of the same botanical family consecutively in the same field. "
            "Alternate deep-rooted taproot crops (e.g. Chickpea, Cotton) with shallow fibrous-rooted cereals (Rice, Maize) "
            "to extract nutrients from varying soil strata."
        )
    }
