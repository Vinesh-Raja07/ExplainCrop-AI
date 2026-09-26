"""
CropMind AI: Crop Disease & Pest Risk Forecasting Matrix
Predicts fungal, bacterial, and insect pest vulnerability based on microclimate indices and crop phenology.
"""

from typing import Dict, Any, List, Optional

# Crop-specific disease and pest susceptibility database
CROP_PATHOGEN_DB = {
    "Rice": [
        {
            "name": "Rice Blast (Magnaporthe oryzae)",
            "type": "Fungal",
            "opt_temp_min": 24.0, "opt_temp_max": 28.0,
            "opt_hum_min": 85.0,
            "min_rain_mm": 50.0,
            "symptoms": "Spindle-shaped lesions with greyish center and brownish margins on leaf blades.",
            "bio_control": "Spray Pseudomonas fluorescens @ 10 g/L or Trichoderma viride.",
            "chemical_cure": "Tricyclazole 75% WP @ 0.6 g/L or Azoxystrobin 23% SC @ 1 mL/L."
        },
        {
            "name": "Bacterial Leaf Blight (Xanthomonas oryzae)",
            "type": "Bacterial",
            "opt_temp_min": 25.0, "opt_temp_max": 34.0,
            "opt_hum_min": 80.0,
            "min_rain_mm": 80.0,
            "symptoms": "Water-soaked streaks on leaf margins turning yellow to straw-white.",
            "bio_control": "Apply fresh cow-dung extract 20% or Streptomyces bio-fungicide.",
            "chemical_cure": "Streptocycline 100 ppm (0.1 g/L) + Copper Oxychloride @ 2.5 g/L."
        },
        {
            "name": "Brown Plant Hopper (Nilaparvata lugens)",
            "type": "Pest",
            "opt_temp_min": 28.0, "opt_temp_max": 35.0,
            "opt_hum_min": 75.0,
            "min_rain_mm": 20.0,
            "symptoms": "Hopper burn, drying of tillers in circular patches across field.",
            "bio_control": "Neem oil 10,000 ppm @ 3 mL/L or Metarhizium anisopliae @ 5 g/L.",
            "chemical_cure": "Triflumezopyrim 10% SC @ 0.5 mL/L or Pymetrozine 50% WDG @ 0.6 g/L."
        }
    ],
    "Maize": [
        {
            "name": "Fall Armyworm (Spodoptera frugiperda)",
            "type": "Pest",
            "opt_temp_min": 25.0, "opt_temp_max": 35.0,
            "opt_hum_min": 60.0,
            "min_rain_mm": 10.0,
            "symptoms": "Pinholes and large irregular ragged leaf feeding with heavy frass in whorls.",
            "bio_control": "Release Trichogramma pretiosum @ 50,000/acre or Beauveria bassiana.",
            "chemical_cure": "Chlorantraniliprole 18.5% SC @ 0.4 mL/L or Emamectin benzoate 5% SG @ 0.4 g/L."
        },
        {
            "name": "Turcicum Leaf Blight (Exserohilum turcicum)",
            "type": "Fungal",
            "opt_temp_min": 20.0, "opt_temp_max": 28.0,
            "opt_hum_min": 80.0,
            "min_rain_mm": 40.0,
            "symptoms": "Long, elliptical grayish-green or tan lesions on leaves.",
            "bio_control": "Trichoderma harzianum soil and foliar application.",
            "chemical_cure": "Mancozeb 75% WP @ 2.5 g/L or Azoxystrobin + Difenoconazole @ 1 mL/L."
        }
    ],
    "Cotton": [
        {
            "name": "Pink Bollworm (Pectinophora gossypiella)",
            "type": "Pest",
            "opt_temp_min": 25.0, "opt_temp_max": 33.0,
            "opt_hum_min": 65.0,
            "min_rain_mm": 15.0,
            "symptoms": "Rosetted flowers, exit holes on bolls with stained lint and damaged seeds.",
            "bio_control": "Install Pheromone traps @ 5/acre and release Trichogrammatoidea bactrae.",
            "chemical_cure": "Profenophos 50% EC @ 2 mL/L or Spinetoram 11.7% SC @ 0.8 mL/L."
        },
        {
            "name": "Cotton Leaf Curl Virus (CLCuV)",
            "type": "Viral/Vector",
            "opt_temp_min": 28.0, "opt_temp_max": 38.0,
            "opt_hum_min": 50.0,
            "min_rain_mm": 0.0,
            "symptoms": "Upward/downward leaf curling, thickening of veins and enations on undersurface.",
            "bio_control": "Control whitefly vector using Yellow Sticky Traps (10/acre) + Neem spray.",
            "chemical_cure": "Diafenthiuron 50% WP @ 1.2 g/L or Afidopyropen 50 g/L DC @ 2 mL/L."
        }
    ],
    "Banana": [
        {
            "name": "Sigatoka Leaf Spot (Mycosphaerella musicola)",
            "type": "Fungal",
            "opt_temp_min": 23.0, "opt_temp_max": 30.0,
            "opt_hum_min": 85.0,
            "min_rain_mm": 70.0,
            "symptoms": "Yellowish-green streaks parallel to leaf veins turning dark brown with grey center.",
            "bio_control": "Pseudomonas fluorescens foliar spray @ 5 g/L with mineral oil 1%.",
            "chemical_cure": "Propiconazole 25% EC @ 1 mL/L or Carbendazim 50% WP @ 1 g/L."
        },
        {
            "name": "Panama Wilt (Fusarium oxysporum f. sp. cubense)",
            "type": "Fungal",
            "opt_temp_min": 24.0, "opt_temp_max": 32.0,
            "opt_hum_min": 70.0,
            "min_rain_mm": 30.0,
            "symptoms": "Yellowing of lower leaves, longitudinal pseudostem splitting and vascular discoloration.",
            "bio_control": "Trichoderma viride enriched FYM @ 5 kg/plant at planting.",
            "chemical_cure": "Soil drenching with Carbendazim 0.2% around root zone."
        }
    ]
}

# Generic pathogen fallback for crops not explicitly mapped
GENERIC_PATHOGENS = [
    {
        "name": "Powdery Mildew (Erysiphales spp.)",
        "type": "Fungal",
        "opt_temp_min": 20.0, "opt_temp_max": 28.0,
        "opt_hum_min": 65.0,
        "min_rain_mm": 10.0,
        "symptoms": "White powdery fungal growth on upper surfaces of leaves and young shoots.",
        "bio_control": "Foliar spray with Bacillus subtilis or baking soda (5 g/L) + soap.",
        "chemical_cure": "Wettable Sulfur 80% WP @ 3 g/L or Hexaconazole 5% EC @ 1 mL/L."
    },
    {
        "name": "Anthracnose & Fruit Rot (Colletotrichum spp.)",
        "type": "Fungal",
        "opt_temp_min": 24.0, "opt_temp_max": 32.0,
        "opt_hum_min": 80.0,
        "min_rain_mm": 60.0,
        "symptoms": "Sunken dark necrotic spots on leaves, stems, and fruits.",
        "bio_control": "Trichoderma viride seed and foliar spray.",
        "chemical_cure": "Azoxystrobin + Difenoconazole @ 1 mL/L or Copper Hydroxide @ 2 g/L."
    },
    {
        "name": "Sucking Pests (Aphids / Thrips / Whiteflies)",
        "type": "Pest",
        "opt_temp_min": 22.0, "opt_temp_max": 34.0,
        "opt_hum_min": 50.0,
        "min_rain_mm": 0.0,
        "symptoms": "Yellowing, curling, stunting, and sticky honeydew secretion with sooty mold.",
        "bio_control": "Neem seed kernel extract (NSKE 5%) or Verticillium lecanii @ 5 g/L.",
        "chemical_cure": "Acetamiprid 20% SP @ 0.3 g/L or Thiamethoxam 25% WG @ 0.3 g/L."
    }
]


def calculate_disease_pest_risk(
    crop_name: str,
    temperature_c: float,
    humidity_pct: float,
    rainfall_14d_mm: float
) -> Dict[str, Any]:
    """
    Evaluates temperature-humidity suitability score for high-impact crop pathogens.
    """
    crop_key = crop_name.strip().capitalize()
    pathogens = CROP_PATHOGEN_DB.get(crop_key, GENERIC_PATHOGENS)
    
    evaluated_risks = []
    max_risk_score = 0.0
    
    for p in pathogens:
        # Temperature score (0 to 100%)
        if p["opt_temp_min"] <= temperature_c <= p["opt_temp_max"]:
            temp_score = 100.0
        elif abs(temperature_c - p["opt_temp_min"]) <= 4.0 or abs(temperature_c - p["opt_temp_max"]) <= 4.0:
            temp_score = 65.0
        else:
            temp_score = 25.0
            
        # Humidity score (0 to 100%)
        if humidity_pct >= p["opt_hum_min"]:
            hum_score = 100.0
        elif humidity_pct >= p["opt_hum_min"] - 15.0:
            hum_score = 60.0
        else:
            hum_score = 20.0
            
        # Rainfall score
        if rainfall_14d_mm >= p["min_rain_mm"]:
            rain_score = 100.0
        elif rainfall_14d_mm >= p["min_rain_mm"] * 0.5:
            rain_score = 55.0
        else:
            rain_score = 25.0
            
        # Composite Disease Severity Index (DSI)
        dsi = (temp_score * 0.40) + (hum_score * 0.40) + (rain_score * 0.20)
        dsi = round(dsi, 1)
        
        if dsi >= 80:
            risk_level = "Severe"
            badge_color = "#E11D48"
        elif dsi >= 60:
            risk_level = "High"
            badge_color = "#F59E0B"
        elif dsi >= 40:
            risk_level = "Moderate"
            badge_color = "#3B82F6"
        else:
            risk_level = "Low"
            badge_color = "#10B981"
            
        if dsi > max_risk_score:
            max_risk_score = dsi
            
        evaluated_risks.append({
            "pathogen_name": p["name"],
            "type": p["type"],
            "dsi_score": dsi,
            "risk_level": risk_level,
            "badge_color": badge_color,
            "symptoms": p["symptoms"],
            "bio_control": p["bio_control"],
            "chemical_cure": p["chemical_cure"]
        })
        
    overall_status = "Severe Risk" if max_risk_score >= 80 else ("High Risk" if max_risk_score >= 60 else ("Moderate Risk" if max_risk_score >= 40 else "Low Risk"))
    
    return {
        "crop": crop_key,
        "meteorological_conditions": {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rainfall_14d_mm": rainfall_14d_mm
        },
        "overall_risk_status": overall_status,
        "max_risk_score": max_risk_score,
        "pathogens": evaluated_risks,
        "ipm_advisory": (
            "Implement Integrated Pest Management (IPM): Conduct bi-weekly scouting, "
            "maintain field sanitation, remove weed reservoirs, and apply bio-control agents prior to chemical intervention."
        )
    }
