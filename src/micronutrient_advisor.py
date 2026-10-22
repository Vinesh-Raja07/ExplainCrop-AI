"""
CropMind AI: Secondary & Micronutrient Diagnostic & Foliar Correction Engine
Diagnoses Zinc (Zn), Iron (Fe), Boron (B), Sulfur (S), Manganese (Mn), and Copper (Cu) deficiencies
and prescribes precision chelated and salt formulations for soil and foliar delivery.
"""

from typing import Dict, Any, List, Optional

MICRONUTRIENT_DEFICIENCY_PROFILES = {
    "Zinc (Zn)": {
        "visual_symptoms": "Interveinal chlorosis on young/middle leaves, 'Khaira' rusty brown pigmentation in rice, rosette little leaf in cotton.",
        "soil_risk_conditions": "High soil pH (> 7.5), high phosphorus overload (P:Zn imbalance), waterlogged submerged soils.",
        "foliar_prescription": "Foliar spray of Zinc Sulfate (ZnSO4 21%) @ 5 g/L + Agricultural Lime (2.5 g/L) OR Zn-EDTA (12%) @ 1.5 g/L.",
        "soil_prescription": "Soil application of Zinc Sulfate @ 10 kg/acre at basal seedbed preparation (lasts 2 years).",
        "critical_threshold_ppm": 0.6
    },
    "Iron (Fe)": {
        "visual_symptoms": "Severe interveinal chlorosis of youngest emerging leaves; in acute cases, leaves turn ivory white (bleaching) without necrosis.",
        "soil_risk_conditions": "Calcareous alkaline soils (pH > 7.8) with high free CaCO3 converting soluble Fe2+ to insoluble Fe3+.",
        "foliar_prescription": "Foliar spray of Ferrous Sulfate (FeSO4 19%) @ 5 g/L + Citric Acid (1 g/L) OR Fe-EDDHA (6%) @ 1.0 g/L.",
        "soil_prescription": "Soil application of Fe-EDDHA @ 2 kg/acre (effective in high pH soils where inorganic FeSO4 is immobilized).",
        "critical_threshold_ppm": 4.5
    },
    "Boron (B)": {
        "visual_symptoms": "Brittle leaves, death of growing shoot tips, fruit cracking (pomegranate/citrus), hollow heart in pulses, poor pollen tube viability.",
        "soil_risk_conditions": "Leached sandy soils, low organic carbon (<0.5%), dry soil conditions during flowering.",
        "foliar_prescription": "Foliar spray of Solubor (Di-Sodium Octaborate Tetrahydrate 20% B) @ 1.5 g/L at pre-flowering and fruit set stages.",
        "soil_prescription": "Soil broadcast of Borax (10.5% B) @ 2.5 kg/acre once per year.",
        "critical_threshold_ppm": 0.5
    },
    "Sulfur (S)": {
        "visual_symptoms": "Uniform pale yellowing starting from upper younger leaves, stunted branching, poor oil synthesis in oilseeds and reduced nodulation in legumes.",
        "soil_risk_conditions": "Light coarse-textured soils with low organic matter, continuous use of sulfur-free fertilizers (Urea + DAP).",
        "foliar_prescription": "Foliar spray of Water Soluble Sulfur (WDG 80%) @ 3 g/L.",
        "soil_prescription": "Soil application of Agricultural Gypsum (CaSO4.2H2O 18% S) @ 50 kg/acre or Bentonite Sulfur 90% @ 10 kg/acre.",
        "critical_threshold_ppm": 10.0
    },
    "Manganese (Mn)": {
        "visual_symptoms": "Grey speck or marsh spot on leaves, brown necrotic specks surrounded by chlorotic halo along veins.",
        "soil_risk_conditions": "Very high soil pH or over-limed soils with high organic matter oxidizing Mn2+ to unabsorbable forms.",
        "foliar_prescription": "Foliar spray of Manganese Sulfate (MnSO4 30%) @ 3 g/L.",
        "soil_prescription": "Soil application of Manganese Sulfate @ 5 kg/acre.",
        "critical_threshold_ppm": 2.0
    },
    "Copper (Cu)": {
        "visual_symptoms": "Exanthema / die-back of young twigs in citrus/orchards, gum pockets under bark, bleaching of cereal leaf tips ('white tip disease').",
        "soil_risk_conditions": "Peat and high organic matter soils (>5%) where copper is strongly bound to humic complexes.",
        "foliar_prescription": "Foliar spray of Copper Oxychloride 50% WP @ 2.5 g/L OR Cu-EDTA @ 1.0 g/L.",
        "soil_prescription": "Soil application of Copper Sulfate (CuSO4) @ 2 kg/acre.",
        "critical_threshold_ppm": 0.2
    }
}

CROP_MICRONUTRIENT_SENSITIVITY: Dict[str, List[str]] = {
    "Rice": ["Zinc (Zn)", "Iron (Fe)"],
    "Wheat": ["Zinc (Zn)", "Copper (Cu)", "Boron (B)"],
    "Maize": ["Zinc (Zn)", "Iron (Fe)", "Manganese (Mn)"],
    "Cotton": ["Zinc (Zn)", "Boron (B)", "Magnesium (Mg)"],
    "Sugarcane": ["Iron (Fe)", "Zinc (Zn)", "Sulfur (S)"],
    "Groundnut": ["Sulfur (S)", "Iron (Fe)", "Boron (B)"],
    "Soybean": ["Iron (Fe)", "Zinc (Zn)", "Sulfur (S)"],
    "Pigeonpea": ["Boron (B)", "Sulfur (S)", "Zinc (Zn)"],
    "Chickpea": ["Iron (Fe)", "Boron (B)", "Sulfur (S)"],
    "Mungbean": ["Zinc (Zn)", "Iron (Fe)"],
    "Blackgram": ["Zinc (Zn)", "Iron (Fe)"],
    "Lentil": ["Zinc (Zn)", "Boron (B)"],
    "Pomegranate": ["Boron (B)", "Zinc (Zn)", "Iron (Fe)"],
    "Banana": ["Zinc (Zn)", "Boron (B)", "Iron (Fe)"],
    "Mango": ["Zinc (Zn)", "Boron (B)", "Copper (Cu)"],
    "Grapes": ["Iron (Fe)", "Zinc (Zn)", "Boron (B)"],
    "Apple": ["Zinc (Zn)", "Boron (B)", "Calcium (Ca)"],
    "Orange": ["Zinc (Zn)", "Iron (Fe)", "Copper (Cu)"],
    "Papaya": ["Boron (B)", "Zinc (Zn)"],
    "Coconut": ["Boron (B)", "Chlorine (Cl)"],
    "Jute": ["Zinc (Zn)", "Sulfur (S)"],
    "Coffee": ["Zinc (Zn)", "Boron (B)", "Iron (Fe)"],
}


def diagnose_micronutrient_deficiencies(
    crop_name: str,
    soil_ph: float,
    organic_matter_pct: float = 0.75,
    soil_zn_ppm: Optional[float] = None,
    soil_fe_ppm: Optional[float] = None,
    soil_b_ppm: Optional[float] = None,
    soil_s_ppm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluates potential micronutrient risk factors based on soil pH, organic matter, and chemical test values.
    """
    crop_key = crop_name.strip().capitalize()
    diagnosed_elements = []
    
    # 1. Zinc Evaluation (High risk if pH > 7.4 or Zn < 0.6 ppm)
    zn_risk = "High Deficit Risk" if (soil_ph >= 7.6 or (soil_zn_ppm is not None and soil_zn_ppm < 0.6)) else ("Moderate Risk" if soil_ph >= 7.2 else "Low Risk / Adequate")
    diagnosed_elements.append({
        "nutrient": "Zinc (Zn)",
        "risk_level": zn_risk,
        "badge_color": "#EF4444" if "High" in zn_risk else ("#F59E0B" if "Moderate" in zn_risk else "#10B981"),
        "visual_symptoms": MICRONUTRIENT_DEFICIENCY_PROFILES["Zinc (Zn)"]["visual_symptoms"],
        "foliar_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Zinc (Zn)"]["foliar_prescription"],
        "soil_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Zinc (Zn)"]["soil_prescription"]
    })
    
    # 2. Iron Evaluation (High risk in alkaline calcareous soils pH > 7.8)
    fe_risk = "High Deficit Risk (Lime-induced Chlorosis)" if soil_ph >= 7.8 else ("Moderate Risk" if soil_ph >= 7.4 else "Low Risk / Adequate")
    diagnosed_elements.append({
        "nutrient": "Iron (Fe)",
        "risk_level": fe_risk,
        "badge_color": "#EF4444" if "High" in fe_risk else ("#F59E0B" if "Moderate" in fe_risk else "#10B981"),
        "visual_symptoms": MICRONUTRIENT_DEFICIENCY_PROFILES["Iron (Fe)"]["visual_symptoms"],
        "foliar_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Iron (Fe)"]["foliar_prescription"],
        "soil_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Iron (Fe)"]["soil_prescription"]
    })
    
    # 3. Boron Evaluation (High risk in low OM soils < 0.6% or sandy soils)
    b_risk = "High Deficit Risk" if (organic_matter_pct < 0.5 or (soil_b_ppm is not None and soil_b_ppm < 0.5)) else ("Moderate Risk" if organic_matter_pct < 0.8 else "Low Risk / Adequate")
    diagnosed_elements.append({
        "nutrient": "Boron (B)",
        "risk_level": b_risk,
        "badge_color": "#EF4444" if "High" in b_risk else ("#F59E0B" if "Moderate" in b_risk else "#10B981"),
        "visual_symptoms": MICRONUTRIENT_DEFICIENCY_PROFILES["Boron (B)"]["visual_symptoms"],
        "foliar_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Boron (B)"]["foliar_prescription"],
        "soil_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Boron (B)"]["soil_prescription"]
    })
    
    # 4. Sulfur Evaluation (High risk in low OM soils or if S < 10 ppm)
    s_risk = "High Deficit Risk" if (organic_matter_pct < 0.6 or (soil_s_ppm is not None and soil_s_ppm < 10.0)) else "Low Risk / Adequate"
    diagnosed_elements.append({
        "nutrient": "Sulfur (S)",
        "risk_level": s_risk,
        "badge_color": "#EF4444" if "High" in s_risk else "#10B981",
        "visual_symptoms": MICRONUTRIENT_DEFICIENCY_PROFILES["Sulfur (S)"]["visual_symptoms"],
        "foliar_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Sulfur (S)"]["foliar_prescription"],
        "soil_remedy": MICRONUTRIENT_DEFICIENCY_PROFILES["Sulfur (S)"]["soil_prescription"]
    })

    high_risk_count = sum(1 for e in diagnosed_elements if "High" in e["risk_level"])
    mod_risk_count = sum(1 for e in diagnosed_elements if "Moderate" in e["risk_level"])
    
    if high_risk_count > 0:
        overall_risk = "High"
    elif mod_risk_count > 0:
        overall_risk = "Moderate"
    else:
        overall_risk = "Low"

    sensitive_nutrients = CROP_MICRONUTRIENT_SENSITIVITY.get(crop_key, ["Zinc (Zn)", "Iron (Fe)", "Boron (B)"])
    crop_sensitivity_profile = {
        "crop": crop_key,
        "high_sensitivity_nutrients": ", ".join(sensitive_nutrients)
    }

    # Structured mapping for tab UI
    deficiencies_detected = {
        "zn": {
            "status": zn_risk,
            "critical_threshold_ppm": "0.6 ppm",
            "observed_ppm": f"{soil_zn_ppm:.2f} ppm" if soil_zn_ppm is not None else "pH Dependent (Alkalinity Gap)" if soil_ph >= 7.6 else "Adequate",
            "impact": MICRONUTRIENT_DEFICIENCY_PROFILES["Zinc (Zn)"]["visual_symptoms"]
        },
        "fe": {
            "status": fe_risk,
            "critical_threshold_ppm": "4.5 ppm",
            "observed_ppm": f"{soil_fe_ppm:.2f} ppm" if soil_fe_ppm is not None else "Lime-Induced Fixation" if soil_ph >= 7.8 else "Adequate",
            "impact": MICRONUTRIENT_DEFICIENCY_PROFILES["Iron (Fe)"]["visual_symptoms"]
        },
        "b": {
            "status": b_risk,
            "critical_threshold_ppm": "0.5 ppm",
            "observed_ppm": f"{soil_b_ppm:.2f} ppm" if soil_b_ppm is not None else "Low Organic Carbon" if organic_matter_pct < 0.5 else "Adequate",
            "impact": MICRONUTRIENT_DEFICIENCY_PROFILES["Boron (B)"]["visual_symptoms"]
        },
        "s": {
            "status": s_risk,
            "critical_threshold_ppm": "10.0 ppm",
            "observed_ppm": f"{soil_s_ppm:.2f} ppm" if soil_s_ppm is not None else "Sub-optimal SOM" if organic_matter_pct < 0.6 else "Adequate",
            "impact": MICRONUTRIENT_DEFICIENCY_PROFILES["Sulfur (S)"]["visual_symptoms"]
        }
    }

    recommended_prescriptions = []
    for elem in diagnosed_elements:
        if "High" in elem["risk_level"] or "Moderate" in elem["risk_level"]:
            recommended_prescriptions.append({
                "Target Nutrient": elem["nutrient"],
                "Deficiency Risk": elem["risk_level"],
                "Foliar Spray Prescription": elem["foliar_remedy"],
                "Soil Corrective Amendment": elem["soil_remedy"]
            })
    
    return {
        "crop": crop_key,
        "soil_conditions": {
            "soil_ph": soil_ph,
            "organic_matter_pct": organic_matter_pct
        },
        "overall_micronutrient_status": f"{high_risk_count} Micronutrient(s) at High Deficit Risk" if high_risk_count > 0 else "Micronutrient Levels Satisfactory",
        "overall_deficiency_risk": overall_risk,
        "crop_sensitivity_profile": crop_sensitivity_profile,
        "deficiencies_detected": deficiencies_detected,
        "recommended_prescriptions": recommended_prescriptions,
        "deficiency_profiles": diagnosed_elements,
        "foliar_spraying_guidance": (
            "Best spraying window: Early morning (7:00 AM - 9:30 AM) or late afternoon (4:00 PM - 6:00 PM). "
            "Always include an agricultural non-ionic wetting agent / surfactant @ 0.5 mL/L to maximize leaf cuticle penetration."
        )
    }
