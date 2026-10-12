"""
CropMind AI: Real-Time Extreme Weather & Climate Anomaly Early Warning System
Detects thermal, moisture, and precipitation stress anomalies and generates emergency agronomic mitigation protocols.
"""

from typing import Dict, Any, List, Optional
import math

# Threshold limits per crop class
CROP_CLIMATE_THRESHOLDS = {
    "Rice": {"heat_max": 38.0, "cold_min": 15.0, "drought_rh_min": 50.0, "flood_rain_max": 250.0, "frost_sensitive": False},
    "Maize": {"heat_max": 36.0, "cold_min": 10.0, "drought_rh_min": 40.0, "flood_rain_max": 180.0, "frost_sensitive": True},
    "Cotton": {"heat_max": 40.0, "cold_min": 12.0, "drought_rh_min": 35.0, "flood_rain_max": 150.0, "frost_sensitive": True},
    "Chickpea": {"heat_max": 32.0, "cold_min": 5.0, "drought_rh_min": 30.0, "flood_rain_max": 100.0, "frost_sensitive": True},
    "Banana": {"heat_max": 38.0, "cold_min": 12.0, "drought_rh_min": 55.0, "flood_rain_max": 280.0, "frost_sensitive": True},
    "Grapes": {"heat_max": 38.0, "cold_min": 4.0, "drought_rh_min": 35.0, "flood_rain_max": 120.0, "frost_sensitive": True},
    "Apple": {"heat_max": 30.0, "cold_min": -2.0, "drought_rh_min": 40.0, "flood_rain_max": 150.0, "frost_sensitive": False},
    "Pomegranate": {"heat_max": 42.0, "cold_min": 5.0, "drought_rh_min": 25.0, "flood_rain_max": 120.0, "frost_sensitive": False},
    "Coffee": {"heat_max": 32.0, "cold_min": 8.0, "drought_rh_min": 50.0, "flood_rain_max": 260.0, "frost_sensitive": True},
}

DEFAULT_THRESHOLDS = {"heat_max": 38.0, "cold_min": 8.0, "drought_rh_min": 35.0, "flood_rain_max": 200.0, "frost_sensitive": True}


def calculate_vpd(temperature_c: float, humidity_pct: float) -> float:
    """Calculates Vapor Pressure Deficit (VPD in kPa)."""
    es = 0.6108 * math.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    ea = es * (humidity_pct / 100.0)
    return round(max(0.05, es - ea), 2)


def evaluate_climate_anomalies(
    crop_name: str,
    temperature_c: float,
    humidity_pct: float,
    rainfall_14d_mm: float,
    wind_speed_kmh: float = 12.0
) -> Dict[str, Any]:
    """
    Scans forecast meteorological conditions for agricultural climate shock events.
    """
    crop_key = crop_name.strip().capitalize()
    limits = CROP_CLIMATE_THRESHOLDS.get(crop_key, DEFAULT_THRESHOLDS)
    vpd = calculate_vpd(temperature_c, humidity_pct)
    
    active_alerts = []
    
    # 1. Extreme Heatwave & High Thermal Stress
    if temperature_c >= limits["heat_max"]:
        severity = "Critical" if temperature_c >= (limits["heat_max"] + 3.0) else "Warning"
        active_alerts.append({
            "anomaly_type": "Heatwave / Thermal Burn",
            "severity": severity,
            "badge_color": "#EF4444" if severity == "Critical" else "#F59E0B",
            "condition": f"Temperature ({temperature_c:.1f}°C) exceeds critical biological threshold ({limits['heat_max']:.1f}°C).",
            "physiological_impact": "Pollen sterility, stomatal closure, premature fruit/flower drop, and thermal enzyme degradation.",
            "emergency_mitigation": (
                "1. Apply Kaolin Clay 3% or Potassium Silicate @ 2 mL/L to reflect solar radiation.\n"
                "2. Run light overhead micro-sprinklers for 15 minutes during solar noon (12:30 - 2:00 PM).\n"
                "3. Avoid daytime urea top-dressing to prevent ammonia leaf scorch."
            )
        })
        
    # 2. Frost & Low Temperature Shock
    if temperature_c <= limits["cold_min"]:
        severity = "Critical" if temperature_c <= (limits["cold_min"] - 3.0) else "Warning"
        active_alerts.append({
            "anomaly_type": "Frost / Cold Shock Anomaly",
            "severity": severity,
            "badge_color": "#38BDF8",
            "condition": f"Temperature ({temperature_c:.1f}°C) at or below critical chilling boundary ({limits['cold_min']:.1f}°C).",
            "physiological_impact": "Intracellular ice crystal formation, cellular membrane rupture, and blackening of tender terminal buds.",
            "emergency_mitigation": (
                "1. Irrigate field immediately before sundown; wet soil conducts heat 3x better than dry soil.\n"
                "2. Create controlled smoke smudges (straw/biomass) along windward field boundaries during early dawn (4:00 - 6:00 AM).\n"
                "3. Spray Thiophenate Methyl or Seaweed extract (2 mL/L) to enhance cryo-protection."
            )
        })

    # 3. Severe Moisture Deficit & High VPD Stress
    if rainfall_14d_mm < 15.0 and (humidity_pct < limits["drought_rh_min"] or vpd > 2.2):
        active_alerts.append({
            "anomaly_type": "Drought & Atmospheric Aridity",
            "severity": "Warning" if vpd < 3.0 else "Critical",
            "badge_color": "#F59E0B",
            "condition": f"High evaporative demand (VPD: {vpd:.2f} kPa, Precipitation: {rainfall_14d_mm:.1f} mm).",
            "physiological_impact": "Severe moisture stress, xylem cavitation, and rapid leaf senescence.",
            "emergency_mitigation": (
                "1. Spread 5-8 cm organic crop residue mulch (paddy straw/bagasse) to suppress soil evaporation.\n"
                "2. Spray anti-transpirant Salicylic Acid @ 100 ppm or Brassinolide (0.1 ppm).\n"
                "3. Shift fertigation strictly to night cycles (8:00 PM - 11:00 PM)."
            )
        })

    # 4. Torrential Rainfall & Waterlogging Inundation Risk
    if rainfall_14d_mm >= limits["flood_rain_max"]:
        active_alerts.append({
            "anomaly_type": "Excessive Precipitation & Waterlogging",
            "severity": "Critical" if rainfall_14d_mm >= (limits["flood_rain_max"] + 50.0) else "Warning",
            "badge_color": "#A78BFA",
            "condition": f"Rainfall sum ({rainfall_14d_mm:.0f} mm) exceeds field percolation capacity ({limits['flood_rain_max']:.0f} mm).",
            "physiological_impact": "Root hypoxia, root rot pathogen proliferation, and rapid nitrogen denitrification.",
            "emergency_mitigation": (
                "1. Excavate emergency cross-drainage furrows (depth 30 cm) to discharge standing surface water.\n"
                "2. Drench root zone with Trichoderma harzianum or Copper Oxychloride 0.25% post drainage.\n"
                "3. Apply foliar Potassium Nitrate (13-0-45) @ 1% once water recedes to restore vigor."
            )
        })

    overall_climate_health = "Safe / Nominal Conditions" if len(active_alerts) == 0 else ("Severe Weather Stress" if any(a["severity"] == "Critical" for a in active_alerts) else "Elevated Climate Stress")

    return {
        "crop": crop_key,
        "meteorological_state": {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "rainfall_14d_mm": rainfall_14d_mm,
            "vapor_pressure_deficit_kpa": vpd,
            "wind_speed_kmh": wind_speed_kmh
        },
        "overall_status": overall_climate_health,
        "total_active_alerts": len(active_alerts),
        "alerts": active_alerts,
        "general_preparedness_note": (
            "Climate risk alerts are computed using dynamic physiological tolerance thresholds. "
            "Execute listed mitigation protocols within 24 hours of forecast onset to minimize yield penalty."
        )
    }
